"""Visualization using Supervision library.

This module provides annotated video output with bounding boxes,
trajectory traces, labels, and optional calibration zone overlay.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from src.config import CalibrationConfig, TrackingConfig, VisualizationConfig
from src.tracker import TrackingResult


@dataclass
class AnnotationStyle:
    """Visual style settings computed from video resolution."""

    thickness: int
    text_scale: float
    text_thickness: int


class TrafficVisualizer:
    """Renders annotated video frames with vehicle tracking visualization.

    Uses Supervision library annotators for:
    - Bounding boxes around detected vehicles
    - Trajectory traces showing vehicle paths
    - Labels with tracker ID, class, speed, and direction
    - Optional calibration zone overlay
    """

    # Color palette for different states
    COLOR_NORMAL = sv.Color(0, 255, 0)  # Green
    COLOR_VIOLATION = sv.Color(0, 0, 255)  # Red
    COLOR_COMMERCIAL = sv.Color(255, 165, 0)  # Orange
    COLOR_CALIBRATION = sv.Color(255, 255, 0)  # Yellow

    def __init__(
        self,
        video_info: sv.VideoInfo,
        vis_config: VisualizationConfig | None = None,
        calibration_config: CalibrationConfig | None = None,
        tracking_config: TrackingConfig | None = None,
    ):
        """Initialize the visualizer.

        Args:
            video_info: Video metadata (resolution, fps).
            vis_config: Visualization settings.
            calibration_config: Calibration settings for zone overlay.
            tracking_config: Tracking settings for violation detection.
        """
        self.video_info = video_info
        self.vis_config = vis_config or VisualizationConfig()
        self.calibration = calibration_config or CalibrationConfig()
        self.tracking = tracking_config or TrackingConfig()

        # Calculate optimal annotation style for resolution
        self.style = self._compute_style()

        # Initialize annotators
        self._init_annotators()

        # Video sink for output
        self._video_sink: sv.VideoSink | None = None

    def _compute_style(self) -> AnnotationStyle:
        """Compute annotation style based on video resolution."""
        thickness = sv.calculate_optimal_line_thickness(
            resolution_wh=self.video_info.resolution_wh
        )
        text_scale = sv.calculate_optimal_text_scale(
            resolution_wh=self.video_info.resolution_wh
        )
        return AnnotationStyle(
            thickness=thickness,
            text_scale=text_scale,
            text_thickness=thickness,
        )

    def _init_annotators(self) -> None:
        """Initialize Supervision annotators."""
        # Bounding box annotator
        self.box_annotator = sv.BoxAnnotator(
            thickness=self.style.thickness,
        )

        # Label annotator
        self.label_annotator = sv.LabelAnnotator(
            text_scale=self.style.text_scale,
            text_thickness=self.style.text_thickness,
            text_position=sv.Position.TOP_LEFT,
            text_padding=5,
        )

        # Trajectory trace annotator
        trace_length = int(self.video_info.fps * self.vis_config.trace_length_seconds)
        self.trace_annotator = sv.TraceAnnotator(
            thickness=self.style.thickness,
            trace_length=trace_length,
            position=sv.Position.BOTTOM_CENTER,
        )

        # Round box annotator for highlighting violations
        self.round_box_annotator = sv.RoundBoxAnnotator(
            thickness=self.style.thickness + 2,
            roundness=0.3,
        )

    def start_video_output(self, output_path: Path | str) -> None:
        """Start video output sink.

        Args:
            output_path: Path for output video file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._video_sink = sv.VideoSink(
            target_path=str(output_path),
            video_info=self.video_info,
        )
        self._video_sink.__enter__()

    def stop_video_output(self) -> None:
        """Stop and close video output sink."""
        if self._video_sink:
            self._video_sink.__exit__(None, None, None)
            self._video_sink = None

    def annotate_frame(
        self,
        frame: np.ndarray,
        tracking_result: TrackingResult,
    ) -> np.ndarray:
        """Annotate a frame with tracking visualization.

        Args:
            frame: Original BGR frame.
            tracking_result: Tracking data for this frame.

        Returns:
            Annotated frame.
        """
        annotated = frame.copy()

        # Draw calibration zone if enabled
        if self.vis_config.show_calibration_zone:
            annotated = self._draw_calibration_zone(annotated)

        detections = tracking_result.detections
        labels = tracking_result.labels

        if len(detections) == 0:
            return annotated

        # Identify violations for special coloring
        violation_mask = self._get_violation_mask(tracking_result)

        # Draw trajectory traces
        annotated = self.trace_annotator.annotate(
            scene=annotated,
            detections=detections,
        )

        # Draw bounding boxes
        annotated = self.box_annotator.annotate(
            scene=annotated,
            detections=detections,
        )

        # Highlight violations with round boxes
        if any(violation_mask):
            violation_detections = detections[violation_mask]
            annotated = self.round_box_annotator.annotate(
                scene=annotated,
                detections=violation_detections,
            )

        # Draw labels
        annotated = self.label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels,
        )

        # Draw stats overlay
        annotated = self._draw_stats_overlay(annotated, tracking_result)

        return annotated

    def _get_violation_mask(self, tracking_result: TrackingResult) -> np.ndarray:
        """Create boolean mask for vehicles with speed violations.

        Args:
            tracking_result: Current tracking data.

        Returns:
            Boolean mask array.
        """
        detections = tracking_result.detections
        mask = np.zeros(len(detections), dtype=bool)

        if detections.tracker_id is None:
            return mask

        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None:
                continue

            vehicle = tracking_result.tracked_vehicles.get(tracker_id)
            if vehicle and vehicle.max_speed_mph:
                if vehicle.max_speed_mph > self.tracking.speed_limit_mph:
                    mask[i] = True

        return mask

    def _draw_calibration_zone(self, frame: np.ndarray) -> np.ndarray:
        """Draw calibration reference zone on frame.

        Args:
            frame: Frame to annotate.

        Returns:
            Frame with calibration zone overlay.
        """
        # Draw vertical reference lines
        y_top = 0
        y_bottom = frame.shape[0]

        # Left reference line
        cv2.line(
            frame,
            (self.calibration.reference_pixel_start_x, y_top),
            (self.calibration.reference_pixel_start_x, y_bottom),
            (0, 255, 255),  # Yellow
            2,
        )

        # Right reference line
        cv2.line(
            frame,
            (self.calibration.reference_pixel_end_x, y_top),
            (self.calibration.reference_pixel_end_x, y_bottom),
            (0, 255, 255),  # Yellow
            2,
        )

        # Horizontal reference line at road level
        cv2.line(
            frame,
            (self.calibration.reference_pixel_start_x, self.calibration.reference_pixel_y),
            (self.calibration.reference_pixel_end_x, self.calibration.reference_pixel_y),
            (0, 255, 255),  # Yellow
            2,
        )

        # Add calibration label
        label = f"Reference: {self.calibration.reference_distance_feet}ft"
        cv2.putText(
            frame,
            label,
            (self.calibration.reference_pixel_start_x + 10, self.calibration.reference_pixel_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.style.text_scale,
            (0, 255, 255),
            self.style.text_thickness,
        )

        return frame

    def _draw_stats_overlay(
        self, frame: np.ndarray, tracking_result: TrackingResult
    ) -> np.ndarray:
        """Draw statistics overlay on frame.

        Args:
            frame: Frame to annotate.
            tracking_result: Current tracking data.

        Returns:
            Frame with stats overlay.
        """
        # Count active tracks
        active_tracks = len(tracking_result.detections)

        # Count violations
        violations = sum(
            1 for v in tracking_result.tracked_vehicles.values()
            if v.max_speed_mph and v.max_speed_mph > self.tracking.speed_limit_mph
        )

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Draw stats text
        stats = [
            f"Active: {active_tracks}",
            f"Total tracked: {len(tracking_result.tracked_vehicles)}",
            f"Violations: {violations}",
            f"Limit: {self.tracking.speed_limit_mph} MPH",
        ]

        y_offset = 30
        for stat in stats:
            cv2.putText(
                frame,
                stat,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y_offset += 20

        return frame

    def write_frame(self, frame: np.ndarray) -> None:
        """Write frame to video output.

        Args:
            frame: Annotated frame to write.
        """
        if self._video_sink:
            self._video_sink.write_frame(frame)

    def show_frame(self, frame: np.ndarray, window_name: str = "Traffic Monitor") -> bool:
        """Display frame in preview window.

        Args:
            frame: Frame to display.
            window_name: Window title.

        Returns:
            False if user pressed 'q' to quit, True otherwise.
        """
        if not self.vis_config.show_preview:
            return True

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        return key != ord("q")

    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_video_output()
        cv2.destroyAllWindows()
