"""Vehicle tracking and speed estimation for side-view footage.

This module handles multi-object tracking using ByteTrack and calculates
vehicle speeds based on horizontal movement across the frame. Unlike
front-facing cameras that track Y-axis movement, side-view footage
requires tracking X-axis movement.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque

import numpy as np
import supervision as sv
from collections import deque

from src.config import (
    CalibrationConfig,
    Direction,
    SpecialVehicleType,
    TrackingConfig,
    VehicleClass,
)
from src.detector import DetectionResult


@dataclass
class TrackedVehicle:
    """Represents a tracked vehicle with historical position data."""

    tracker_id: int
    vehicle_class: VehicleClass
    first_seen: datetime
    last_seen: datetime
    positions_x: Deque[float] = field(default_factory=lambda: deque(maxlen=120))
    positions_y: Deque[float] = field(default_factory=lambda: deque(maxlen=120))
    speeds_mph: list[float] = field(default_factory=list)
    direction: Direction | None = None
    is_commercial: bool = False
    special_type: SpecialVehicleType | None = None
    bbox_width_pixels: float = 0.0
    bbox_height_pixels: float = 0.0
    # Current bounding box for leading corner calculation
    current_bbox: tuple[float, float, float, float] | None = None

    @property
    def current_speed_mph(self) -> float | None:
        """Get the most recent calculated speed."""
        return self.speeds_mph[-1] if self.speeds_mph else None

    @property
    def average_speed_mph(self) -> float | None:
        """Get average speed across all measurements."""
        return sum(self.speeds_mph) / len(self.speeds_mph) if self.speeds_mph else None

    @property
    def max_speed_mph(self) -> float | None:
        """Get maximum recorded speed."""
        return max(self.speeds_mph) if self.speeds_mph else None

    def get_leading_corner(self) -> tuple[float, float] | None:
        """Get the leading corner position based on travel direction.

        For side-view footage:
        - Eastbound (moving right): leading corner is BOTTOM_RIGHT
        - Westbound (moving left): leading corner is BOTTOM_LEFT
        - Unknown direction: use BOTTOM_CENTER

        Returns:
            (x, y) coordinates of the leading corner, or None if no bbox.
        """
        if self.current_bbox is None:
            return None

        x1, y1, x2, y2 = self.current_bbox

        if self.direction == Direction.EASTBOUND:
            # Moving right - front of car is on the right
            return (x2, y2)  # Bottom-right
        elif self.direction == Direction.WESTBOUND:
            # Moving left - front of car is on the left
            return (x1, y2)  # Bottom-left
        else:
            # Unknown direction - use bottom center
            return ((x1 + x2) / 2, y2)


@dataclass
class TrackingResult:
    """Container for tracking results for a single frame."""

    detections: sv.Detections
    tracked_vehicles: dict[int, TrackedVehicle]
    frame_index: int
    labels: list[str]


class VehicleTracker:
    """Tracks vehicles across frames and estimates speed.

    Uses ByteTrack for multi-object tracking and calculates speed based on
    horizontal pixel movement converted to real-world distance using
    calibration settings.
    """

    def __init__(
        self,
        calibration_config: CalibrationConfig | None = None,
        tracking_config: TrackingConfig | None = None,
        fps: float = 30.0,
    ):
        """Initialize the tracker.

        Args:
            calibration_config: Calibration settings for pixel-to-feet conversion.
            tracking_config: Tracking behavior settings.
            fps: Video frame rate for time calculations.
        """
        self.calibration = calibration_config or CalibrationConfig()
        self.config = tracking_config or TrackingConfig()
        self.fps = fps

        # Initialize ByteTrack
        self.byte_track = sv.ByteTrack(
            frame_rate=int(fps),
            track_activation_threshold=0.25,
            lost_track_buffer=self.config.track_buffer,
            minimum_matching_threshold=0.8,
        )

        # Track history: tracker_id -> TrackedVehicle
        self.tracked_vehicles: dict[int, TrackedVehicle] = {}

        # Position history for speed calculation (X coordinates)
        self.position_history: dict[int, Deque[tuple[float, int]]] = defaultdict(
            lambda: deque(maxlen=int(fps * 2))  # 2 seconds of history
        )

        # Vehicle class mapping from last detection
        self._last_vehicle_classes: dict[int, VehicleClass] = {}

    def update(self, detection_result: DetectionResult) -> TrackingResult:
        """Update tracking with new detections.

        Args:
            detection_result: Detection result from the detector.

        Returns:
            TrackingResult with updated tracking information.
        """
        detections = detection_result.detections
        frame_index = detection_result.frame_index

        # Update tracks using ByteTrack
        if len(detections) > 0:
            detections = self.byte_track.update_with_detections(detections)

        # Update vehicle class mapping
        tracker_ids = detections.tracker_id if detections.tracker_id is not None else []
        for i, (tracker_id, vehicle_class) in enumerate(
            zip(tracker_ids, detection_result.vehicle_classes)
        ):
            if tracker_id is not None:
                self._last_vehicle_classes[tracker_id] = vehicle_class

        # Process each tracked detection
        self._update_tracked_vehicles(detections, frame_index)

        # Calculate speeds and generate labels
        labels = self._generate_labels(detections)

        return TrackingResult(
            detections=detections,
            tracked_vehicles=self.tracked_vehicles,
            frame_index=frame_index,
            labels=labels,
        )

    def _update_tracked_vehicles(
        self, detections: sv.Detections, frame_index: int
    ) -> None:
        """Update tracked vehicle records with new positions.

        Args:
            detections: Tracked detections from ByteTrack.
            frame_index: Current frame index.
        """
        if detections.tracker_id is None:
            return

        now = datetime.now()

        # Get anchor points (bottom center of bounding boxes)
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None:
                continue

            x, y = points[i]
            bbox = detections.xyxy[i]
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]

            # Get or create tracked vehicle record
            if tracker_id not in self.tracked_vehicles:
                vehicle_class = self._last_vehicle_classes.get(
                    tracker_id, VehicleClass.CAR
                )
                self.tracked_vehicles[tracker_id] = TrackedVehicle(
                    tracker_id=tracker_id,
                    vehicle_class=vehicle_class,
                    first_seen=now,
                    last_seen=now,
                )

            vehicle = self.tracked_vehicles[tracker_id]
            vehicle.last_seen = now
            vehicle.positions_x.append(x)
            vehicle.positions_y.append(y)
            vehicle.bbox_width_pixels = bbox_width
            vehicle.bbox_height_pixels = bbox_height
            vehicle.current_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])

            # Store position with frame index for speed calculation
            self.position_history[tracker_id].append((x, frame_index))

            # Determine direction FIRST (needed for speed calc context)
            self._determine_direction(vehicle)

            # Calculate speed if enough history
            self._calculate_speed(vehicle, tracker_id)

            # Check if commercial vehicle
            self._check_commercial_status(vehicle)

    def _calculate_speed(self, vehicle: TrackedVehicle, tracker_id: int) -> None:
        """Calculate speed based on horizontal movement.

        For side-view footage, speed is calculated from X-axis movement:
        - Get positions from recent history
        - Convert pixel distance to feet using calibration
        - Calculate speed in MPH

        Args:
            vehicle: The tracked vehicle record.
            tracker_id: Tracker ID for position history lookup.
        """
        history = self.position_history[tracker_id]

        if len(history) < self.config.min_frames_for_speed:
            return

        # Get oldest and newest positions from history
        oldest_x, oldest_frame = history[0]
        newest_x, newest_frame = history[-1]

        # Calculate pixel distance traveled (X-axis for side view)
        pixel_distance = abs(newest_x - oldest_x)

        # Convert pixels to feet using calibration
        feet_distance = pixel_distance / self.calibration.pixels_per_foot

        # Calculate time elapsed
        frames_elapsed = newest_frame - oldest_frame
        if frames_elapsed <= 0:
            return

        seconds_elapsed = frames_elapsed / self.fps

        if seconds_elapsed <= 0:
            return

        # Calculate speed: feet per second -> miles per hour
        # 1 mile = 5280 feet, 1 hour = 3600 seconds
        # mph = (feet / seconds) * (3600 / 5280) = fps * 0.6818
        feet_per_second = feet_distance / seconds_elapsed
        speed_mph = feet_per_second * 0.6818

        # Sanity check: ignore unrealistic speeds
        if 0 < speed_mph < 150:  # Max reasonable speed
            vehicle.speeds_mph.append(speed_mph)

    def _determine_direction(self, vehicle: TrackedVehicle) -> None:
        """Determine vehicle travel direction based on movement.

        Direction is determined by net horizontal movement:
        - Moving RIGHT (increasing X) = EASTBOUND
        - Moving LEFT (decreasing X) = WESTBOUND

        Uses a minimum movement threshold to avoid noise from stationary vehicles.

        Args:
            vehicle: The tracked vehicle record.
        """
        # Need at least 3 frames to determine direction
        if len(vehicle.positions_x) < 3:
            return

        # Use all available positions for more stable direction
        positions = list(vehicle.positions_x)
        first_x = positions[0]
        last_x = positions[-1]
        net_movement = last_x - first_x

        # Minimum pixel movement to confidently assign direction
        # Lower threshold = faster direction detection
        min_movement_pixels = 5

        if net_movement > min_movement_pixels:
            # Moving right (positive X direction) = EASTBOUND
            vehicle.direction = Direction.EASTBOUND
        elif net_movement < -min_movement_pixels:
            # Moving left (negative X direction) = WESTBOUND
            vehicle.direction = Direction.WESTBOUND
        # else: keep previous direction or None if movement is too small

    def _check_commercial_status(self, vehicle: TrackedVehicle) -> None:
        """Check if vehicle should be classified as commercial.

        Uses bounding box size as a proxy for vehicle length.

        Args:
            vehicle: The tracked vehicle record.
        """
        # Estimate vehicle length from bounding box width
        estimated_length_feet = (
            vehicle.bbox_width_pixels / self.calibration.pixels_per_foot
        )

        if estimated_length_feet >= self.config.commercial_vehicle_min_length_feet:
            vehicle.is_commercial = True
            vehicle.special_type = SpecialVehicleType.COMMERCIAL

    def _generate_labels(self, detections: sv.Detections) -> list[str]:
        """Generate display labels for tracked vehicles.

        Args:
            detections: Current frame detections.

        Returns:
            List of label strings for each detection.
        """
        labels = []

        if detections.tracker_id is None:
            return labels

        for tracker_id in detections.tracker_id:
            if tracker_id is None:
                labels.append("")
                continue

            vehicle = self.tracked_vehicles.get(tracker_id)
            if vehicle is None:
                labels.append(f"#{tracker_id}")
                continue

            # Build label
            parts = [f"#{tracker_id}"]

            # Add vehicle class
            parts.append(vehicle.vehicle_class.value.upper())

            # Add speed if available
            if vehicle.current_speed_mph is not None:
                speed = vehicle.current_speed_mph
                parts.append(f"{speed:.0f} MPH")

                # Add violation indicator
                if speed > self.config.speed_limit_mph:
                    parts.append("VIOLATION")

            # Add direction
            if vehicle.direction:
                parts.append(vehicle.direction.value[0].upper())  # E or W

            labels.append(" | ".join(parts))

        return labels

    def get_speed_violations(self) -> list[TrackedVehicle]:
        """Get all vehicles that exceeded the speed limit.

        Returns:
            List of vehicles with speed violations.
        """
        violations = []
        for vehicle in self.tracked_vehicles.values():
            if vehicle.max_speed_mph and vehicle.max_speed_mph > self.config.speed_limit_mph:
                violations.append(vehicle)
        return violations

    def get_vehicles_by_direction(self) -> dict[Direction, list[TrackedVehicle]]:
        """Group tracked vehicles by direction.

        Returns:
            Dictionary mapping direction to list of vehicles.
        """
        by_direction: dict[Direction, list[TrackedVehicle]] = {
            Direction.EASTBOUND: [],
            Direction.WESTBOUND: [],
        }

        for vehicle in self.tracked_vehicles.values():
            if vehicle.direction:
                by_direction[vehicle.direction].append(vehicle)

        return by_direction

    def reset(self) -> None:
        """Reset tracker state for new video or time window."""
        self.byte_track = sv.ByteTrack(
            frame_rate=int(self.fps),
            track_activation_threshold=0.25,
            lost_track_buffer=self.config.track_buffer,
            minimum_matching_threshold=0.8,
        )
        self.tracked_vehicles.clear()
        self.position_history.clear()
        self._last_vehicle_classes.clear()
