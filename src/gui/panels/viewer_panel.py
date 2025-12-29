"""Live Viewer Panel for DadBot Traffic Monitor."""

import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import cv2
import numpy as np
import supervision as sv
from PIL import Image, ImageTk

from src.utils import get_logger
from src.config import (
    CalibrationConfig,
    DetectionConfig,
    TrackingConfig,
    VisualizationConfig,
    ZoneConfig,
    VEHICLE_CLASS_NAMES,
    VehicleClass,
)
from src.detector import VehicleDetector, DetectionResult
from src.gui.components import ScrollableFrame
from src.gui.styles import COLORS, FONTS
from src.settings import AppSettings, LeadCornerMode, LabelDisplayMode
from src.tracker import VehicleTracker
from src.utils import get_video_info
from src.visualizer import TrafficVisualizer

# Import InferencePipeline for real-time streaming
try:
    from inference import InferencePipeline
    from inference.core.interfaces.camera.entities import VideoFrame
    INFERENCE_PIPELINE_AVAILABLE = True
except ImportError:
    INFERENCE_PIPELINE_AVAILABLE = False
    InferencePipeline = None
    VideoFrame = None

log = get_logger("viewer")


class LiveViewerPanel(ttk.Frame):
    """Panel for live video preview with real-time configuration."""

    def __init__(
        self,
        parent,
        settings: AppSettings,
        on_change: Callable,
        **kwargs
    ):
        super().__init__(parent, style="Main.TFrame", **kwargs)

        self.settings = settings
        self.on_change = on_change

        self.video_path: str | None = None
        self.cap: cv2.VideoCapture | None = None
        self.running = False
        self.photo_image = None
        self.frame_delay = 33  # ~30fps
        self.total_frames = 0
        self.frame_index = 0

        # RTSP stream state
        self.is_rtsp_mode = False
        self.rtsp_url: str | None = None
        self.rtsp_reconnect_attempts = 0
        self.rtsp_max_reconnects = 5
        self.rtsp_reconnect_delay = 2000  # ms

        # ML pipeline components (lazy initialized)
        self.detector: VehicleDetector | None = None
        self.tracker: VehicleTracker | None = None
        self.visualizer: TrafficVisualizer | None = None
        self.video_info: sv.VideoInfo | None = None
        self._pipeline_initialized = False
        self._pipeline_error: str | None = None

        # Inference FPS cap - only run detection every N ms
        self._last_inference_time: float = 0.0
        self._last_tracking_result = None  # Cache last result for display

        # FPS tracking for display
        self._frame_times: list[float] = []
        self._fps_update_interval = 0.5  # Update FPS display every 0.5 seconds
        self._last_fps_update: float = 0.0
        self._current_fps: float = 0.0

        # Lens correction maps (cached for performance)
        self._lens_map1 = None
        self._lens_map2 = None
        self._lens_k1 = None
        self._lens_k2 = None
        self._lens_frame_size = None

        # InferencePipeline for real-time RTSP (much faster than HTTP per-frame)
        self._inference_pipeline: InferencePipeline | None = None
        self._pipeline_queue: queue.Queue = queue.Queue(maxsize=2)  # Small buffer
        self._pipeline_fps_monitor = sv.FPSMonitor()
        self._use_inference_pipeline = INFERENCE_PIPELINE_AVAILABLE

        # Detection controls list (populated in _create_detection_section)
        self._detection_controls: list = []

        self._create_layout()

    def _create_layout(self):
        """Create the panel layout."""
        # Header
        header = ttk.Frame(self, style="Main.TFrame")
        header.pack(fill="x", pady=(0, 15))

        ttk.Label(
            header,
            text="Live Viewer",
            style="Heading.TLabel",
        ).pack(side="left")

        # Main content
        content = ttk.Frame(self, style="Main.TFrame")
        content.pack(fill="both", expand=True)

        # Left side - Video preview
        preview_frame = ttk.Frame(content, style="Card.TFrame")
        preview_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Video canvas
        self.canvas = tk.Canvas(
            preview_frame,
            bg=COLORS["bg_dark"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        self.canvas.create_text(
            400, 300,
            text="Load a video to preview",
            font=FONTS["body"],
            fill=COLORS["text_muted"],
            tags="placeholder",
        )

        # Playback controls
        controls = ttk.Frame(preview_frame, style="CardInner.TFrame")
        controls.pack(fill="x", padx=10, pady=10)

        self.play_btn = ttk.Button(
            controls,
            text="▶ Play",
            command=self._toggle_playback,
        )
        self.play_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(
            controls,
            text="⏹ Stop",
            command=self._stop_video,
        )
        self.stop_btn.pack(side="left", padx=5)

        self.frame_label = tk.Label(
            controls,
            text="Frame: 0 / 0",
            font=FONTS["mono_small"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_medium"],
        )
        self.frame_label.pack(side="right", padx=10)

        # Right side - Configuration panel (scrollable)
        config_frame = ttk.Frame(content, style="Card.TFrame", width=320)
        config_frame.pack(side="right", fill="y")
        config_frame.pack_propagate(False)

        # Scrollable config area
        self.config_scroll = ScrollableFrame(config_frame, bg_color=COLORS["bg_medium"])
        self.config_scroll.pack(fill="both", expand=True)

        scrollable_frame = self.config_scroll.scrollable_frame

        # Config sections
        self._create_video_section(scrollable_frame)
        self._create_visualization_section(scrollable_frame)
        self._create_detection_section(scrollable_frame)

    def _create_video_section(self, parent):
        """Create video/RTSP source section."""
        section = ttk.LabelFrame(parent, text="Video Source", style="Card.TLabelframe")
        section.pack(fill="x", padx=10, pady=10)

        inner = ttk.Frame(section, style="CardInner.TFrame")
        inner.pack(fill="x", padx=10, pady=10)

        # File loading
        load_btn = ttk.Button(
            inner,
            text="📁 Load Video File",
            command=self._load_video,
        )
        load_btn.pack(fill="x", pady=5)

        ttk.Separator(inner, orient="horizontal").pack(fill="x", pady=10)

        # RTSP section
        ttk.Label(inner, text="RTSP Stream:", style="Body.TLabel").pack(anchor="w")

        self.rtsp_var = tk.StringVar(value=self.settings.last_rtsp_url or "")
        rtsp_entry = tk.Entry(
            inner,
            textvariable=self.rtsp_var,
            font=FONTS["small"],
            bg=COLORS["bg_dark"],
            fg=COLORS["text_primary"],
            insertbackground=COLORS["text_primary"],  # Cursor color
            relief="flat",
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["accent"],
        )
        rtsp_entry.pack(fill="x", pady=5, ipady=4)
        rtsp_entry.bind("<Return>", lambda e: self._connect_rtsp())

        # RTSP buttons
        rtsp_btns = ttk.Frame(inner, style="CardInner.TFrame")
        rtsp_btns.pack(fill="x", pady=5)

        self.rtsp_connect_btn = ttk.Button(
            rtsp_btns,
            text="▶ Connect",
            command=self._connect_rtsp,
        )
        self.rtsp_connect_btn.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.rtsp_disconnect_btn = ttk.Button(
            rtsp_btns,
            text="⏹ Disconnect",
            command=self._disconnect_rtsp,
            state="disabled",
        )
        self.rtsp_disconnect_btn.pack(side="left", fill="x", expand=True, padx=(2, 0))

        # Hint
        tk.Label(
            inner,
            text="e.g., rtsp://user:pass@192.168.1.100:554/stream",
            font=FONTS["small"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg_medium"],
        ).pack(anchor="w")

        ttk.Separator(inner, orient="horizontal").pack(fill="x", pady=10)

        # Status label
        self.video_info_label = tk.Label(
            inner,
            text="No source connected",
            font=FONTS["small"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg_medium"],
            wraplength=260,
        )
        self.video_info_label.pack(anchor="w", pady=5)

    def _create_visualization_section(self, parent):
        """Create visualization options section."""
        section = ttk.LabelFrame(parent, text="Visualization", style="Card.TLabelframe")
        section.pack(fill="x", padx=10, pady=10)

        inner = ttk.Frame(section, style="CardInner.TFrame")
        inner.pack(fill="x", padx=10, pady=10)

        # Lead corner mode
        ttk.Label(inner, text="Trail Mode:", style="Body.TLabel").pack(anchor="w")
        self.lead_mode_var = tk.StringVar(value=self.settings.visualization.lead_corner_mode.value)
        lead_combo = ttk.Combobox(
            inner,
            textvariable=self.lead_mode_var,
            values=[mode.value for mode in LeadCornerMode],
            state="readonly",
            font=FONTS["small"],
        )
        lead_combo.pack(fill="x", pady=5)
        lead_combo.bind("<<ComboboxSelected>>", self._on_lead_mode_change)

        # Label mode
        ttk.Label(inner, text="Label Display:", style="Body.TLabel").pack(anchor="w", pady=(10, 0))
        self.label_mode_var = tk.StringVar(value=self.settings.visualization.label_mode.value)
        label_combo = ttk.Combobox(
            inner,
            textvariable=self.label_mode_var,
            values=[mode.value for mode in LabelDisplayMode],
            state="readonly",
            font=FONTS["small"],
        )
        label_combo.pack(fill="x", pady=5)
        label_combo.bind("<<ComboboxSelected>>", self._on_label_mode_change)

        ttk.Separator(inner, orient="horizontal").pack(fill="x", pady=10)

        # Toggle options
        self.show_boxes_var = tk.BooleanVar(value=self.settings.visualization.show_bounding_boxes)
        ttk.Checkbutton(
            inner,
            text="Show bounding boxes",
            variable=self.show_boxes_var,
            command=self._on_vis_toggle,
            style="Modern.TCheckbutton",
        ).pack(anchor="w", pady=2)

        self.show_traces_var = tk.BooleanVar(value=self.settings.visualization.show_traces)
        ttk.Checkbutton(
            inner,
            text="Show trajectory traces",
            variable=self.show_traces_var,
            command=self._on_vis_toggle,
            style="Modern.TCheckbutton",
        ).pack(anchor="w", pady=2)

        self.show_labels_var = tk.BooleanVar(value=self.settings.visualization.show_labels)
        ttk.Checkbutton(
            inner,
            text="Show labels",
            variable=self.show_labels_var,
            command=self._on_vis_toggle,
            style="Modern.TCheckbutton",
        ).pack(anchor="w", pady=2)

        self.show_zone_var = tk.BooleanVar(value=self.settings.visualization.show_zone_overlay)
        ttk.Checkbutton(
            inner,
            text="Show zone overlay",
            variable=self.show_zone_var,
            command=self._on_vis_toggle,
            style="Modern.TCheckbutton",
        ).pack(anchor="w", pady=2)

        self.show_stats_var = tk.BooleanVar(value=self.settings.visualization.show_stats_overlay)
        ttk.Checkbutton(
            inner,
            text="Show stats overlay",
            variable=self.show_stats_var,
            command=self._on_vis_toggle,
            style="Modern.TCheckbutton",
        ).pack(anchor="w", pady=2)

        self.highlight_violations_var = tk.BooleanVar(value=self.settings.visualization.highlight_violations)
        ttk.Checkbutton(
            inner,
            text="Highlight speed violations",
            variable=self.highlight_violations_var,
            command=self._on_vis_toggle,
            style="Modern.TCheckbutton",
        ).pack(anchor="w", pady=2)

    def _create_detection_section(self, parent):
        """Create detection options section."""
        section = ttk.LabelFrame(parent, text="Detection", style="Card.TLabelframe")
        section.pack(fill="x", padx=10, pady=10)

        inner = ttk.Frame(section, style="CardInner.TFrame")
        inner.pack(fill="x", padx=10, pady=10)

        # Clear and populate detection controls list for enabling/disabling during stream
        self._detection_controls.clear()

        # Detection toggle
        self.detection_enabled_var = tk.BooleanVar(value=self.settings.detection.detection_enabled)
        detection_cb = ttk.Checkbutton(
            inner,
            text="Enable ML Detection",
            variable=self.detection_enabled_var,
            command=self._on_detection_toggle,
            style="Modern.TCheckbutton",
        )
        detection_cb.pack(anchor="w", pady=2)
        self._detection_controls.append(detection_cb)

        # Inference mode toggle (InferencePipeline vs OpenCV)
        self.use_inference_pipeline_var = tk.BooleanVar(value=self._use_inference_pipeline)
        pipeline_cb = ttk.Checkbutton(
            inner,
            text="Use InferencePipeline (real-time)",
            variable=self.use_inference_pipeline_var,
            command=self._on_pipeline_mode_toggle,
            style="Modern.TCheckbutton",
        )
        pipeline_cb.pack(anchor="w", pady=2)
        self._detection_controls.append(pipeline_cb)

        # Cloud inference toggle (only for OpenCV mode)
        self.cloud_inference_var = tk.BooleanVar(value=self.settings.detection.use_cloud_inference)
        self.cloud_inference_cb = ttk.Checkbutton(
            inner,
            text="Use Cloud GPU (OpenCV mode only)",
            variable=self.cloud_inference_var,
            command=self._on_cloud_toggle,
            style="Modern.TCheckbutton",
        )
        self.cloud_inference_cb.pack(anchor="w", pady=2)
        self._detection_controls.append(self.cloud_inference_cb)

        # Inference resolution dropdown
        res_frame = ttk.Frame(inner, style="CardInner.TFrame")
        res_frame.pack(anchor="w", pady=(5, 10))

        ttk.Label(res_frame, text="Inference Res:", style="Body.TLabel").pack(side="left")
        self.inference_res_var = tk.StringVar(value="640px max")
        res_combo = ttk.Combobox(
            res_frame,
            textvariable=self.inference_res_var,
            values=["320px max", "480px max", "640px max", "720px max", "960px max", "1080px max", "Original"],
            width=10,
            state="readonly",
        )
        res_combo.pack(side="left", padx=(5, 0))
        self._detection_controls.append(res_combo)

        # Hint about modes
        self.mode_hint_label = tk.Label(
            inner,
            text="InferencePipeline: Local model, optimized for RTSP\nOpenCV: Frame-by-frame, supports cloud GPU",
            font=FONTS["small"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg_medium"],
            justify="left",
        )
        self.mode_hint_label.pack(anchor="w", pady=(0, 5))

        ttk.Separator(inner, orient="horizontal").pack(fill="x", pady=5)

        # Confidence threshold
        ttk.Label(inner, text="Confidence Threshold:", style="Body.TLabel").pack(anchor="w")

        conf_frame = ttk.Frame(inner, style="CardInner.TFrame")
        conf_frame.pack(fill="x", pady=5)

        self.conf_var = tk.DoubleVar(value=self.settings.detection.confidence_threshold)
        self.conf_label = tk.Label(
            conf_frame,
            text=f"{self.conf_var.get():.2f}",
            font=FONTS["mono"],
            fg=COLORS["accent"],
            bg=COLORS["bg_medium"],
            width=5,
        )
        self.conf_label.pack(side="right")

        self.conf_scale = ttk.Scale(
            conf_frame,
            from_=0.1,
            to=0.9,
            variable=self.conf_var,
            orient="horizontal",
            command=self._on_conf_change,
        )
        self.conf_scale.pack(side="left", fill="x", expand=True)
        self._detection_controls.append(self.conf_scale)

        # Speed limit
        ttk.Label(inner, text="Speed Limit (MPH):", style="Body.TLabel").pack(anchor="w", pady=(10, 0))

        speed_frame = ttk.Frame(inner, style="CardInner.TFrame")
        speed_frame.pack(fill="x", pady=5)

        self.speed_var = tk.StringVar(value=str(int(self.settings.tracking.speed_limit_mph)))
        self.speed_entry = ttk.Entry(speed_frame, textvariable=self.speed_var, width=8, font=FONTS["mono"])
        self.speed_entry.pack(side="left")
        self.speed_entry.bind("<Return>", self._on_speed_change)
        self.speed_entry.bind("<FocusOut>", self._on_speed_change)

        ttk.Label(speed_frame, text="mph", style="Muted.TLabel").pack(side="left", padx=5)

        # Update cloud toggle state based on pipeline mode
        self._update_cloud_toggle_state()

    def _settings_to_configs(self) -> tuple[
        DetectionConfig, CalibrationConfig, TrackingConfig, VisualizationConfig, ZoneConfig
    ]:
        """Convert AppSettings to config objects for ML pipeline."""
        detection_config = DetectionConfig(
            model_id=self.settings.detection.model_id,
            confidence_threshold=self.settings.detection.confidence_threshold,
            iou_threshold=self.settings.detection.iou_threshold,
        )

        calibration_config = CalibrationConfig(
            reference_distance_feet=self.settings.calibration.reference_distance_feet,
            reference_pixel_start_x=self.settings.calibration.reference_pixel_start_x,
            reference_pixel_end_x=self.settings.calibration.reference_pixel_end_x,
            reference_pixel_y=self.settings.calibration.reference_pixel_y,
        )

        tracking_config = TrackingConfig(
            min_frames_for_speed=self.settings.tracking.min_frames_for_speed,
            track_buffer=self.settings.tracking.track_buffer,
            speed_limit_mph=self.settings.tracking.speed_limit_mph,
            commercial_vehicle_min_length_feet=self.settings.tracking.commercial_vehicle_min_length_feet,
        )

        vis_config = VisualizationConfig(
            show_preview=True,
            show_calibration_zone=self.settings.visualization.show_calibration_overlay,
            trace_length_seconds=self.settings.visualization.trace_length_seconds,
        )

        zone_config = ZoneConfig(
            enabled=self.settings.zone.enabled,
            polygon_points=self.settings.zone.polygon_points,
            show_zone=self.settings.visualization.show_zone_overlay,
        )

        return detection_config, calibration_config, tracking_config, vis_config, zone_config

    def _init_pipeline(self, fps: float) -> None:
        """Initialize the ML pipeline components."""
        log.info(f"Initializing ML pipeline at {fps:.1f} FPS")
        try:
            detection_cfg, calibration_cfg, tracking_cfg, vis_cfg, zone_cfg = self._settings_to_configs()

            # Initialize detector (use cloud GPU or local CPU)
            # Parse inference resolution (format: "640px max" or "Original")
            res_str = self.inference_res_var.get()
            if res_str == "Original":
                max_inference_dim = None
            else:
                max_inference_dim = int(res_str.split("px")[0])

            self.detector = VehicleDetector(
                config=detection_cfg,
                zone_config=zone_cfg,
                use_cloud=self.settings.detection.use_cloud_inference,
                max_inference_dim=max_inference_dim,
            )

            # Initialize tracker
            self.tracker = VehicleTracker(
                calibration_config=calibration_cfg,
                tracking_config=tracking_cfg,
                fps=fps,
            )

            # Initialize visualizer
            self.visualizer = TrafficVisualizer(
                video_info=self.video_info,
                vis_config=vis_cfg,
                calibration_config=calibration_cfg,
                tracking_config=tracking_cfg,
                zone_config=zone_cfg,
            )

            self._pipeline_initialized = True
            self._pipeline_error = None
            log.info("ML pipeline initialized successfully")

        except Exception as e:
            self._pipeline_initialized = False
            self._pipeline_error = str(e)
            log.error(f"ML pipeline initialization failed: {e}")

    def _reset_pipeline(self) -> None:
        """Reset the ML pipeline for a new video."""
        # Stop inference pipeline if running
        self._stop_inference_pipeline()

        self.detector = None
        self.tracker = None
        self.visualizer = None
        self._pipeline_initialized = False
        self._pipeline_error = None
        self.frame_index = 0
        # Reset inference cache
        self._last_inference_time = 0.0
        self._last_tracking_result = None
        # Reset FPS tracking
        self._frame_times = []
        self._last_fps_update = 0.0
        self._current_fps = 0.0
        # Reset lens correction cache
        self._lens_map1 = None
        self._lens_map2 = None
        # Clear pipeline queue
        while not self._pipeline_queue.empty():
            try:
                self._pipeline_queue.get_nowait()
            except queue.Empty:
                break
        # Reset display loop flag
        if hasattr(self, '_display_loop_started'):
            delattr(self, '_display_loop_started')

    def _apply_lens_correction(self, frame: np.ndarray) -> np.ndarray:
        """Apply lens distortion correction if enabled."""
        lens = self.settings.lens
        if not lens.enabled:
            return frame

        k1 = lens.distortion_k1
        k2 = lens.distortion_k2

        if k1 == 0 and k2 == 0:
            return frame

        h, w = frame.shape[:2]
        frame_size = (w, h)

        # Use cached maps if coefficients and frame size haven't changed
        if (self._lens_map1 is not None and
            self._lens_k1 == k1 and
            self._lens_k2 == k2 and
            self._lens_frame_size == frame_size):
            return cv2.remap(frame, self._lens_map1, self._lens_map2, cv2.INTER_LINEAR)

        # Build camera matrix
        cx = lens.center_x if lens.center_x else w / 2
        cy = lens.center_y if lens.center_y else h / 2
        fx = fy = max(w, h)

        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float32)

        # Get optimal new camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )

        # Compute and cache undistortion maps
        self._lens_map1, self._lens_map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
        )
        self._lens_k1 = k1
        self._lens_k2 = k2
        self._lens_frame_size = frame_size

        return cv2.remap(frame, self._lens_map1, self._lens_map2, cv2.INTER_LINEAR)

    # =========================================================================
    # InferencePipeline for Real-Time RTSP (runs locally, ~25+ FPS)
    # =========================================================================

    def _on_pipeline_prediction(self, result, frame) -> None:
        """Callback from InferencePipeline - runs in pipeline thread.

        This is called for each frame processed by the pipeline.
        We queue the result for the GUI thread to display.

        Note: After inference v0.9.18, result and frame can be lists for multi-source pipelines.
        """
        # Immediate log to confirm callback is invoked
        log.debug(f"_on_pipeline_prediction called: result_type={type(result)}, frame_type={type(frame)}")

        try:
            # Handle list format (v0.9.18+) - extract first item for single-source
            if isinstance(result, list):
                result = result[0] if result else {}
            if isinstance(frame, list):
                frame = frame[0] if frame else None

            if frame is None:
                log.warning("Received None frame in prediction callback")
                return

            self._pipeline_fps_monitor.tick()

            # Log frames periodically
            if self.frame_index == 0:
                log.info(f"First frame received! Shape: {frame.image.shape}")
                log.info(f"Result keys: {result.keys() if isinstance(result, dict) else type(result)}")
            elif self.frame_index % 100 == 0:
                log.debug(f"Pipeline processed {self.frame_index} frames, queue size: {self._pipeline_queue.qsize()}")
        except Exception as e:
            log.error(f"Error in prediction callback setup: {e}", exc_info=True)
            return

        try:
            # Get the frame image
            image = frame.image.copy()

            # Apply lens correction if enabled
            image = self._apply_lens_correction(image)

            # Convert inference result to sv.Detections
            detections = sv.Detections.from_inference(result)

            # Filter to vehicle classes only
            if len(detections) > 0:
                class_names = detections.data.get("class_name", [])
                vehicle_class_names = set(VEHICLE_CLASS_NAMES.keys())
                vehicle_mask = np.array([
                    str(name).lower() in vehicle_class_names
                    for name in class_names
                ])
                detections = detections[vehicle_mask]

            # Update tracker
            if self.tracker and len(detections) > 0:
                # Create a minimal DetectionResult for tracker
                vehicle_classes = [
                    VEHICLE_CLASS_NAMES.get(str(name).lower(), VehicleClass.CAR)
                    for name in detections.data.get("class_name", [])
                ]
                detection_result = DetectionResult(
                    detections=detections,
                    vehicle_classes=vehicle_classes,
                    frame_index=self.frame_index,
                )
                tracking_result = self.tracker.update(detection_result)
            else:
                tracking_result = None

            # Queue frame and result for GUI display (non-blocking)
            try:
                self._pipeline_queue.put_nowait({
                    "frame": image,
                    "detections": detections,
                    "tracking_result": tracking_result,
                    "fps": self._pipeline_fps_monitor.fps,
                    "frame_index": self.frame_index,
                })
            except queue.Full:
                # Drop frame if queue is full (keep latest)
                try:
                    self._pipeline_queue.get_nowait()
                    self._pipeline_queue.put_nowait({
                        "frame": image,
                        "detections": detections,
                        "tracking_result": tracking_result,
                        "fps": self._pipeline_fps_monitor.fps,
                        "frame_index": self.frame_index,
                    })
                except (queue.Empty, queue.Full):
                    pass

            self.frame_index += 1

        except Exception as e:
            log.error(f"Pipeline prediction error: {e}")

    def _start_inference_pipeline_async(self, rtsp_url: str) -> None:
        """Start the InferencePipeline in a background thread (non-blocking).

        This runs initialization in a thread since model download can be slow.
        """
        def _init_and_start():
            try:
                # Use RF-DETR model from settings
                model_id = self.settings.detection.model_id

                api_key = os.environ.get("ROBOFLOW_API_KEY")
                if not api_key:
                    log.error("ROBOFLOW_API_KEY not set")
                    self.after(0, lambda: self._on_pipeline_failed("ROBOFLOW_API_KEY not set"))
                    return

                # Check if model is cached
                from inference.core.env import MODEL_CACHE_DIR
                model_cache_path = os.path.join(MODEL_CACHE_DIR, model_id.replace("/", "--"))
                is_cached = os.path.exists(model_cache_path) and os.listdir(model_cache_path)

                if is_cached:
                    log.info(f"Model {model_id} found in cache")
                    self.after(0, lambda: self._update_pipeline_status("Loading cached model..."))
                else:
                    log.info(f"Downloading model {model_id} (first run)...")
                    self.after(0, lambda: self._update_pipeline_status("Downloading model (first run)..."))

                # Status handler to track pipeline events
                def on_status_update(status):
                    event_type = status.event_type
                    payload = status.payload
                    log.info(f"Pipeline event: {event_type} | {payload}")

                    if "ERROR" in event_type:
                        error_msg = payload.get("error", "Unknown error")
                        self.after(0, lambda e=error_msg: self._update_pipeline_status(f"Error: {e}"))
                    elif "VIDEO_SOURCE" in event_type:
                        self.after(0, lambda: self._update_pipeline_status("Video source ready"))
                    elif "INFERENCE_THREAD_STARTED" in event_type:
                        self.after(0, lambda: self._update_pipeline_status("Inference thread started"))

                # Wrap callback to ensure it works with InferencePipeline threading
                prediction_callback = self._on_pipeline_prediction

                def on_prediction_wrapper(result, frame):
                    """Wrapper to ensure callback invocation is logged."""
                    # Use print as fallback in case logging fails in thread
                    print(f"[PREDICTION] on_prediction called: result_type={type(result)}, frame_type={type(frame)}")
                    try:
                        prediction_callback(result, frame)
                    except Exception as e:
                        print(f"[PREDICTION ERROR] {e}")
                        log.error(f"Exception in prediction callback: {e}", exc_info=True)

                log.info(f"Creating InferencePipeline with model={model_id}, video={rtsp_url}")

                self._inference_pipeline = InferencePipeline.init(
                    model_id=model_id,
                    video_reference=rtsp_url,
                    on_prediction=on_prediction_wrapper,
                    confidence=self.settings.detection.confidence_threshold,
                    iou_threshold=self.settings.detection.iou_threshold,
                    api_key=api_key,
                    status_update_handlers=[on_status_update],
                )

                log.debug("Pipeline initialized, starting...")
                self.after(0, lambda: self._update_pipeline_status("Starting inference..."))
                self._inference_pipeline.start()
                log.info("InferencePipeline running")

                # Notify GUI thread that pipeline is ready
                self.after(0, self._on_pipeline_started)

            except Exception as e:
                log.error(f"Pipeline init failed: {e}", exc_info=True)
                self.after(0, lambda: self._on_pipeline_failed(str(e)))

        # Run in background thread
        thread = threading.Thread(target=_init_and_start, daemon=True)
        thread.start()

    def _update_pipeline_status(self, status: str) -> None:
        """Update the pipeline status display."""
        model_id = self.settings.detection.model_id
        self.video_info_label.configure(
            text=f"RTSP Stream (InferencePipeline)\nModel: {model_id}\nStatus: {status}",
            fg=COLORS["text_secondary"],
        )

    def _on_pipeline_started(self) -> None:
        """Called on GUI thread when pipeline starts successfully."""
        model_id = self.settings.detection.model_id
        log.info(f"Pipeline started ({model_id}), beginning display loop")
        log.info(f"Pipeline object exists: {self._inference_pipeline is not None}")
        log.info(f"Running state: {self.running}")
        self.video_info_label.configure(
            text=f"RTSP Stream (InferencePipeline)\nModel: {model_id}\nStatus: Running",
            fg=COLORS["success"],
        )
        self._pipeline_display_loop()

    def _on_pipeline_failed(self, error: str) -> None:
        """Called on GUI thread when pipeline fails to start."""
        log.warning(f"Pipeline failed: {error}, falling back to OpenCV")
        self._inference_pipeline = None

        # Fall back to OpenCV mode
        if self.rtsp_url:
            self._connect_rtsp_opencv_fallback()

    def _connect_rtsp_opencv_fallback(self) -> None:
        """Fallback to OpenCV-based RTSP capture."""
        url = self.rtsp_url
        log.info(f"Using OpenCV fallback for: {url}")

        try:
            self._set_rtsp_options()
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                raise ValueError("Could not connect to stream")

            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

            self.video_info = sv.VideoInfo(width=width, height=height, fps=fps, total_frames=0)
            self.frame_delay = int(1000 / fps)
            self._init_pipeline(fps)

            log.info(f"OpenCV connected: {width}x{height} @ {fps:.1f}fps")

            self.video_info_label.configure(
                text=f"RTSP Stream (OpenCV)\n{width}x{height} @ {fps:.1f}fps",
                fg=COLORS["warning"],
            )

            self._play_loop()

        except Exception as e:
            log.error(f"OpenCV fallback failed: {e}")
            self.video_info_label.configure(
                text=f"Connection failed:\n{e}",
                fg=COLORS["error"],
            )

    def _start_inference_pipeline(self, rtsp_url: str) -> bool:
        """Start the InferencePipeline for real-time RTSP inference.

        Returns True if pipeline initialization was started (async).
        """
        if not INFERENCE_PIPELINE_AVAILABLE:
            log.warning("InferencePipeline not available, falling back to HTTP")
            return False

        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            log.error("ROBOFLOW_API_KEY not set, cannot use InferencePipeline")
            return False

        # Start async initialization
        self._start_inference_pipeline_async(rtsp_url)
        return True

    def _stop_inference_pipeline(self) -> None:
        """Stop the running InferencePipeline."""
        if self._inference_pipeline is not None:
            try:
                log.info("Stopping InferencePipeline...")
                self._inference_pipeline.terminate()
                self._inference_pipeline.join(timeout=2.0)
            except Exception as e:
                log.warning(f"Error stopping pipeline: {e}")
            finally:
                self._inference_pipeline = None

    def _pipeline_display_loop(self) -> None:
        """GUI loop to display frames from the pipeline queue."""
        if not self.running:
            log.debug("Display loop stopped: not running")
            return

        if self._inference_pipeline is None:
            log.debug("Display loop stopped: no pipeline")
            return

        # Log first iteration
        if not hasattr(self, '_display_loop_started'):
            self._display_loop_started = True
            log.info("Display loop started, waiting for frames...")

        try:
            # Get frame from queue (non-blocking)
            data = self._pipeline_queue.get_nowait()

            frame = data["frame"]
            tracking_result = data["tracking_result"]
            fps = data["fps"]
            frame_idx = data["frame_index"]

            # Log periodically
            if frame_idx % 100 == 0:
                log.debug(f"Display loop got frame {frame_idx}, shape: {frame.shape}")

            # Annotate frame if we have tracking results
            if tracking_result and self.visualizer:
                frame = self.visualizer.annotate_frame(frame, tracking_result)
            elif self.show_zone_var.get() and self.settings.zone.polygon_points:
                frame = self._draw_zone_overlay(frame)

            # Update FPS display
            self._current_fps = fps
            self.frame_label.configure(
                text=f"LIVE | {fps:.1f} FPS | Frame: {frame_idx}"
            )

            # Display frame
            self._display_frame(frame)

        except queue.Empty:
            # No frame available yet - this is normal while waiting
            pass
        except Exception as e:
            log.error(f"Display loop error: {e}", exc_info=True)

        # Schedule next update (~60 FPS display rate)
        if self.running and self._inference_pipeline is not None:
            self.after(16, self._pipeline_display_loop)
        else:
            log.debug("Display loop ending")

    def _display_frame(self, frame: np.ndarray) -> None:
        """Display a frame on the canvas."""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1:
            log.warning(f"Canvas not ready: {canvas_w}x{canvas_h}")
            return

        h, w = frame.shape[:2]
        scale = min(canvas_w / w, canvas_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))

        # Convert and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        self.photo_image = ImageTk.PhotoImage(image)

        self.canvas.delete("all")
        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=self.photo_image)
        self.canvas.update_idletasks()

    def _load_video(self):
        """Load a video file."""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("All files", "*.*"),
        ]

        initial_dir = None
        if self.settings.last_video_path:
            initial_dir = str(Path(self.settings.last_video_path).parent)

        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes,
            initialdir=initial_dir,
        )

        if not path:
            return

        log.info(f"Loading video: {path}")
        try:
            self._stop_video()
            self._reset_pipeline()
            self.is_rtsp_mode = False

            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise ValueError("Could not open video")

            self.video_path = path
            info = get_video_info(path)
            log.debug(f"Video info: {info['width']}x{info['height']} @ {info['fps']:.1f}fps, "
                      f"{info['total_frames']} frames")

            self.video_info_label.configure(
                text=f"{Path(path).name}\n{info['width']}x{info['height']} @ {info['fps']:.1f}fps\n{info['total_frames']} frames",
                fg=COLORS["text_primary"],
            )

            self.frame_delay = int(1000 / info['fps'])
            self.total_frames = info['total_frames']
            fps = info['fps']

            # Create VideoInfo for visualizer
            self.video_info = sv.VideoInfo(
                width=info['width'],
                height=info['height'],
                fps=fps,
                total_frames=info['total_frames'],
            )

            # Initialize ML pipeline
            self._init_pipeline(fps)

            if self._pipeline_error:
                self.video_info_label.configure(
                    text=f"{Path(path).name}\n{info['width']}x{info['height']} @ {fps:.1f}fps\nML: {self._pipeline_error}",
                    fg=COLORS["warning"],
                )

            self.settings.last_video_path = path
            self.on_change()

            # Show first frame
            self._show_frame()

        except Exception as e:
            log.error(f"Failed to load video: {e}")
            messagebox.showerror("Error", f"Failed to load video: {e}")

    def _set_rtsp_options(self):
        """Set FFmpeg options for RTSP streaming via environment variable."""
        # Use TCP transport with buffer for stable streaming
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|"
            "buffer_size;8192000|"
            "fflags;discardcorrupt"
        )

    def _clear_rtsp_options(self):
        """Clear FFmpeg RTSP options from environment."""
        if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
            del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]

    def _connect_rtsp(self):
        """Connect to an RTSP stream using InferencePipeline for real-time inference."""
        url = self.rtsp_var.get().strip()
        if not url:
            messagebox.showwarning("Warning", "Please enter an RTSP URL")
            return

        # Validate URL format
        if not url.startswith(("rtsp://", "rtsps://", "http://", "https://")):
            messagebox.showwarning(
                "Warning",
                "URL should start with rtsp://, rtsps://, http://, or https://"
            )
            return

        log.info(f"Connecting to RTSP stream: {url}")
        try:
            self._stop_video()
            self._reset_pipeline()
            self.is_rtsp_mode = True
            self.rtsp_url = url
            self.rtsp_reconnect_attempts = 0

            self.video_info_label.configure(
                text="Connecting to stream...",
                fg=COLORS["text_secondary"],
            )
            self.update_idletasks()

            detection_enabled = self.detection_enabled_var.get()

            # Try to use InferencePipeline for real-time inference (much faster)
            if detection_enabled and self._use_inference_pipeline:
                log.info("Using InferencePipeline for real-time inference (~25+ FPS)")

                # Initialize tracker and visualizer for pipeline
                fps = 30.0
                self.video_info = sv.VideoInfo(width=1920, height=1080, fps=fps, total_frames=0)
                self._init_pipeline(fps)

                # Start the inference pipeline (async - will call _on_pipeline_started when ready)
                if self._start_inference_pipeline(url):
                    # Update button states
                    self.rtsp_connect_btn.configure(state="disabled")
                    self.rtsp_disconnect_btn.configure(state="normal")

                    # Disable detection controls while streaming
                    self._set_detection_controls_state(False)

                    # Save URL for next session
                    self.settings.last_rtsp_url = url
                    self.on_change()

                    # Show loading status (pipeline starts async)
                    model_id = self.settings.detection.model_id
                    self.video_info_label.configure(
                        text=f"RTSP Stream (InferencePipeline)\nModel: {model_id}\nStatus: Loading model...",
                        fg=COLORS["text_secondary"],
                    )

                    # Mark as running - display loop starts when pipeline is ready
                    self.running = True
                    self.play_btn.configure(text="⏸ Pause")
                    return
                else:
                    log.warning("InferencePipeline not available, falling back to OpenCV")

            # Fallback: Use OpenCV for stream capture (slower, HTTP-based inference)
            log.info("Using OpenCV capture (fallback mode)")
            self._set_rtsp_options()

            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                raise ValueError("Could not connect to stream")

            # Get stream properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

            self.video_info = sv.VideoInfo(
                width=width,
                height=height,
                fps=fps,
                total_frames=0,
            )

            self.frame_delay = int(1000 / fps)
            self.total_frames = 0

            # Initialize ML pipeline
            self._init_pipeline(fps)

            log.info(f"RTSP connected (OpenCV): {width}x{height} @ {fps:.1f}fps")

            # Update UI
            mode = "OpenCV + HTTP" if detection_enabled else "OpenCV (no ML)"
            status = f"RTSP Stream\n{width}x{height} @ {fps:.1f}fps\nMode: {mode}"
            if self._pipeline_error:
                status += f"\nML: {self._pipeline_error}"
                self.video_info_label.configure(text=status, fg=COLORS["warning"])
            else:
                self.video_info_label.configure(text=status, fg=COLORS["success"])

            # Update button states
            self.rtsp_connect_btn.configure(state="disabled")
            self.rtsp_disconnect_btn.configure(state="normal")

            # Disable detection controls while streaming
            self._set_detection_controls_state(False)

            # Save URL for next session
            self.settings.last_rtsp_url = url
            self.on_change()

            # Start playback automatically
            self.running = True
            self.play_btn.configure(text="⏸ Pause")
            self._play_loop()

        except Exception as e:
            log.error(f"RTSP connection failed: {e}")
            self.is_rtsp_mode = False
            self.rtsp_url = None
            self._stop_inference_pipeline()
            self.video_info_label.configure(
                text=f"Connection failed:\n{e}",
                fg=COLORS["error"],
            )
            if self.cap:
                self.cap.release()
                self.cap = None

    def _disconnect_rtsp(self):
        """Disconnect from RTSP stream."""
        log.info("Disconnecting from RTSP stream")
        self._stop_video()
        self._stop_inference_pipeline()
        self._clear_rtsp_options()
        self.is_rtsp_mode = False
        self.rtsp_url = None
        self.rtsp_reconnect_attempts = 0

        # Update button states
        self.rtsp_connect_btn.configure(state="normal")
        self.rtsp_disconnect_btn.configure(state="disabled")

        # Re-enable detection controls
        self._set_detection_controls_state(True)
        self._update_cloud_toggle_state()

        self.video_info_label.configure(
            text="Disconnected",
            fg=COLORS["text_muted"],
        )

        # Clear the canvas
        self.canvas.delete("all")
        self.canvas.create_text(
            400, 300,
            text="Load a video or connect to RTSP stream",
            font=FONTS["body"],
            fill=COLORS["text_muted"],
            tags="placeholder",
        )

    def _attempt_rtsp_reconnect(self):
        """Attempt to reconnect to RTSP stream after failure."""
        if not self.is_rtsp_mode or not self.rtsp_url:
            return

        self.rtsp_reconnect_attempts += 1
        log.warning(f"RTSP stream interrupted, reconnect attempt "
                    f"{self.rtsp_reconnect_attempts}/{self.rtsp_max_reconnects}")

        if self.rtsp_reconnect_attempts > self.rtsp_max_reconnects:
            log.error(f"RTSP max reconnect attempts ({self.rtsp_max_reconnects}) exceeded")
            self.video_info_label.configure(
                text=f"Stream lost. Max reconnect attempts ({self.rtsp_max_reconnects}) exceeded.",
                fg=COLORS["error"],
            )
            self._disconnect_rtsp()
            return

        self.video_info_label.configure(
            text=f"Stream interrupted. Reconnecting... ({self.rtsp_reconnect_attempts}/{self.rtsp_max_reconnects})",
            fg=COLORS["warning"],
        )

        # Release old capture
        if self.cap:
            self.cap.release()
            self.cap = None

        # Reset tracker for fresh start
        if self.tracker:
            self.tracker.reset()
        self.frame_index = 0

        # Schedule reconnection attempt
        self.after(self.rtsp_reconnect_delay, self._do_rtsp_reconnect)

    def _do_rtsp_reconnect(self):
        """Perform the actual RTSP reconnection."""
        if not self.is_rtsp_mode or not self.rtsp_url:
            return

        try:
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                raise ValueError("Reconnection failed")

            # Reset reconnect counter on success
            self.rtsp_reconnect_attempts = 0
            log.info("RTSP stream reconnected successfully")

            self.video_info_label.configure(
                text=f"RTSP Stream (reconnected)\n{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
                fg=COLORS["success"],
            )

            # Resume playback
            self.running = True
            self.play_btn.configure(text="⏸ Pause")
            self._play_loop()

        except Exception:
            # Try again
            self._attempt_rtsp_reconnect()

    def _toggle_playback(self):
        """Toggle video playback."""
        if self.cap is None:
            return

        if self.running:
            self.running = False
            self.play_btn.configure(text="▶ Play")
        else:
            self.running = True
            self.play_btn.configure(text="⏸ Pause")
            self._play_loop()

    def _play_loop(self):
        """Video playback loop."""
        if not self.running or self.cap is None:
            return

        ret = self._show_frame()

        if ret:
            self.after(self.frame_delay, self._play_loop)
        else:
            if self.is_rtsp_mode:
                # RTSP stream failed - attempt reconnection
                self.running = False
                self._attempt_rtsp_reconnect()
            else:
                # Video ended, loop back to beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_index = 0
                # Reset tracker for fresh start
                if self.tracker:
                    self.tracker.reset()
                self.after(self.frame_delay, self._play_loop)

    def _show_frame(self) -> bool:
        """Show the current frame with full ML annotation pipeline."""
        if self.cap is None:
            return False

        ret, frame = self.cap.read()
        if not ret:
            return False

        # Apply lens correction if enabled
        frame = self._apply_lens_correction(frame)

        # Calculate and update FPS display
        current_time = time.time()
        self._frame_times.append(current_time)
        # Keep only last 30 frame times for rolling average
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)

        # Update FPS display periodically (not every frame to reduce overhead)
        if current_time - self._last_fps_update >= self._fps_update_interval:
            if len(self._frame_times) >= 2:
                time_span = self._frame_times[-1] - self._frame_times[0]
                if time_span > 0:
                    self._current_fps = (len(self._frame_times) - 1) / time_span
            self._last_fps_update = current_time

        # Update frame label with FPS
        if self.is_rtsp_mode:
            self.frame_label.configure(text=f"LIVE | {self._current_fps:.1f} FPS | Frame: {self.frame_index}")
        else:
            self.frame_label.configure(text=f"{self._current_fps:.1f} FPS | Frame: {self.frame_index} / {self.total_frames}")

        # Run ML pipeline if initialized and detection is enabled
        detection_enabled = self.detection_enabled_var.get()
        pipeline_ready = self._pipeline_initialized and self.detector and self.tracker and self.visualizer

        if detection_enabled and pipeline_ready:
            try:
                # Check if we should run inference (FPS cap)
                current_time = time.time()
                min_interval = 1.0 / self.settings.detection.max_inference_fps
                should_infer = (current_time - self._last_inference_time) >= min_interval

                if should_infer:
                    # Run detection and tracking
                    detection_result = self.detector.detect(frame, self.frame_index)
                    tracking_result = self.tracker.update(detection_result)
                    self._last_tracking_result = tracking_result
                    self._last_inference_time = current_time

                # Annotate frame with current or cached tracking result
                if self._last_tracking_result is not None:
                    frame = self.visualizer.annotate_frame(frame, self._last_tracking_result)
                elif self.show_zone_var.get() and self.settings.zone.polygon_points:
                    frame = self._draw_zone_overlay(frame)

            except Exception:
                # Fall back to simple zone overlay on error
                if self.show_zone_var.get() and self.settings.zone.polygon_points:
                    frame = self._draw_zone_overlay(frame)
        else:
            # No ML pipeline or detection disabled - just draw zone overlay if enabled
            if self.show_zone_var.get() and self.settings.zone.polygon_points:
                frame = self._draw_zone_overlay(frame)

        self.frame_index += 1

        # Resize to fit canvas
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w > 1 and canvas_h > 1:
            h, w = frame.shape[:2]
            scale = min(canvas_w / w, canvas_h / h, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

            # Convert and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            self.photo_image = ImageTk.PhotoImage(image)

            self.canvas.delete("all")
            x_offset = (canvas_w - new_w) // 2
            y_offset = (canvas_h - new_h) // 2
            self.canvas.create_image(x_offset, y_offset, anchor="nw", image=self.photo_image)
            self.canvas.update_idletasks()

        return True

    def _draw_zone_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw zone overlay on frame (fallback when ML pipeline not available)."""
        points = self.settings.zone.polygon_points
        if not points or len(points) < 3:
            return frame

        pts = np.array(points, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 255))
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        return frame

    def _stop_video(self):
        """Stop video playback."""
        self.running = False
        self.play_btn.configure(text="▶ Play")

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _on_lead_mode_change(self, event=None):
        """Handle lead corner mode change."""
        self.settings.visualization.lead_corner_mode = LeadCornerMode(self.lead_mode_var.get())
        self.on_change()

    def _on_label_mode_change(self, event=None):
        """Handle label mode change."""
        self.settings.visualization.label_mode = LabelDisplayMode(self.label_mode_var.get())
        self.on_change()

    def _on_vis_toggle(self):
        """Handle visualization toggle changes."""
        self.settings.visualization.show_bounding_boxes = self.show_boxes_var.get()
        self.settings.visualization.show_traces = self.show_traces_var.get()
        self.settings.visualization.show_labels = self.show_labels_var.get()
        self.settings.visualization.show_zone_overlay = self.show_zone_var.get()
        self.settings.visualization.show_stats_overlay = self.show_stats_var.get()
        self.settings.visualization.highlight_violations = self.highlight_violations_var.get()
        self.on_change()

    def _on_detection_toggle(self):
        """Handle detection enable/disable toggle."""
        enabled = self.detection_enabled_var.get()
        self.settings.detection.detection_enabled = enabled
        log.info(f"ML detection {'enabled' if enabled else 'disabled'}")
        # Clear cached results when toggling off
        if not enabled:
            self._last_tracking_result = None
        self.on_change()

    def _on_cloud_toggle(self):
        """Handle cloud/local inference toggle."""
        use_cloud = self.cloud_inference_var.get()
        self.settings.detection.use_cloud_inference = use_cloud
        mode = "cloud GPU" if use_cloud else "local CPU"
        log.info(f"Switching to {mode} inference")
        # Reinitialize detector with new setting
        if self._pipeline_initialized and self.video_info:
            self._reset_pipeline()
            self._init_pipeline(self.video_info.fps)
        self.on_change()

    def _on_pipeline_mode_toggle(self):
        """Handle InferencePipeline vs OpenCV mode toggle."""
        use_pipeline = self.use_inference_pipeline_var.get()
        self._use_inference_pipeline = use_pipeline
        mode = "InferencePipeline" if use_pipeline else "OpenCV"
        log.info(f"Switching to {mode} mode")
        self._update_cloud_toggle_state()
        self.on_change()

    def _update_cloud_toggle_state(self):
        """Update cloud toggle state based on pipeline mode."""
        if hasattr(self, 'cloud_inference_cb'):
            # Cloud GPU only works in OpenCV mode
            if self.use_inference_pipeline_var.get():
                self.cloud_inference_cb.configure(state="disabled")
            else:
                self.cloud_inference_cb.configure(state="normal")

    def _set_detection_controls_state(self, enabled: bool):
        """Enable or disable detection controls during streaming.

        Args:
            enabled: True to enable controls, False to disable.
        """
        state = "normal" if enabled else "disabled"
        for control in self._detection_controls:
            try:
                control.configure(state=state)
            except tk.TclError:
                pass  # Some widgets may not support state

        # Also update the hint label appearance
        if hasattr(self, 'mode_hint_label'):
            if enabled:
                self.mode_hint_label.configure(fg=COLORS["text_muted"])
            else:
                self.mode_hint_label.configure(fg=COLORS["text_muted"])

    def _on_conf_change(self, value):
        """Handle confidence threshold change."""
        self.conf_label.configure(text=f"{float(value):.2f}")
        self.settings.detection.confidence_threshold = float(value)
        self.on_change()

    def _on_speed_change(self, event=None):
        """Handle speed limit change."""
        try:
            speed = float(self.speed_var.get())
            if speed > 0:
                self.settings.tracking.speed_limit_mph = speed
                self.on_change()
        except ValueError:
            pass

    def destroy(self):
        """Clean up resources."""
        self._stop_video()
        self._clear_rtsp_options()
        if self.visualizer:
            self.visualizer.cleanup()
        self._reset_pipeline()
        super().destroy()
