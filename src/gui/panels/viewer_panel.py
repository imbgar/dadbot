"""Live Viewer Panel for DadBot Traffic Monitor."""

import os
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import cv2
import numpy as np
import supervision as sv
from PIL import Image, ImageTk

from src.config import (
    CalibrationConfig,
    DetectionConfig,
    TrackingConfig,
    VisualizationConfig,
    ZoneConfig,
)
from src.detector import VehicleDetector
from src.gui.components import ScrollableFrame
from src.gui.styles import COLORS, FONTS
from src.settings import AppSettings, LeadCornerMode, LabelDisplayMode
from src.tracker import VehicleTracker
from src.utils import get_video_info
from src.visualizer import TrafficVisualizer


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

        # Detection toggle
        self.detection_enabled_var = tk.BooleanVar(value=self.settings.detection.detection_enabled)
        ttk.Checkbutton(
            inner,
            text="Enable ML Detection",
            variable=self.detection_enabled_var,
            command=self._on_detection_toggle,
            style="Modern.TCheckbutton",
        ).pack(anchor="w", pady=(0, 10))

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

        conf_scale = ttk.Scale(
            conf_frame,
            from_=0.1,
            to=0.9,
            variable=self.conf_var,
            orient="horizontal",
            command=self._on_conf_change,
        )
        conf_scale.pack(side="left", fill="x", expand=True)

        # Speed limit
        ttk.Label(inner, text="Speed Limit (MPH):", style="Body.TLabel").pack(anchor="w", pady=(10, 0))

        speed_frame = ttk.Frame(inner, style="CardInner.TFrame")
        speed_frame.pack(fill="x", pady=5)

        self.speed_var = tk.StringVar(value=str(int(self.settings.tracking.speed_limit_mph)))
        speed_entry = ttk.Entry(speed_frame, textvariable=self.speed_var, width=8, font=FONTS["mono"])
        speed_entry.pack(side="left")
        speed_entry.bind("<Return>", self._on_speed_change)
        speed_entry.bind("<FocusOut>", self._on_speed_change)

        ttk.Label(speed_frame, text="mph", style="Muted.TLabel").pack(side="left", padx=5)

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
        try:
            detection_cfg, calibration_cfg, tracking_cfg, vis_cfg, zone_cfg = self._settings_to_configs()

            # Initialize detector
            self.detector = VehicleDetector(
                config=detection_cfg,
                zone_config=zone_cfg,
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

        except Exception as e:
            self._pipeline_initialized = False
            self._pipeline_error = str(e)

    def _reset_pipeline(self) -> None:
        """Reset the ML pipeline for a new video."""
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

        try:
            self._stop_video()
            self._reset_pipeline()
            self.is_rtsp_mode = False

            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise ValueError("Could not open video")

            self.video_path = path
            info = get_video_info(path)

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
        """Connect to an RTSP stream."""
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

            # Set FFmpeg options for TCP transport
            self._set_rtsp_options()

            # Connect to RTSP stream
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                raise ValueError("Could not connect to stream")

            # Get stream properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

            # Create VideoInfo for visualizer
            self.video_info = sv.VideoInfo(
                width=width,
                height=height,
                fps=fps,
                total_frames=0,  # Unknown for streams
            )

            self.frame_delay = int(1000 / fps)
            self.total_frames = 0

            # Initialize ML pipeline
            self._init_pipeline(fps)

            # Update UI
            status = f"RTSP Stream\n{width}x{height} @ {fps:.1f}fps"
            if self._pipeline_error:
                status += f"\nML: {self._pipeline_error}"
                self.video_info_label.configure(text=status, fg=COLORS["warning"])
            else:
                status += "\nML: Ready"
                self.video_info_label.configure(text=status, fg=COLORS["success"])

            # Update button states
            self.rtsp_connect_btn.configure(state="disabled")
            self.rtsp_disconnect_btn.configure(state="normal")

            # Save URL for next session
            self.settings.last_rtsp_url = url
            self.on_change()

            # Start playback automatically
            self.running = True
            self.play_btn.configure(text="⏸ Pause")
            self._play_loop()

        except Exception as e:
            self.is_rtsp_mode = False
            self.rtsp_url = None
            self.video_info_label.configure(
                text=f"Connection failed:\n{e}",
                fg=COLORS["error"],
            )
            if self.cap:
                self.cap.release()
                self.cap = None

    def _disconnect_rtsp(self):
        """Disconnect from RTSP stream."""
        self._stop_video()
        self._clear_rtsp_options()
        self.is_rtsp_mode = False
        self.rtsp_url = None
        self.rtsp_reconnect_attempts = 0

        # Update button states
        self.rtsp_connect_btn.configure(state="normal")
        self.rtsp_disconnect_btn.configure(state="disabled")

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

        if self.rtsp_reconnect_attempts > self.rtsp_max_reconnects:
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
        self.settings.detection.detection_enabled = self.detection_enabled_var.get()
        # Clear cached results when toggling off
        if not self.detection_enabled_var.get():
            self._last_tracking_result = None
        self.on_change()

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
