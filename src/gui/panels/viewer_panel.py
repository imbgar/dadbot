"""Live Viewer Panel for DadBot Traffic Monitor."""

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.gui.styles import COLORS, FONTS
from src.settings import AppSettings, LeadCornerMode, LabelDisplayMode
from src.utils import get_video_info


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

        # Right side - Configuration panel
        config_frame = ttk.Frame(content, style="Card.TFrame", width=320)
        config_frame.pack(side="right", fill="y")
        config_frame.pack_propagate(False)

        # Scrollable config area
        config_canvas = tk.Canvas(
            config_frame,
            bg=COLORS["bg_medium"],
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=config_canvas.yview)
        scrollable_frame = ttk.Frame(config_canvas, style="CardInner.TFrame")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: config_canvas.configure(scrollregion=config_canvas.bbox("all"))
        )

        config_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=300)
        config_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        config_canvas.pack(side="left", fill="both", expand=True)

        # Config sections
        self._create_video_section(scrollable_frame)
        self._create_visualization_section(scrollable_frame)
        self._create_detection_section(scrollable_frame)

    def _create_video_section(self, parent):
        """Create video loading section."""
        section = ttk.LabelFrame(parent, text="Video Source", style="Card.TLabelframe")
        section.pack(fill="x", padx=10, pady=10)

        inner = ttk.Frame(section, style="CardInner.TFrame")
        inner.pack(fill="x", padx=10, pady=10)

        load_btn = ttk.Button(
            inner,
            text="📁 Load Video",
            command=self._load_video,
        )
        load_btn.pack(fill="x", pady=5)

        self.video_info_label = tk.Label(
            inner,
            text="No video loaded",
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

            self.settings.last_video_path = path
            self.on_change()

            # Show first frame
            self._show_frame()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {e}")

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
            # Video ended, loop back
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.after(self.frame_delay, self._play_loop)

    def _show_frame(self) -> bool:
        """Show the current frame."""
        if self.cap is None:
            return False

        ret, frame = self.cap.read()
        if not ret:
            return False

        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.frame_label.configure(text=f"Frame: {current_frame} / {self.total_frames}")

        # Apply visualization settings (simplified - just zone overlay for now)
        if self.show_zone_var.get() and self.settings.zone.polygon_points:
            frame = self._draw_zone_overlay(frame)

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
        """Draw zone overlay on frame."""
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
        super().destroy()
