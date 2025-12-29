"""Lens Calibration Panel for DadBot Traffic Monitor."""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.gui.components import ScrollableFrame
from src.gui.styles import COLORS, FONTS
from src.settings import AppSettings, CAMERA_PRESETS
from src.utils import extract_first_frame


class LensCalibrationPanel(ttk.Frame):
    """Panel for calibrating lens distortion correction."""

    def __init__(
        self,
        parent,
        settings: AppSettings,
        on_change: Callable,
        on_status: Callable[[str], None] = None,
        **kwargs
    ):
        super().__init__(parent, style="Main.TFrame", **kwargs)

        self.settings = settings
        self.on_change = on_change
        self.on_status = on_status

        self.video_path: str | None = None
        self.original_frame: np.ndarray | None = None
        self.corrected_frame: np.ndarray | None = None
        self.photo_image = None
        self.frame_height = 0
        self.frame_width = 0

        # Reference points for calibration line
        self.reference_points: list[tuple[int, int]] = []
        self.scale_factor = 1.0
        self._x_offset = 0
        self._y_offset = 0

        # Correction matrices (cached for performance)
        self._map1 = None
        self._map2 = None
        self._last_k1 = None
        self._last_k2 = None

        # Load existing reference points
        if self.settings.lens.reference_points:
            self.reference_points = [tuple(p) for p in self.settings.lens.reference_points]

        self._create_layout()

    def _create_layout(self):
        """Create the panel layout."""
        # Header
        header = ttk.Frame(self, style="Main.TFrame")
        header.pack(fill="x", pady=(0, 15))

        ttk.Label(
            header,
            text="Lens Calibration",
            style="Heading.TLabel",
        ).pack(side="left")

        # Main content
        content = ttk.Frame(self, style="Main.TFrame")
        content.pack(fill="both", expand=True)

        # Left side - Image preview
        preview_frame = ttk.Frame(content, style="Card.TFrame")
        preview_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Canvas for image
        self.canvas = tk.Canvas(
            preview_frame,
            bg=COLORS["bg_dark"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # Bind mouse events
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_motion)

        # Placeholder text
        self.canvas.create_text(
            400, 300,
            text="Load a video to calibrate lens distortion\n\nClick to place reference points along\na road edge that should be straight",
            font=FONTS["body"],
            fill=COLORS["text_muted"],
            tags="placeholder",
            justify="center",
        )

        # Right side - Controls (scrollable)
        controls_frame = ttk.Frame(content, style="Card.TFrame", width=320)
        controls_frame.pack(side="right", fill="y")
        controls_frame.pack_propagate(False)

        controls_scroll = ScrollableFrame(controls_frame, bg_color=COLORS["bg_medium"])
        controls_scroll.pack(fill="both", expand=True)

        scrollable = controls_scroll.scrollable_frame

        # Video source section
        self._create_source_section(scrollable)

        # Camera preset section
        self._create_preset_section(scrollable)

        # Distortion controls section
        self._create_distortion_section(scrollable)

        # Reference points section
        self._create_reference_section(scrollable)

        # Actions section
        self._create_actions_section(scrollable)

    def _create_source_section(self, parent):
        """Create video source section."""
        section = ttk.LabelFrame(parent, text="Video Source", style="Card.TLabelframe")
        section.pack(fill="x", padx=10, pady=10)

        inner = ttk.Frame(section, style="CardInner.TFrame")
        inner.pack(fill="x", padx=10, pady=10)

        load_btn = ttk.Button(
            inner,
            text="Load Video Frame",
            command=self._load_video,
        )
        load_btn.pack(fill="x", pady=5)

        self.source_label = tk.Label(
            inner,
            text="No video loaded",
            font=FONTS["small"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg_medium"],
            wraplength=260,
        )
        self.source_label.pack(anchor="w", pady=5)

    def _create_preset_section(self, parent):
        """Create camera preset section."""
        section = ttk.LabelFrame(parent, text="Camera Preset", style="Card.TLabelframe")
        section.pack(fill="x", padx=10, pady=10)

        inner = ttk.Frame(section, style="CardInner.TFrame")
        inner.pack(fill="x", padx=10, pady=10)

        # Preset dropdown
        ttk.Label(inner, text="Camera Model:", style="Body.TLabel").pack(anchor="w")

        preset_names = list(CAMERA_PRESETS.keys())
        preset_display = [p.replace("_", " ").title() for p in preset_names]

        self.preset_var = tk.StringVar(value=self.settings.lens.camera_preset)
        self.preset_combo = ttk.Combobox(
            inner,
            textvariable=self.preset_var,
            values=preset_names,
            state="readonly",
            font=FONTS["small"],
        )
        self.preset_combo.pack(fill="x", pady=5)
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_change)

        # Preset hint
        tk.Label(
            inner,
            text="Select your camera model for quick setup,\nor use 'custom' for manual adjustment",
            font=FONTS["small"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg_medium"],
            wraplength=260,
            justify="left",
        ).pack(anchor="w", pady=(5, 0))

    def _create_distortion_section(self, parent):
        """Create distortion controls section."""
        section = ttk.LabelFrame(parent, text="Distortion Correction", style="Card.TLabelframe")
        section.pack(fill="x", padx=10, pady=10)

        inner = ttk.Frame(section, style="CardInner.TFrame")
        inner.pack(fill="x", padx=10, pady=10)

        # Enable toggle
        self.enabled_var = tk.BooleanVar(value=self.settings.lens.enabled)
        ttk.Checkbutton(
            inner,
            text="Enable Lens Correction",
            variable=self.enabled_var,
            command=self._on_enabled_toggle,
            style="Modern.TCheckbutton",
        ).pack(anchor="w", pady=(0, 10))

        ttk.Separator(inner, orient="horizontal").pack(fill="x", pady=5)

        # Primary distortion slider (k1)
        ttk.Label(inner, text="Barrel/Pincushion (k1):", style="Body.TLabel").pack(anchor="w")

        k1_frame = ttk.Frame(inner, style="CardInner.TFrame")
        k1_frame.pack(fill="x", pady=5)

        self.k1_var = tk.DoubleVar(value=self.settings.lens.distortion_k1)
        self.k1_label = tk.Label(
            k1_frame,
            text=f"{self.k1_var.get():.3f}",
            font=FONTS["mono"],
            fg=COLORS["accent"],
            bg=COLORS["bg_medium"],
            width=7,
        )
        self.k1_label.pack(side="right")

        self.k1_scale = ttk.Scale(
            k1_frame,
            from_=-0.5,
            to=0.5,
            variable=self.k1_var,
            orient="horizontal",
            command=self._on_k1_change,
        )
        self.k1_scale.pack(side="left", fill="x", expand=True)

        # K1 hint labels
        hint_frame = ttk.Frame(inner, style="CardInner.TFrame")
        hint_frame.pack(fill="x")
        tk.Label(
            hint_frame,
            text="Barrel",
            font=FONTS["small"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg_medium"],
        ).pack(side="left")
        tk.Label(
            hint_frame,
            text="Pincushion",
            font=FONTS["small"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg_medium"],
        ).pack(side="right")

        ttk.Separator(inner, orient="horizontal").pack(fill="x", pady=10)

        # Secondary distortion slider (k2)
        ttk.Label(inner, text="Fine Adjustment (k2):", style="Body.TLabel").pack(anchor="w")

        k2_frame = ttk.Frame(inner, style="CardInner.TFrame")
        k2_frame.pack(fill="x", pady=5)

        self.k2_var = tk.DoubleVar(value=self.settings.lens.distortion_k2)
        self.k2_label = tk.Label(
            k2_frame,
            text=f"{self.k2_var.get():.3f}",
            font=FONTS["mono"],
            fg=COLORS["accent"],
            bg=COLORS["bg_medium"],
            width=7,
        )
        self.k2_label.pack(side="right")

        self.k2_scale = ttk.Scale(
            k2_frame,
            from_=-0.2,
            to=0.2,
            variable=self.k2_var,
            orient="horizontal",
            command=self._on_k2_change,
        )
        self.k2_scale.pack(side="left", fill="x", expand=True)

        ttk.Separator(inner, orient="horizontal").pack(fill="x", pady=10)

        # Preview toggle
        self.preview_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            inner,
            text="Show corrected preview",
            variable=self.preview_var,
            command=self._update_display,
            style="Modern.TCheckbutton",
        ).pack(anchor="w", pady=2)

        self.grid_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            inner,
            text="Show grid overlay",
            variable=self.grid_var,
            command=self._update_display,
            style="Modern.TCheckbutton",
        ).pack(anchor="w", pady=2)

    def _create_reference_section(self, parent):
        """Create reference points section."""
        section = ttk.LabelFrame(parent, text="Reference Line", style="Card.TLabelframe")
        section.pack(fill="x", padx=10, pady=10)

        inner = ttk.Frame(section, style="CardInner.TFrame")
        inner.pack(fill="x", padx=10, pady=10)

        tk.Label(
            inner,
            text="Click 3+ points along a road edge\nthat should be a straight line",
            font=FONTS["small"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_medium"],
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        self.points_label = tk.Label(
            inner,
            text="Points: 0",
            font=FONTS["mono"],
            fg=COLORS["accent"],
            bg=COLORS["bg_medium"],
        )
        self.points_label.pack(anchor="w")

        # Deviation indicator (how far from straight)
        self.deviation_label = tk.Label(
            inner,
            text="Deviation: --",
            font=FONTS["mono"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg_medium"],
        )
        self.deviation_label.pack(anchor="w", pady=(5, 0))

        ttk.Button(
            inner,
            text="Clear Points",
            command=self._clear_points,
        ).pack(fill="x", pady=(10, 0))

    def _create_actions_section(self, parent):
        """Create actions section."""
        section = ttk.LabelFrame(parent, text="Actions", style="Card.TLabelframe")
        section.pack(fill="x", padx=10, pady=10)

        inner = ttk.Frame(section, style="CardInner.TFrame")
        inner.pack(fill="x", padx=10, pady=10)

        ttk.Button(
            inner,
            text="Apply & Save",
            command=self._save_calibration,
            style="Accent.TButton",
        ).pack(fill="x", pady=2)

        ttk.Button(
            inner,
            text="Reset to Defaults",
            command=self._reset_calibration,
        ).pack(fill="x", pady=2)

    def _load_video(self):
        """Load a video file for calibration."""
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
            self.original_frame = extract_first_frame(path)
            if self.original_frame is None:
                raise ValueError("Could not extract frame")

            self.video_path = path
            self.frame_height, self.frame_width = self.original_frame.shape[:2]

            self.source_label.configure(
                text=f"{Path(path).name}\n{self.frame_width}x{self.frame_height}",
                fg=COLORS["text_primary"],
            )

            self.settings.last_video_path = path
            self.on_change()

            # Clear cached correction maps
            self._map1 = None
            self._map2 = None

            self._update_display()

        except Exception as e:
            self.source_label.configure(
                text=f"Error: {e}",
                fg=COLORS["error"],
            )

    def _on_preset_change(self, event=None):
        """Handle camera preset change."""
        preset = self.preset_var.get()
        if preset in CAMERA_PRESETS:
            k1, k2 = CAMERA_PRESETS[preset]
            self.k1_var.set(k1)
            self.k2_var.set(k2)
            self.k1_label.configure(text=f"{k1:.3f}")
            self.k2_label.configure(text=f"{k2:.3f}")

            self.settings.lens.camera_preset = preset
            self.settings.lens.distortion_k1 = k1
            self.settings.lens.distortion_k2 = k2

            # Clear cached maps to force recalculation
            self._map1 = None
            self._map2 = None

            self._update_display()
            self.on_change()

    def _on_enabled_toggle(self):
        """Handle enable/disable toggle."""
        self.settings.lens.enabled = self.enabled_var.get()
        self._update_display()
        self.on_change()

    def _on_k1_change(self, value):
        """Handle k1 slider change."""
        k1 = float(value)
        self.k1_label.configure(text=f"{k1:.3f}")
        self.settings.lens.distortion_k1 = k1
        self.settings.lens.camera_preset = "custom"
        self.preset_var.set("custom")

        # Clear cached maps
        self._map1 = None
        self._map2 = None

        self._update_display()

    def _on_k2_change(self, value):
        """Handle k2 slider change."""
        k2 = float(value)
        self.k2_label.configure(text=f"{k2:.3f}")
        self.settings.lens.distortion_k2 = k2
        self.settings.lens.camera_preset = "custom"
        self.preset_var.set("custom")

        # Clear cached maps
        self._map1 = None
        self._map2 = None

        self._update_display()

    def _on_canvas_click(self, event):
        """Handle canvas click to add reference point."""
        if self.original_frame is None:
            return

        # Convert canvas coords to image coords
        img_x = int((event.x - self._x_offset) / self.scale_factor)
        img_y = int((event.y - self._y_offset) / self.scale_factor)

        # Validate within bounds
        if 0 <= img_x < self.frame_width and 0 <= img_y < self.frame_height:
            self.reference_points.append((img_x, img_y))
            self.points_label.configure(text=f"Points: {len(self.reference_points)}")
            self._update_deviation()
            self._update_display()

    def _on_canvas_motion(self, event):
        """Handle mouse motion for coordinate display."""
        if self.original_frame is None:
            return

        img_x = int((event.x - self._x_offset) / self.scale_factor)
        img_y = int((event.y - self._y_offset) / self.scale_factor)

        if 0 <= img_x < self.frame_width and 0 <= img_y < self.frame_height:
            self.canvas.configure(cursor="crosshair")
        else:
            self.canvas.configure(cursor="arrow")

    def _clear_points(self):
        """Clear all reference points."""
        self.reference_points = []
        self.points_label.configure(text="Points: 0")
        self.deviation_label.configure(text="Deviation: --", fg=COLORS["text_muted"])
        self._update_display()

    def _update_deviation(self):
        """Calculate and display deviation from straight line."""
        if len(self.reference_points) < 3:
            self.deviation_label.configure(text="Deviation: --", fg=COLORS["text_muted"])
            return

        # Calculate deviation from best-fit line
        points = np.array(self.reference_points, dtype=np.float32)

        # Fit line using least squares
        if len(points) >= 2:
            vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

            # Calculate perpendicular distances from each point to the line
            distances = []
            for px, py in points:
                # Distance from point to line
                t = vx[0] * (px - x0[0]) + vy[0] * (py - y0[0])
                closest_x = x0[0] + t * vx[0]
                closest_y = y0[0] + t * vy[0]
                dist = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
                distances.append(dist)

            avg_deviation = np.mean(distances)
            max_deviation = np.max(distances)

            # Color code based on deviation
            if max_deviation < 5:
                color = COLORS["success"]
                status = "Excellent"
            elif max_deviation < 15:
                color = COLORS["accent"]
                status = "Good"
            elif max_deviation < 30:
                color = COLORS["warning"]
                status = "Fair"
            else:
                color = COLORS["error"]
                status = "Poor"

            self.deviation_label.configure(
                text=f"Deviation: {max_deviation:.1f}px ({status})",
                fg=color,
            )

    def _apply_correction(self, frame: np.ndarray) -> np.ndarray:
        """Apply lens distortion correction to frame."""
        k1 = self.k1_var.get()
        k2 = self.k2_var.get()

        if k1 == 0 and k2 == 0:
            return frame

        h, w = frame.shape[:2]

        # Use cached maps if coefficients haven't changed
        if self._map1 is not None and self._last_k1 == k1 and self._last_k2 == k2:
            return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)

        # Build camera matrix (assume principal point at center)
        cx = self.settings.lens.center_x if self.settings.lens.center_x else w / 2
        cy = self.settings.lens.center_y if self.settings.lens.center_y else h / 2
        fx = fy = max(w, h)  # Focal length approximation

        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Distortion coefficients [k1, k2, p1, p2, k3]
        dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float32)

        # Get optimal new camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )

        # Compute undistortion maps
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
        )
        self._last_k1 = k1
        self._last_k2 = k2

        return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)

    def _draw_grid(self, frame: np.ndarray) -> np.ndarray:
        """Draw a grid overlay on the frame."""
        h, w = frame.shape[:2]
        frame = frame.copy()

        # Draw grid lines
        grid_spacing = 50
        color = (100, 100, 100)

        for x in range(0, w, grid_spacing):
            cv2.line(frame, (x, 0), (x, h), color, 1)
        for y in range(0, h, grid_spacing):
            cv2.line(frame, (0, y), (w, y), color, 1)

        return frame

    def _update_display(self):
        """Update the canvas display."""
        if self.original_frame is None:
            return

        # Get display frame (corrected or original)
        if self.preview_var.get() and self.enabled_var.get():
            display_frame = self._apply_correction(self.original_frame.copy())
        else:
            display_frame = self.original_frame.copy()

        # Apply grid if enabled
        if self.grid_var.get():
            display_frame = self._draw_grid(display_frame)

        # Draw reference points and line
        if self.reference_points:
            # If previewing correction, transform the reference points too
            if self.preview_var.get() and self.enabled_var.get():
                transformed_points = self._transform_points(self.reference_points)
            else:
                transformed_points = self.reference_points

            # Draw the line connecting points
            if len(transformed_points) >= 2:
                pts = np.array(transformed_points, dtype=np.int32)
                cv2.polylines(display_frame, [pts], False, (0, 255, 255), 2)

            # Draw individual points
            for i, (x, y) in enumerate(transformed_points):
                cv2.circle(display_frame, (int(x), int(y)), 8, (0, 255, 0), -1)
                cv2.circle(display_frame, (int(x), int(y)), 8, (255, 255, 255), 2)
                cv2.putText(
                    display_frame,
                    str(i + 1),
                    (int(x) + 12, int(y) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        # Resize to fit canvas
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w > 1 and canvas_h > 1:
            h, w = display_frame.shape[:2]
            self.scale_factor = min(canvas_w / w, canvas_h / h, 1.0)
            new_w, new_h = int(w * self.scale_factor), int(h * self.scale_factor)
            display_frame = cv2.resize(display_frame, (new_w, new_h))

            # Convert and display
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            self.photo_image = ImageTk.PhotoImage(image)

            self.canvas.delete("all")
            self._x_offset = (canvas_w - new_w) // 2
            self._y_offset = (canvas_h - new_h) // 2
            self.canvas.create_image(
                self._x_offset, self._y_offset,
                anchor="nw",
                image=self.photo_image,
            )

    def _transform_points(self, points: list[tuple[int, int]]) -> list[tuple[float, float]]:
        """Transform points using current distortion correction."""
        if not points:
            return []

        k1 = self.k1_var.get()
        k2 = self.k2_var.get()

        if k1 == 0 and k2 == 0:
            return points

        h, w = self.frame_height, self.frame_width
        cx = self.settings.lens.center_x if self.settings.lens.center_x else w / 2
        cy = self.settings.lens.center_y if self.settings.lens.center_y else h / 2
        fx = fy = max(w, h)

        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float32)

        # Convert points to the format expected by undistortPoints
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

        # Undistort points
        undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)

        return [(p[0][0], p[0][1]) for p in undistorted]

    def _save_calibration(self):
        """Save the current calibration."""
        self.settings.lens.enabled = self.enabled_var.get()
        self.settings.lens.distortion_k1 = self.k1_var.get()
        self.settings.lens.distortion_k2 = self.k2_var.get()
        self.settings.lens.camera_preset = self.preset_var.get()
        self.settings.lens.reference_points = [list(p) for p in self.reference_points]

        self.on_change()

        if self.on_status:
            self.on_status("Lens calibration saved")

    def _reset_calibration(self):
        """Reset calibration to defaults."""
        self.settings.lens.enabled = False
        self.settings.lens.distortion_k1 = 0.0
        self.settings.lens.distortion_k2 = 0.0
        self.settings.lens.camera_preset = "custom"
        self.settings.lens.reference_points = []

        self.enabled_var.set(False)
        self.k1_var.set(0.0)
        self.k2_var.set(0.0)
        self.k1_label.configure(text="0.000")
        self.k2_label.configure(text="0.000")
        self.preset_var.set("custom")

        self._clear_points()
        self._map1 = None
        self._map2 = None

        self._update_display()
        self.on_change()

        if self.on_status:
            self.on_status("Lens calibration reset")
