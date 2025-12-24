"""Calibration Panel for DadBot Traffic Monitor."""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.gui.styles import COLORS, FONTS
from src.settings import AppSettings
from src.utils import extract_first_frame, get_video_info


class CalibrationPanel(ttk.Frame):
    """Panel for calibrating distance measurements."""

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
        self.original_frame: np.ndarray | None = None
        self.photo_image = None

        # Calibration line points
        self.point1: tuple[int, int] | None = None
        self.point2: tuple[int, int] | None = None
        self.scale_factor = 1.0
        self._x_offset = 0
        self._y_offset = 0

        # Load existing calibration
        cal = settings.calibration
        if cal.reference_pixel_start_x != cal.reference_pixel_end_x:
            self.point1 = (cal.reference_pixel_start_x, cal.reference_pixel_y)
            self.point2 = (cal.reference_pixel_end_x, cal.reference_pixel_y)

        self._create_layout()

    def _create_layout(self):
        """Create the panel layout."""
        # Header
        header = ttk.Frame(self, style="Main.TFrame")
        header.pack(fill="x", pady=(0, 15))

        ttk.Label(
            header,
            text="Distance Calibration",
            style="Heading.TLabel",
        ).pack(side="left")

        # Main content
        content = ttk.Frame(self, style="Main.TFrame")
        content.pack(fill="both", expand=True)

        # Left side - Canvas
        canvas_frame = ttk.Frame(content, style="Card.TFrame")
        canvas_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Instructions
        instructions = tk.Label(
            canvas_frame,
            text="Click two points to define a reference distance (e.g., a vehicle's length)",
            font=FONTS["small"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_medium"],
        )
        instructions.pack(pady=10)

        # Canvas
        self.canvas = tk.Canvas(
            canvas_frame,
            bg=COLORS["bg_dark"],
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.canvas.create_text(
            400, 300,
            text="Load a video to calibrate distance",
            font=FONTS["body"],
            fill=COLORS["text_muted"],
            tags="placeholder",
        )

        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Right side - Controls
        controls_frame = ttk.Frame(content, style="Card.TFrame", width=300)
        controls_frame.pack(side="right", fill="y")
        controls_frame.pack_propagate(False)

        controls_inner = ttk.Frame(controls_frame, style="CardInner.TFrame")
        controls_inner.pack(fill="both", expand=True, padx=15, pady=15)

        # Load video
        ttk.Label(
            controls_inner,
            text="Video Source",
            style="Subheading.TLabel",
        ).pack(anchor="w", pady=(0, 10))

        load_btn = ttk.Button(
            controls_inner,
            text="📁 Load Video File",
            command=self._load_video,
            style="Action.TButton",
        )
        load_btn.pack(fill="x", pady=5)

        self.video_label = tk.Label(
            controls_inner,
            text="No video loaded",
            font=FONTS["small"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg_medium"],
            wraplength=270,
        )
        self.video_label.pack(anchor="w", pady=5)

        ttk.Separator(controls_inner, orient="horizontal").pack(fill="x", pady=15)

        # Measurement section
        ttk.Label(
            controls_inner,
            text="Reference Measurement",
            style="Subheading.TLabel",
        ).pack(anchor="w", pady=(0, 10))

        # Pixel distance (read-only)
        pixel_frame = ttk.Frame(controls_inner, style="CardInner.TFrame")
        pixel_frame.pack(fill="x", pady=5)

        ttk.Label(
            pixel_frame,
            text="Pixel Distance:",
            style="Body.TLabel",
        ).pack(side="left")

        self.pixel_label = tk.Label(
            pixel_frame,
            text="0 px",
            font=FONTS["mono"],
            fg=COLORS["accent"],
            bg=COLORS["bg_medium"],
        )
        self.pixel_label.pack(side="right")

        # Known distance input
        distance_frame = ttk.Frame(controls_inner, style="CardInner.TFrame")
        distance_frame.pack(fill="x", pady=5)

        ttk.Label(
            distance_frame,
            text="Known Distance (ft):",
            style="Body.TLabel",
        ).pack(side="left")

        self.distance_var = tk.StringVar(value=str(self.settings.calibration.reference_distance_feet))
        self.distance_entry = ttk.Entry(
            distance_frame,
            textvariable=self.distance_var,
            width=10,
            font=FONTS["mono"],
        )
        self.distance_entry.pack(side="right")

        # Calculated pixels per foot
        ppf_frame = ttk.Frame(controls_inner, style="CardInner.TFrame")
        ppf_frame.pack(fill="x", pady=5)

        ttk.Label(
            ppf_frame,
            text="Pixels per Foot:",
            style="Body.TLabel",
        ).pack(side="left")

        self.ppf_label = tk.Label(
            ppf_frame,
            text=f"{self._calculate_ppf():.2f}",
            font=FONTS["mono"],
            fg=COLORS["success"],
            bg=COLORS["bg_medium"],
        )
        self.ppf_label.pack(side="right")

        # Tips
        tips_frame = ttk.LabelFrame(
            controls_inner,
            text="Tips",
            style="Card.TLabelframe",
        )
        tips_frame.pack(fill="x", pady=15)

        tips_text = tk.Label(
            tips_frame,
            text="• Use a vehicle of known length\n"
                 "• Standard car: ~15 ft\n"
                 "• Pickup truck: ~18-22 ft\n"
                 "• Semi trailer: ~53 ft\n\n"
                 "Click two points on the\n"
                 "object ends to measure.",
            font=FONTS["small"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_medium"],
            justify="left",
        )
        tips_text.pack(padx=10, pady=10)

        ttk.Separator(controls_inner, orient="horizontal").pack(fill="x", pady=15)

        # Action buttons
        clear_btn = ttk.Button(
            controls_inner,
            text="🗑️ Clear Points",
            command=self._clear_points,
        )
        clear_btn.pack(fill="x", pady=3)

        save_btn = ttk.Button(
            controls_inner,
            text="💾 Save Calibration",
            command=self._save_calibration,
            style="Accent.TButton",
        )
        save_btn.pack(fill="x", pady=10)

        # Update display
        self._update_pixel_display()

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
            self.original_frame = extract_first_frame(path)
            self.video_path = path

            info = get_video_info(path)
            self.video_label.configure(
                text=f"{Path(path).name}\n{info['width']}x{info['height']} @ {info['fps']:.1f}fps",
                fg=COLORS["text_primary"],
            )

            self.settings.last_video_path = path
            self.on_change()

            self._update_display()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {e}")

    def _update_display(self):
        """Update the canvas display."""
        if self.original_frame is None:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1:
            self.after(100, self._update_display)
            return

        frame_h, frame_w = self.original_frame.shape[:2]
        scale_w = canvas_w / frame_w
        scale_h = canvas_h / frame_h
        self.scale_factor = min(scale_w, scale_h, 1.0)

        new_w = int(frame_w * self.scale_factor)
        new_h = int(frame_h * self.scale_factor)
        display = cv2.resize(self.original_frame, (new_w, new_h))

        # Draw calibration line
        display = self._draw_calibration_line(display)

        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(display_rgb)
        self.photo_image = ImageTk.PhotoImage(image)

        self.canvas.delete("all")

        self._x_offset = (canvas_w - new_w) // 2
        self._y_offset = (canvas_h - new_h) // 2
        self.canvas.create_image(self._x_offset, self._y_offset, anchor="nw", image=self.photo_image)

    def _draw_calibration_line(self, frame: np.ndarray) -> np.ndarray:
        """Draw the calibration line on the frame."""
        if self.point1:
            x1, y1 = int(self.point1[0] * self.scale_factor), int(self.point1[1] * self.scale_factor)
            cv2.circle(frame, (x1, y1), 10, (0, 255, 0), -1)
            cv2.circle(frame, (x1, y1), 10, (255, 255, 255), 2)
            cv2.putText(frame, "1", (x1 + 15, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.point2:
            x2, y2 = int(self.point2[0] * self.scale_factor), int(self.point2[1] * self.scale_factor)
            cv2.circle(frame, (x2, y2), 10, (0, 0, 255), -1)
            cv2.circle(frame, (x2, y2), 10, (255, 255, 255), 2)
            cv2.putText(frame, "2", (x2 + 15, y2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.point1 and self.point2:
            x1, y1 = int(self.point1[0] * self.scale_factor), int(self.point1[1] * self.scale_factor)
            x2, y2 = int(self.point2[0] * self.scale_factor), int(self.point2[1] * self.scale_factor)

            # Draw line with arrows
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

            # Draw distance label
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            pixel_dist = self._calculate_pixel_distance()
            cv2.putText(
                frame,
                f"{pixel_dist} px",
                (mid_x + 10, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        return frame

    def _on_click(self, event):
        """Handle mouse click."""
        if self.original_frame is None:
            return

        x = int((event.x - self._x_offset) / self.scale_factor)
        y = int((event.y - self._y_offset) / self.scale_factor)

        h, w = self.original_frame.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return

        if self.point1 is None:
            self.point1 = (x, y)
        elif self.point2 is None:
            self.point2 = (x, y)
        else:
            # Reset and start new measurement
            self.point1 = (x, y)
            self.point2 = None

        self._update_pixel_display()
        self._update_display()

    def _on_canvas_resize(self, event):
        """Handle canvas resize."""
        if self.original_frame is not None:
            self._update_display()

    def _calculate_pixel_distance(self) -> int:
        """Calculate the pixel distance between the two points."""
        if self.point1 is None or self.point2 is None:
            return 0
        dx = self.point2[0] - self.point1[0]
        dy = self.point2[1] - self.point1[1]
        return int((dx**2 + dy**2) ** 0.5)

    def _calculate_ppf(self) -> float:
        """Calculate pixels per foot."""
        pixel_dist = self._calculate_pixel_distance()
        try:
            feet_dist = float(self.distance_var.get())
            if feet_dist > 0 and pixel_dist > 0:
                return pixel_dist / feet_dist
        except ValueError:
            pass
        return 0.0

    def _update_pixel_display(self):
        """Update the pixel distance display."""
        pixel_dist = self._calculate_pixel_distance()
        self.pixel_label.configure(text=f"{pixel_dist} px")
        self.ppf_label.configure(text=f"{self._calculate_ppf():.2f}")

    def _clear_points(self):
        """Clear calibration points."""
        self.point1 = None
        self.point2 = None
        self._update_pixel_display()
        if self.original_frame is not None:
            self._update_display()

    def _save_calibration(self):
        """Save the calibration settings."""
        if self.point1 is None or self.point2 is None:
            messagebox.showwarning("Warning", "Please define two calibration points")
            return

        try:
            feet_dist = float(self.distance_var.get())
            if feet_dist <= 0:
                raise ValueError("Distance must be positive")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid distance value: {e}")
            return

        # Update settings - store X coordinates for horizontal measurement
        x1, y1 = self.point1
        x2, y2 = self.point2

        self.settings.calibration.reference_distance_feet = feet_dist
        self.settings.calibration.reference_pixel_start_x = min(x1, x2)
        self.settings.calibration.reference_pixel_end_x = max(x1, x2)
        self.settings.calibration.reference_pixel_y = (y1 + y2) // 2

        self.on_change()

        ppf = self._calculate_ppf()
        messagebox.showinfo(
            "Success",
            f"Calibration saved!\n\n"
            f"Pixel distance: {self._calculate_pixel_distance()} px\n"
            f"Real distance: {feet_dist} ft\n"
            f"Pixels per foot: {ppf:.2f}"
        )
