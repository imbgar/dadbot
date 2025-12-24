"""Zone Definition Panel for DadBot Traffic Monitor."""

import json
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


class ZoneDefinitionPanel(ttk.Frame):
    """Panel for defining the road detection zone."""

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
        self.display_frame: np.ndarray | None = None
        self.photo_image = None

        # Zone points being edited
        self.points: list[tuple[int, int]] = []
        self.scale_factor = 1.0

        # Load existing points from settings
        if settings.zone.polygon_points:
            self.points = [tuple(p) for p in settings.zone.polygon_points]

        self._create_layout()

    def _create_layout(self):
        """Create the panel layout."""
        # Header
        header = ttk.Frame(self, style="Main.TFrame")
        header.pack(fill="x", pady=(0, 15))

        ttk.Label(
            header,
            text="Zone Definition",
            style="Heading.TLabel",
        ).pack(side="left")

        # Main content - split view
        content = ttk.Frame(self, style="Main.TFrame")
        content.pack(fill="both", expand=True)

        # Left side - Canvas for zone drawing
        canvas_frame = ttk.Frame(content, style="Card.TFrame")
        canvas_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Instructions above canvas
        instructions = tk.Label(
            canvas_frame,
            text="Click to add points • Right-click to remove last point • Draw clockwise around the road",
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

        # Placeholder text
        self.canvas.create_text(
            400, 300,
            text="Load a video to define the zone",
            font=FONTS["body"],
            fill=COLORS["text_muted"],
            tags="placeholder",
        )

        # Bind mouse events
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Right side - Controls
        controls_frame = ttk.Frame(content, style="Card.TFrame", width=280)
        controls_frame.pack(side="right", fill="y")
        controls_frame.pack_propagate(False)

        controls_inner = ttk.Frame(controls_frame, style="CardInner.TFrame")
        controls_inner.pack(fill="both", expand=True, padx=15, pady=15)

        # Load video button
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
            wraplength=250,
        )
        self.video_label.pack(anchor="w", pady=5)

        ttk.Separator(controls_inner, orient="horizontal").pack(fill="x", pady=15)

        # Zone points info
        ttk.Label(
            controls_inner,
            text="Zone Points",
            style="Subheading.TLabel",
        ).pack(anchor="w", pady=(0, 10))

        self.points_label = tk.Label(
            controls_inner,
            text="0 points defined",
            font=FONTS["body"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_medium"],
        )
        self.points_label.pack(anchor="w")

        # Points listbox
        points_list_frame = ttk.Frame(controls_inner, style="CardInner.TFrame")
        points_list_frame.pack(fill="x", pady=10)

        self.points_listbox = tk.Listbox(
            points_list_frame,
            height=6,
            font=FONTS["mono_small"],
            bg=COLORS["bg_dark"],
            fg=COLORS["text_primary"],
            selectbackground=COLORS["accent"],
            borderwidth=0,
            highlightthickness=0,
        )
        self.points_listbox.pack(fill="x")

        ttk.Separator(controls_inner, orient="horizontal").pack(fill="x", pady=15)

        # Action buttons
        ttk.Label(
            controls_inner,
            text="Actions",
            style="Subheading.TLabel",
        ).pack(anchor="w", pady=(0, 10))

        clear_btn = ttk.Button(
            controls_inner,
            text="🗑️ Clear All Points",
            command=self._clear_points,
        )
        clear_btn.pack(fill="x", pady=3)

        save_btn = ttk.Button(
            controls_inner,
            text="💾 Save Zone",
            command=self._save_zone,
            style="Accent.TButton",
        )
        save_btn.pack(fill="x", pady=10)

        # Zone enabled checkbox
        self.zone_enabled_var = tk.BooleanVar(value=self.settings.zone.enabled)
        zone_cb = ttk.Checkbutton(
            controls_inner,
            text="Enable zone filtering",
            variable=self.zone_enabled_var,
            command=self._on_zone_toggle,
            style="Modern.TCheckbutton",
        )
        zone_cb.pack(anchor="w", pady=5)

        # Show overlay checkbox
        self.show_overlay_var = tk.BooleanVar(value=self.settings.zone.show_overlay)
        overlay_cb = ttk.Checkbutton(
            controls_inner,
            text="Show overlay in output",
            variable=self.show_overlay_var,
            command=self._on_overlay_toggle,
            style="Modern.TCheckbutton",
        )
        overlay_cb.pack(anchor="w", pady=5)

        # Update points display
        self._update_points_display()

    def _load_video(self):
        """Load a video file and extract first frame."""
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
            # Extract first frame
            self.original_frame = extract_first_frame(path)
            self.video_path = path

            # Get video info
            info = get_video_info(path)
            self.video_label.configure(
                text=f"{Path(path).name}\n{info['width']}x{info['height']} @ {info['fps']:.1f}fps",
                fg=COLORS["text_primary"],
            )

            # Save last video path
            self.settings.last_video_path = path
            self.on_change()

            # Display frame
            self._update_display()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {e}")

    def _update_display(self):
        """Update the canvas display with the current frame and zone."""
        if self.original_frame is None:
            return

        # Get canvas size
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1:
            # Canvas not ready yet
            self.after(100, self._update_display)
            return

        # Calculate scale to fit
        frame_h, frame_w = self.original_frame.shape[:2]
        scale_w = canvas_w / frame_w
        scale_h = canvas_h / frame_h
        self.scale_factor = min(scale_w, scale_h, 1.0)  # Don't upscale

        # Resize frame
        new_w = int(frame_w * self.scale_factor)
        new_h = int(frame_h * self.scale_factor)
        self.display_frame = cv2.resize(self.original_frame, (new_w, new_h))

        # Draw zone overlay
        display = self._draw_zone_overlay(self.display_frame.copy())

        # Convert to PhotoImage
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(display_rgb)
        self.photo_image = ImageTk.PhotoImage(image)

        # Update canvas
        self.canvas.delete("all")

        # Center the image
        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=self.photo_image)

        # Store offset for point calculations
        self._x_offset = x_offset
        self._y_offset = y_offset

    def _draw_zone_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw the zone polygon on the frame."""
        if not self.points:
            return frame

        # Scale points for display
        scaled_points = [
            (int(x * self.scale_factor), int(y * self.scale_factor))
            for x, y in self.points
        ]

        # Draw filled polygon with transparency
        if len(scaled_points) >= 3:
            overlay = frame.copy()
            pts = np.array(scaled_points, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 255))  # Yellow
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

            # Draw polygon outline
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        # Draw points
        for i, (x, y) in enumerate(scaled_points):
            # Color: green for first, red for last, blue for others
            if i == 0:
                color = (0, 255, 0)  # Green
            elif i == len(scaled_points) - 1:
                color = (0, 0, 255)  # Red
            else:
                color = (255, 0, 0)  # Blue

            cv2.circle(frame, (x, y), 8, color, -1)
            cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)

            # Draw point number
            cv2.putText(
                frame,
                str(i + 1),
                (x + 12, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        return frame

    def _on_left_click(self, event):
        """Handle left mouse click to add a point."""
        if self.original_frame is None:
            return

        # Convert canvas coordinates to original image coordinates
        x = int((event.x - self._x_offset) / self.scale_factor)
        y = int((event.y - self._y_offset) / self.scale_factor)

        # Check bounds
        h, w = self.original_frame.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            self.points.append((x, y))
            self._update_points_display()
            self._update_display()

    def _on_right_click(self, event):
        """Handle right mouse click to remove last point."""
        if self.points:
            self.points.pop()
            self._update_points_display()
            self._update_display()

    def _on_canvas_resize(self, event):
        """Handle canvas resize."""
        if self.original_frame is not None:
            self._update_display()

    def _update_points_display(self):
        """Update the points list display."""
        self.points_label.configure(text=f"{len(self.points)} points defined")

        self.points_listbox.delete(0, tk.END)
        for i, (x, y) in enumerate(self.points):
            self.points_listbox.insert(tk.END, f"  {i + 1}: ({x}, {y})")

    def _clear_points(self):
        """Clear all zone points."""
        if messagebox.askyesno("Confirm", "Clear all zone points?"):
            self.points = []
            self._update_points_display()
            self._update_display()

    def _save_zone(self):
        """Save the zone configuration."""
        if len(self.points) < 3:
            messagebox.showwarning("Warning", "Zone requires at least 3 points")
            return

        # Update settings
        self.settings.zone.polygon_points = [list(p) for p in self.points]
        self.settings.zone.enabled = self.zone_enabled_var.get()
        self.settings.zone.show_overlay = self.show_overlay_var.get()

        self.on_change()

        messagebox.showinfo("Success", f"Zone saved with {len(self.points)} points")

    def _on_zone_toggle(self):
        """Handle zone enabled checkbox toggle."""
        self.settings.zone.enabled = self.zone_enabled_var.get()
        self.on_change()

    def _on_overlay_toggle(self):
        """Handle overlay checkbox toggle."""
        self.settings.zone.show_overlay = self.show_overlay_var.get()
        self.on_change()
