"""Calibration Panel for DadBot Traffic Monitor."""

from datetime import datetime
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.gui.components import ScrollableFrame
from src.gui.styles import COLORS, FONTS
from src.settings import AppSettings, SavedCalibration
from src.utils import extract_first_frame, get_video_info


# State for undo/redo
class CalibrationState:
    """Represents a calibration state for undo/redo."""

    def __init__(self, point1, point2, distance_feet):
        self.point1 = point1
        self.point2 = point2
        self.distance_feet = distance_feet


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

        # Undo/Redo stacks (max 50 steps)
        self._undo_stack: list[CalibrationState] = []
        self._redo_stack: list[CalibrationState] = []
        self._max_undo_steps = 50

        # Load existing calibration
        cal = settings.calibration
        if cal.reference_pixel_start_x != cal.reference_pixel_end_x:
            self.point1 = (cal.reference_pixel_start_x, cal.reference_pixel_y)
            self.point2 = (cal.reference_pixel_end_x, cal.reference_pixel_y)

        self._create_layout()

        # Bind keyboard shortcuts (Cmd on macOS, Ctrl on others)
        self.bind_all("<Command-z>", self._on_undo_key)
        self.bind_all("<Command-Z>", self._on_undo_key)
        self.bind_all("<Command-Shift-z>", self._on_redo_key)
        self.bind_all("<Command-Shift-Z>", self._on_redo_key)
        self.bind_all("<Command-s>", self._on_save_key)
        self.bind_all("<Command-S>", self._on_save_key)
        # Also support Ctrl for non-macOS
        self.bind_all("<Control-z>", self._on_undo_key)
        self.bind_all("<Control-Z>", self._on_undo_key)
        self.bind_all("<Control-Shift-z>", self._on_redo_key)
        self.bind_all("<Control-Shift-Z>", self._on_redo_key)
        self.bind_all("<Control-s>", self._on_save_key)
        self.bind_all("<Control-S>", self._on_save_key)

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

        # Right side - Controls (scrollable)
        controls_frame = ttk.Frame(content, style="Card.TFrame", width=300)
        controls_frame.pack(side="right", fill="y")
        controls_frame.pack_propagate(False)

        self.controls_scroll = ScrollableFrame(controls_frame, bg_color=COLORS["bg_medium"])
        self.controls_scroll.pack(fill="both", expand=True)

        controls_inner = self.controls_scroll.scrollable_frame
        controls_inner.configure(padding=(15, 15))

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

        ttk.Separator(controls_inner, orient="horizontal").pack(fill="x", pady=10)

        # Undo/Redo buttons
        undo_redo_frame = ttk.Frame(controls_inner, style="CardInner.TFrame")
        undo_redo_frame.pack(fill="x", pady=5)

        self.undo_btn = ttk.Button(
            undo_redo_frame,
            text="↶ Undo",
            command=self._undo,
            width=8,
        )
        self.undo_btn.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.redo_btn = ttk.Button(
            undo_redo_frame,
            text="↷ Redo",
            command=self._redo,
            width=8,
        )
        self.redo_btn.pack(side="left", fill="x", expand=True, padx=(2, 0))

        # Action buttons
        clear_btn = ttk.Button(
            controls_inner,
            text="🗑️ Clear Points",
            command=self._clear_points,
        )
        clear_btn.pack(fill="x", pady=3)

        ttk.Separator(controls_inner, orient="horizontal").pack(fill="x", pady=10)

        # Save section
        ttk.Label(
            controls_inner,
            text="Save Calibration",
            style="Subheading.TLabel",
        ).pack(anchor="w", pady=(0, 5))

        # Name entry
        name_frame = ttk.Frame(controls_inner, style="CardInner.TFrame")
        name_frame.pack(fill="x", pady=3)

        ttk.Label(
            name_frame,
            text="Name:",
            style="Body.TLabel",
        ).pack(side="left")

        self.calibration_name_var = tk.StringVar(value="Default")
        self.calibration_name_entry = ttk.Entry(
            name_frame,
            textvariable=self.calibration_name_var,
            font=FONTS["body"],
        )
        self.calibration_name_entry.pack(side="right", fill="x", expand=True, padx=(5, 0))

        save_btn = ttk.Button(
            controls_inner,
            text="💾 Save",
            command=self._save_calibration,
            style="Accent.TButton",
        )
        save_btn.pack(fill="x", pady=3)

        ttk.Separator(controls_inner, orient="horizontal").pack(fill="x", pady=10)

        # Saved calibrations section
        ttk.Label(
            controls_inner,
            text="Saved Calibrations",
            style="Subheading.TLabel",
        ).pack(anchor="w", pady=(0, 5))

        self.saved_calibrations_listbox = tk.Listbox(
            controls_inner,
            height=4,
            font=FONTS["small"],
            bg=COLORS["bg_dark"],
            fg=COLORS["text_primary"],
            selectbackground=COLORS["accent"],
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        self.saved_calibrations_listbox.pack(fill="x", pady=3)
        self.saved_calibrations_listbox.bind("<Double-Button-1>", self._load_saved_calibration)

        saved_btns = ttk.Frame(controls_inner, style="CardInner.TFrame")
        saved_btns.pack(fill="x", pady=3)

        load_saved_btn = ttk.Button(
            saved_btns,
            text="Load",
            command=self._load_saved_calibration,
            width=6,
        )
        load_saved_btn.pack(side="left", fill="x", expand=True, padx=(0, 2))

        delete_saved_btn = ttk.Button(
            saved_btns,
            text="Delete",
            command=self._delete_saved_calibration,
            width=6,
        )
        delete_saved_btn.pack(side="left", fill="x", expand=True, padx=(2, 0))

        ttk.Separator(controls_inner, orient="horizontal").pack(fill="x", pady=10)

        # History section
        ttk.Label(
            controls_inner,
            text="Save History",
            style="Subheading.TLabel",
        ).pack(anchor="w", pady=(0, 5))

        self.history_listbox = tk.Listbox(
            controls_inner,
            height=4,
            font=FONTS["small"],
            bg=COLORS["bg_dark"],
            fg=COLORS["text_primary"],
            selectbackground=COLORS["accent"],
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
        )
        self.history_listbox.pack(fill="x", pady=3)
        self.history_listbox.bind("<Double-Button-1>", self._load_from_history)

        load_history_btn = ttk.Button(
            controls_inner,
            text="Restore Selected",
            command=self._load_from_history,
        )
        load_history_btn.pack(fill="x", pady=3)

        ttk.Separator(controls_inner, orient="horizontal").pack(fill="x", pady=10)

        # Tips
        tips_frame = ttk.LabelFrame(
            controls_inner,
            text="Tips",
            style="Card.TLabelframe",
        )
        tips_frame.pack(fill="x", pady=5)

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

        # Update displays
        self._update_pixel_display()
        self._update_saved_calibrations_list()
        self._update_history_list()
        self._update_undo_redo_buttons()

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

        # Push undo state before modifying
        self._push_undo_state()

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
        if self.point1 is not None or self.point2 is not None:
            self._push_undo_state()
            self.point1 = None
            self.point2 = None
            self._update_pixel_display()
            if self.original_frame is not None:
                self._update_display()

    def _save_calibration(self):
        """Save the calibration settings and add to history."""
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

        name = self.calibration_name_var.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter a name for the calibration")
            return

        # Check if name already exists
        existing_names = [c.name for c in self.settings.calibration.saved_calibrations]
        if name in existing_names:
            if not messagebox.askyesno("Overwrite", f"Calibration '{name}' already exists. Overwrite?"):
                return
            # Remove existing
            self.settings.calibration.saved_calibrations = [
                c for c in self.settings.calibration.saved_calibrations if c.name != name
            ]

        # Create saved calibration
        saved_cal = SavedCalibration(
            name=name,
            reference_distance_feet=feet_dist,
            point1=list(self.point1),
            point2=list(self.point2),
            saved_at=datetime.now().isoformat(),
        )

        # Add to saved calibrations
        self.settings.calibration.saved_calibrations.append(saved_cal)

        # Update settings - store X coordinates for horizontal measurement
        x1, y1 = self.point1
        x2, y2 = self.point2

        self.settings.calibration.reference_distance_feet = feet_dist
        self.settings.calibration.reference_pixel_start_x = min(x1, x2)
        self.settings.calibration.reference_pixel_end_x = max(x1, x2)
        self.settings.calibration.reference_pixel_y = (y1 + y2) // 2

        # Add to history
        self._add_to_history(name)

        self.on_change()
        self._update_saved_calibrations_list()
        self._update_history_list()

        ppf = self._calculate_ppf()
        messagebox.showinfo(
            "Success",
            f"Calibration '{name}' saved!\n\n"
            f"Pixel distance: {self._calculate_pixel_distance()} px\n"
            f"Real distance: {feet_dist} ft\n"
            f"Pixels per foot: {ppf:.2f}"
        )

    # ==================== Undo/Redo Methods ====================

    def _push_undo_state(self):
        """Push current state to undo stack."""
        try:
            distance = float(self.distance_var.get())
        except ValueError:
            distance = 0.0

        state = CalibrationState(
            point1=self.point1,
            point2=self.point2,
            distance_feet=distance,
        )
        self._undo_stack.append(state)

        # Limit stack size
        if len(self._undo_stack) > self._max_undo_steps:
            self._undo_stack.pop(0)

        # Clear redo stack when new action is performed
        self._redo_stack.clear()

        self._update_undo_redo_buttons()

    def _undo(self):
        """Undo the last action."""
        if not self._undo_stack:
            return

        # Push current state to redo stack
        try:
            distance = float(self.distance_var.get())
        except ValueError:
            distance = 0.0

        current_state = CalibrationState(
            point1=self.point1,
            point2=self.point2,
            distance_feet=distance,
        )
        self._redo_stack.append(current_state)

        # Pop and restore previous state
        state = self._undo_stack.pop()
        self.point1 = state.point1
        self.point2 = state.point2
        self.distance_var.set(str(state.distance_feet))

        self._update_pixel_display()
        self._update_display()
        self._update_undo_redo_buttons()

    def _redo(self):
        """Redo the last undone action."""
        if not self._redo_stack:
            return

        # Push current state to undo stack
        try:
            distance = float(self.distance_var.get())
        except ValueError:
            distance = 0.0

        current_state = CalibrationState(
            point1=self.point1,
            point2=self.point2,
            distance_feet=distance,
        )
        self._undo_stack.append(current_state)

        # Pop and restore redo state
        state = self._redo_stack.pop()
        self.point1 = state.point1
        self.point2 = state.point2
        self.distance_var.set(str(state.distance_feet))

        self._update_pixel_display()
        self._update_display()
        self._update_undo_redo_buttons()

    def _on_undo_key(self, event):
        """Handle Cmd+Z keyboard shortcut."""
        self._undo()
        return "break"

    def _on_redo_key(self, event):
        """Handle Cmd+Shift+Z keyboard shortcut."""
        self._redo()
        return "break"

    def _on_save_key(self, event):
        """Handle Cmd+S keyboard shortcut."""
        self._save_calibration()
        return "break"

    def _update_undo_redo_buttons(self):
        """Update the enabled state of undo/redo buttons."""
        if hasattr(self, 'undo_btn'):
            if self._undo_stack:
                self.undo_btn.configure(state="normal")
            else:
                self.undo_btn.configure(state="disabled")

        if hasattr(self, 'redo_btn'):
            if self._redo_stack:
                self.redo_btn.configure(state="normal")
            else:
                self.redo_btn.configure(state="disabled")

    # ==================== Save/Load Named Calibrations ====================

    def _update_saved_calibrations_list(self):
        """Update the saved calibrations listbox."""
        if not hasattr(self, 'saved_calibrations_listbox'):
            return

        self.saved_calibrations_listbox.delete(0, tk.END)
        for cal in self.settings.calibration.saved_calibrations:
            self.saved_calibrations_listbox.insert(tk.END, f"  {cal.name}")

    def _load_saved_calibration(self, event=None):
        """Load a saved calibration."""
        selection = self.saved_calibrations_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        if idx >= len(self.settings.calibration.saved_calibrations):
            return

        cal = self.settings.calibration.saved_calibrations[idx]

        # Push current state to undo
        self._push_undo_state()

        # Load the calibration
        self.point1 = tuple(cal.point1)
        self.point2 = tuple(cal.point2)
        self.distance_var.set(str(cal.reference_distance_feet))

        self._update_pixel_display()
        self._update_display()

    def _delete_saved_calibration(self):
        """Delete a saved calibration."""
        selection = self.saved_calibrations_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        if idx >= len(self.settings.calibration.saved_calibrations):
            return

        cal = self.settings.calibration.saved_calibrations[idx]

        if messagebox.askyesno("Confirm", f"Delete saved calibration '{cal.name}'?"):
            self.settings.calibration.saved_calibrations.pop(idx)
            self.on_change()
            self._update_saved_calibrations_list()

    # ==================== History Methods ====================

    def _add_to_history(self, name: str):
        """Add current calibration to save history."""
        if self.point1 is None or self.point2 is None:
            return

        try:
            feet_dist = float(self.distance_var.get())
        except ValueError:
            feet_dist = 0.0

        saved_cal = SavedCalibration(
            name=name,
            reference_distance_feet=feet_dist,
            point1=list(self.point1),
            point2=list(self.point2),
            saved_at=datetime.now().isoformat(),
        )

        # Insert at beginning (most recent first)
        self.settings.calibration.save_history.insert(0, saved_cal)

        # Limit history size
        max_history = self.settings.calibration.max_history
        if len(self.settings.calibration.save_history) > max_history:
            self.settings.calibration.save_history = self.settings.calibration.save_history[:max_history]

    def _update_history_list(self):
        """Update the history listbox."""
        if not hasattr(self, 'history_listbox'):
            return

        self.history_listbox.delete(0, tk.END)
        for cal in self.settings.calibration.save_history:
            # Format timestamp
            try:
                dt = datetime.fromisoformat(cal.saved_at)
                time_str = dt.strftime("%m/%d %H:%M")
            except Exception:
                time_str = "Unknown"

            self.history_listbox.insert(tk.END, f"  {time_str} - {cal.name}")

    def _load_from_history(self, event=None):
        """Load a calibration from history."""
        selection = self.history_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        if idx >= len(self.settings.calibration.save_history):
            return

        cal = self.settings.calibration.save_history[idx]

        # Push current state to undo
        self._push_undo_state()

        # Load the calibration
        self.point1 = tuple(cal.point1)
        self.point2 = tuple(cal.point2)
        self.distance_var.set(str(cal.reference_distance_feet))

        self._update_pixel_display()
        self._update_display()
