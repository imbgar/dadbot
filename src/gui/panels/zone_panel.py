"""Zone Definition Panel for DadBot Traffic Monitor."""

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
from src.settings import AppSettings, SavedZone
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

        # Undo/Redo stacks (max 50 steps)
        self._undo_stack: list[list[tuple[int, int]]] = []
        self._redo_stack: list[list[tuple[int, int]]] = []
        self._max_undo_steps = 50

        # Drag state
        self._dragging_index: int | None = None
        self._dragging_zone = False  # True when dragging entire zone
        self._drag_start_pos: tuple[int, int] | None = None
        self._drag_start_points: list[tuple[int, int]] | None = None  # Original points when zone drag started
        self._point_hit_radius = 15  # Pixels to detect point click

        # Canvas offsets (set during display)
        self._x_offset = 0
        self._y_offset = 0

        # Load existing points from settings
        if settings.zone.polygon_points:
            self.points = [tuple(p) for p in settings.zone.polygon_points]

        self._create_layout()

        # Bind keyboard shortcuts (Cmd on macOS, Ctrl on others)
        # macOS uses Command key
        self.bind_all("<Command-z>", self._on_undo_key)
        self.bind_all("<Command-Z>", self._on_undo_key)
        self.bind_all("<Command-Shift-z>", self._on_redo_key)
        self.bind_all("<Command-Shift-Z>", self._on_redo_key)
        self.bind_all("<Command-s>", self._on_save_key)
        self.bind_all("<Command-S>", self._on_save_key)
        # Also support Ctrl for non-macOS or external keyboards
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
            text="Click to add points • Drag points to move • Drag inside zone to reposition • Right-click to undo",
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
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Right side - Controls (scrollable)
        controls_frame = ttk.Frame(content, style="Card.TFrame", width=280)
        controls_frame.pack(side="right", fill="y")
        controls_frame.pack_propagate(False)

        self.controls_scroll = ScrollableFrame(controls_frame, bg_color=COLORS["bg_medium"])
        self.controls_scroll.pack(fill="both", expand=True)

        controls_inner = self.controls_scroll.scrollable_frame
        # Add padding inside scrollable area
        controls_inner.configure(padding=(15, 15))

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

        clear_btn = ttk.Button(
            controls_inner,
            text="🗑️ Clear All",
            command=self._clear_points,
        )
        clear_btn.pack(fill="x", pady=3)

        ttk.Separator(controls_inner, orient="horizontal").pack(fill="x", pady=10)

        # Save section
        ttk.Label(
            controls_inner,
            text="Save Zone",
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

        self.zone_name_var = tk.StringVar(value="Default")
        self.zone_name_entry = ttk.Entry(
            name_frame,
            textvariable=self.zone_name_var,
            font=FONTS["body"],
        )
        self.zone_name_entry.pack(side="right", fill="x", expand=True, padx=(5, 0))

        save_btn = ttk.Button(
            controls_inner,
            text="💾 Save",
            command=self._save_zone,
            style="Accent.TButton",
        )
        save_btn.pack(fill="x", pady=3)

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
        overlay_cb.pack(anchor="w", pady=3)

        ttk.Separator(controls_inner, orient="horizontal").pack(fill="x", pady=10)

        # Saved zones section
        ttk.Label(
            controls_inner,
            text="Saved Zones",
            style="Subheading.TLabel",
        ).pack(anchor="w", pady=(0, 5))

        self.saved_zones_listbox = tk.Listbox(
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
        self.saved_zones_listbox.pack(fill="x", pady=3)
        self.saved_zones_listbox.bind("<Double-Button-1>", self._load_saved_zone)

        saved_btns = ttk.Frame(controls_inner, style="CardInner.TFrame")
        saved_btns.pack(fill="x", pady=3)

        load_saved_btn = ttk.Button(
            saved_btns,
            text="Load",
            command=self._load_saved_zone,
            width=6,
        )
        load_saved_btn.pack(side="left", fill="x", expand=True, padx=(0, 2))

        delete_saved_btn = ttk.Button(
            saved_btns,
            text="Delete",
            command=self._delete_saved_zone,
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

        # Update displays
        self._update_points_display()
        self._update_saved_zones_list()
        self._update_history_list()
        self._update_undo_redo_buttons()

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

    def _get_point_at_position(self, canvas_x: int, canvas_y: int) -> int | None:
        """Check if canvas position is near a point. Returns point index or None."""
        for i, (px, py) in enumerate(self.points):
            # Convert point to canvas coordinates
            screen_x = px * self.scale_factor + self._x_offset
            screen_y = py * self.scale_factor + self._y_offset

            # Check distance
            dist = ((canvas_x - screen_x) ** 2 + (canvas_y - screen_y) ** 2) ** 0.5
            if dist <= self._point_hit_radius:
                return i
        return None

    def _is_inside_polygon(self, canvas_x: int, canvas_y: int) -> bool:
        """Check if canvas position is inside the zone polygon using ray casting."""
        if len(self.points) < 3:
            return False

        # Convert canvas coords to image coords
        x = (canvas_x - self._x_offset) / self.scale_factor
        y = (canvas_y - self._y_offset) / self.scale_factor

        # Ray casting algorithm
        n = len(self.points)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = self.points[i]
            xj, yj = self.points[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    def _on_left_click(self, event):
        """Handle left mouse click to add or start dragging a point/zone."""
        if self.original_frame is None:
            return

        # Check if clicking on an existing point (highest priority)
        hit_index = self._get_point_at_position(event.x, event.y)

        if hit_index is not None:
            # Start dragging this point
            self._push_undo_state()
            self._dragging_index = hit_index
            self._dragging_zone = False
            self._drag_start_pos = (event.x, event.y)
            self.canvas.configure(cursor="fleur")  # Move cursor
        elif self._is_inside_polygon(event.x, event.y):
            # Start dragging the entire zone
            self._push_undo_state()
            self._dragging_zone = True
            self._dragging_index = None
            self._drag_start_pos = (event.x, event.y)
            self._drag_start_points = list(self.points)  # Copy original points
            self.canvas.configure(cursor="fleur")
        else:
            # Add a new point
            x = int((event.x - self._x_offset) / self.scale_factor)
            y = int((event.y - self._y_offset) / self.scale_factor)

            # Check bounds
            h, w = self.original_frame.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                self._push_undo_state()
                self.points.append((x, y))
                self._update_points_display()
                self._update_display()

    def _on_mouse_drag(self, event):
        """Handle mouse drag to move a point or the entire zone."""
        if self.original_frame is None:
            return

        if self._dragging_zone and self._drag_start_pos and self._drag_start_points:
            # Dragging the entire zone
            # Calculate offset in image coordinates
            dx = int((event.x - self._drag_start_pos[0]) / self.scale_factor)
            dy = int((event.y - self._drag_start_pos[1]) / self.scale_factor)

            h, w = self.original_frame.shape[:2]

            # Calculate new positions for all points
            new_points = []
            for ox, oy in self._drag_start_points:
                nx = ox + dx
                ny = oy + dy
                # Clamp to image bounds
                nx = max(0, min(nx, w - 1))
                ny = max(0, min(ny, h - 1))
                new_points.append((nx, ny))

            self.points = new_points
            self._update_points_display()
            self._update_display()

        elif self._dragging_index is not None:
            # Dragging a single point
            x = int((event.x - self._x_offset) / self.scale_factor)
            y = int((event.y - self._y_offset) / self.scale_factor)

            # Clamp to image bounds
            h, w = self.original_frame.shape[:2]
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))

            # Update the point position
            self.points[self._dragging_index] = (x, y)
            self._update_points_display()
            self._update_display()

    def _on_mouse_release(self, event):
        """Handle mouse release to stop dragging."""
        if self._dragging_index is not None or self._dragging_zone:
            self._dragging_index = None
            self._dragging_zone = False
            self._drag_start_pos = None
            self._drag_start_points = None
            self.canvas.configure(cursor="crosshair")

    def _on_mouse_move(self, event):
        """Handle mouse move to update cursor when hovering over points or inside zone."""
        if self.original_frame is None:
            return

        # Don't change cursor while dragging
        if self._dragging_index is not None or self._dragging_zone:
            return

        hit_index = self._get_point_at_position(event.x, event.y)
        if hit_index is not None:
            self.canvas.configure(cursor="fleur")  # Move cursor for points
        elif self._is_inside_polygon(event.x, event.y):
            self.canvas.configure(cursor="fleur")  # Move cursor for zone
        else:
            self.canvas.configure(cursor="crosshair")

    def _on_right_click(self, event):
        """Handle right mouse click to remove last point."""
        if self.points:
            self._push_undo_state()
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
        if self.points and messagebox.askyesno("Confirm", "Clear all zone points?"):
            self._push_undo_state()
            self.points = []
            self._update_points_display()
            self._update_display()

    def _save_zone(self):
        """Save the zone configuration to settings and add to history."""
        if len(self.points) < 3:
            messagebox.showwarning("Warning", "Zone requires at least 3 points")
            return

        name = self.zone_name_var.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter a name for the zone")
            return

        # Check if name already exists in saved zones
        existing_names = [z.name for z in self.settings.zone.saved_zones]
        if name in existing_names:
            if not messagebox.askyesno("Overwrite", f"Zone '{name}' already exists. Overwrite?"):
                return
            # Remove existing
            self.settings.zone.saved_zones = [
                z for z in self.settings.zone.saved_zones if z.name != name
            ]

        # Create saved zone
        saved_zone = SavedZone(
            name=name,
            points=[list(p) for p in self.points],
            saved_at=datetime.now().isoformat(),
        )

        # Add to saved zones
        self.settings.zone.saved_zones.append(saved_zone)

        # Update settings
        self.settings.zone.polygon_points = [list(p) for p in self.points]
        self.settings.zone.enabled = self.zone_enabled_var.get()
        self.settings.zone.show_overlay = self.show_overlay_var.get()

        # Add to history
        self._add_to_history(name)

        self.on_change()
        self._update_saved_zones_list()
        self._update_history_list()

        messagebox.showinfo("Success", f"Zone '{name}' saved with {len(self.points)} points")

    def _on_zone_toggle(self):
        """Handle zone enabled checkbox toggle."""
        self.settings.zone.enabled = self.zone_enabled_var.get()
        self.on_change()

    def _on_overlay_toggle(self):
        """Handle overlay checkbox toggle."""
        self.settings.zone.show_overlay = self.show_overlay_var.get()
        self.on_change()

    # ==================== Undo/Redo Methods ====================

    def _push_undo_state(self):
        """Push current state to undo stack."""
        # Save current points
        self._undo_stack.append(list(self.points))

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
        self._redo_stack.append(list(self.points))

        # Pop and restore previous state
        self.points = self._undo_stack.pop()

        self._update_points_display()
        self._update_display()
        self._update_undo_redo_buttons()

    def _redo(self):
        """Redo the last undone action."""
        if not self._redo_stack:
            return

        # Push current state to undo stack
        self._undo_stack.append(list(self.points))

        # Pop and restore redo state
        self.points = self._redo_stack.pop()

        self._update_points_display()
        self._update_display()
        self._update_undo_redo_buttons()

    def _on_undo_key(self, event):
        """Handle Ctrl+Z keyboard shortcut."""
        self._undo()
        return "break"

    def _on_redo_key(self, event):
        """Handle Cmd+Shift+Z keyboard shortcut."""
        self._redo()
        return "break"

    def _on_save_key(self, event):
        """Handle Cmd+S keyboard shortcut."""
        self._save_zone()
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

    # ==================== Save/Load Named Zones ====================

    def _update_saved_zones_list(self):
        """Update the saved zones listbox."""
        if not hasattr(self, 'saved_zones_listbox'):
            return

        self.saved_zones_listbox.delete(0, tk.END)
        for zone in self.settings.zone.saved_zones:
            self.saved_zones_listbox.insert(tk.END, f"  {zone.name}")

    def _load_saved_zone(self, event=None):
        """Load a saved zone."""
        selection = self.saved_zones_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        if idx >= len(self.settings.zone.saved_zones):
            return

        zone = self.settings.zone.saved_zones[idx]

        # Push current state to undo
        self._push_undo_state()

        # Load the zone
        self.points = [tuple(p) for p in zone.points]

        self._update_points_display()
        self._update_display()

    def _delete_saved_zone(self):
        """Delete a saved zone."""
        selection = self.saved_zones_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        if idx >= len(self.settings.zone.saved_zones):
            return

        zone = self.settings.zone.saved_zones[idx]

        if messagebox.askyesno("Confirm", f"Delete saved zone '{zone.name}'?"):
            self.settings.zone.saved_zones.pop(idx)
            self.on_change()
            self._update_saved_zones_list()

    # ==================== History Methods ====================

    def _add_to_history(self, name: str):
        """Add current zone to save history."""
        saved_zone = SavedZone(
            name=name,
            points=[list(p) for p in self.points],
            saved_at=datetime.now().isoformat(),
        )

        # Insert at beginning (most recent first)
        self.settings.zone.save_history.insert(0, saved_zone)

        # Limit history size
        max_history = self.settings.zone.max_history
        if len(self.settings.zone.save_history) > max_history:
            self.settings.zone.save_history = self.settings.zone.save_history[:max_history]

    def _update_history_list(self):
        """Update the history listbox."""
        if not hasattr(self, 'history_listbox'):
            return

        self.history_listbox.delete(0, tk.END)
        for zone in self.settings.zone.save_history:
            # Format timestamp
            try:
                dt = datetime.fromisoformat(zone.saved_at)
                time_str = dt.strftime("%m/%d %H:%M")
            except Exception:
                time_str = "Unknown"

            self.history_listbox.insert(tk.END, f"  {time_str} - {zone.name}")

    def _load_from_history(self, event=None):
        """Load a zone from history."""
        selection = self.history_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        if idx >= len(self.settings.zone.save_history):
            return

        zone = self.settings.zone.save_history[idx]

        # Push current state to undo
        self._push_undo_state()

        # Load the zone
        self.points = [tuple(p) for p in zone.points]

        self._update_points_display()
        self._update_display()
