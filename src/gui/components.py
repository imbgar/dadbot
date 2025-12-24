"""Reusable GUI components for DadBot Traffic Monitor."""

import tkinter as tk
from tkinter import ttk
from typing import Callable

from src.gui.styles import COLORS, FONTS, ToolTip


class ActionButton(ttk.Frame):
    """Large action button with icon and description."""

    def __init__(
        self,
        parent,
        title: str,
        description: str,
        icon: str,
        command: Callable,
        accent: bool = False,
        **kwargs
    ):
        super().__init__(parent, style="Card.TFrame", **kwargs)

        self.command = command
        self._hover = False

        # Create canvas for custom drawing
        self.canvas = tk.Canvas(
            self,
            width=280,
            height=100,
            bg=COLORS["bg_medium"],
            highlightthickness=0,
            cursor="hand2",
        )
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)

        # Store colors
        self.bg_normal = COLORS["bg_medium"]
        self.bg_hover = COLORS["bg_light"]
        self.accent_color = COLORS["accent"] if accent else COLORS["text_secondary"]

        # Draw content
        self._draw_content(title, description, icon)

        # Bind events
        self.canvas.bind("<Enter>", self._on_enter)
        self.canvas.bind("<Leave>", self._on_leave)
        self.canvas.bind("<Button-1>", self._on_click)

    def _draw_content(self, title: str, description: str, icon: str):
        """Draw button content."""
        self.canvas.delete("all")

        bg = self.bg_hover if self._hover else self.bg_normal

        # Background
        self.canvas.create_rectangle(
            0, 0, 280, 100,
            fill=bg,
            outline=COLORS["border"],
            width=1,
        )

        # Icon (emoji)
        self.canvas.create_text(
            30, 50,
            text=icon,
            font=("Segoe UI Emoji", 24),
            fill=self.accent_color,
            anchor="center",
        )

        # Title
        self.canvas.create_text(
            70, 35,
            text=title,
            font=FONTS["subheading"],
            fill=COLORS["text_primary"],
            anchor="w",
        )

        # Description
        self.canvas.create_text(
            70, 60,
            text=description,
            font=FONTS["small"],
            fill=COLORS["text_secondary"],
            anchor="w",
            width=190,
        )

    def _on_enter(self, event):
        self._hover = True
        self._draw_content(
            self.canvas.itemcget(2, "text"),
            self.canvas.itemcget(3, "text"),
            self.canvas.itemcget(1, "text"),
        )

    def _on_leave(self, event):
        self._hover = False
        self._draw_content(
            self.canvas.itemcget(2, "text"),
            self.canvas.itemcget(3, "text"),
            self.canvas.itemcget(1, "text"),
        )

    def _on_click(self, event):
        if self.command:
            self.command()


class SettingsPanel(ttk.LabelFrame):
    """Collapsible settings panel."""

    def __init__(self, parent, title: str, **kwargs):
        super().__init__(parent, text=title, style="Card.TLabelframe", **kwargs)

        self.inner_frame = ttk.Frame(self, style="CardInner.TFrame")
        self.inner_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def add_row(self, label: str, widget: tk.Widget, tooltip: str = None) -> ttk.Frame:
        """Add a label-widget row to the panel."""
        row = ttk.Frame(self.inner_frame, style="CardInner.TFrame")
        row.pack(fill="x", pady=5)

        lbl = ttk.Label(row, text=label, style="Body.TLabel", width=20)
        lbl.pack(side="left")

        widget.pack(side="right", fill="x", expand=True)

        if tooltip:
            ToolTip(lbl, tooltip)

        return row

    def add_checkbox(self, label: str, variable: tk.BooleanVar, tooltip: str = None) -> ttk.Checkbutton:
        """Add a checkbox row."""
        row = ttk.Frame(self.inner_frame, style="CardInner.TFrame")
        row.pack(fill="x", pady=5)

        cb = ttk.Checkbutton(
            row,
            text=label,
            variable=variable,
            style="Modern.TCheckbutton",
        )
        cb.pack(side="left")

        if tooltip:
            ToolTip(cb, tooltip)

        return cb

    def add_separator(self):
        """Add a separator line."""
        sep = ttk.Separator(self.inner_frame, orient="horizontal", style="Modern.TSeparator")
        sep.pack(fill="x", pady=10)


class ConsoleOutput(ttk.Frame):
    """Console-style output panel with scrolling."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, style="Card.TFrame", **kwargs)

        # Create text widget with scrollbar
        self.text = tk.Text(
            self,
            bg=COLORS["bg_dark"],
            fg=COLORS["text_primary"],
            font=FONTS["mono_small"],
            insertbackground=COLORS["text_primary"],
            selectbackground=COLORS["accent"],
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=10,
            wrap="word",
            state="disabled",
        )

        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

        # Configure tags
        self.text.tag_configure("info", foreground=COLORS["text_primary"])
        self.text.tag_configure("success", foreground=COLORS["success"])
        self.text.tag_configure("warning", foreground=COLORS["warning"])
        self.text.tag_configure("error", foreground=COLORS["error"])
        self.text.tag_configure("muted", foreground=COLORS["text_muted"])
        self.text.tag_configure("accent", foreground=COLORS["accent"])

    def write(self, text: str, tag: str = "info"):
        """Write text to the console."""
        self.text.configure(state="normal")
        self.text.insert("end", text, tag)
        self.text.see("end")
        self.text.configure(state="disabled")

    def writeln(self, text: str, tag: str = "info"):
        """Write a line to the console."""
        self.write(text + "\n", tag)

    def clear(self):
        """Clear the console."""
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")


class StatusBar(ttk.Frame):
    """Status bar with multiple sections."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, style="Card.TFrame", **kwargs)

        self.sections: dict[str, ttk.Label] = {}

    def add_section(self, name: str, text: str = "", width: int = 20) -> ttk.Label:
        """Add a status section."""
        label = ttk.Label(
            self,
            text=text,
            style="Muted.TLabel",
            width=width,
            anchor="center",
        )
        label.pack(side="left", padx=5)

        # Add separator if not first section
        if self.sections:
            sep = ttk.Separator(self, orient="vertical")
            sep.pack(side="left", fill="y", padx=5)

        self.sections[name] = label
        return label

    def update_section(self, name: str, text: str):
        """Update a status section."""
        if name in self.sections:
            self.sections[name].configure(text=text)


class VideoPreview(ttk.Frame):
    """Video preview panel with controls."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, style="Card.TFrame", **kwargs)

        # Canvas for video display
        self.canvas = tk.Canvas(
            self,
            bg=COLORS["bg_dark"],
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)

        # Placeholder text
        self.canvas.create_text(
            320, 240,
            text="No video loaded",
            font=FONTS["body"],
            fill=COLORS["text_muted"],
            tags="placeholder",
        )

        self._image = None  # Keep reference to prevent garbage collection

    def set_frame(self, photo_image):
        """Set the current frame to display."""
        self.canvas.delete("all")
        self._image = photo_image

        # Center the image
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=photo_image, anchor="center")

    def set_placeholder(self, text: str):
        """Set placeholder text."""
        self.canvas.delete("all")
        self._image = None

        canvas_w = self.canvas.winfo_width() or 640
        canvas_h = self.canvas.winfo_height() or 480

        self.canvas.create_text(
            canvas_w // 2, canvas_h // 2,
            text=text,
            font=FONTS["body"],
            fill=COLORS["text_muted"],
        )
