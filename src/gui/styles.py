"""GUI styling and theming for DadBot Traffic Monitor."""

import tkinter as tk
from tkinter import ttk


# Color palette - Modern dark theme
COLORS = {
    "bg_dark": "#1a1a2e",
    "bg_medium": "#16213e",
    "bg_light": "#0f3460",
    "accent": "#e94560",
    "accent_hover": "#ff6b6b",
    "text_primary": "#ffffff",
    "text_secondary": "#a0a0a0",
    "text_muted": "#606060",
    "success": "#4ecca3",
    "warning": "#ffc107",
    "error": "#e94560",
    "border": "#2a2a4a",
}

# Font settings
FONTS = {
    "heading": ("Segoe UI", 16, "bold"),
    "subheading": ("Segoe UI", 12, "bold"),
    "body": ("Segoe UI", 10),
    "small": ("Segoe UI", 9),
    "mono": ("Consolas", 10),
    "mono_small": ("Consolas", 9),
}


def configure_styles():
    """Configure ttk styles for the application."""
    style = ttk.Style()

    # Try to use a modern theme as base
    available_themes = style.theme_names()
    if "clam" in available_themes:
        style.theme_use("clam")

    # Configure main frame style
    style.configure(
        "Main.TFrame",
        background=COLORS["bg_dark"],
    )

    style.configure(
        "Card.TFrame",
        background=COLORS["bg_medium"],
        relief="flat",
    )

    style.configure(
        "CardInner.TFrame",
        background=COLORS["bg_medium"],
    )

    # Configure label styles
    style.configure(
        "Heading.TLabel",
        background=COLORS["bg_dark"],
        foreground=COLORS["text_primary"],
        font=FONTS["heading"],
    )

    style.configure(
        "Subheading.TLabel",
        background=COLORS["bg_medium"],
        foreground=COLORS["text_primary"],
        font=FONTS["subheading"],
    )

    style.configure(
        "Body.TLabel",
        background=COLORS["bg_medium"],
        foreground=COLORS["text_primary"],
        font=FONTS["body"],
    )

    style.configure(
        "Muted.TLabel",
        background=COLORS["bg_medium"],
        foreground=COLORS["text_secondary"],
        font=FONTS["small"],
    )

    # Configure button styles
    style.configure(
        "Action.TButton",
        font=FONTS["body"],
        padding=(20, 10),
    )

    style.configure(
        "Accent.TButton",
        font=FONTS["subheading"],
        padding=(30, 15),
    )

    style.map(
        "Accent.TButton",
        background=[("active", COLORS["accent_hover"])],
    )

    # Configure entry styles
    style.configure(
        "Modern.TEntry",
        font=FONTS["body"],
        padding=5,
    )

    # Configure checkbutton styles
    style.configure(
        "Modern.TCheckbutton",
        background=COLORS["bg_medium"],
        foreground=COLORS["text_primary"],
        font=FONTS["body"],
    )

    # Configure combobox styles
    style.configure(
        "Modern.TCombobox",
        font=FONTS["body"],
        padding=5,
    )

    # Configure labelframe styles
    style.configure(
        "Card.TLabelframe",
        background=COLORS["bg_medium"],
        foreground=COLORS["text_primary"],
        font=FONTS["subheading"],
    )

    style.configure(
        "Card.TLabelframe.Label",
        background=COLORS["bg_medium"],
        foreground=COLORS["accent"],
        font=FONTS["subheading"],
    )

    # Configure scale/slider styles
    style.configure(
        "Modern.Horizontal.TScale",
        background=COLORS["bg_medium"],
        troughcolor=COLORS["bg_light"],
    )

    # Configure notebook (tabs) styles
    style.configure(
        "Modern.TNotebook",
        background=COLORS["bg_dark"],
        tabmargins=[2, 5, 2, 0],
    )

    style.configure(
        "Modern.TNotebook.Tab",
        background=COLORS["bg_medium"],
        foreground=COLORS["text_secondary"],
        padding=[15, 8],
        font=FONTS["body"],
    )

    style.map(
        "Modern.TNotebook.Tab",
        background=[("selected", COLORS["bg_light"])],
        foreground=[("selected", COLORS["text_primary"])],
    )

    # Configure separator
    style.configure(
        "Modern.TSeparator",
        background=COLORS["border"],
    )

    # Configure progressbar
    style.configure(
        "Modern.Horizontal.TProgressbar",
        background=COLORS["accent"],
        troughcolor=COLORS["bg_light"],
    )

    return style


class StyledText(tk.Text):
    """A styled text widget for console output."""

    def __init__(self, parent, **kwargs):
        # Set defaults for dark theme
        defaults = {
            "bg": COLORS["bg_dark"],
            "fg": COLORS["text_primary"],
            "font": FONTS["mono"],
            "insertbackground": COLORS["text_primary"],
            "selectbackground": COLORS["accent"],
            "selectforeground": COLORS["text_primary"],
            "relief": "flat",
            "borderwidth": 0,
            "padx": 10,
            "pady": 10,
        }
        defaults.update(kwargs)
        super().__init__(parent, **defaults)

        # Configure tags for different message types
        self.tag_configure("info", foreground=COLORS["text_primary"])
        self.tag_configure("success", foreground=COLORS["success"])
        self.tag_configure("warning", foreground=COLORS["warning"])
        self.tag_configure("error", foreground=COLORS["error"])
        self.tag_configure("muted", foreground=COLORS["text_muted"])


class ToolTip:
    """Simple tooltip implementation."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        x, y, _, _ = self.widget.bbox("insert") if hasattr(self.widget, "bbox") else (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            self.tooltip,
            text=self.text,
            background=COLORS["bg_light"],
            foreground=COLORS["text_primary"],
            relief="solid",
            borderwidth=1,
            font=FONTS["small"],
            padx=8,
            pady=4,
        )
        label.pack()

    def hide(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None
