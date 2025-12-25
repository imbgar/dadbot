"""Main GUI Application for DadBot Traffic Monitor."""

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from src.gui.components import ActionButton, ConsoleOutput, ScrollableFrame, SettingsPanel, StatusBar
from src.gui.styles import COLORS, FONTS, configure_styles
from src.gui.panels.zone_panel import ZoneDefinitionPanel
from src.gui.panels.calibration_panel import CalibrationPanel
from src.gui.panels.viewer_panel import LiveViewerPanel
from src.gui.panels.processor_panel import ProcessorPanel
from src.settings import AppSettings, load_or_create_settings

# Get the assets directory
ASSETS_DIR = Path(__file__).parent.parent.parent / "assets"


class DadBotApp:
    """Main application window for DadBot Traffic Monitor."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DadBot Traffic Monitor")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # Set window icon (if available)
        try:
            # You can add an icon file later
            pass
        except Exception:
            pass

        # Configure dark theme background
        self.root.configure(bg=COLORS["bg_dark"])

        # Initialize styles
        self.style = configure_styles()

        # Load settings
        self.settings = load_or_create_settings()

        # Create main layout
        self._create_layout()

        # Track active panel
        self.active_panel = None

    def _create_layout(self):
        """Create the main application layout."""
        # Main container
        self.main_container = ttk.Frame(self.root, style="Main.TFrame")
        self.main_container.pack(fill="both", expand=True)

        # Left sidebar with navigation
        self._create_sidebar()

        # Right content area
        self._create_content_area()

        # Status bar at bottom
        self._create_status_bar()

    def _create_sidebar(self):
        """Create the left navigation sidebar."""
        sidebar = ttk.Frame(self.main_container, style="Card.TFrame", width=300)
        sidebar.pack(side="left", fill="y", padx=0, pady=0)
        sidebar.pack_propagate(False)

        # Logo display (fixed at top)
        logo_frame = ttk.Frame(sidebar, style="Card.TFrame")
        logo_frame.pack(fill="x", padx=5, pady=10)

        # Create canvas for logo (800x500 aspect ratio)
        self.logo_canvas = tk.Canvas(
            logo_frame,
            width=290,
            height=181,
            bg=COLORS["bg_medium"],
            highlightthickness=0,
        )
        self.logo_canvas.pack(fill="x", pady=5)

        # Load and display the logo
        self._load_logo()

        # Separator
        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=20, pady=10)

        # Scrollable navigation area
        self.sidebar_scroll = ScrollableFrame(sidebar, bg_color=COLORS["bg_medium"])
        self.sidebar_scroll.pack(fill="both", expand=True)

        nav_frame = self.sidebar_scroll.scrollable_frame
        nav_frame.configure(padding=(10, 10))

        # Zone Definition Button
        ActionButton(
            nav_frame,
            title="Zone Definition",
            description="Define road detection area",
            icon="📍",
            command=self._show_zone_panel,
        ).pack(fill="x", pady=5)

        # Calibration Button
        ActionButton(
            nav_frame,
            title="Calibration",
            description="Set distance measurements",
            icon="📏",
            command=self._show_calibration_panel,
        ).pack(fill="x", pady=5)

        # Live Viewer Button
        ActionButton(
            nav_frame,
            title="Live Viewer",
            description="Preview with real-time config",
            icon="👁️",
            command=self._show_viewer_panel,
        ).pack(fill="x", pady=5)

        # Process Video Button
        ActionButton(
            nav_frame,
            title="Process Video",
            description="Analyze and generate reports",
            icon="▶️",
            command=self._show_processor_panel,
            accent=True,
        ).pack(fill="x", pady=5)

        # Settings button at bottom (fixed)
        settings_frame = ttk.Frame(sidebar, style="Card.TFrame")
        settings_frame.pack(fill="x", padx=20, pady=20)

        settings_btn = ttk.Button(
            settings_frame,
            text="⚙️ Settings",
            command=self._show_settings,
            style="Action.TButton",
        )
        settings_btn.pack(fill="x")

    def _create_content_area(self):
        """Create the main content area."""
        self.content_frame = ttk.Frame(self.main_container, style="Main.TFrame")
        self.content_frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)

        # Welcome screen (shown initially)
        self._show_welcome()

    def _create_status_bar(self):
        """Create the status bar."""
        self.status_bar = StatusBar(self.root)
        self.status_bar.pack(side="bottom", fill="x", padx=10, pady=5)

        self.status_bar.add_section("status", "Ready", width=25)
        self.status_bar.add_section("video", "No video loaded", width=35)
        self.status_bar.add_section("zone", "Zone: Enabled", width=15)
        self.status_bar.add_section("save_status", "", width=30)

    def _clear_content(self):
        """Clear the content area."""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        self.active_panel = None

    def _show_welcome(self):
        """Show the welcome screen."""
        self._clear_content()

        welcome_frame = ttk.Frame(self.content_frame, style="Main.TFrame")
        welcome_frame.pack(expand=True)

        # Welcome message
        welcome_label = tk.Label(
            welcome_frame,
            text="Welcome to DadBot Traffic Monitor",
            font=("Segoe UI", 20, "bold"),
            fg=COLORS["text_primary"],
            bg=COLORS["bg_dark"],
        )
        welcome_label.pack(pady=20)

        description = tk.Label(
            welcome_frame,
            text="Select an option from the sidebar to get started.\n\n"
                 "1. Define your road zone for accurate detection\n"
                 "2. Calibrate distance measurements for speed accuracy\n"
                 "3. Use Live Viewer to preview and adjust settings\n"
                 "4. Process videos to generate traffic reports",
            font=FONTS["body"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_dark"],
            justify="left",
        )
        description.pack(pady=20)

    def _show_zone_panel(self):
        """Show the zone definition panel."""
        self._clear_content()
        self.active_panel = ZoneDefinitionPanel(
            self.content_frame,
            self.settings,
            self._on_settings_changed,
            self._update_save_status,
        )
        self.active_panel.pack(fill="both", expand=True)
        self.status_bar.update_section("status", "Zone Definition")
        self.status_bar.update_section("save_status", "")

    def _show_calibration_panel(self):
        """Show the calibration panel."""
        self._clear_content()
        self.active_panel = CalibrationPanel(
            self.content_frame,
            self.settings,
            self._on_settings_changed,
            self._update_save_status,
        )
        self.active_panel.pack(fill="both", expand=True)
        self.status_bar.update_section("status", "Calibration")
        self.status_bar.update_section("save_status", "")

    def _show_viewer_panel(self):
        """Show the live viewer panel."""
        self._clear_content()
        self.active_panel = LiveViewerPanel(
            self.content_frame,
            self.settings,
            self._on_settings_changed,
        )
        self.active_panel.pack(fill="both", expand=True)
        self.status_bar.update_section("status", "Live Viewer")

    def _show_processor_panel(self):
        """Show the video processor panel."""
        self._clear_content()
        self.active_panel = ProcessorPanel(
            self.content_frame,
            self.settings,
            self._on_settings_changed,
        )
        self.active_panel.pack(fill="both", expand=True)
        self.status_bar.update_section("status", "Processing")

    def _show_settings(self):
        """Show the settings dialog."""
        # Create a settings dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Settings")
        dialog.geometry("500x400")
        dialog.configure(bg=COLORS["bg_dark"])
        dialog.transient(self.root)
        dialog.grab_set()

        # Settings content
        content = ttk.Frame(dialog, style="Main.TFrame")
        content.pack(fill="both", expand=True, padx=20, pady=20)

        # Config file path
        ttk.Label(
            content,
            text="Configuration File",
            style="Subheading.TLabel",
        ).pack(anchor="w", pady=(0, 10))

        path_frame = ttk.Frame(content, style="CardInner.TFrame")
        path_frame.pack(fill="x", pady=5)

        config_path = tk.StringVar(value=str(AppSettings.get_default_path()))
        path_entry = ttk.Entry(path_frame, textvariable=config_path, font=FONTS["mono_small"])
        path_entry.pack(side="left", fill="x", expand=True)

        def browse_config():
            path = filedialog.askopenfilename(
                filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
            )
            if path:
                config_path.set(path)

        ttk.Button(path_frame, text="Browse", command=browse_config).pack(side="right", padx=5)

        # Buttons
        btn_frame = ttk.Frame(content, style="Main.TFrame")
        btn_frame.pack(fill="x", pady=20)

        def load_config():
            try:
                self.settings = AppSettings.load(config_path.get())
                messagebox.showinfo("Success", "Settings loaded successfully!")
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load settings: {e}")

        def save_config():
            try:
                self.settings.save(config_path.get())
                messagebox.showinfo("Success", "Settings saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save settings: {e}")

        def reset_config():
            if messagebox.askyesno("Confirm", "Reset all settings to defaults?"):
                self.settings = AppSettings()
                messagebox.showinfo("Success", "Settings reset to defaults!")

        ttk.Button(btn_frame, text="Load", command=load_config).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Save", command=save_config).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Reset Defaults", command=reset_config).pack(side="right", padx=5)

    def _on_settings_changed(self):
        """Callback when settings are modified."""
        # Auto-save settings
        try:
            self.settings.save(AppSettings.get_default_path())
        except Exception:
            pass  # Silently fail auto-save

        # Update status bar
        zone_status = "Zone: Enabled" if self.settings.zone.enabled else "Zone: Disabled"
        self.status_bar.update_section("zone", zone_status)

    def _update_save_status(self, message: str):
        """Update the save status in the status bar."""
        self.status_bar.update_section("save_status", message)

    def _load_logo(self):
        """Load and display the logo image."""
        png_path = ASSETS_DIR / "dadbot-logo.png"
        gif_path = ASSETS_DIR / "dadbot-logo.gif"

        # Try animated GIF first
        if gif_path.exists():
            try:
                self._load_animated_gif(gif_path)
                return
            except Exception:
                pass

        # Fall back to static PNG
        if png_path.exists():
            try:
                img = Image.open(png_path)
                img = img.resize((290, 181), Image.Resampling.LANCZOS)
                self._logo_image = ImageTk.PhotoImage(img)
                self.logo_canvas.create_image(145, 91, image=self._logo_image)
                return
            except Exception:
                pass

        self._draw_fallback_logo()

    def _load_animated_gif(self, gif_path):
        """Load and animate a GIF image."""
        self._gif_frames = []
        self._gif_index = 0

        img = Image.open(gif_path)

        # Extract all frames
        try:
            while True:
                frame = img.copy()
                frame = frame.resize((290, 181), Image.Resampling.LANCZOS)
                self._gif_frames.append(ImageTk.PhotoImage(frame))
                img.seek(img.tell() + 1)
        except EOFError:
            pass

        if not self._gif_frames:
            raise ValueError("No frames in GIF")

        # Get frame delay (default 100ms if not specified)
        try:
            self._gif_delay = img.info.get("duration", 100)
        except Exception:
            self._gif_delay = 100

        # Display first frame and start animation
        self._logo_image_id = self.logo_canvas.create_image(145, 91, image=self._gif_frames[0])
        self._animate_gif()

    def _animate_gif(self):
        """Cycle through GIF frames."""
        if hasattr(self, "_gif_frames") and self._gif_frames:
            self._gif_index = (self._gif_index + 1) % len(self._gif_frames)
            self.logo_canvas.itemconfig(self._logo_image_id, image=self._gif_frames[self._gif_index])
            self.root.after(self._gif_delay, self._animate_gif)

    def _draw_fallback_logo(self):
        """Draw a simple text logo as fallback."""
        self.logo_canvas.create_text(
            145,
            70,
            text="DadBot",
            font=("Segoe UI", 28, "bold"),
            fill=COLORS["accent"],
        )
        self.logo_canvas.create_text(
            145,
            110,
            text="Traffic Monitor",
            font=("Segoe UI", 12),
            fill=COLORS["text_secondary"],
        )

    def run(self):
        """Start the application main loop."""
        self.root.mainloop()


def main():
    """Main entry point for the GUI application."""
    app = DadBotApp()
    app.run()


if __name__ == "__main__":
    main()
