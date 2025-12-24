"""Main GUI Application for DadBot Traffic Monitor."""

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.gui.components import ActionButton, ConsoleOutput, SettingsPanel, StatusBar
from src.gui.styles import COLORS, FONTS, configure_styles
from src.gui.panels.zone_panel import ZoneDefinitionPanel
from src.gui.panels.calibration_panel import CalibrationPanel
from src.gui.panels.viewer_panel import LiveViewerPanel
from src.gui.panels.processor_panel import ProcessorPanel
from src.settings import AppSettings, load_or_create_settings


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

        # App title
        title_frame = ttk.Frame(sidebar, style="Card.TFrame")
        title_frame.pack(fill="x", padx=20, pady=20)

        title_label = tk.Label(
            title_frame,
            text="DadBot",
            font=("Segoe UI", 24, "bold"),
            fg=COLORS["accent"],
            bg=COLORS["bg_medium"],
        )
        title_label.pack(anchor="w")

        subtitle_label = tk.Label(
            title_frame,
            text="Traffic Monitor",
            font=FONTS["body"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_medium"],
        )
        subtitle_label.pack(anchor="w")

        # Separator
        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=20, pady=10)

        # Navigation buttons
        nav_frame = ttk.Frame(sidebar, style="Card.TFrame")
        nav_frame.pack(fill="x", padx=10, pady=10)

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

        # Spacer
        ttk.Frame(sidebar, style="Card.TFrame").pack(fill="both", expand=True)

        # Settings button at bottom
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

        self.status_bar.add_section("status", "Ready", width=30)
        self.status_bar.add_section("video", "No video loaded", width=40)
        self.status_bar.add_section("zone", "Zone: Enabled", width=20)

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
        )
        self.active_panel.pack(fill="both", expand=True)
        self.status_bar.update_section("status", "Zone Definition")

    def _show_calibration_panel(self):
        """Show the calibration panel."""
        self._clear_content()
        self.active_panel = CalibrationPanel(
            self.content_frame,
            self.settings,
            self._on_settings_changed,
        )
        self.active_panel.pack(fill="both", expand=True)
        self.status_bar.update_section("status", "Calibration")

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

    def run(self):
        """Start the application main loop."""
        self.root.mainloop()


def main():
    """Main entry point for the GUI application."""
    app = DadBotApp()
    app.run()


if __name__ == "__main__":
    main()
