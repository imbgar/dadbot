"""Video Processor Panel for DadBot Traffic Monitor."""

import os
import queue
import sys
import threading
import tkinter as tk
from datetime import datetime
from io import StringIO
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from src.gui.components import ConsoleOutput
from src.gui.styles import COLORS, FONTS
from src.settings import AppSettings


class ProcessorPanel(ttk.Frame):
    """Panel for processing videos and generating reports."""

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
        self.processing = False
        self.process_thread: threading.Thread | None = None
        self.output_queue: queue.Queue = queue.Queue()

        self._create_layout()

    def _create_layout(self):
        """Create the panel layout."""
        # Header
        header = ttk.Frame(self, style="Main.TFrame")
        header.pack(fill="x", pady=(0, 15))

        ttk.Label(
            header,
            text="Process Video",
            style="Heading.TLabel",
        ).pack(side="left")

        # Main content - split horizontally
        content = ttk.Frame(self, style="Main.TFrame")
        content.pack(fill="both", expand=True)

        # Top section - Settings
        settings_frame = ttk.Frame(content, style="Card.TFrame")
        settings_frame.pack(fill="x", pady=(0, 10))

        settings_inner = ttk.Frame(settings_frame, style="CardInner.TFrame")
        settings_inner.pack(fill="x", padx=15, pady=15)

        # Video source row
        source_row = ttk.Frame(settings_inner, style="CardInner.TFrame")
        source_row.pack(fill="x", pady=5)

        ttk.Label(source_row, text="Video Source:", style="Body.TLabel", width=15).pack(side="left")

        self.video_path_var = tk.StringVar()
        video_entry = ttk.Entry(source_row, textvariable=self.video_path_var, font=FONTS["small"])
        video_entry.pack(side="left", fill="x", expand=True, padx=5)

        browse_btn = ttk.Button(source_row, text="Browse", command=self._browse_video)
        browse_btn.pack(side="right")

        # Output directory row
        output_row = ttk.Frame(settings_inner, style="CardInner.TFrame")
        output_row.pack(fill="x", pady=5)

        ttk.Label(output_row, text="Output Directory:", style="Body.TLabel", width=15).pack(side="left")

        self.output_path_var = tk.StringVar(value=self.settings.reporting.output_dir)
        output_entry = ttk.Entry(output_row, textvariable=self.output_path_var, font=FONTS["small"])
        output_entry.pack(side="left", fill="x", expand=True, padx=5)

        output_browse_btn = ttk.Button(output_row, text="Browse", command=self._browse_output)
        output_browse_btn.pack(side="right")

        # Options row
        options_row = ttk.Frame(settings_inner, style="CardInner.TFrame")
        options_row.pack(fill="x", pady=10)

        self.save_video_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_row,
            text="Save annotated video",
            variable=self.save_video_var,
            style="Modern.TCheckbutton",
        ).pack(side="left", padx=10)

        # Action buttons row
        action_row = ttk.Frame(settings_inner, style="CardInner.TFrame")
        action_row.pack(fill="x", pady=10)

        self.start_btn = ttk.Button(
            action_row,
            text="▶ Start Processing",
            command=self._start_processing,
            style="Accent.TButton",
        )
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(
            action_row,
            text="⏹ Stop",
            command=self._stop_processing,
            state="disabled",
        )
        self.stop_btn.pack(side="left", padx=5)

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            action_row,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
            style="Modern.Horizontal.TProgressbar",
        )
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=20)

        self.progress_label = tk.Label(
            action_row,
            text="0%",
            font=FONTS["mono"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_medium"],
            width=6,
        )
        self.progress_label.pack(side="right")

        # Bottom section - Console output
        console_frame = ttk.LabelFrame(content, text="Console Output", style="Card.TLabelframe")
        console_frame.pack(fill="both", expand=True)

        self.console = ConsoleOutput(console_frame)
        self.console.pack(fill="both", expand=True, padx=5, pady=5)

        # Initial message
        self.console.writeln("DadBot Traffic Monitor - Video Processor", "accent")
        self.console.writeln("=" * 50, "muted")
        self.console.writeln("")
        self.console.writeln("Select a video file and click 'Start Processing' to begin.", "info")
        self.console.writeln("")

        # Start checking output queue
        self._check_output_queue()

    def _browse_video(self):
        """Browse for video file."""
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

        if path:
            self.video_path_var.set(path)
            self.video_path = path
            self.settings.last_video_path = path
            self.on_change()

    def _browse_output(self):
        """Browse for output directory."""
        path = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_path_var.get(),
        )

        if path:
            self.output_path_var.set(path)
            self.settings.reporting.output_dir = path
            self.on_change()

    def _start_processing(self):
        """Start video processing."""
        video_path = self.video_path_var.get()
        if not video_path:
            messagebox.showwarning("Warning", "Please select a video file")
            return

        if not Path(video_path).exists():
            messagebox.showerror("Error", f"Video file not found: {video_path}")
            return

        # Clear console and update UI
        self.console.clear()
        self.console.writeln(f"Starting processing: {Path(video_path).name}", "accent")
        self.console.writeln(f"Output directory: {self.output_path_var.get()}", "info")
        self.console.writeln("")

        self.processing = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress_var.set(0)

        # Start processing in background thread
        self.process_thread = threading.Thread(target=self._process_video, daemon=True)
        self.process_thread.start()

    def _stop_processing(self):
        """Stop video processing."""
        self.processing = False
        self.output_queue.put(("info", "Stopping..."))

    def _process_video(self):
        """Process video in background thread."""
        try:
            video_path = self.video_path_var.get()
            output_dir = Path(self.output_path_var.get())
            output_dir.mkdir(parents=True, exist_ok=True)

            # Import processing modules
            self.output_queue.put(("info", "Loading modules..."))

            import cv2
            import supervision as sv
            from dotenv import load_dotenv

            load_dotenv()

            from src.config import (
                CalibrationConfig,
                DetectionConfig,
                ReportingConfig,
                TrackingConfig,
                VisualizationConfig,
                ZoneConfig,
            )
            from src.detector import VehicleDetector
            from src.reporter import TrafficReporter
            from src.tracker import VehicleTracker
            from src.visualizer import TrafficVisualizer

            # Build configuration from settings
            self.output_queue.put(("info", "Building configuration..."))

            calibration = CalibrationConfig(
                reference_distance_feet=self.settings.calibration.reference_distance_feet,
                reference_pixel_start_x=self.settings.calibration.reference_pixel_start_x,
                reference_pixel_end_x=self.settings.calibration.reference_pixel_end_x,
            )

            api_key = os.environ.get("ROBOFLOW_API_KEY")
            if not api_key:
                self.output_queue.put(("error", "ROBOFLOW_API_KEY not set in environment"))
                return

            detection = DetectionConfig(
                roboflow_api_key=api_key,
                model_id=self.settings.detection.model_id,
                confidence_threshold=self.settings.detection.confidence_threshold,
            )

            tracking = TrackingConfig(
                speed_limit_mph=self.settings.tracking.speed_limit_mph,
            )

            reporting = ReportingConfig(
                output_dir=output_dir,
            )

            zone = ZoneConfig(
                enabled=self.settings.zone.enabled,
                polygon_points=self.settings.zone.polygon_points,
                show_zone=self.settings.visualization.show_zone_overlay,
            )

            # Note: OpenCV imshow conflicts with tkinter, so disable preview in GUI mode
            visualization = VisualizationConfig(
                show_preview=False,
                save_video=self.save_video_var.get(),
            )

            # Load video
            self.output_queue.put(("info", f"Loading video: {video_path}"))
            video_info = sv.VideoInfo.from_video_path(video_path)
            self.output_queue.put(("info", f"Resolution: {video_info.resolution_wh}"))
            self.output_queue.put(("info", f"FPS: {video_info.fps}"))
            self.output_queue.put(("info", f"Total frames: {video_info.total_frames}"))
            self.output_queue.put(("info", f"Pixels per foot: {calibration.pixels_per_foot:.2f}"))
            self.output_queue.put(("info", ""))

            # Initialize components
            self.output_queue.put(("info", "Initializing detector..."))
            detector = VehicleDetector(detection, zone_config=zone)

            self.output_queue.put(("info", "Initializing tracker..."))
            tracker = VehicleTracker(
                calibration_config=calibration,
                tracking_config=tracking,
                fps=video_info.fps,
            )

            reporter = TrafficReporter(config=reporting, speed_limit_mph=tracking.speed_limit_mph)
            visualizer = TrafficVisualizer(
                video_info=video_info,
                vis_config=visualization,
                calibration_config=calibration,
                tracking_config=tracking,
                zone_config=zone,
            )

            # Start session
            report_path = reporter.start_session()
            self.output_queue.put(("success", f"Reports: {report_path}"))

            if visualization.save_video:
                output_video = output_dir / "annotated_output.mp4"
                visualizer.start_video_output(output_video)
                self.output_queue.put(("success", f"Video: {output_video}"))

            self.output_queue.put(("info", ""))
            self.output_queue.put(("accent", "Processing started..."))
            self.output_queue.put(("info", ""))

            # Process frames
            frame_generator = sv.get_video_frames_generator(source_path=video_path)
            total_frames = video_info.total_frames

            for frame_index, frame in enumerate(frame_generator):
                if not self.processing:
                    self.output_queue.put(("warning", "Processing stopped by user"))
                    break

                # Update progress
                progress = (frame_index / total_frames) * 100
                self.output_queue.put(("progress", progress))

                # Detect, track, record
                detection_result = detector.detect(frame, frame_index)
                tracking_result = tracker.update(detection_result)

                for vehicle in tracking_result.tracked_vehicles.values():
                    reporter.record_vehicle(vehicle)

                # Visualize
                annotated_frame = visualizer.annotate_frame(frame, tracking_result)

                if visualization.save_video:
                    visualizer.write_frame(annotated_frame)

                if visualization.show_preview:
                    if not visualizer.show_frame(annotated_frame):
                        self.processing = False
                        break

                # Log progress periodically
                if frame_index > 0 and frame_index % 100 == 0:
                    summary = reporter.get_summary()
                    self.output_queue.put((
                        "info",
                        f"Frame {frame_index}/{total_frames} | "
                        f"Vehicles: {summary.get('vehicles_counted', 0)} | "
                        f"Violations: {summary.get('violations', 0)}"
                    ))

            # Finalize
            self.output_queue.put(("info", ""))
            self.output_queue.put(("info", "Finalizing..."))

            final_report = reporter.end_session()
            visualizer.cleanup()

            if final_report:
                self.output_queue.put(("success", f"Total vehicles: {final_report['total_vehicles']}"))
                self.output_queue.put(("success", f"Eastbound: {final_report['vehicles_by_direction'].get('eastbound', 0)}"))
                self.output_queue.put(("success", f"Westbound: {final_report['vehicles_by_direction'].get('westbound', 0)}"))
                self.output_queue.put(("success", f"Top speed: {final_report['top_speed_mph']} MPH"))
                self.output_queue.put(("success", f"Violations: {len(final_report['speed_violations'])}"))

            self.output_queue.put(("info", ""))
            self.output_queue.put(("accent", "Processing complete!"))
            self.output_queue.put(("progress", 100))

        except Exception as e:
            self.output_queue.put(("error", f"Error: {e}"))
            import traceback
            self.output_queue.put(("muted", traceback.format_exc()))

        finally:
            self.output_queue.put(("done", None))

    def _check_output_queue(self):
        """Check output queue for messages from processing thread."""
        try:
            while True:
                msg_type, msg = self.output_queue.get_nowait()

                if msg_type == "progress":
                    self.progress_var.set(msg)
                    self.progress_label.configure(text=f"{msg:.0f}%")
                elif msg_type == "done":
                    self.processing = False
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                else:
                    self.console.writeln(msg, msg_type)

        except queue.Empty:
            pass

        # Schedule next check
        self.after(100, self._check_output_queue)

    def destroy(self):
        """Clean up resources."""
        self.processing = False
        super().destroy()
