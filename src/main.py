"""Main entry point for the traffic monitoring pipeline.

This module orchestrates the complete pipeline:
1. Load video source
2. Detect vehicles using RF-DETR via Roboflow
3. Track vehicles and estimate speeds
4. Generate annotated visualization
5. Output aggregated reports
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import supervision as sv
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.config import (
    AppConfig,
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


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Traffic monitoring system for residential street footage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output
    parser.add_argument(
        "--source-video",
        type=str,
        required=True,
        help="Path to source video file or camera index (0 for webcam)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory for output files",
    )

    # Model settings
    parser.add_argument(
        "--model-id",
        type=str,
        default="rfdetr-base",
        help="Roboflow model ID (e.g., rfdetr-base, yolov8n-640)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Detection confidence threshold",
    )

    # Calibration (defaults based on 20.16ft truck = 269 pixels)
    parser.add_argument(
        "--reference-distance",
        type=float,
        default=20.16,
        help="Known reference distance in feet (calibrated: truck length)",
    )
    parser.add_argument(
        "--reference-start-x",
        type=int,
        default=0,
        help="X pixel of reference line start",
    )
    parser.add_argument(
        "--reference-end-x",
        type=int,
        default=269,
        help="X pixel of reference line end",
    )

    # Speed settings
    parser.add_argument(
        "--speed-limit",
        type=float,
        default=25.0,
        help="Speed limit in MPH for violation detection",
    )

    # Visualization
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable live preview window",
    )
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Disable saving annotated video",
    )
    parser.add_argument(
        "--show-calibration",
        action="store_true",
        help="Show calibration zone overlay",
    )

    # Reporting
    parser.add_argument(
        "--aggregation-window",
        type=int,
        default=300,
        help="Aggregation window in seconds",
    )

    # Road zone filtering
    parser.add_argument(
        "--zone",
        type=str,
        default=None,
        help="Road zone polygon as JSON: '[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]'. Uses default if not specified.",
    )
    parser.add_argument(
        "--no-zone",
        action="store_true",
        help="Disable road zone filtering (include all detections)",
    )
    parser.add_argument(
        "--show-zone",
        action="store_true",
        help="Show road zone overlay on output",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    """Build configuration from command line arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        Complete application configuration.
    """
    import json

    # Get API key from environment
    api_key = os.environ.get("ROBOFLOW_API_KEY")

    calibration = CalibrationConfig(
        reference_distance_feet=args.reference_distance,
        reference_pixel_start_x=args.reference_start_x,
        reference_pixel_end_x=args.reference_end_x,
    )

    detection = DetectionConfig(
        roboflow_api_key=api_key,
        model_id=args.model_id,
        confidence_threshold=args.confidence,
    )

    tracking = TrackingConfig(
        speed_limit_mph=args.speed_limit,
    )

    reporting = ReportingConfig(
        output_dir=Path(args.output_dir),
        aggregation_window_seconds=args.aggregation_window,
    )

    # Zone configuration: use defaults unless overridden
    if args.no_zone:
        # Explicitly disabled
        zone = ZoneConfig(enabled=False, polygon_points=[], show_zone=False)
    elif args.zone:
        # Custom zone provided
        try:
            zone_points = json.loads(args.zone)
            zone = ZoneConfig(
                enabled=True,
                polygon_points=zone_points,
                show_zone=args.show_zone,
            )
        except json.JSONDecodeError:
            print(f"Warning: Invalid zone JSON, using default zone: {args.zone}")
            zone = ZoneConfig(show_zone=args.show_zone)
    else:
        # Use default zone from config
        zone = ZoneConfig(show_zone=args.show_zone)

    visualization = VisualizationConfig(
        show_preview=not args.no_preview,
        save_video=not args.no_save_video,
        show_calibration_zone=args.show_calibration,
    )

    return AppConfig(
        calibration=calibration,
        detection=detection,
        tracking=tracking,
        reporting=reporting,
        zone=zone,
        visualization=visualization,
    )


def run_pipeline(source_path: str, config: AppConfig) -> None:
    """Run the traffic monitoring pipeline.

    Args:
        source_path: Path to video source.
        config: Application configuration.
    """
    # Load video info
    print(f"Loading video: {source_path}")
    video_info = sv.VideoInfo.from_video_path(video_path=source_path)
    print(f"Resolution: {video_info.resolution_wh}")
    print(f"FPS: {video_info.fps}")
    print(f"Total frames: {video_info.total_frames}")

    # Calculate calibration
    print(f"\nCalibration:")
    print(f"  Reference distance: {config.calibration.reference_distance_feet} ft")
    print(f"  Pixel range: {config.calibration.reference_pixel_start_x} - {config.calibration.reference_pixel_end_x}")
    print(f"  Pixels per foot: {config.calibration.pixels_per_foot:.2f}")

    # Zone filtering info
    if config.zone.enabled:
        print(f"\nRoad Zone: Enabled ({len(config.zone.polygon_points)} points)")
    else:
        print(f"\nRoad Zone: Disabled (all detections included)")

    # Initialize components
    print("\nInitializing components...")
    detector = VehicleDetector(config.detection, zone_config=config.zone)
    tracker = VehicleTracker(
        calibration_config=config.calibration,
        tracking_config=config.tracking,
        fps=video_info.fps,
    )
    reporter = TrafficReporter(
        config=config.reporting,
        speed_limit_mph=config.tracking.speed_limit_mph,
    )
    visualizer = TrafficVisualizer(
        video_info=video_info,
        vis_config=config.visualization,
        calibration_config=config.calibration,
        tracking_config=config.tracking,
        zone_config=config.zone,
    )

    # Start reporting session
    report_path = reporter.start_session()
    print(f"Reports will be saved to: {report_path}")

    # Start video output if enabled
    if config.visualization.save_video:
        output_video_path = config.reporting.output_dir / "annotated_output.mp4"
        visualizer.start_video_output(output_video_path)
        print(f"Annotated video will be saved to: {output_video_path}")

    # Process video
    print("\nProcessing video...")
    print("Press 'q' to quit\n")

    frame_generator = sv.get_video_frames_generator(source_path=source_path)

    try:
        for frame_index, frame in enumerate(frame_generator):
            # Detect vehicles
            detection_result = detector.detect(frame, frame_index)

            # Track and estimate speeds
            tracking_result = tracker.update(detection_result)

            # Record vehicles in report
            for vehicle in tracking_result.tracked_vehicles.values():
                reporter.record_vehicle(vehicle)

            # Check if aggregation window complete
            if reporter.check_window_complete():
                window_data = reporter.finalize_window()
                if window_data:
                    print(f"\n[Window Complete] Vehicles: {window_data['total_vehicles']}, "
                          f"Top Speed: {window_data['top_speed_mph']} MPH, "
                          f"Violations: {len(window_data['speed_violations'])}")

            # Visualize
            annotated_frame = visualizer.annotate_frame(frame, tracking_result)

            # Write to video
            if config.visualization.save_video:
                visualizer.write_frame(annotated_frame)

            # Show preview
            if config.visualization.show_preview:
                if not visualizer.show_frame(annotated_frame):
                    print("\nUser quit")
                    break

            # Progress update
            if frame_index > 0 and frame_index % 100 == 0:
                summary = reporter.get_summary()
                print(f"Frame {frame_index}/{video_info.total_frames} | "
                      f"Vehicles: {summary.get('vehicles_counted', 0)} | "
                      f"Violations: {summary.get('violations', 0)}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        print("\nFinalizing...")
        final_report = reporter.end_session()
        visualizer.cleanup()

        if final_report:
            print(f"\nFinal window: {final_report['total_vehicles']} vehicles")

        print(f"\nReport saved to: {report_path}")
        if config.visualization.save_video:
            print(f"Video saved to: {config.reporting.output_dir / 'annotated_output.mp4'}")


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    # Validate source
    source_path = args.source_video
    if not source_path.isdigit() and not Path(source_path).exists():
        print(f"Error: Source video not found: {source_path}")
        sys.exit(1)

    # Build configuration
    config = build_config(args)

    # Check for API key
    if config.detection.roboflow_api_key is None:
        print("Error: ROBOFLOW_API_KEY environment variable not set")
        print("Please set it with: export ROBOFLOW_API_KEY=your_api_key")
        sys.exit(1)

    # Run pipeline
    try:
        run_pipeline(source_path, config)
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
