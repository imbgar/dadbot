"""Configuration management using Pydantic settings.

This module defines all configurable parameters for the traffic monitoring system,
including calibration settings, detection thresholds, and reporting options.
"""

from enum import Enum
from pathlib import Path
from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VehicleClass(str, Enum):
    """Standard vehicle classifications from COCO dataset."""

    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"


class SpecialVehicleType(str, Enum):
    """Extended vehicle classifications for special reporting."""

    EMERGENCY = "emergency"
    SCHOOL_BUS = "school_bus"
    TOW_TRUCK = "tow_truck"
    COMMERCIAL = "commercial"


class Direction(str, Enum):
    """Vehicle travel direction relative to camera view."""

    EASTBOUND = "eastbound"  # Left to right in frame
    WESTBOUND = "westbound"  # Right to left in frame


# COCO class IDs for vehicles (used by RF-DETR)
COCO_VEHICLE_CLASS_IDS = {
    2: VehicleClass.CAR,       # car
    3: VehicleClass.MOTORCYCLE,  # motorcycle
    5: VehicleClass.BUS,       # bus
    7: VehicleClass.TRUCK,     # truck
}


class CalibrationConfig(BaseSettings):
    """Calibration settings for speed estimation.

    For side-view footage, we need to map pixel distances to real-world distances.
    The calibration uses a known reference measurement (e.g., road width) to
    establish the pixels-per-foot ratio.
    """

    model_config = SettingsConfigDict(env_prefix="DADBOT_CALIBRATION_")

    # Known real-world distance for calibration (in feet)
    # Calibrated using truck length: 20.16 ft = 269 pixels
    reference_distance_feet: Annotated[
        float, Field(description="Known real-world distance in feet (e.g., vehicle length)")
    ] = 20.16

    # Pixel coordinates defining the calibration reference line
    # These should be set based on the camera view - measured horizontally
    # Default calibration: 269 pixels for 20.16 ft (13.34 pixels/foot)
    reference_pixel_start_x: Annotated[
        int, Field(description="X pixel coordinate of reference line start")
    ] = 0

    reference_pixel_end_x: Annotated[
        int, Field(description="X pixel coordinate of reference line end")
    ] = 269

    # Y-coordinate for the calibration line (typically at road level)
    reference_pixel_y: Annotated[
        int, Field(description="Y pixel coordinate of reference line (road level)")
    ] = 400

    @property
    def pixels_per_foot(self) -> float:
        """Calculate pixels per foot based on reference measurement."""
        pixel_distance = abs(self.reference_pixel_end_x - self.reference_pixel_start_x)
        return pixel_distance / self.reference_distance_feet


class DetectionConfig(BaseSettings):
    """Detection model configuration."""

    model_config = SettingsConfigDict(env_prefix="DADBOT_DETECTION_")

    # Roboflow API key for cloud inference
    roboflow_api_key: Annotated[
        str | None, Field(description="Roboflow API key for cloud inference")
    ] = None

    # Model selection - use YOLOv8 with COCO classes
    # Format: yolov8{n,s,m,l,x}-{resolution} e.g., yolov8n-640, yolov8x-1280
    model_id: Annotated[
        str, Field(description="Roboflow model ID")
    ] = "yolov8n-640"

    # Detection thresholds
    confidence_threshold: Annotated[
        float, Field(ge=0.0, le=1.0, description="Minimum confidence for detections")
    ] = 0.3

    iou_threshold: Annotated[
        float, Field(ge=0.0, le=1.0, description="IOU threshold for NMS")
    ] = 0.5


class TrackingConfig(BaseSettings):
    """Tracking and speed estimation configuration."""

    model_config = SettingsConfigDict(env_prefix="DADBOT_TRACKING_")

    # Minimum frames to track before calculating speed
    min_frames_for_speed: Annotated[
        int, Field(ge=1, description="Minimum frames before speed calculation")
    ] = 10

    # Maximum track age before deletion (in frames)
    track_buffer: Annotated[
        int, Field(ge=1, description="Maximum frames to keep lost tracks")
    ] = 30

    # Speed limit for violation detection (MPH)
    speed_limit_mph: Annotated[
        float, Field(ge=0.0, description="Speed limit in MPH for violation detection")
    ] = 25.0

    # Threshold for commercial vehicle detection (approximate length in feet)
    commercial_vehicle_min_length_feet: Annotated[
        float, Field(ge=0.0, description="Minimum length to classify as commercial")
    ] = 25.0


class ReportingConfig(BaseSettings):
    """Reporting and output configuration."""

    model_config = SettingsConfigDict(env_prefix="DADBOT_REPORTING_")

    # Output directory for reports
    output_dir: Annotated[
        Path, Field(description="Directory for output files")
    ] = Path("./output")

    # Aggregation window in seconds
    aggregation_window_seconds: Annotated[
        int, Field(ge=1, description="Aggregation window in seconds")
    ] = 300  # 5 minutes

    # Direction labels (can be customized per installation)
    left_to_right_label: Annotated[
        str, Field(description="Label for left-to-right movement")
    ] = "eastbound"

    right_to_left_label: Annotated[
        str, Field(description="Label for right-to-left movement")
    ] = "westbound"


class VisualizationConfig(BaseSettings):
    """Visualization settings for annotated output."""

    model_config = SettingsConfigDict(env_prefix="DADBOT_VIS_")

    # Show live preview window
    show_preview: Annotated[
        bool, Field(description="Show live preview window")
    ] = True

    # Save annotated video
    save_video: Annotated[
        bool, Field(description="Save annotated video to file")
    ] = True

    # Trace length (in seconds)
    trace_length_seconds: Annotated[
        float, Field(ge=0.0, description="Length of trajectory trace in seconds")
    ] = 2.0

    # Draw calibration zone overlay
    show_calibration_zone: Annotated[
        bool, Field(description="Show calibration reference zone overlay")
    ] = False


class AppConfig(BaseSettings):
    """Main application configuration combining all sub-configs."""

    model_config = SettingsConfigDict(
        env_prefix="DADBOT_",
        env_nested_delimiter="__",
    )

    calibration: CalibrationConfig = CalibrationConfig()
    detection: DetectionConfig = DetectionConfig()
    tracking: TrackingConfig = TrackingConfig()
    reporting: ReportingConfig = ReportingConfig()
    visualization: VisualizationConfig = VisualizationConfig()

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        return cls()
