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


# Vehicle class names to VehicleClass mapping
# Using class names instead of IDs for model compatibility
VEHICLE_CLASS_NAMES = {
    "car": VehicleClass.CAR,
    "truck": VehicleClass.TRUCK,
    "bus": VehicleClass.BUS,
    "motorcycle": VehicleClass.MOTORCYCLE,
    # Additional aliases
    "motorbike": VehicleClass.MOTORCYCLE,
    "van": VehicleClass.TRUCK,  # Map van to truck
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

    # Model selection - RF-DETR (Roboflow's SOTA real-time detector)
    # Options: rfdetr-base, rfdetr-large, or yolov8{n,s,m,l,x}-{resolution}
    model_id: Annotated[
        str, Field(description="Roboflow model ID")
    ] = "rfdetr-base"

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


class ZoneConfig(BaseSettings):
    """Road zone configuration for filtering detections.

    Defines a polygon representing the road surface. Only vehicles
    within this zone are tracked and counted.
    """

    model_config = SettingsConfigDict(env_prefix="DADBOT_ZONE_")

    # Enable zone filtering
    enabled: Annotated[
        bool, Field(description="Enable road zone filtering")
    ] = True

    # Road zone polygon points as list of [x, y] coordinates
    # Define clockwise or counter-clockwise from any corner
    # Default: calibrated road zone for reference camera position
    polygon_points: Annotated[
        list[list[int]], Field(description="Polygon points defining road zone [[x,y], ...]")
    ] = [[3, 287], [1078, 412], [1081, 554], [4, 411]]

    # Show zone overlay on video
    show_zone: Annotated[
        bool, Field(description="Draw road zone overlay on output")
    ] = True


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
    zone: ZoneConfig = ZoneConfig()
    visualization: VisualizationConfig = VisualizationConfig()

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        return cls()


class LensCorrector:
    """Coordinate correction for lens distortion.

    This class provides mathematical coordinate transformation to correct
    for barrel/pincushion distortion without modifying the actual video frames.
    Use this for accurate speed calculations across the entire frame.
    """

    def __init__(
        self,
        k1: float = 0.0,
        k2: float = 0.0,
        frame_width: int = 1920,
        frame_height: int = 1080,
        center_x: float | None = None,
        center_y: float | None = None,
    ):
        """Initialize the lens corrector.

        Args:
            k1: Primary radial distortion coefficient (negative = barrel)
            k2: Secondary radial distortion coefficient
            frame_width: Width of the frame in pixels
            frame_height: Height of the frame in pixels
            center_x: Distortion center X (defaults to frame center)
            center_y: Distortion center Y (defaults to frame center)
        """
        self.k1 = k1
        self.k2 = k2
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cx = center_x if center_x is not None else frame_width / 2
        self.cy = center_y if center_y is not None else frame_height / 2
        self.max_dim = max(frame_width, frame_height)

        # Check if correction is effectively disabled
        self.enabled = abs(k1) > 1e-6 or abs(k2) > 1e-6

    def undistort_point(self, x: float, y: float) -> tuple[float, float]:
        """Undistort a single point coordinate.

        Transforms a point from distorted (observed) coordinates to
        undistorted (true) coordinates.

        Args:
            x: X coordinate in distorted frame
            y: Y coordinate in distorted frame

        Returns:
            Tuple of (undistorted_x, undistorted_y)
        """
        if not self.enabled:
            return x, y

        # Normalize to center
        xn = (x - self.cx) / self.max_dim
        yn = (y - self.cy) / self.max_dim

        # Radial distance squared
        r2 = xn * xn + yn * yn
        r4 = r2 * r2

        # Distortion factor
        factor = 1 + self.k1 * r2 + self.k2 * r4

        # Apply correction
        xu = xn * factor
        yu = yn * factor

        # Convert back to pixel coordinates
        return xu * self.max_dim + self.cx, yu * self.max_dim + self.cy

    def undistort_points(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Undistort multiple points.

        Args:
            points: List of (x, y) coordinate tuples

        Returns:
            List of undistorted (x, y) coordinate tuples
        """
        return [self.undistort_point(x, y) for x, y in points]

    def undistort_bbox_center(self, x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
        """Get undistorted center of a bounding box.

        Args:
            x1, y1: Top-left corner
            x2, y2: Bottom-right corner

        Returns:
            Undistorted (center_x, center_y)
        """
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return self.undistort_point(cx, cy)

    def correct_pixel_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate the true pixel distance between two points.

        This corrects for lens distortion when measuring distances
        for speed calculations.

        Args:
            x1, y1: First point coordinates
            x2, y2: Second point coordinates

        Returns:
            Corrected distance in pixels
        """
        ux1, uy1 = self.undistort_point(x1, y1)
        ux2, uy2 = self.undistort_point(x2, y2)
        return ((ux2 - ux1) ** 2 + (uy2 - uy1) ** 2) ** 0.5
