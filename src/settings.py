"""Persistent settings management using YAML configuration files.

This module provides a unified configuration system that can be saved/loaded
from YAML files, with sensible defaults for all settings.
"""

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LeadCornerMode(str, Enum):
    """Mode for determining which corner to draw trails from."""

    DIRECTION_BASED = "direction_based"  # Bottom-right for eastbound, bottom-left for westbound
    CENTER_BOTTOM = "center_bottom"  # Always use bottom center
    CENTER = "center"  # Always use center of bounding box
    BOTTOM_LEFT = "bottom_left"  # Always use bottom-left
    BOTTOM_RIGHT = "bottom_right"  # Always use bottom-right


class LabelDisplayMode(str, Enum):
    """What information to show in vehicle labels."""

    FULL = "full"  # ID, class, speed, direction
    SPEED_ONLY = "speed_only"  # Just speed
    ID_SPEED = "id_speed"  # ID and speed
    MINIMAL = "minimal"  # Just ID
    NONE = "none"  # No labels


class SavedZone(BaseModel):
    """A saved zone configuration."""

    name: str
    points: list[list[int]]
    saved_at: str  # ISO format timestamp


class SavedCalibration(BaseModel):
    """A saved calibration configuration."""

    name: str
    reference_distance_feet: float
    point1: list[int]  # [x, y]
    point2: list[int]  # [x, y]
    saved_at: str  # ISO format timestamp


class ZoneSettings(BaseModel):
    """Road zone filtering settings."""

    enabled: bool = True
    polygon_points: list[list[int]] = Field(
        default=[[3, 287], [1078, 412], [1081, 554], [4, 411]]
    )
    show_overlay: bool = True
    overlay_color: list[int] = [0, 255, 255]  # BGR Yellow
    overlay_opacity: float = 0.2

    # Named saved zones
    saved_zones: list[SavedZone] = Field(default_factory=list)
    # History of saved states (most recent first)
    save_history: list[SavedZone] = Field(default_factory=list)
    max_history: int = 20


class CalibrationSettings(BaseModel):
    """Speed calibration settings."""

    reference_distance_feet: float = 20.16
    reference_pixel_start_x: int = 0
    reference_pixel_end_x: int = 269
    reference_pixel_y: int = 400

    # Named saved calibrations
    saved_calibrations: list["SavedCalibration"] = Field(default_factory=list)
    # History of saved states (most recent first)
    save_history: list["SavedCalibration"] = Field(default_factory=list)
    max_history: int = 20


class DetectionSettings(BaseModel):
    """Object detection settings."""

    model_id: str = "rfdetr-base"
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.5


class TrackingSettings(BaseModel):
    """Vehicle tracking settings."""

    min_frames_for_speed: int = 10
    track_buffer: int = 30
    speed_limit_mph: float = 25.0
    commercial_vehicle_min_length_feet: float = 25.0


class VisualizationSettings(BaseModel):
    """Visualization and annotation settings."""

    # Trail settings
    lead_corner_mode: LeadCornerMode = LeadCornerMode.DIRECTION_BASED
    trace_length_seconds: float = 2.0
    show_traces: bool = True

    # Label settings
    label_mode: LabelDisplayMode = LabelDisplayMode.FULL
    show_labels: bool = True

    # Bounding box settings
    show_bounding_boxes: bool = True
    highlight_violations: bool = True

    # Zone overlays
    show_zone_overlay: bool = True
    show_calibration_overlay: bool = False

    # Stats overlay
    show_stats_overlay: bool = True

    # Colors (BGR format)
    color_eastbound: list[int] = [0, 255, 0]  # Green
    color_westbound: list[int] = [255, 0, 0]  # Blue
    color_violation: list[int] = [0, 0, 255]  # Red
    color_unknown: list[int] = [128, 128, 128]  # Gray


class ReportingSettings(BaseModel):
    """Report generation settings."""

    output_dir: str = "./output"
    aggregation_window_seconds: int = 300
    left_to_right_label: str = "eastbound"
    right_to_left_label: str = "westbound"


class AppSettings(BaseModel):
    """Complete application settings."""

    zone: ZoneSettings = Field(default_factory=ZoneSettings)
    calibration: CalibrationSettings = Field(default_factory=CalibrationSettings)
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    visualization: VisualizationSettings = Field(default_factory=VisualizationSettings)
    reporting: ReportingSettings = Field(default_factory=ReportingSettings)

    # Last used paths (for GUI convenience)
    last_video_path: str | None = None
    last_output_path: str | None = None
    last_rtsp_url: str | None = None

    def save(self, path: str | Path) -> None:
        """Save settings to a YAML file.

        Args:
            path: Path to save the settings file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and handle enums
        data = self._to_serializable(self.model_dump())

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> "AppSettings":
        """Load settings from a YAML file.

        Args:
            path: Path to the settings file.

        Returns:
            Loaded AppSettings instance.
        """
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls.model_validate(data)

    @classmethod
    def get_default_path(cls) -> Path:
        """Get the default settings file path."""
        return Path.home() / ".dadbot" / "settings.yaml"

    def _to_serializable(self, obj: Any) -> Any:
        """Convert object to YAML-serializable format."""
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_serializable(v) for v in obj]
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, Path):
            return str(obj)
        return obj


def load_or_create_settings(path: str | Path | None = None) -> AppSettings:
    """Load settings from file or create defaults.

    Args:
        path: Optional path to settings file. Uses default if not provided.

    Returns:
        AppSettings instance.
    """
    if path is None:
        path = AppSettings.get_default_path()
    else:
        path = Path(path)

    if path.exists():
        return AppSettings.load(path)
    else:
        settings = AppSettings()
        settings.save(path)
        return settings
