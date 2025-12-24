"""Aggregated reporting for traffic monitoring.

This module handles aggregation of vehicle tracking data into time-windowed
reports, output as JSON Lines format for easy downstream processing.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TextIO

from src.config import Direction, ReportingConfig, SpecialVehicleType, VehicleClass
from src.tracker import TrackedVehicle


@dataclass
class SpeedViolation:
    """Record of a speed violation."""

    vehicle_id: str
    speed_mph: float
    timestamp: str
    vehicle_class: str
    direction: str | None


@dataclass
class SpecialVehicleSighting:
    """Record of a special vehicle sighting."""

    vehicle_type: str
    timestamp: str
    direction: str | None
    vehicle_id: str


@dataclass
class AggregationWindow:
    """Aggregated data for a single time window."""

    window_start: datetime
    window_end: datetime | None = None
    total_vehicles: int = 0
    vehicles_by_direction: dict[str, int] = field(
        default_factory=lambda: {"eastbound": 0, "westbound": 0}
    )
    vehicles_by_type: dict[str, int] = field(default_factory=dict)
    speeds_mph: list[float] = field(default_factory=list)
    speed_violations: list[SpeedViolation] = field(default_factory=list)
    special_vehicles: list[SpecialVehicleSighting] = field(default_factory=list)
    commercial_trucks: int = 0
    _seen_tracker_ids: set[int] = field(default_factory=set)
    _direction_assigned_ids: set[int] = field(default_factory=set)

    @property
    def top_speed_mph(self) -> float:
        """Get maximum speed observed in window."""
        return max(self.speeds_mph) if self.speeds_mph else 0.0

    @property
    def average_speed_mph(self) -> float:
        """Get average speed observed in window."""
        return sum(self.speeds_mph) / len(self.speeds_mph) if self.speeds_mph else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat() if self.window_end else None,
            "total_vehicles": self.total_vehicles,
            "vehicles_by_direction": self.vehicles_by_direction,
            "vehicles_by_type": self.vehicles_by_type,
            "top_speed_mph": round(self.top_speed_mph, 1),
            "average_speed_mph": round(self.average_speed_mph, 1),
            "speed_violations": [
                {
                    "vehicle_id": v.vehicle_id,
                    "speed_mph": round(v.speed_mph, 1),
                    "timestamp": v.timestamp,
                    "vehicle_class": v.vehicle_class,
                    "direction": v.direction,
                }
                for v in self.speed_violations
            ],
            "special_vehicles": [
                {
                    "type": s.vehicle_type,
                    "timestamp": s.timestamp,
                    "direction": s.direction,
                    "vehicle_id": s.vehicle_id,
                }
                for s in self.special_vehicles
            ],
            "commercial_trucks": self.commercial_trucks,
        }


class TrafficReporter:
    """Aggregates and reports traffic monitoring data.

    Collects vehicle tracking data into time-windowed aggregations
    and writes reports as JSON Lines files.
    """

    def __init__(self, config: ReportingConfig | None = None, speed_limit_mph: float = 25.0):
        """Initialize the reporter.

        Args:
            config: Reporting configuration.
            speed_limit_mph: Speed limit for violation detection.
        """
        self.config = config or ReportingConfig()
        self.speed_limit_mph = speed_limit_mph

        # Current aggregation window
        self.current_window: AggregationWindow | None = None

        # File handle for output
        self._output_file: TextIO | None = None
        self._output_path: Path | None = None

    def start_session(self, session_name: str | None = None) -> Path:
        """Start a new reporting session.

        Args:
            session_name: Optional name for the session file.

        Returns:
            Path to the output file.
        """
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = session_name or f"traffic_report_{timestamp}"
        self._output_path = self.config.output_dir / f"{filename}.jsonl"

        # Open file for writing
        self._output_file = open(self._output_path, "w")

        # Start first window
        self._start_new_window()

        return self._output_path

    def _start_new_window(self) -> None:
        """Start a new aggregation window."""
        self.current_window = AggregationWindow(window_start=datetime.now())

    def record_vehicle(self, vehicle: TrackedVehicle) -> None:
        """Record a vehicle in the current aggregation window.

        Args:
            vehicle: Tracked vehicle data.
        """
        if self.current_window is None:
            self._start_new_window()

        window = self.current_window

        # Check if we've already counted this vehicle in this window
        already_seen = vehicle.tracker_id in window._seen_tracker_ids

        if already_seen:
            # Still update speed if new measurement
            if vehicle.current_speed_mph:
                window.speeds_mph.append(vehicle.current_speed_mph)

            # Check if direction was assigned after initial recording
            # Direction may be determined after a few frames of movement
            if (vehicle.direction and
                vehicle.tracker_id not in window._direction_assigned_ids):
                direction_key = vehicle.direction.value
                window.vehicles_by_direction[direction_key] = (
                    window.vehicles_by_direction.get(direction_key, 0) + 1
                )
                window._direction_assigned_ids.add(vehicle.tracker_id)
            return

        # Mark as seen
        window._seen_tracker_ids.add(vehicle.tracker_id)

        # Count total
        window.total_vehicles += 1

        # Count by direction (if known at first sighting)
        if vehicle.direction:
            direction_key = vehicle.direction.value
            window.vehicles_by_direction[direction_key] = (
                window.vehicles_by_direction.get(direction_key, 0) + 1
            )
            window._direction_assigned_ids.add(vehicle.tracker_id)

        # Count by type
        type_key = vehicle.vehicle_class.value
        window.vehicles_by_type[type_key] = window.vehicles_by_type.get(type_key, 0) + 1

        # Record speeds
        if vehicle.current_speed_mph:
            window.speeds_mph.append(vehicle.current_speed_mph)

        # Check for speed violation
        if vehicle.max_speed_mph and vehicle.max_speed_mph > self.speed_limit_mph:
            violation = SpeedViolation(
                vehicle_id=f"#{vehicle.tracker_id}",
                speed_mph=vehicle.max_speed_mph,
                timestamp=vehicle.last_seen.isoformat(),
                vehicle_class=vehicle.vehicle_class.value,
                direction=vehicle.direction.value if vehicle.direction else None,
            )
            window.speed_violations.append(violation)

        # Check for special vehicle
        if vehicle.special_type:
            sighting = SpecialVehicleSighting(
                vehicle_type=vehicle.special_type.value,
                timestamp=vehicle.last_seen.isoformat(),
                direction=vehicle.direction.value if vehicle.direction else None,
                vehicle_id=f"#{vehicle.tracker_id}",
            )
            window.special_vehicles.append(sighting)

        # Count commercial trucks
        if vehicle.is_commercial:
            window.commercial_trucks += 1

    def check_window_complete(self) -> bool:
        """Check if current window duration has elapsed.

        Returns:
            True if window should be finalized.
        """
        if self.current_window is None:
            return False

        elapsed = datetime.now() - self.current_window.window_start
        return elapsed.total_seconds() >= self.config.aggregation_window_seconds

    def finalize_window(self) -> dict | None:
        """Finalize and write the current window.

        Returns:
            The window data as dictionary, or None if no window.
        """
        if self.current_window is None:
            return None

        # Set end time
        self.current_window.window_end = datetime.now()

        # Convert to dict
        window_data = self.current_window.to_dict()

        # Write to file
        if self._output_file:
            self._output_file.write(json.dumps(window_data) + "\n")
            self._output_file.flush()

        # Start new window
        self._start_new_window()

        return window_data

    def end_session(self) -> dict | None:
        """End the reporting session and close files.

        Returns:
            Final window data, or None.
        """
        # Finalize current window
        final_data = None
        if self.current_window and self.current_window.total_vehicles > 0:
            final_data = self.finalize_window()

        # Close file
        if self._output_file:
            self._output_file.close()
            self._output_file = None

        return final_data

    def get_summary(self) -> dict:
        """Get summary of current window (for display purposes).

        Returns:
            Current window statistics.
        """
        if self.current_window is None:
            return {}

        return {
            "vehicles_counted": self.current_window.total_vehicles,
            "violations": len(self.current_window.speed_violations),
            "top_speed": round(self.current_window.top_speed_mph, 1),
            "eastbound": self.current_window.vehicles_by_direction.get("eastbound", 0),
            "westbound": self.current_window.vehicles_by_direction.get("westbound", 0),
        }
