"""Vehicle detection using Roboflow Inference with RF-DETR model.

This module handles vehicle detection using Roboflow's RF-DETR model,
filtering for vehicle classes only (car, truck, bus, motorcycle).
Optionally filters to a road zone polygon.
"""

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import supervision as sv
from inference import get_model

from src.config import VEHICLE_CLASS_NAMES, DetectionConfig, VehicleClass, ZoneConfig


@dataclass
class DetectionResult:
    """Container for detection results with vehicle-specific metadata."""

    detections: sv.Detections
    vehicle_classes: list[VehicleClass]
    frame_index: int


class VehicleDetector:
    """Detects vehicles in frames using Roboflow RF-DETR model.

    This class wraps the Roboflow Inference API to detect vehicles,
    filtering results to only include relevant vehicle classes.
    Optionally filters detections to a road zone polygon.
    """

    # Vehicle class names (model-agnostic)
    VEHICLE_CLASS_NAMES = set(VEHICLE_CLASS_NAMES.keys())

    def __init__(
        self,
        config: DetectionConfig | None = None,
        zone_config: ZoneConfig | None = None,
    ):
        """Initialize the detector with configuration.

        Args:
            config: Detection configuration. Uses defaults if not provided.
            zone_config: Road zone configuration for spatial filtering.
        """
        self.config = config or DetectionConfig()
        self.zone_config = zone_config or ZoneConfig()
        self._model = None
        self._polygon_zone: sv.PolygonZone | None = None
        self._ensure_api_key()
        self._init_zone()

    def _ensure_api_key(self) -> None:
        """Ensure Roboflow API key is available from config or environment."""
        if self.config.roboflow_api_key is None:
            api_key = os.environ.get("ROBOFLOW_API_KEY")
            if api_key:
                self.config.roboflow_api_key = api_key

    def _init_zone(self) -> None:
        """Initialize polygon zone for road filtering if configured."""
        if self.zone_config.enabled and self.zone_config.polygon_points:
            polygon = np.array(self.zone_config.polygon_points, dtype=np.int32)
            self._polygon_zone = sv.PolygonZone(polygon=polygon)

    @property
    def model(self) -> Any:
        """Lazy-load the detection model.

        Returns:
            The loaded Roboflow inference model.

        Raises:
            ValueError: If no API key is available for cloud inference.
        """
        if self._model is None:
            if self.config.roboflow_api_key is None:
                raise ValueError(
                    "Roboflow API key is required. Set ROBOFLOW_API_KEY environment "
                    "variable or provide it in DetectionConfig."
                )
            self._model = get_model(
                model_id=self.config.model_id,
                api_key=self.config.roboflow_api_key,
            )
        return self._model

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> DetectionResult:
        """Detect vehicles in a single frame.

        Args:
            frame: BGR image as numpy array (OpenCV format).
            frame_index: Current frame index for tracking.

        Returns:
            DetectionResult containing filtered vehicle detections.
        """
        # Run inference
        results = self.model.infer(frame, confidence=self.config.confidence_threshold)

        # Handle different result formats from inference API
        if isinstance(results, list) and len(results) > 0:
            results = results[0]

        # Convert to supervision Detections
        detections = sv.Detections.from_inference(results)

        # Filter to vehicle classes only (using class names from inference)
        detections, vehicle_classes = self._filter_vehicles(detections)

        # Apply NMS to remove duplicates
        if len(detections) > 0:
            detections = detections.with_nms(threshold=self.config.iou_threshold)
            # Re-map vehicle classes after NMS using class names
            class_names = detections.data.get("class_name", [])
            vehicle_classes = [
                VEHICLE_CLASS_NAMES.get(name.lower(), VehicleClass.CAR)
                for name in class_names
            ]

        # Apply road zone filtering if configured
        if len(detections) > 0 and self._polygon_zone is not None:
            detections, vehicle_classes = self._filter_by_zone(detections, vehicle_classes)

        return DetectionResult(
            detections=detections,
            vehicle_classes=vehicle_classes,
            frame_index=frame_index,
        )

    def _filter_by_zone(
        self, detections: sv.Detections, vehicle_classes: list[VehicleClass]
    ) -> tuple[sv.Detections, list[VehicleClass]]:
        """Filter detections to only those within the road zone.

        Args:
            detections: Vehicle detections.
            vehicle_classes: Corresponding vehicle classes.

        Returns:
            Tuple of (filtered detections, filtered vehicle classes).
        """
        if self._polygon_zone is None or len(detections) == 0:
            return detections, vehicle_classes

        # Get mask of detections inside the zone
        zone_mask = self._polygon_zone.trigger(detections)

        # Filter detections
        filtered_detections = detections[zone_mask]

        # Filter vehicle classes
        filtered_classes = [
            vc for vc, in_zone in zip(vehicle_classes, zone_mask) if in_zone
        ]

        return filtered_detections, filtered_classes

    def _filter_vehicles(
        self, detections: sv.Detections
    ) -> tuple[sv.Detections, list[VehicleClass]]:
        """Filter detections to only include vehicle classes.

        Uses class names from inference results for model-agnostic filtering.

        Args:
            detections: Raw detections from the model.

        Returns:
            Tuple of (filtered detections, list of vehicle classes).
        """
        if len(detections) == 0:
            return detections, []

        # Get class names from detections data
        class_names = detections.data.get("class_name", [])
        if len(class_names) == 0:
            return detections, []

        # Create mask for vehicle classes based on class names
        vehicle_mask = np.array([
            name.lower() in self.VEHICLE_CLASS_NAMES
            for name in class_names
        ])

        # Filter detections
        filtered = detections[vehicle_mask]

        # Get filtered class names
        filtered_names = [name for name, is_vehicle in zip(class_names, vehicle_mask) if is_vehicle]

        # Map class names to VehicleClass enum
        vehicle_classes = [
            VEHICLE_CLASS_NAMES.get(name.lower(), VehicleClass.CAR)
            for name in filtered_names
        ]

        return filtered, vehicle_classes
