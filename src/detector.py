"""Vehicle detection using Roboflow Inference with RF-DETR model.

This module handles vehicle detection using Roboflow's RF-DETR model,
filtering for vehicle classes only (car, truck, bus, motorcycle).
"""

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import supervision as sv
from inference import get_model

from src.config import COCO_VEHICLE_CLASS_IDS, DetectionConfig, VehicleClass


@dataclass
class DetectionResult:
    """Container for detection results with vehicle-specific metadata."""

    detections: sv.Detections
    vehicle_classes: list[VehicleClass]
    frame_index: int


class VehicleDetector:
    """Detects vehicles in frames using Roboflow RF-DETR model.

    This class wraps the Roboflow Inference API to detect vehicles,
    filtering results to only include relevant vehicle classes from COCO.
    """

    # COCO class IDs for vehicles
    VEHICLE_CLASS_IDS = set(COCO_VEHICLE_CLASS_IDS.keys())

    def __init__(self, config: DetectionConfig | None = None):
        """Initialize the detector with configuration.

        Args:
            config: Detection configuration. Uses defaults if not provided.
        """
        self.config = config or DetectionConfig()
        self._model = None
        self._ensure_api_key()

    def _ensure_api_key(self) -> None:
        """Ensure Roboflow API key is available from config or environment."""
        if self.config.roboflow_api_key is None:
            api_key = os.environ.get("ROBOFLOW_API_KEY")
            if api_key:
                self.config.roboflow_api_key = api_key

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

        # Filter to vehicle classes only
        detections, vehicle_classes = self._filter_vehicles(detections)

        # Apply NMS to remove duplicates
        if len(detections) > 0:
            detections = detections.with_nms(threshold=self.config.iou_threshold)
            # Re-filter vehicle classes after NMS (indices may have changed)
            vehicle_classes = [
                COCO_VEHICLE_CLASS_IDS.get(class_id, VehicleClass.CAR)
                for class_id in detections.class_id
            ]

        return DetectionResult(
            detections=detections,
            vehicle_classes=vehicle_classes,
            frame_index=frame_index,
        )

    def _filter_vehicles(
        self, detections: sv.Detections
    ) -> tuple[sv.Detections, list[VehicleClass]]:
        """Filter detections to only include vehicle classes.

        Args:
            detections: Raw detections from the model.

        Returns:
            Tuple of (filtered detections, list of vehicle classes).
        """
        if len(detections) == 0:
            return detections, []

        # Create mask for vehicle classes
        vehicle_mask = np.array([
            class_id in self.VEHICLE_CLASS_IDS
            for class_id in detections.class_id
        ])

        # Filter detections
        filtered = detections[vehicle_mask]

        # Map class IDs to VehicleClass enum
        vehicle_classes = [
            COCO_VEHICLE_CLASS_IDS.get(class_id, VehicleClass.CAR)
            for class_id in filtered.class_id
        ]

        return filtered, vehicle_classes

    def get_class_name(self, class_id: int) -> str:
        """Get human-readable class name from COCO class ID.

        Args:
            class_id: COCO class ID.

        Returns:
            Human-readable class name.
        """
        vehicle_class = COCO_VEHICLE_CLASS_IDS.get(class_id)
        if vehicle_class:
            return vehicle_class.value
        return f"unknown_{class_id}"
