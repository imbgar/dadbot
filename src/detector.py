"""Vehicle detection using Roboflow Inference with RF-DETR model.

This module handles vehicle detection using Roboflow's RF-DETR model,
filtering for vehicle classes only (car, truck, bus, motorcycle).
Optionally filters to a road zone polygon.

Supports both local inference and cloud GPU inference via Roboflow API.
"""

import os
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import supervision as sv
from inference import get_model
from inference_sdk import InferenceHTTPClient

from src.config import VEHICLE_CLASS_NAMES, DetectionConfig, VehicleClass, ZoneConfig
from src.utils import get_logger

log = get_logger("detector")


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

    Supports both local CPU inference and cloud GPU inference.
    """

    # Vehicle class names (model-agnostic)
    VEHICLE_CLASS_NAMES = set(VEHICLE_CLASS_NAMES.keys())

    # Roboflow cloud API endpoint
    CLOUD_API_URL = "https://detect.roboflow.com"

    def __init__(
        self,
        config: DetectionConfig | None = None,
        zone_config: ZoneConfig | None = None,
        use_cloud: bool = True,
    ):
        """Initialize the detector with configuration.

        Args:
            config: Detection configuration. Uses defaults if not provided.
            zone_config: Road zone configuration for spatial filtering.
            use_cloud: If True, use Roboflow cloud GPU. If False, use local inference.
        """
        self.config = config or DetectionConfig()
        self.zone_config = zone_config or ZoneConfig()
        self.use_cloud = use_cloud
        self._model = None
        self._cloud_client: InferenceHTTPClient | None = None
        self._polygon_zone: sv.PolygonZone | None = None
        self._ensure_api_key()
        self._init_zone()

        mode = "cloud GPU" if use_cloud else "local"
        log.info(f"VehicleDetector initialized: model={self.config.model_id}, mode={mode}")
        log.debug(f"Detection config: confidence={self.config.confidence_threshold}, "
                  f"iou={self.config.iou_threshold}")

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
    def cloud_client(self) -> InferenceHTTPClient:
        """Lazy-load the cloud inference client.

        Returns:
            The Roboflow HTTP client for cloud inference.

        Raises:
            ValueError: If no API key is available.
        """
        if self._cloud_client is None:
            if self.config.roboflow_api_key is None:
                raise ValueError(
                    "Roboflow API key is required. Set ROBOFLOW_API_KEY environment "
                    "variable or provide it in DetectionConfig."
                )
            log.info(f"Initializing Roboflow cloud client: {self.CLOUD_API_URL}")
            self._cloud_client = InferenceHTTPClient(
                api_url=self.CLOUD_API_URL,
                api_key=self.config.roboflow_api_key,
            )
            log.debug("Cloud client initialized successfully")
        return self._cloud_client

    @property
    def model(self) -> Any:
        """Lazy-load the local detection model.

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
            log.info(f"Loading local model: {self.config.model_id}")
            self._model = get_model(
                model_id=self.config.model_id,
                api_key=self.config.roboflow_api_key,
            )
            log.debug("Local model loaded successfully")
        return self._model

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> DetectionResult:
        """Detect vehicles in a single frame.

        Args:
            frame: BGR image as numpy array (OpenCV format).
            frame_index: Current frame index for tracking.

        Returns:
            DetectionResult containing filtered vehicle detections.
        """
        start_time = time.perf_counter()

        # Run inference (cloud or local)
        inference_start = time.perf_counter()
        if self.use_cloud:
            h, w = frame.shape[:2]

            # Downscale for cloud inference (faster upload, similar accuracy)
            max_dim = 640  # RF-DETR works well at 640
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                inference_frame = cv2.resize(frame, (new_w, new_h))
                log.debug(f"Downscaled {w}x{h} -> {new_w}x{new_h} for cloud")
            else:
                inference_frame = frame
                log.debug(f"Sending {w}x{h} frame to cloud")

            results = self.cloud_client.infer(
                inference_frame,
                model_id=self.config.model_id,
            )

            # Scale bounding boxes back to original resolution
            if max(h, w) > max_dim:
                scale_back = max(h, w) / max_dim
                if isinstance(results, dict) and "predictions" in results:
                    for pred in results["predictions"]:
                        pred["x"] *= scale_back
                        pred["y"] *= scale_back
                        pred["width"] *= scale_back
                        pred["height"] *= scale_back
        else:
            results = self.model.infer(frame, confidence=self.config.confidence_threshold)
        inference_time = (time.perf_counter() - inference_start) * 1000

        # Handle different result formats from inference API
        if isinstance(results, list) and len(results) > 0:
            results = results[0]

        # Convert to supervision Detections
        detections = sv.Detections.from_inference(results)
        raw_count = len(detections)

        # Filter to vehicle classes only (using class names from inference)
        detections, vehicle_classes = self._filter_vehicles(detections)

        # Apply NMS to remove duplicates
        if len(detections) > 0:
            pre_nms = len(detections)
            detections = detections.with_nms(threshold=self.config.iou_threshold)
            # Re-map vehicle classes after NMS using class names
            class_names = detections.data.get("class_name", [])
            vehicle_classes = [
                VEHICLE_CLASS_NAMES.get(name.lower(), VehicleClass.CAR)
                for name in class_names
            ]
            if pre_nms != len(detections):
                log.debug(f"NMS reduced detections: {pre_nms} -> {len(detections)}")

        # Apply road zone filtering if configured
        if len(detections) > 0 and self._polygon_zone is not None:
            pre_zone = len(detections)
            detections, vehicle_classes = self._filter_by_zone(detections, vehicle_classes)
            if pre_zone != len(detections):
                log.debug(f"Zone filter reduced detections: {pre_zone} -> {len(detections)}")

        total_time = (time.perf_counter() - start_time) * 1000
        mode = "cloud" if self.use_cloud else "local"
        log.debug(f"Frame {frame_index}: {len(detections)} vehicles detected "
                  f"(raw={raw_count}) [{mode} inference={inference_time:.1f}ms, "
                  f"total={total_time:.1f}ms]")

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
            log.debug("No class_name in detections data")
            return detections, []

        # Log detected classes for debugging
        unique_classes = set(str(name).lower() for name in class_names)
        log.debug(f"Detected classes: {unique_classes}, looking for: {self.VEHICLE_CLASS_NAMES}")

        # Create mask for vehicle classes based on class names
        vehicle_mask = np.array([
            str(name).lower() in self.VEHICLE_CLASS_NAMES
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
