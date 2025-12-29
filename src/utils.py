"""Utility functions for the traffic monitoring system.

Includes:
- Video utilities (extract frames, get info)
- Centralized logging configuration for DadBot

All application modules should use get_logger() to obtain a child logger.

Example:
    from src.utils import get_logger

    log = get_logger("detector")
    log.info("Initializing detector...")
    log.debug(f"Model: {model_id}")
"""

import logging
import sys
from collections.abc import Callable
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# Logging Configuration
# =============================================================================

# Log directory - use ~/.dadbot/logs/
LOG_DIR = Path.home() / ".dadbot" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Root logger for the application
logger = logging.getLogger("dadbot")
logger.setLevel(logging.DEBUG)

# Global formatter for consistent log format
formatter = logging.Formatter(
    fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Shorter formatter for console (less verbose)
console_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Prevent duplicate handlers if module is reloaded
if not logger.handlers:
    # Console handler - outputs to stdout (INFO level to reduce noise)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Main log file - all logs (rotating, 5MB max, keep 3 backups)
    main_log_file = LOG_DIR / "dadbot.log"
    file_handler = RotatingFileHandler(
        main_log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Viewer-specific log file for debugging pipeline issues
    viewer_log_file = LOG_DIR / "viewer.log"
    viewer_handler = RotatingFileHandler(
        viewer_log_file,
        maxBytes=2 * 1024 * 1024,  # 2MB
        backupCount=2,
        encoding="utf-8",
    )
    viewer_handler.setLevel(logging.DEBUG)
    viewer_handler.setFormatter(formatter)
    # Only capture viewer logs
    viewer_handler.addFilter(lambda record: "viewer" in record.name)
    logger.addHandler(viewer_handler)

    # Inference-specific log file
    inference_log_file = LOG_DIR / "inference.log"
    inference_handler = RotatingFileHandler(
        inference_log_file,
        maxBytes=2 * 1024 * 1024,  # 2MB
        backupCount=2,
        encoding="utf-8",
    )
    inference_handler.setLevel(logging.DEBUG)
    inference_handler.setFormatter(formatter)
    # Capture inference-related logs
    inference_handler.addFilter(
        lambda record: any(x in record.name.lower() for x in ["inference", "detector", "pipeline"])
    )
    logger.addHandler(inference_handler)

    # Log startup info
    logger.info(f"Logging initialized. Log directory: {LOG_DIR}")


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a module.

    Args:
        name: Module name (e.g., "detector", "viewer", "tracker")

    Returns:
        A logger instance named "dadbot.{name}"

    Example:
        log = get_logger("detector")
        log.info("Detection started")
    """
    return logging.getLogger(f"dadbot.{name}")


class GUILogHandler(logging.Handler):
    """Custom handler that sends log messages to a GUI callback.

    Used to display log messages in the application's console panel.
    """

    def __init__(self, callback: Callable[[str], None]):
        """Initialize with a callback function.

        Args:
            callback: Function that receives formatted log messages
        """
        super().__init__()
        self.callback = callback
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record by calling the callback with formatted message."""
        try:
            msg = self.format(record)
            self.callback(msg)
        except Exception:
            # Silently ignore errors to prevent logging loops
            pass


def add_gui_handler(callback: Callable[[str], None]) -> GUILogHandler:
    """Add a GUI handler to receive log messages.

    Args:
        callback: Function to receive formatted log messages

    Returns:
        The created handler (keep reference for removal)

    Example:
        def on_log(msg: str):
            console.append(msg)

        handler = add_gui_handler(on_log)
        # Later: remove_gui_handler(handler)
    """
    handler = GUILogHandler(callback)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return handler


def remove_gui_handler(handler: GUILogHandler) -> None:
    """Remove a previously added GUI handler.

    Args:
        handler: The handler returned by add_gui_handler()
    """
    logger.removeHandler(handler)


# =============================================================================
# Video Utilities
# =============================================================================


def extract_first_frame(video_path: str | Path) -> np.ndarray:
    """Extract the first frame from a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        First frame as BGR numpy array.

    Raises:
        FileNotFoundError: If video file doesn't exist.
        ValueError: If video cannot be read.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise ValueError(f"Could not read first frame from: {video_path}")

    return frame


def save_frame(frame: np.ndarray, output_path: str | Path) -> Path:
    """Save a frame to an image file.

    Args:
        frame: BGR image as numpy array.
        output_path: Path to save the image.

    Returns:
        Path to the saved image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    return output_path


def get_video_info(video_path: str | Path) -> dict:
    """Get video metadata.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary with width, height, fps, total_frames.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }

    cap.release()
    return info
