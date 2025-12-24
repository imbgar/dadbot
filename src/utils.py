"""Utility functions for the traffic monitoring system."""

from pathlib import Path

import cv2
import numpy as np


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
