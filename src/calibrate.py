"""Calibration utility for setting up pixel-to-feet conversion.

This interactive tool helps users identify the pixel coordinates
corresponding to known real-world distances in their camera view.

Usage:
    python -m src.calibrate --image path/to/frame.png --distance 35

Click on two points that represent a known distance (e.g., curb to curb).
The tool will output the calibration parameters to use.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


class CalibrationTool:
    """Interactive tool for calibrating pixel-to-feet conversion."""

    def __init__(self, image_path: str, known_distance_feet: float):
        """Initialize calibration tool.

        Args:
            image_path: Path to reference frame image.
            known_distance_feet: Real-world distance between calibration points.
        """
        self.image_path = image_path
        self.known_distance = known_distance_feet
        self.points: list[tuple[int, int]] = []
        self.image = None
        self.display_image = None

    def load_image(self) -> bool:
        """Load the reference image.

        Returns:
            True if image loaded successfully.
        """
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            print(f"Error: Could not load image: {self.image_path}")
            return False

        print(f"Loaded image: {self.image.shape[1]}x{self.image.shape[0]}")
        return True

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse click events.

        Args:
            event: OpenCV mouse event type.
            x: X coordinate of click.
            y: Y coordinate of click.
            flags: OpenCV event flags.
            param: Additional parameters.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                print(f"Point {len(self.points)}: ({x}, {y})")
                self.update_display()

    def update_display(self) -> None:
        """Update the display with current calibration points."""
        self.display_image = self.image.copy()

        # Draw points
        for i, point in enumerate(self.points):
            color = (0, 255, 0) if i == 0 else (0, 0, 255)
            cv2.circle(self.display_image, point, 8, color, -1)
            cv2.putText(
                self.display_image,
                f"P{i + 1}",
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        # Draw line between points
        if len(self.points) == 2:
            cv2.line(self.display_image, self.points[0], self.points[1], (255, 255, 0), 2)

            # Calculate and display results
            self.display_results()

        # Draw instructions
        self._draw_instructions()

    def _draw_instructions(self) -> None:
        """Draw instruction text on image."""
        instructions = [
            "Click two points representing a known distance",
            f"Known distance: {self.known_distance} feet",
            f"Points selected: {len(self.points)}/2",
            "Press 'r' to reset, 'q' to quit",
        ]

        y_offset = 30
        for instruction in instructions:
            cv2.putText(
                self.display_image,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            y_offset += 25

    def display_results(self) -> None:
        """Calculate and display calibration results."""
        if len(self.points) != 2:
            return

        x1, y1 = self.points[0]
        x2, y2 = self.points[1]

        # Calculate pixel distance
        pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        horizontal_distance = abs(x2 - x1)

        # Calculate pixels per foot
        pixels_per_foot = pixel_distance / self.known_distance
        horizontal_ppf = horizontal_distance / self.known_distance

        # Draw results on image
        results = [
            f"Pixel distance: {pixel_distance:.1f}px",
            f"Horizontal: {horizontal_distance}px",
            f"Pixels/foot: {pixels_per_foot:.2f}",
            f"Horizontal pixels/foot: {horizontal_ppf:.2f}",
        ]

        y_offset = self.image.shape[0] - 120
        for result in results:
            cv2.putText(
                self.display_image,
                result,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            y_offset += 25

        # Print to console
        print("\n" + "=" * 50)
        print("CALIBRATION RESULTS")
        print("=" * 50)
        print(f"Point 1 (left):  ({x1}, {y1})")
        print(f"Point 2 (right): ({x2}, {y2})")
        print(f"Known distance:  {self.known_distance} feet")
        print(f"Pixel distance:  {pixel_distance:.1f} pixels")
        print(f"Pixels per foot: {pixels_per_foot:.2f}")
        print()
        print("Use these command-line arguments:")
        print(f"  --reference-distance {self.known_distance}")
        print(f"  --reference-start-x {min(x1, x2)}")
        print(f"  --reference-end-x {max(x1, x2)}")
        print()
        print("Or set these environment variables:")
        print(f"  export DADBOT_CALIBRATION_REFERENCE_DISTANCE_FEET={self.known_distance}")
        print(f"  export DADBOT_CALIBRATION_REFERENCE_PIXEL_START_X={min(x1, x2)}")
        print(f"  export DADBOT_CALIBRATION_REFERENCE_PIXEL_END_X={max(x1, x2)}")
        print("=" * 50)

    def run(self) -> None:
        """Run the interactive calibration tool."""
        if not self.load_image():
            return

        self.display_image = self.image.copy()
        self._draw_instructions()

        window_name = "Calibration Tool"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\nCalibration Tool")
        print("=" * 50)
        print("Click on two points that represent a known distance")
        print(f"(e.g., curb to curb = {self.known_distance} feet)")
        print("Press 'r' to reset points, 'q' to quit")
        print("=" * 50)

        while True:
            cv2.imshow(window_name, self.display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                self.points = []
                self.display_image = self.image.copy()
                self._draw_instructions()
                print("\nPoints reset. Click again to set new calibration points.")

        cv2.destroyAllWindows()


def main():
    """Main entry point for calibration tool."""
    parser = argparse.ArgumentParser(
        description="Interactive calibration tool for traffic monitoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to reference frame image",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=35.0,
        help="Known real-world distance in feet",
    )

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    tool = CalibrationTool(args.image, args.distance)
    tool.run()


if __name__ == "__main__":
    main()
