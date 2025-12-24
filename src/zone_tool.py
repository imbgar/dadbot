"""Interactive tool for defining road zone polygons.

This tool allows users to click on a video frame or image to define
the road zone polygon used for filtering vehicle detections.

Usage:
    python -m src.zone_tool --image sample_frame.png
    python -m src.zone_tool --video reference_footage/clipped_1.mp4
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv


class ZoneDefinitionTool:
    """Interactive tool for defining road zone polygons."""

    def __init__(self, image: np.ndarray, window_name: str = "Zone Definition Tool"):
        """Initialize the tool.

        Args:
            image: Reference frame image.
            window_name: Window title.
        """
        self.image = image
        self.window_name = window_name
        self.points: list[tuple[int, int]] = []
        self.display_image = None
        self.completed = False

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse events.

        Left click: Add point
        Right click: Remove last point
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add new point
            self.points.append((x, y))
            print(f"Point {len(self.points)}: ({x}, {y})")
            self.update_display()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove last point
            if self.points:
                removed = self.points.pop()
                print(f"Removed point: {removed}")
                self.update_display()

    def update_display(self) -> None:
        """Update the display with current polygon."""
        self.display_image = self.image.copy()

        # Draw instructions
        self._draw_instructions()

        if not self.points:
            return

        # Draw points
        for i, point in enumerate(self.points):
            # Color: green for first, blue for others, red for last
            if i == 0:
                color = (0, 255, 0)  # Green - start
            elif i == len(self.points) - 1:
                color = (0, 0, 255)  # Red - current
            else:
                color = (255, 0, 0)  # Blue - middle

            cv2.circle(self.display_image, point, 8, color, -1)
            cv2.putText(
                self.display_image,
                str(i + 1),
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Draw polygon edges
        if len(self.points) >= 2:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(self.display_image, [pts], isClosed=False, color=(255, 255, 0), thickness=2)

        # Draw closing line if 3+ points (preview of complete polygon)
        if len(self.points) >= 3:
            cv2.line(
                self.display_image,
                self.points[-1],
                self.points[0],
                (255, 255, 0),  # Yellow dashed preview
                1,
            )

            # Draw semi-transparent fill preview
            overlay = self.display_image.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            self.display_image = cv2.addWeighted(overlay, 0.2, self.display_image, 0.8, 0)

    def _draw_instructions(self) -> None:
        """Draw instruction overlay."""
        instructions = [
            "LEFT CLICK: Add point",
            "RIGHT CLICK: Remove last point",
            f"Points: {len(self.points)}/4+",
            "Press 'c' to complete (need 3+ points)",
            "Press 'r' to reset all points",
            "Press 'q' to quit without saving",
        ]

        # Draw background
        overlay = self.display_image.copy()
        cv2.rectangle(overlay, (5, 5), (320, 140), (0, 0, 0), -1)
        self.display_image = cv2.addWeighted(overlay, 0.7, self.display_image, 0.3, 0)

        y_offset = 25
        for instruction in instructions:
            cv2.putText(
                self.display_image,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y_offset += 20

    def run(self) -> list[list[int]] | None:
        """Run the interactive zone definition tool.

        Returns:
            List of [x, y] coordinates, or None if cancelled.
        """
        self.display_image = self.image.copy()
        self._draw_instructions()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n" + "=" * 60)
        print("ROAD ZONE DEFINITION TOOL")
        print("=" * 60)
        print("Define a polygon around the road surface.")
        print("Click points in order (clockwise or counter-clockwise).")
        print("")
        print("Controls:")
        print("  LEFT CLICK  - Add point")
        print("  RIGHT CLICK - Remove last point")
        print("  'c'         - Complete and save")
        print("  'r'         - Reset all points")
        print("  'q'         - Quit without saving")
        print("=" * 60)

        while True:
            cv2.imshow(self.window_name, self.display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\nCancelled - no zone saved")
                cv2.destroyAllWindows()
                return None

            elif key == ord("r"):
                self.points = []
                self.update_display()
                print("\nPoints reset")

            elif key == ord("c"):
                if len(self.points) >= 3:
                    self.completed = True
                    break
                else:
                    print("Need at least 3 points to complete polygon")

        cv2.destroyAllWindows()

        # Convert to list format
        polygon = [[p[0], p[1]] for p in self.points]
        return polygon


def main():
    """Main entry point for zone definition tool."""
    parser = argparse.ArgumentParser(
        description="Interactive tool for defining road zone polygons",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image",
        type=str,
        help="Path to reference frame image",
    )
    group.add_argument(
        "--video",
        type=str,
        help="Path to video (will use first frame)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for zone config (default: print to stdout)",
    )

    args = parser.parse_args()

    # Load image
    if args.image:
        if not Path(args.image).exists():
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image: {args.image}")
            sys.exit(1)
        print(f"Loaded image: {args.image}")

    else:  # args.video
        if not Path(args.video).exists():
            print(f"Error: Video not found: {args.video}")
            sys.exit(1)
        # Get first frame
        cap = cv2.VideoCapture(args.video)
        ret, image = cap.read()
        cap.release()
        if not ret or image is None:
            print(f"Error: Could not read video: {args.video}")
            sys.exit(1)
        print(f"Loaded first frame from: {args.video}")

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Run tool
    tool = ZoneDefinitionTool(image)
    polygon = tool.run()

    if polygon is None:
        sys.exit(0)

    # Output results
    print("\n" + "=" * 60)
    print("ZONE DEFINED SUCCESSFULLY")
    print("=" * 60)
    print(f"Points: {len(polygon)}")
    for i, point in enumerate(polygon):
        print(f"  {i + 1}: ({point[0]}, {point[1]})")

    # Generate JSON
    zone_json = json.dumps(polygon)
    print(f"\nJSON for --zone argument:")
    print(f"  '{zone_json}'")

    print(f"\nFull command example:")
    print(f"  uv run python -m src.main --source-video VIDEO.mp4 --zone '{zone_json}' --show-zone")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump({"polygon_points": polygon}, f, indent=2)
        print(f"\nSaved to: {output_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
