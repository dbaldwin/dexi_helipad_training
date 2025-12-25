#!/usr/bin/env python3
"""
Simple Image Labeling Tool for Helipad Detection

Click and drag to draw bounding boxes around helipads.
Labels are saved in YOLO format.

Usage:
    python label_images.py source_images/

Controls:
    - Click and drag to draw bounding box
    - 's' - Save current label and move to next image
    - 'r' - Reset/clear current boxes
    - 'u' - Undo last box
    - 'n' - Next image (skip without saving)
    - 'p' - Previous image
    - 'q' - Quit
"""

import os
import cv2
import argparse
from pathlib import Path


class LabelingTool:
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        self.images = self._find_images()
        self.current_idx = 0
        self.boxes = []
        self.drawing = False
        self.start_point = None
        self.current_box = None
        self.window_name = "Helipad Labeling - Press 'h' for help"

        # Class ID for helipad
        self.class_id = 0

    def _find_images(self):
        """Find all images in directory."""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = sorted([f for f in self.image_dir.iterdir()
                        if f.suffix.lower() in extensions])
        return images

    def _load_existing_labels(self, image_path):
        """Load existing labels for an image."""
        label_path = image_path.with_suffix('.txt')
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        boxes.append([class_id, x_center, y_center, width, height])
        return boxes

    def _save_labels(self, image_path, boxes):
        """Save labels in YOLO format."""
        label_path = image_path.with_suffix('.txt')
        with open(label_path, 'w') as f:
            for box in boxes:
                class_id, x_center, y_center, width, height = box
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        print(f"Saved: {label_path}")

    def _draw_boxes(self, image, boxes, img_w, img_h):
        """Draw bounding boxes on image."""
        display = image.copy()

        for box in boxes:
            class_id, x_center, y_center, width, height = box

            # Convert normalized to pixel coordinates
            x1 = int((x_center - width / 2) * img_w)
            y1 = int((y_center - height / 2) * img_h)
            x2 = int((x_center + width / 2) * img_w)
            y2 = int((y_center + height / 2) * img_h)

            # Draw box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, "helipad", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return display

    def _pixel_to_yolo(self, x1, y1, x2, y2, img_w, img_h):
        """Convert pixel coordinates to YOLO format."""
        # Ensure x1 < x2 and y1 < y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h

        return [self.class_id, x_center, y_center, width, height]

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing boxes."""
        image, img_w, img_h = param

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_box = None

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_box = (self.start_point[0], self.start_point[1], x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point and abs(x - self.start_point[0]) > 5 and abs(y - self.start_point[1]) > 5:
                box = self._pixel_to_yolo(self.start_point[0], self.start_point[1],
                                          x, y, img_w, img_h)
                self.boxes.append(box)
            self.current_box = None

    def run(self):
        """Main labeling loop."""
        if not self.images:
            print(f"No images found in {self.image_dir}")
            return

        print(f"Found {len(self.images)} images")
        print("\nControls:")
        print("  Click+drag - Draw bounding box")
        print("  s - Save and next")
        print("  r - Reset boxes")
        print("  u - Undo last box")
        print("  n - Next (skip)")
        print("  p - Previous")
        print("  q - Quit")

        cv2.namedWindow(self.window_name)

        while 0 <= self.current_idx < len(self.images):
            image_path = self.images[self.current_idx]
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Could not load {image_path}")
                self.current_idx += 1
                continue

            img_h, img_w = image.shape[:2]

            # Load existing labels
            self.boxes = self._load_existing_labels(image_path)

            cv2.setMouseCallback(self.window_name, self._mouse_callback,
                                (image, img_w, img_h))

            while True:
                # Draw current state
                display = self._draw_boxes(image, self.boxes, img_w, img_h)

                # Draw current box being drawn
                if self.current_box:
                    x1, y1, x2, y2 = self.current_box
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # Draw info
                info = f"[{self.current_idx + 1}/{len(self.images)}] {image_path.name} - {len(self.boxes)} boxes"
                cv2.putText(display, info, (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow(self.window_name, display)
                key = cv2.waitKey(30) & 0xFF

                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return

                elif key == ord('s'):
                    self._save_labels(image_path, self.boxes)
                    self.current_idx += 1
                    break

                elif key == ord('n'):
                    self.current_idx += 1
                    break

                elif key == ord('p'):
                    self.current_idx = max(0, self.current_idx - 1)
                    break

                elif key == ord('r'):
                    self.boxes = []

                elif key == ord('u'):
                    if self.boxes:
                        self.boxes.pop()

                elif key == ord('h'):
                    print("\nControls:")
                    print("  Click+drag - Draw bounding box")
                    print("  s - Save and next")
                    print("  r - Reset boxes")
                    print("  u - Undo last box")
                    print("  n - Next (skip)")
                    print("  p - Previous")
                    print("  q - Quit")

        cv2.destroyAllWindows()
        print("\nLabeling complete!")


def main():
    parser = argparse.ArgumentParser(description='Label helipad images')
    parser.add_argument('image_dir', type=str, nargs='?', default='source_images',
                        help='Directory containing images to label')
    args = parser.parse_args()

    tool = LabelingTool(args.image_dir)
    tool.run()


if __name__ == '__main__':
    main()
