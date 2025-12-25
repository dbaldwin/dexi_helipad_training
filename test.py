#!/usr/bin/env python3
"""
Test Helipad Detection Model

Run inference on test images or video to validate the trained model.
Displays bounding box center offset from image center (useful for flight control).

Usage:
    python test.py --image test.jpg
    python test.py --image test.jpg --model models/best.pt
    python test.py --dir test_images/
"""

import cv2
import argparse
from pathlib import Path


def calculate_centering_error(box, img_w, img_h):
    """
    Calculate the error between helipad center and image center.

    Returns:
        (error_x, error_y): Normalized error (-1 to 1)
            Negative x = helipad is left of center
            Positive x = helipad is right of center
            Negative y = helipad is above center
            Positive y = helipad is below center
    """
    x_center, y_center = box[0], box[1]  # Normalized 0-1

    # Image center is at (0.5, 0.5)
    error_x = (x_center - 0.5) * 2  # Scale to -1 to 1
    error_y = (y_center - 0.5) * 2

    return error_x, error_y


def run_detection(image_path, model, conf_threshold=0.5):
    """Run detection on a single image."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not load {image_path}")
        return None

    img_h, img_w = image.shape[:2]

    # Run inference
    results = model(image, verbose=False)

    # Process results
    detections = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf >= conf_threshold:
                # Get normalized coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                detections.append({
                    'confidence': conf,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'pixel_box': (int(x1), int(y1), int(x2), int(y2))
                })

    return image, detections, img_w, img_h


def draw_results(image, detections, img_w, img_h):
    """Draw detection results on image."""
    display = image.copy()

    # Draw image center crosshair
    cx, cy = img_w // 2, img_h // 2
    cv2.line(display, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
    cv2.line(display, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)

    for det in detections:
        x1, y1, x2, y2 = det['pixel_box']
        conf = det['confidence']

        # Draw bounding box
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label = f"helipad {conf:.2f}"
        cv2.putText(display, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw center point of helipad
        hcx = (x1 + x2) // 2
        hcy = (y1 + y2) // 2
        cv2.circle(display, (hcx, hcy), 5, (0, 0, 255), -1)

        # Draw line from image center to helipad center
        cv2.line(display, (cx, cy), (hcx, hcy), (0, 0, 255), 2)

        # Calculate and display centering error
        error_x, error_y = calculate_centering_error(
            (det['x_center'], det['y_center']), img_w, img_h
        )

        error_text = f"Error: X={error_x:+.2f} Y={error_y:+.2f}"
        cv2.putText(display, error_text, (10, img_h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Centering guidance
        guidance = []
        if abs(error_x) > 0.05:
            guidance.append("LEFT" if error_x < 0 else "RIGHT")
        if abs(error_y) > 0.05:
            guidance.append("FORWARD" if error_y < 0 else "BACKWARD")

        if guidance:
            guide_text = f"Move: {', '.join(guidance)}"
        else:
            guide_text = "CENTERED - Ready for descent"

        cv2.putText(display, guide_text, (10, img_h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if not detections:
        cv2.putText(display, "No helipad detected", (10, img_h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return display


def main():
    parser = argparse.ArgumentParser(description='Test helipad detection')
    parser.add_argument('--image', type=str, help='Single image to test')
    parser.add_argument('--dir', type=str, help='Directory of images to test')
    parser.add_argument('--model', type=str, default='models/best.pt',
                        help='Model path')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--save', action='store_true',
                        help='Save annotated images')
    args = parser.parse_args()

    # Load model
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics not installed. Run: pip install -r requirements.txt")
        return

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train a model first: python train.py")
        return

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Collect images to test
    images = []
    if args.image:
        images.append(Path(args.image))
    elif args.dir:
        dir_path = Path(args.dir)
        images = list(dir_path.glob('*.jpg')) + list(dir_path.glob('*.png'))
    else:
        print("Specify --image or --dir")
        return

    print(f"Testing on {len(images)} images...")

    for img_path in images:
        result = run_detection(img_path, model, args.conf)
        if result is None:
            continue

        image, detections, img_w, img_h = result

        print(f"\n{img_path.name}:")
        if detections:
            for det in detections:
                error_x, error_y = calculate_centering_error(
                    (det['x_center'], det['y_center']), img_w, img_h
                )
                print(f"  Helipad detected (conf={det['confidence']:.2f})")
                print(f"  Center: ({det['x_center']:.3f}, {det['y_center']:.3f})")
                print(f"  Centering error: X={error_x:+.3f}, Y={error_y:+.3f}")
        else:
            print("  No helipad detected")

        # Display
        display = draw_results(image, detections, img_w, img_h)

        if args.save:
            out_path = img_path.parent / f"{img_path.stem}_detected{img_path.suffix}"
            cv2.imwrite(str(out_path), display)
            print(f"  Saved: {out_path}")
        else:
            cv2.imshow("Helipad Detection", display)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
