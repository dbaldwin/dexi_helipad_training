#!/usr/bin/env python3
"""
Helipad Training Data Augmentation

Generates augmented training images from source images captured at different altitudes.
Applies translations, rotations, scaling, and color adjustments while properly
transforming bounding box labels.

Usage:
    python augment.py --count 100
    python augment.py --count 200 --split 0.8
"""

import os
import cv2
import numpy as np
import argparse
import random
from pathlib import Path


def load_yolo_label(label_path):
    """Load YOLO format label file."""
    boxes = []
    if os.path.exists(label_path):
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


def save_yolo_label(label_path, boxes):
    """Save YOLO format label file."""
    with open(label_path, 'w') as f:
        for box in boxes:
            class_id, x_center, y_center, width, height = box
            # Clamp values to valid range
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def rotate_box(box, angle, img_w, img_h):
    """Rotate bounding box by angle (degrees) around image center."""
    class_id, x_center, y_center, width, height = box

    # Convert to pixel coordinates
    cx = x_center * img_w
    cy = y_center * img_h
    w = width * img_w
    h = height * img_h

    # Image center
    img_cx = img_w / 2
    img_cy = img_h / 2

    # Rotate center point around image center
    angle_rad = np.radians(-angle)  # Negative for clockwise rotation
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Translate to origin, rotate, translate back
    cx_new = cos_a * (cx - img_cx) - sin_a * (cy - img_cy) + img_cx
    cy_new = sin_a * (cx - img_cx) + cos_a * (cy - img_cy) + img_cy

    # For rotated rectangle, compute new bounding box
    # Get corners of original box
    corners = [
        (cx - w/2, cy - h/2),
        (cx + w/2, cy - h/2),
        (cx + w/2, cy + h/2),
        (cx - w/2, cy + h/2)
    ]

    # Rotate corners
    rotated_corners = []
    for px, py in corners:
        px_new = cos_a * (px - img_cx) - sin_a * (py - img_cy) + img_cx
        py_new = sin_a * (px - img_cx) + cos_a * (py - img_cy) + img_cy
        rotated_corners.append((px_new, py_new))

    # Get bounding box of rotated corners
    xs = [c[0] for c in rotated_corners]
    ys = [c[1] for c in rotated_corners]
    new_w = max(xs) - min(xs)
    new_h = max(ys) - min(ys)

    # Convert back to normalized coordinates
    x_center_new = cx_new / img_w
    y_center_new = cy_new / img_h
    width_new = new_w / img_w
    height_new = new_h / img_h

    return [class_id, x_center_new, y_center_new, width_new, height_new]


def translate_box(box, tx, ty):
    """Translate bounding box by normalized amounts."""
    class_id, x_center, y_center, width, height = box
    return [class_id, x_center + tx, y_center + ty, width, height]


def scale_box(box, scale, img_w, img_h):
    """Scale bounding box around image center."""
    class_id, x_center, y_center, width, height = box

    # Scale position relative to center
    x_center_new = 0.5 + (x_center - 0.5) * scale
    y_center_new = 0.5 + (y_center - 0.5) * scale

    # Scale dimensions
    width_new = width * scale
    height_new = height * scale

    return [class_id, x_center_new, y_center_new, width_new, height_new]


def is_box_valid(box, min_visibility=0.3):
    """Check if box is still sufficiently visible after transformation."""
    class_id, x_center, y_center, width, height = box

    # Calculate visible portion
    x_min = x_center - width / 2
    x_max = x_center + width / 2
    y_min = y_center - height / 2
    y_max = y_center + height / 2

    # Clamp to image bounds
    x_min_vis = max(0, x_min)
    x_max_vis = min(1, x_max)
    y_min_vis = max(0, y_min)
    y_max_vis = min(1, y_max)

    if x_max_vis <= x_min_vis or y_max_vis <= y_min_vis:
        return False

    # Calculate visibility ratio
    original_area = width * height
    visible_area = (x_max_vis - x_min_vis) * (y_max_vis - y_min_vis)

    return visible_area / original_area >= min_visibility


def augment_image(image, boxes, img_w, img_h):
    """Apply random augmentations to image and boxes."""
    augmented_img = image.copy()
    augmented_boxes = [box.copy() for box in boxes]

    # Random rotation (0-360 degrees)
    angle = random.uniform(0, 360)
    matrix = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1.0)
    augmented_img = cv2.warpAffine(augmented_img, matrix, (img_w, img_h),
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(128, 128, 128))
    augmented_boxes = [rotate_box(box, angle, img_w, img_h) for box in augmented_boxes]

    # Random translation (-30% to +30%)
    tx = random.uniform(-0.3, 0.3)
    ty = random.uniform(-0.3, 0.3)
    tx_pixels = int(tx * img_w)
    ty_pixels = int(ty * img_h)
    matrix = np.float32([[1, 0, tx_pixels], [0, 1, ty_pixels]])
    augmented_img = cv2.warpAffine(augmented_img, matrix, (img_w, img_h),
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(128, 128, 128))
    augmented_boxes = [translate_box(box, tx, ty) for box in augmented_boxes]

    # Random scale (0.7 to 1.3)
    scale = random.uniform(0.7, 1.3)
    matrix = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), 0, scale)
    augmented_img = cv2.warpAffine(augmented_img, matrix, (img_w, img_h),
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(128, 128, 128))
    augmented_boxes = [scale_box(box, scale, img_w, img_h) for box in augmented_boxes]

    # Random brightness (-40 to +40)
    brightness = random.randint(-40, 40)
    augmented_img = np.clip(augmented_img.astype(np.int16) + brightness, 0, 255).astype(np.uint8)

    # Random contrast (0.7 to 1.3)
    contrast = random.uniform(0.7, 1.3)
    augmented_img = np.clip((augmented_img.astype(np.float32) - 128) * contrast + 128, 0, 255).astype(np.uint8)

    # Random blur (15% chance)
    if random.random() < 0.15:
        kernel_size = random.choice([3, 5])
        augmented_img = cv2.GaussianBlur(augmented_img, (kernel_size, kernel_size), 0)

    # Random noise (15% chance)
    if random.random() < 0.15:
        noise = np.random.normal(0, 10, augmented_img.shape).astype(np.int16)
        augmented_img = np.clip(augmented_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Filter out boxes that are no longer visible
    augmented_boxes = [box for box in augmented_boxes if is_box_valid(box)]

    return augmented_img, augmented_boxes


def main():
    parser = argparse.ArgumentParser(description='Augment helipad training data')
    parser.add_argument('--source', type=str, default='source_images',
                        help='Source images directory')
    parser.add_argument('--output', type=str, default='dataset',
                        help='Output dataset directory')
    parser.add_argument('--count', type=int, default=100,
                        help='Number of augmentations per source image')
    parser.add_argument('--split', type=float, default=0.8,
                        help='Train/val split ratio (default 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    # Create output directories
    train_img_dir = output_dir / 'train' / 'images'
    train_lbl_dir = output_dir / 'train' / 'labels'
    val_img_dir = output_dir / 'val' / 'images'
    val_lbl_dir = output_dir / 'val' / 'labels'

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Find source images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    source_images = [f for f in source_dir.iterdir()
                     if f.suffix.lower() in image_extensions]

    if not source_images:
        print(f"No images found in {source_dir}")
        print("Please add source images with corresponding .txt label files")
        return

    print(f"Found {len(source_images)} source images")
    print(f"Generating {args.count} augmentations per image...")

    total_train = 0
    total_val = 0

    for img_path in source_images:
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Warning: Could not load {img_path}")
            continue

        img_h, img_w = image.shape[:2]

        # Load corresponding label
        label_path = img_path.with_suffix('.txt')
        boxes = load_yolo_label(label_path)

        if not boxes:
            print(f"  Warning: No labels found for {img_path.name}")
            continue

        print(f"  Processing {img_path.name} ({len(boxes)} boxes)...")

        # Generate augmentations
        for i in range(args.count):
            aug_img, aug_boxes = augment_image(image, boxes, img_w, img_h)

            # Skip if no valid boxes remain
            if not aug_boxes:
                continue

            # Determine train or val
            is_train = random.random() < args.split

            if is_train:
                img_out_dir = train_img_dir
                lbl_out_dir = train_lbl_dir
                total_train += 1
            else:
                img_out_dir = val_img_dir
                lbl_out_dir = val_lbl_dir
                total_val += 1

            # Save augmented image and label
            out_name = f"{img_path.stem}_aug_{i:04d}"
            cv2.imwrite(str(img_out_dir / f"{out_name}.jpg"), aug_img)
            save_yolo_label(lbl_out_dir / f"{out_name}.txt", aug_boxes)

    print(f"\nAugmentation complete!")
    print(f"  Training images: {total_train}")
    print(f"  Validation images: {total_val}")
    print(f"\nNext step: python train.py")


if __name__ == '__main__':
    main()
