# Helipad Detection Training

Train a YOLOv8n model to detect helipads for autonomous drone landing.

## Overview

This project trains a single-class YOLO model to detect helipads from aerial footage.
It's designed for a two-phase landing approach:

1. **Phase 1 (YOLO)**: Detect helipad at distance, center drone over it
2. **Phase 2 (AprilTag)**: Precision landing using AprilTag when close enough

## Quick Start

```bash
# 1. Setup
cd ~/_dev/dexi_helipad_training
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Add source images to source_images/ folder
#    Capture from Unity at 5m, 10m, 15m, 20m altitudes

# 3. Label images
python label_images.py source_images/

# 4. Generate augmented dataset
python augment.py --count 100

# 5. Train
python train.py --epochs 50 --device mps

# 6. Test
python test.py --image source_images/helipad_10m.png
```

## Using with dexi_yolo

The trained model can be used with the `dexi_yolo` ROS2 package.

### Run directly with the helipad model

```bash
ros2 run dexi_yolo dexi_yolo_node_onnx.py --ros-args \
  -p model_path:=/path/to/dexi_helipad_training/models/best.onnx \
  -p class_names:=helipad
```

### Launch with dexi_bringup (CM4)

```bash
ros2 launch dexi_bringup dexi_bringup_ark_cm4.launch.py \
  yolo:=true
```

To use the helipad model instead of the default 6-class model, override parameters:

```bash
ros2 launch dexi_bringup dexi_bringup_ark_cm4.launch.py yolo:=true \
  --ros-args \
  -p /dexi_yolo_node:model_path:=/path/to/dexi_helipad_training/models/best.onnx \
  -p /dexi_yolo_node:class_names:=helipad
```

### Detection output

The node publishes to `/yolo_detections` with:
- `class_name`: "helipad"
- `confidence`: detection confidence (0-1)
- `bbox`: [x1, y1, x2, y2] normalized coordinates

## Project Structure

```
dexi_helipad_training/
├── source_images/          # Original captures from Unity
│   ├── helipad_05m.png
│   ├── helipad_05m.txt     # YOLO label
│   └── ...
├── models/                 # Trained models (after training)
│   ├── best.pt             # PyTorch model
│   └── best.onnx           # ONNX model for deployment
├── augment.py              # Data augmentation
├── train.py                # Model training
├── label_images.py         # Labeling tool
├── test.py                 # Test/validate model
├── clean.py                # Clean generated files
└── requirements.txt
```

## Training Workflow

### 1. Capture Source Images

From Unity simulator, capture helipad images at different altitudes:
- 5m - Helipad large in frame
- 10m - Medium size
- 15m - Smaller
- 20m - Small (limit of useful detection)

Crop to just the camera viewport and save to `source_images/`.

### 2. Label Images

```bash
python label_images.py source_images/
```

Controls:
- Click and drag to draw bounding box around helipad
- `s` - Save and next image
- `r` - Reset/clear boxes
- `u` - Undo last box
- `q` - Quit

### 3. Augment Dataset

```bash
python augment.py --count 100
```

Generates training data with rotation, translation, scale, brightness/contrast variations.

### 4. Train Model

```bash
python train.py --epochs 50 --device mps
```

Options:
- `--epochs N` - Training epochs (default: 50)
- `--batch N` - Batch size (default: 16)
- `--device mps|cuda|cpu` - Training device
- `--freeze N` - Layers to freeze (default: 10)

### 5. Test Model

```bash
python test.py --image source_images/helipad_10m.png
```

### 6. Re-train (if needed)

```bash
python clean.py              # Remove generated files
python augment.py --count 100
python train.py --epochs 50
```

## Centering Logic for Flight Control

Use the bounding box center to position the drone over the helipad:

```python
# Calculate centering error from detection bbox
bbox_center_x = (bbox[0] + bbox[2]) / 2  # normalized 0-1
bbox_center_y = (bbox[1] + bbox[3]) / 2

# Error from image center (-1 to +1)
error_x = (bbox_center_x - 0.5) * 2
error_y = (bbox_center_y - 0.5) * 2

# Convert to velocity commands
velocity_x = -error_y * gain  # Forward/back
velocity_y = -error_x * gain  # Left/right

# When centered, begin descent
if abs(error_x) < 0.05 and abs(error_y) < 0.05:
    begin_descent()
```

## Training Results

After training with accurate labels:
- Precision: 99.9%
- Recall: 100%
- mAP50: 99.5%
- mAP50-95: 90.3%
