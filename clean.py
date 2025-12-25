#!/usr/bin/env python3
"""
Clean generated training artifacts.

Removes:
- dataset/ (augmented images)
- results/ (training outputs)
- models/ (trained models)

Preserves:
- source_images/ (your original captures and labels)
- All .py scripts

Usage:
    python clean.py
    python clean.py --all  # Also removes models/
"""

import shutil
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Clean training artifacts')
    parser.add_argument('--all', action='store_true',
                        help='Also remove trained models')
    args = parser.parse_args()

    project_dir = Path(__file__).parent

    # Directories to clean
    dirs_to_clean = [
        'dataset/train',
        'dataset/val',
        'results',
        'runs',
    ]

    if args.all:
        dirs_to_clean.append('models')

    cleaned = []
    for dir_name in dirs_to_clean:
        dir_path = project_dir / dir_name
        if dir_path.exists():
            shutil.rmtree(dir_path)
            cleaned.append(dir_name)

    # Remove dataset.yaml if exists
    yaml_path = project_dir / 'dataset' / 'dataset.yaml'
    if yaml_path.exists():
        yaml_path.unlink()
        cleaned.append('dataset/dataset.yaml')

    if cleaned:
        print("Cleaned:")
        for item in cleaned:
            print(f"  - {item}")
    else:
        print("Nothing to clean")

    print("\nPreserved:")
    print("  - source_images/")
    print("  - All .py scripts")
    if not args.all:
        print("  - models/ (use --all to remove)")


if __name__ == '__main__':
    main()
