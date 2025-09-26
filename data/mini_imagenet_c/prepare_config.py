#!/usr/bin/env python3
"""
Prepare dataset configuration files for MiniImageNet-C generation
This script copies necessary configuration files to the output directory
"""

import shutil
import argparse
from pathlib import Path


def prepare_dataset_config(output_dir: str):
    """
    Copy necessary configuration files to the dataset output directory
    
    Args:
        output_dir: Output directory where dataset will be generated
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get the data directory (parent of mini_imagenet_c directory)
    data_dir = Path(__file__).parent.parent
    
    # Copy imagenet class index
    class_index_src = data_dir / "imagenet_class_index.json"
    class_index_dst = output_path / "imagenet_class_index.json"
    
    if class_index_src.exists():
        shutil.copy2(class_index_src, class_index_dst)
        print(f"✓ Copied class index: {class_index_dst}")
    else:
        print(f"✗ Class index not found: {class_index_src}")
        return False
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset configuration')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for dataset generation')
    
    args = parser.parse_args()
    
    success = prepare_dataset_config(args.output_dir)
    if success:
        print(f"✓ Dataset configuration prepared in: {args.output_dir}")
    else:
        print("✗ Failed to prepare dataset configuration")
        exit(1)