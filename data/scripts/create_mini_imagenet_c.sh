#!/bin/bash

# Example usage script for MiniImageNet-C generator
# This script demonstrates how to use the mini_imagenet_c.py script

# Configuration
SOURCE_IMAGENET_C="/path/to/original/ImageNet-C"  # Replace with actual path
OUTPUT_DIR="./mini_imagenet_c_dataset"           # Output directory
RANDOM_SEED=7600                                  # Fixed seed for reproducibility

echo "MiniImageNet-C Dataset Generator Usage Example"
echo "================================================"

# Check if source directory exists (in real usage)
# if [ ! -d "$SOURCE_IMAGENET_C" ]; then
#     echo "Error: Source ImageNet-C directory not found: $SOURCE_IMAGENET_C"
#     echo "Please download ImageNet-C and update the SOURCE_IMAGENET_C variable"
#     exit 1
# fi

echo "This script would create MiniImageNet-C dataset with the following configuration:"
echo "  - Source: $SOURCE_IMAGENET_C"
echo "  - Output: $OUTPUT_DIR"
echo "  - Random seed: $RANDOM_SEED"
echo "  - Images per class: 50"
echo "  - Severity level: 5 (highest)"
echo "  - Corruption types: All 15 types"

echo ""
echo "To actually run the generator, execute:"
echo "  python data/mini_imagenet_c/mini_imagenet_c.py --source $SOURCE_IMAGENET_C --output $OUTPUT_DIR --seed $RANDOM_SEED"

echo ""
echo "To process only specific corruptions:"
echo "  python data/mini_imagenet_c/mini_imagenet_c.py --source $SOURCE_IMAGENET_C --output $OUTPUT_DIR --seed $RANDOM_SEED --corruptions gaussian_noise shot_noise"

echo ""
echo "After generation, you can test the loader with:"
echo "  python data/mini_imagenet_c/mini_imagenet_c_loader.py --data_dir $OUTPUT_DIR --corruption gaussian_noise"

echo ""
echo "Expected output structure:"
echo "$OUTPUT_DIR/"
echo "└── mini-imagenet-c/"
echo "    ├── dataset_info.json"
echo "    ├── gaussian_noise/"
echo "    │   └── 5/"
echo "    │       ├── n01440764/  (50 images)"
echo "    │       ├── n01443537/  (50 images)"
echo "    │       └── ... (1000 classes total)"
echo "    ├── shot_noise/"
echo "    │   └── 5/"
echo "    │       └── ... (same structure)"
echo "    └── ... (15 corruption types total)"

echo ""
echo "Dataset statistics:"
echo "  - Total corruptions: 15"
echo "  - Total classes: 1000 (ImageNet classes)"
echo "  - Images per class per corruption: 50"
echo "  - Total images: 15 × 1000 × 50 = 750,000"
echo "  - Size: Approximately 1/50th of original ImageNet-C"