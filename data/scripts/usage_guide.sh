#!/bin/bash

# Updated usage example for MiniImageNet-C generator with new file structure
# This script shows the complete workflow including path considerations

set -e

echo "MiniImageNet-C Dataset Generator - Updated Usage Guide"
echo "====================================================="

# Configuration
SOURCE_IMAGENET_C="/path/to/original/ImageNet-C"  # Replace with actual path
OUTPUT_DIR="./mini_imagenet_c_dataset"           # Output directory
RANDOM_SEED=7600                                  # Fixed seed for reproducibility

echo ""
echo "File Structure After Reorganization:"
echo "======================================"
echo "data/"
echo "├── mini_imagenet_c/"
echo "│   ├── mini_imagenet_c.py          # Main generator script"
echo "│   ├── mini_imagenet_c_loader.py   # PyTorch loader"
echo "│   └── prepare_config.py           # Config preparation utility"
echo "├── huggingface/"
echo "│   └── ... (HF integration files)"
echo "├── scripts/"
echo "│   └── ... (utility scripts)"
echo "└── imagenet_class_index.json       # Class index file"

echo ""
echo "Step-by-Step Usage:"
echo "==================="

echo ""
echo "1. Generate MiniImageNet-C dataset:"
echo "   cd /path/to/COCA_reprod"
echo "   python data/mini_imagenet_c/mini_imagenet_c.py \\"
echo "       --source $SOURCE_IMAGENET_C \\"
echo "       --output $OUTPUT_DIR \\"
echo "       --seed $RANDOM_SEED"

echo ""
echo "2. (Optional) Process only specific corruptions:"
echo "   python data/mini_imagenet_c/mini_imagenet_c.py \\"
echo "       --source $SOURCE_IMAGENET_C \\"
echo "       --output $OUTPUT_DIR \\"
echo "       --seed $RANDOM_SEED \\"
echo "       --corruptions gaussian_noise shot_noise defocus_blur"

echo ""
echo "3. Test the generated dataset:"
echo "   python data/mini_imagenet_c/mini_imagenet_c_loader.py \\"
echo "       --data_dir $OUTPUT_DIR \\"
echo "       --corruption gaussian_noise"

echo ""
echo "4. Use in training pipeline:"
echo "   # In your training script:"
echo "   from data.mini_imagenet_c.mini_imagenet_c_loader import get_mini_imagenet_c_loader"
echo "   loader = get_mini_imagenet_c_loader("
echo "       data_dir='$OUTPUT_DIR',"
echo "       corruption_type='gaussian_noise',"
echo "       batch_size=32"
echo "   )"

echo ""
echo "5. Upload to Hugging Face (optional):"
echo "   ./data/scripts/hf_upload_setup.sh"

echo ""
echo "Path Resolution Features:"
echo "========================="
echo "✓ Automatic class index file detection from multiple locations"
echo "✓ Auto-copy configuration files to output directory"
echo "✓ Relative import handling for utils.augmentations"
echo "✓ Robust path resolution for different execution contexts"

echo ""
echo "Expected Output Structure:"
echo "=========================="
echo "$OUTPUT_DIR/"
echo "└── mini-imagenet-c/"
echo "    ├── dataset_info.json"
echo "    ├── imagenet_class_index.json  # Auto-copied"
echo "    ├── gaussian_noise/"
echo "    │   └── 5/"
echo "    │       ├── n01440764/  (50 images)"
echo "    │       └── ... (1000 classes)"
echo "    └── ... (15 corruption types)"

echo ""
echo "Notes:"
echo "======"
echo "- Run from project root directory (COCA_reprod/)"
echo "- Class index file is automatically located and copied"
echo "- All paths are now organized in logical subdirectories"
echo "- Scripts handle relative imports correctly"
echo "- Configuration files are managed automatically"