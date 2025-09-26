#!/usr/bin/env bash

# Hugging Face Upload Instructions for MiniImageNet-C Dataset
# Make sure to run this script after generating the dataset

set -e

echo "=== MiniImageNet-C Hugging Face Upload Setup ==="

# Step 1: Install required dependencies
echo "Installing required packages..."
pip install huggingface_hub datasets pillow

# Step 2: Login to Hugging Face (you'll need to provide your token)
echo "Logging in to Hugging Face..."
echo "You can get your token from: https://huggingface.co/settings/tokens"
huggingface-cli login

# Step 3: Set up environment variables
export HF_DATASETS_CACHE="/tmp/hf_cache"  # Optional: set cache directory

# Step 4: Verify dataset structure
echo "Verifying dataset structure..."
DATASET_PATH="./mini_imagenet_c_dataset"

if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset not found at $DATASET_PATH"
    echo "Please run the dataset generation script first:"
    echo "python data/mini_imagenet_c/mini_imagenet_c.py --source /path/to/imagenet-c --output $DATASET_PATH"
    exit 1
fi

echo "Dataset found at: $DATASET_PATH"
echo "Dataset structure:"
find $DATASET_PATH -type d -maxdepth 3 | head -20

# Step 5: Upload dataset
echo "Uploading dataset to Hugging Face..."
echo "Usage: python data/huggingface/upload_to_hf.py --dataset_path $DATASET_PATH [options]"

# Example upload commands:
echo ""
echo "=== Upload Commands ==="
echo ""
echo "1. Basic upload (public repository):"
echo "   python data/huggingface/upload_to_hf.py --dataset_path $DATASET_PATH --repo_name mini-imagenet-c"
echo ""
echo "2. Upload to organization:"
echo "   python data/huggingface/upload_to_hf.py --dataset_path $DATASET_PATH --repo_name mini-imagenet-c --organization your-org"
echo ""
echo "3. Private repository:"
echo "   python data/huggingface/upload_to_hf.py --dataset_path $DATASET_PATH --repo_name mini-imagenet-c --private"
echo ""

# Step 6: Test dataset loading
echo "To test the uploaded dataset, use:"
echo "python -c \"from datasets import load_dataset; ds = load_dataset('your-username/mini-imagenet-c'); print(ds)\""

echo ""
echo "=== Additional Files Created ==="
echo "- data/huggingface/dataset_card.md: Dataset documentation for Hugging Face"
echo "- data/mini_imagenet_c/mini_imagenet_c.py: Dataset generator script"
echo "- data/huggingface/upload_to_hf.py: Upload script"
echo ""
echo "Make sure to copy dataset_card.md to README.md in your Hugging Face repository!"