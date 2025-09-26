"""
MiniImageNet-C Dataset Loader

This module provides a dataset loader for MiniImageNet-C that is compatible
with the existing COCA training pipeline.
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import json
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.augmentations import get_transform


class MiniImageNetC(Dataset):
    """
    MiniImageNet-C Dataset Loader
    
    A compact version of ImageNet-C with:
    - Only severity level 5
    - 50 images per class per corruption
    - Same structure as ImageNet-C but smaller
    """
    
    def __init__(self, root, corruption_type, severity=5, transform_anchor=None, 
                 transform_aux=None, single_model=False, class_index_path=None, 
                 n_examples=None):
        """
        Initialize MiniImageNet-C dataset
        
        Args:
            root: Root directory of MiniImageNet-C dataset
            corruption_type: Type of corruption to load
            severity: Severity level (should be 5 for MiniImageNet-C)
            transform_anchor: Transform for anchor model
            transform_aux: Transform for auxiliary model
            single_model: Whether using single model (no aux transforms)
            class_index_path: Path to class index JSON file
            n_examples: Maximum number of examples to load (optional)
        """
        if severity != 5:
            print(f"Warning: MiniImageNet-C only contains severity=5 data, "
                  f"but severity={severity} was requested")
        
        self.root = Path(root) / "mini-imagenet-c"
        self.corruption_path = self.root / corruption_type / "5"
        self.transform_anchor = transform_anchor
        self.transform_aux = transform_aux
        self.single_model = single_model
        self.image_paths = []
        self.labels = []
        
        # Load class index mapping
        if class_index_path is None:
            class_index_path = Path(__file__).parent.parent / "imagenet_class_index.json"
        
        with open(class_index_path, 'r') as f:
            dictionary = json.load(f)
        self.synset_to_class = {v[0]: int(k) for k, v in dictionary.items()}
        
        # Check if corruption path exists
        if not self.corruption_path.exists():
            raise FileNotFoundError(f"MiniImageNet-C corruption path not found: {self.corruption_path}")
        
        # Load images from class directories
        self._load_images()
        
        # Limit examples if requested
        if n_examples:
            self.image_paths = self.image_paths[:n_examples]
            self.labels = self.labels[:n_examples]
    
    def _load_images(self):
        """Load all images from the corruption directory"""
        class_dirs = sorted([d for d in self.corruption_path.iterdir() if d.is_dir()])
        
        for class_dir in class_dirs:
            synset_id = class_dir.name
            
            if synset_id not in self.synset_to_class:
                print(f"Warning: Unknown synset ID: {synset_id}")
                continue
            
            class_label = self.synset_to_class[synset_id]
            
            # Get all image files in this class directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPEG', '.PNG', '.JPG'}
            image_files = sorted([
                f for f in class_dir.iterdir() 
                if f.is_file() and f.suffix in image_extensions
            ])
            
            for img_file in image_files:
                self.image_paths.append(str(img_file))
                self.labels.append(class_label)
        
        print(f"Loaded {len(self.image_paths)} images from {len(class_dirs)} classes")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and convert image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        img_anchor = self.transform_anchor(image) if self.transform_anchor else image
        
        if self.single_model:
            return img_anchor, label
        
        img_aux = self.transform_aux(image) if self.transform_aux else image
        return img_anchor, img_aux, label


def get_mini_imagenet_c_loader(data_dir, corruption_type, severity=5, batch_size=32, 
                               num_workers=8, anchor_model_name='vit_base_patch16_224', 
                               aux_model_name=None, shuffle=False, n_examples=None):
    """
    Create a DataLoader for MiniImageNet-C dataset
    
    Args:
        data_dir: Root directory containing mini-imagenet-c
        corruption_type: Type of corruption to load
        severity: Severity level (should be 5)
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        anchor_model_name: Name of anchor model for transforms
        aux_model_name: Name of auxiliary model for transforms (optional)
        shuffle: Whether to shuffle the data
        n_examples: Maximum number of examples to load (optional)
    
    Returns:
        DataLoader for MiniImageNet-C
    """
    transform_anchor = get_transform(anchor_model_name)
    
    single_model = aux_model_name is None
    transform_aux = None
    if not single_model:
        transform_aux = get_transform(aux_model_name)
    
    dataset = MiniImageNetC(
        root=data_dir,
        corruption_type=corruption_type,
        severity=severity,
        transform_anchor=transform_anchor,
        transform_aux=transform_aux,
        single_model=single_model,
        n_examples=n_examples
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return loader


def list_available_corruptions(data_dir):
    """
    List all available corruption types in MiniImageNet-C dataset
    
    Args:
        data_dir: Root directory containing mini-imagenet-c
    
    Returns:
        List of available corruption types
    """
    mini_root = Path(data_dir) / "mini-imagenet-c"
    
    if not mini_root.exists():
        return []
    
    corruptions = [
        d.name for d in mini_root.iterdir() 
        if d.is_dir() and (d / "5").exists()
    ]
    
    return sorted(corruptions)


def get_dataset_stats(data_dir):
    """
    Get statistics about the MiniImageNet-C dataset
    
    Args:
        data_dir: Root directory containing mini-imagenet-c
    
    Returns:
        Dictionary with dataset statistics
    """
    mini_root = Path(data_dir) / "mini-imagenet-c"
    
    if not mini_root.exists():
        return {"error": "MiniImageNet-C not found"}
    
    # Check for dataset info file
    info_file = mini_root / "dataset_info.json"
    if info_file.exists():
        with open(info_file, 'r') as f:
            return json.load(f)
    
    # Manually count if no info file
    stats = {
        "corruption_types": [],
        "total_images": 0,
        "classes_per_corruption": {}
    }
    
    corruptions = list_available_corruptions(data_dir)
    stats["corruption_types"] = corruptions
    
    for corruption in corruptions:
        corruption_path = mini_root / corruption / "5"
        if corruption_path.exists():
            class_count = len([d for d in corruption_path.iterdir() if d.is_dir()])
            image_count = sum(
                len([f for f in class_dir.iterdir() if f.is_file()])
                for class_dir in corruption_path.iterdir()
                if class_dir.is_dir()
            )
            stats["classes_per_corruption"][corruption] = {
                "classes": class_count,
                "images": image_count
            }
            stats["total_images"] += image_count
    
    return stats


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MiniImageNet-C dataset loader')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing mini-imagenet-c')
    parser.add_argument('--corruption', type=str, default='gaussian_noise',
                        help='Corruption type to test')
    
    args = parser.parse_args()
    
    # List available corruptions
    corruptions = list_available_corruptions(args.data_dir)
    print(f"Available corruptions: {corruptions}")
    
    # Get dataset stats
    stats = get_dataset_stats(args.data_dir)
    print(f"Dataset stats: {json.dumps(stats, indent=2)}")
    
    # Test loader
    if args.corruption in corruptions:
        try:
            loader = get_mini_imagenet_c_loader(
                args.data_dir, 
                args.corruption, 
                batch_size=4, 
                num_workers=0
            )
            
            print(f"\nTesting loader for corruption: {args.corruption}")
            print(f"Dataset size: {len(loader.dataset)}")
            
            # Get a batch
            for batch_idx, (images, labels) in enumerate(loader):
                print(f"Batch {batch_idx}: {images.shape}, labels: {labels}")
                if batch_idx >= 2:  # Only show first few batches
                    break
                    
        except Exception as e:
            print(f"Error testing loader: {e}")
    else:
        print(f"Corruption {args.corruption} not available")