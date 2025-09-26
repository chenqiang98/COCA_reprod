"""
MiniImageNet-C Dataset Generator

This script creates a compact version of ImageNet-C called MiniImageNet-C with the following properties:
- Only uses severity level 5 (the highest corruption level)
- Randomly selects 50 images per corruption per class (using fixed seed for reproducibility)
- Maintains the same file structure as the original ImageNet-C
- Uses random seed 7600 for reproducible sampling

The original ImageNet-C dataset has the following structure:
<root>/<corruption_type>/5/<synset_id>/*.JPEG

This script will create:
<output_root>/mini-imagenet-c/<corruption_type>/5/<synset_id>/*.JPEG

With exactly 50 images per class per corruption type.
"""

import os
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List
import argparse


class MiniImageNetCGenerator:
    """Generator for MiniImageNet-C dataset"""
    
    def __init__(self, source_root: str, output_root: str, random_seed: int = 7600):
        """
        Initialize the MiniImageNet-C generator
        
        Args:
            source_root: Path to the original ImageNet-C dataset
            output_root: Path where MiniImageNet-C will be created
            random_seed: Random seed for reproducible sampling (default: 7600)
        """
        self.source_root = Path(source_root)
        self.output_root = Path(output_root) / "mini-imagenet-c"
        self.random_seed = random_seed
        self.images_per_class = 50
        self.target_severity = 5
        
        # ImageNet-C corruption types (15 total)
        self.corruption_types = [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
        
        # Load ImageNet class mappings
        self.load_class_mappings()
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        
    def load_class_mappings(self):
        """Load ImageNet class index mappings"""
        # Try multiple possible locations for the class index file
        possible_paths = [
            Path(__file__).parent.parent / "imagenet_class_index.json",  # data/imagenet_class_index.json
            Path(self.output_root) / "imagenet_class_index.json",  # in output directory
            Path("data/imagenet_class_index.json"),  # relative to project root
            Path("imagenet_class_index.json"),  # current directory
        ]
        
        class_index_path = None
        for path in possible_paths:
            if path.exists():
                class_index_path = path
                break
        
        if class_index_path is None:
            raise FileNotFoundError("imagenet_class_index.json not found in any of the expected locations: " +
                                  ", ".join(str(p) for p in possible_paths))
        
        print(f"Using class index file: {class_index_path}")
            
        with open(class_index_path, 'r') as f:
            class_dict = json.load(f)
            
        # Create mapping from synset ID to class index
        self.synset_to_class = {v[0]: int(k) for k, v in class_dict.items()}
        self.class_to_synset = {int(k): v[0] for k, v in class_dict.items()}
        self.synset_names = {v[0]: v[1] for v in class_dict.values()}
        
        print(f"Loaded {len(self.synset_to_class)} class mappings")
        
    def get_corruption_path(self, corruption_type: str) -> Path:
        """Get the path for a specific corruption type at severity 5"""
        return self.source_root / corruption_type / str(self.target_severity)
        
    def sample_images_from_class(self, class_path: Path, synset_id: str) -> List[Path]:
        """
        Sample exactly 50 images from a class directory
        
        Args:
            class_path: Path to the class directory
            synset_id: Synset ID for the class
            
        Returns:
            List of selected image paths
        """
        if not class_path.exists() or not class_path.is_dir():
            print(f"Warning: Class directory not found: {class_path}")
            return []
            
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPEG', '.PNG', '.JPG'}
        all_images = [
            img for img in class_path.iterdir() 
            if img.is_file() and img.suffix in image_extensions
        ]
        
        if len(all_images) == 0:
            print(f"Warning: No images found in {class_path}")
            return []
            
        if len(all_images) < self.images_per_class:
            print(f"Warning: Only {len(all_images)} images available for {synset_id}, using all")
            return all_images
            
        # Use a deterministic seed based on synset_id for consistent sampling across runs
        # but still maintain overall randomness
        temp_state = random.getstate()
        random.seed(self.random_seed + hash(synset_id) % 10000)
        selected_images = random.sample(all_images, self.images_per_class)
        random.setstate(temp_state)
        
        return selected_images
        
    def create_mini_dataset_for_corruption(self, corruption_type: str) -> bool:
        """
        Create mini dataset for a specific corruption type
        
        Args:
            corruption_type: Name of the corruption type
            
        Returns:
            True if successful, False otherwise
        """
        print(f"\nProcessing corruption: {corruption_type}")
        
        # Source and target paths
        source_corruption_path = self.get_corruption_path(corruption_type)
        target_corruption_path = self.output_root / corruption_type / str(self.target_severity)
        
        if not source_corruption_path.exists():
            print(f"Warning: Source corruption path not found: {source_corruption_path}")
            return False
            
        # Create target directory
        target_corruption_path.mkdir(parents=True, exist_ok=True)
        
        # Get all class directories
        class_dirs = [d for d in source_corruption_path.iterdir() if d.is_dir()]
        
        if len(class_dirs) == 0:
            print(f"Warning: No class directories found in {source_corruption_path}")
            return False
            
        total_images_copied = 0
        processed_classes = 0
        
        for class_dir in sorted(class_dirs):
            synset_id = class_dir.name
            
            if synset_id not in self.synset_to_class:
                print(f"Warning: Unknown synset ID: {synset_id}")
                continue
                
            # Sample images from this class
            selected_images = self.sample_images_from_class(class_dir, synset_id)
            
            if len(selected_images) == 0:
                continue
                
            # Create target class directory
            target_class_dir = target_corruption_path / synset_id
            target_class_dir.mkdir(exist_ok=True)
            
            # Copy selected images
            for img_path in selected_images:
                target_img_path = target_class_dir / img_path.name
                shutil.copy2(img_path, target_img_path)
                
            total_images_copied += len(selected_images)
            processed_classes += 1
            
            if processed_classes % 100 == 0:
                print(f"  Processed {processed_classes} classes, copied {total_images_copied} images")
                
        print(f"  Completed {corruption_type}: {processed_classes} classes, {total_images_copied} images")
        return True
        
    def create_mini_dataset(self) -> Dict[str, bool]:
        """
        Create the complete MiniImageNet-C dataset
        
        Returns:
            Dictionary with corruption types as keys and success status as values
        """
        print(f"Creating MiniImageNet-C dataset")
        print(f"Source: {self.source_root}")
        print(f"Output: {self.output_root}")
        print(f"Random seed: {self.random_seed}")
        print(f"Images per class: {self.images_per_class}")
        print(f"Target severity: {self.target_severity}")
        
        # Create output directory
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for corruption_type in self.corruption_types:
            results[corruption_type] = self.create_mini_dataset_for_corruption(corruption_type)
            
        return results
        
    def create_dataset_info(self):
        """Create a JSON file with dataset information"""
        info = {
            "name": "MiniImageNet-C",
            "description": "Compact version of ImageNet-C with 50 images per class per corruption at severity 5",
            "source_dataset": "ImageNet-C",
            "severity_level": self.target_severity,
            "images_per_class": self.images_per_class,
            "num_classes": len(self.synset_to_class),
            "num_corruptions": len(self.corruption_types),
            "corruption_types": self.corruption_types,
            "random_seed": self.random_seed,
            "total_expected_images": len(self.synset_to_class) * self.images_per_class * len(self.corruption_types)
        }
        
        info_path = self.output_root / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
            
        print(f"\nDataset info saved to: {info_path}")
        
    def validate_dataset(self) -> Dict[str, int]:
        """
        Validate the created dataset and return statistics
        
        Returns:
            Dictionary with statistics
        """
        print("\nValidating created dataset...")
        
        stats = {
            "total_images": 0,
            "corruptions_created": 0,
            "classes_per_corruption": {},
            "images_per_corruption": {}
        }
        
        for corruption_type in self.corruption_types:
            corruption_path = self.output_root / corruption_type / str(self.target_severity)
            
            if not corruption_path.exists():
                print(f"Warning: Corruption {corruption_type} was not created")
                continue
                
            stats["corruptions_created"] += 1
            
            class_dirs = [d for d in corruption_path.iterdir() if d.is_dir()]
            stats["classes_per_corruption"][corruption_type] = len(class_dirs)
            
            corruption_images = 0
            for class_dir in class_dirs:
                image_files = [f for f in class_dir.iterdir() if f.is_file()]
                corruption_images += len(image_files)
                
            stats["images_per_corruption"][corruption_type] = corruption_images
            stats["total_images"] += corruption_images
            
        print(f"Dataset validation completed:")
        print(f"  Total corruptions: {stats['corruptions_created']}/{len(self.corruption_types)}")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Average images per corruption: {stats['total_images'] / max(1, stats['corruptions_created']):.1f}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Create MiniImageNet-C dataset')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to the original ImageNet-C dataset')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for MiniImageNet-C')
    parser.add_argument('--seed', type=int, default=7600,
                        help='Random seed for reproducible sampling (default: 7600)')
    parser.add_argument('--corruptions', nargs='*', default=None,
                        help='Specific corruptions to process (default: all)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = MiniImageNetCGenerator(args.source, args.output, args.seed)
    
    # Override corruption types if specified
    if args.corruptions:
        # Validate corruption types
        invalid_corruptions = set(args.corruptions) - set(generator.corruption_types)
        if invalid_corruptions:
            print(f"Error: Invalid corruption types: {invalid_corruptions}")
            print(f"Valid corruption types: {generator.corruption_types}")
            return
        generator.corruption_types = args.corruptions
        print(f"Processing only specified corruptions: {args.corruptions}")
    
    try:
        # Copy configuration files to output directory if needed
        class_index_src = Path(__file__).parent.parent / "imagenet_class_index.json"
        class_index_dst = Path(args.output) / "mini-imagenet-c" / "imagenet_class_index.json"
        
        if class_index_src.exists() and not class_index_dst.exists():
            class_index_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(class_index_src, class_index_dst)
            print(f"Copied class index to output directory: {class_index_dst}")
        
        # Create the dataset
        results = generator.create_mini_dataset()
        
        # Create dataset info
        generator.create_dataset_info()
        
        # Validate the dataset
        stats = generator.validate_dataset()
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        successful = sum(results.values())
        total = len(results)
        
        print(f"Successfully processed: {successful}/{total} corruptions")
        print(f"Total images created: {stats['total_images']}")
        print(f"Output directory: {generator.output_root}")
        
        if successful < total:
            print("\nFailed corruptions:")
            for corruption, success in results.items():
                if not success:
                    print(f"  - {corruption}")
                    
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()