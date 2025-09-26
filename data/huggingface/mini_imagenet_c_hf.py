#!/usr/bin/env python3
"""
Hugging Face dataset loader for MiniImageNet-C
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Iterator

import datasets
from PIL import Image


_DESCRIPTION = """\
MiniImageNet-C is a compact version of the ImageNet-C robustness benchmark dataset.
It contains 50 randomly selected images per class per corruption type at severity level 5,
totaling 750,000 images across 15 corruption types and 1000 ImageNet classes.
"""

_HOMEPAGE = "https://github.com/hendrycks/robustness"

_LICENSE = "MIT"

_CITATION = """\
@article{hendrycks2019robustness,
  title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
  author={Dan Hendrycks and Thomas Dietterich},
  journal={International Conference on Learning Representations},
  year={2019}
}
"""

# Corruption types in ImageNet-C
_CORRUPTION_TYPES = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression"
]


class MiniImageNetCConfig(datasets.BuilderConfig):
    """BuilderConfig for MiniImageNet-C."""
    
    def __init__(self, **kwargs):
        """BuilderConfig for MiniImageNet-C.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MiniImageNetCConfig, self).__init__(**kwargs)


class MiniImageNetC(datasets.GeneratorBasedBuilder):
    """MiniImageNet-C dataset."""
    
    BUILDER_CONFIGS = [
        MiniImageNetCConfig(
            name="default",
            version=datasets.Version("1.0.0"),
            description="MiniImageNet-C dataset with severity 5 corruptions",
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "default"
    
    def _info(self) -> datasets.DatasetInfo:
        """Return the dataset info."""
        # Load class names from ImageNet
        features = datasets.Features({
            "image": datasets.Image(),
            "label": datasets.ClassLabel(num_classes=1000),
            "corruption_type": datasets.Value("string"),
            "severity": datasets.Value("int32"),
            "class_name": datasets.Value("string"),
        })
        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )
    
    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Return SplitGenerators."""
        # The data is expected to be in the repository already
        data_dir = Path(self.config.data_dir or ".")
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_dir": data_dir,
                },
            ),
        ]
    
    def _generate_examples(self, data_dir: Path) -> Iterator[tuple[str, Dict[str, Any]]]:
        """Generate examples."""
        # Load class index mapping - try multiple locations
        class_index_file = data_dir / "imagenet_class_index.json"
        if not class_index_file.exists():
            # Try parent directory structure
            class_index_file = data_dir.parent / "imagenet_class_index.json"
        if not class_index_file.exists():
            # Try data directory in parent
            class_index_file = data_dir.parent / "data" / "imagenet_class_index.json"
            
        if class_index_file.exists():
            with open(class_index_file, 'r') as f:
                class_index = json.load(f)
        else:
            # Fallback to generic class names
            class_index = {str(i): [f"class_{i}", f"class_{i}"] for i in range(1000)}
        
        idx = 0
        
        # Iterate through each corruption type
        for corruption_type in _CORRUPTION_TYPES:
            corruption_dir = data_dir / corruption_type / "severity_5"
            
            if not corruption_dir.exists():
                continue
                
            # Iterate through each class directory
            for class_dir in sorted(corruption_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                    
                class_name = class_dir.name
                # Extract class index from directory name (assuming format like "n01440764")
                class_idx = None
                for idx_str, (wnid, readable_name) in class_index.items():
                    if wnid == class_name:
                        class_idx = int(idx_str)
                        class_readable_name = readable_name
                        break
                
                if class_idx is None:
                    continue
                
                # Iterate through images in this class
                image_files = sorted([f for f in class_dir.iterdir() 
                                    if f.suffix.lower() in ['.jpeg', '.jpg', '.png']])
                
                for image_file in image_files:
                    try:
                        image = Image.open(image_file)
                        
                        example = {
                            "image": image,
                            "label": class_idx,
                            "corruption_type": corruption_type,
                            "severity": 5,
                            "class_name": class_readable_name,
                        }
                        
                        yield f"{corruption_type}_{class_name}_{image_file.stem}", example
                        idx += 1
                        
                    except Exception as e:
                        print(f"Error loading image {image_file}: {e}")
                        continue


if __name__ == "__main__":
    # Test the dataset loader
    dataset = MiniImageNetC()
    print("Dataset info:", dataset.info)