---
license: mit
task_categories:
- image-classification
task_ids:
- multi-class-image-classification
pretty_name: MiniImageNet-C
size_categories:
- 100K<n<1M
tags:
- computer-vision
- robustness
- corruption
- imagenet
- benchmark
configs:
- config_name: default
  data_files:
  - split: test
    path: "*/severity_5/*/*.JPEG"
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype: class_label
    names:
      _type: Value
      dtype: string
  - name: corruption_type
    dtype: string
  - name: severity
    dtype: int32
  splits:
  - name: test
    num_bytes: 750000000
    num_examples: 750000
  download_size: 750000000
  dataset_size: 750000000
---

# MiniImageNet-C Dataset

## Dataset Description

MiniImageNet-C is a compact version of the ImageNet-C robustness benchmark dataset. It contains corrupted images from ImageNet designed to test the robustness of computer vision models to various types of image corruptions.

### Dataset Summary

This dataset is a subset of the original ImageNet-C dataset, containing:
- **15 corruption types**: gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression
- **1 severity level**: Only severity level 5 (most severe)
- **50 images per class per corruption**: Randomly selected from the original dataset
- **1000 classes**: All ImageNet classes
- **Total images**: 750,000 (15 corruptions × 50 images × 1000 classes)

The dataset uses a fixed random seed (7600) for reproducible image selection.

### Supported Tasks and Leaderboards

- **Image Classification**: Multi-class image classification with 1000 classes
- **Robustness Evaluation**: Testing model performance under various image corruptions
- **Benchmarking**: Comparing model robustness across different corruption types

### Languages

Not applicable (computer vision dataset).

## Dataset Structure

### Data Instances

Each instance contains:
- `image`: A PIL Image object
- `label`: Integer class label (0-999)
- `corruption_type`: String indicating the type of corruption applied
- `severity`: Integer indicating corruption severity (always 5)

### Data Fields

- `image` (PIL Image): The corrupted image
- `label` (int): Class label corresponding to ImageNet classes
- `corruption_type` (string): One of 15 corruption types
- `severity` (int): Corruption severity level (always 5)

### Data Splits

The dataset contains only a test split with 750,000 images total.

## Dataset Creation

### Curation Rationale

This dataset was created to provide a more manageable subset of ImageNet-C for:
- Quick robustness evaluation during development
- Reduced computational requirements for benchmarking
- Educational purposes and prototyping

### Source Data

#### Initial Data Collection and Normalization

The source data comes from ImageNet-C, which applies algorithmic corruptions to the original ImageNet validation set.

#### Who are the source language producers?

Not applicable.

### Annotations

#### Annotation process

Labels are inherited from the original ImageNet dataset.

#### Who are the annotators?

Original ImageNet annotators.

### Personal and Sensitive Information

The dataset contains no personal or sensitive information.

## Considerations for Using the Data

### Social Impact of Dataset

This dataset is intended for research purposes to improve the robustness of computer vision models.

### Discussion of Biases

Inherits any biases present in the original ImageNet dataset.

### Other Known Limitations

- Limited to severity level 5 only
- Reduced number of images per class may not capture full diversity
- May not be representative of real-world corruptions

## Additional Information

### Dataset Curators

Created for research purposes based on ImageNet-C.

### Licensing Information

MIT License

### Citation Information

```bibtex
@article{hendrycks2019robustness,
  title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
  author={Dan Hendrycks and Thomas Dietterich},
  journal={International Conference on Learning Representations},
  year={2019}
}
```

### Contributions

This compact version was created to provide an accessible subset of ImageNet-C for rapid prototyping and development.