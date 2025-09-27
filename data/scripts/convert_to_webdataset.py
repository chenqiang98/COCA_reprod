
import os
import tarfile
from pathlib import Path
import argparse
from tqdm import tqdm
import json

def create_webdataset(source_dir: str, output_dir: str, shard_size: int = 5000):
    """
    Converts an image folder dataset to the WebDataset format.

    The script will create TAR archives (shards) of images and their corresponding
    class labels. The structure will be:
    
    output_dir/
    ├── corruption_1/
    │   ├── severity_1/
    │   │   ├── 00000.tar
    │   │   ├── 00001.tar
    │   │   └── ...
    │   └── severity_2/
    │       └── ...
    └── corruption_2/
        └── ...

    Args:
        source_dir (str): The root directory of the MiniImageNet-C dataset.
        output_dir (str): The directory where the WebDataset shards will be saved.
        shard_size (int): The maximum number of images per TAR shard.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory not found at {source_path}")
        return

    print(f"Starting WebDataset conversion for {source_path}")
    print(f"Output will be saved to {output_path}")
    print(f"Shard size set to {shard_size} images per .tar file.")

    # Find class index from the directory structure of the first available corruption/severity
    try:
        sample_path = next(p for p in source_path.glob('*/*') if p.is_dir())
        class_dirs = sorted([d.name for d in sample_path.iterdir() if d.is_dir()])
        class_to_idx = {class_name: i for i, class_name in enumerate(class_dirs)}
        print(f"Found {len(class_to_idx)} classes.")
    except StopIteration:
        print("Error: Could not find any class subdirectories in the source dataset.")
        return

    corruption_dirs = [d for d in source_path.iterdir() if d.is_dir()]

    for corruption_dir in tqdm(corruption_dirs, desc="Corruptions"):
        for severity_dir in tqdm(corruption_dir.iterdir(), desc=f"Severities in {corruption_dir.name}", leave=False):
            if not severity_dir.is_dir():
                continue

            shard_output_path = output_path / corruption_dir.name / severity_dir.name
            shard_output_path.mkdir(parents=True, exist_ok=True)

            files_to_archive = []
            for class_dir in severity_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                class_idx = class_to_idx.get(class_dir.name)
                if class_idx is None:
                    continue
                
                for image_file in class_dir.glob('*.JPEG'):
                    files_to_archive.append((image_file, class_idx, image_file.stem))

            shard_count = 0
            for i in tqdm(range(0, len(files_to_archive), shard_size), desc="Shards", leave=False):
                shard_files = files_to_archive[i:i+shard_size]
                shard_filename = shard_output_path / f"{shard_count:05d}.tar"
                
                with tarfile.open(shard_filename, "w") as tar:
                    for image_path, class_idx, base_name in shard_files:
                        # Add image file
                        arcname_img = f"{base_name}.jpg"
                        tar.add(image_path, arcname=arcname_img)
                        
                        # Add class file
                        arcname_cls = f"{base_name}.cls"
                        class_bytes = str(class_idx).encode('utf-8')
                        tarinfo = tarfile.TarInfo(name=arcname_cls)
                        tarinfo.size = len(class_bytes)
                        tar.addfile(tarinfo, fileobj=io.BytesIO(class_bytes))

                shard_count += 1
    
    print("\nWebDataset conversion complete!")
    print(f"Shards are saved in: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert an image dataset to WebDataset format.")
    parser.add_argument(
        "source_dir",
        type=str,
        help="Path to the source image dataset directory (e.g., mini_imagenet_c)."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the WebDataset .tar shards."
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=5000,
        help="Maximum number of images per shard. Default is 5000."
    )
    args = parser.parse_args()

    create_webdataset(args.source_dir, args.output_dir, args.shard_size)

if __name__ == "__main__":
    # Add io import for the script to run
    import io
    main()
