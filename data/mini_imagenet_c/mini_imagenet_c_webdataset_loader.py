
import webdataset as wds
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import io

def identity(x):
    return x

def pil_decoder(key, data):
    """Decodes image data from bytes to a PIL Image."""
    if not key.endswith((".jpg", ".jpeg", ".png")):
        return None
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None

def cls_decoder(key, data):
    """Decodes class label from bytes."""
    if not key.endswith(".cls"):
        return None
    try:
        return int(data.decode('utf-8'))
    except (ValueError, UnicodeDecodeError):
        return None

class MiniImageNetCWebDataset(torch.utils.data.IterableDataset):
    """
    A PyTorch Dataset for the WebDataset version of MiniImageNet-C.

    Args:
        root (str): The root directory of the WebDataset shards.
        corruption (str): The corruption type to load (e.g., 'gaussian_noise').
        severity (int): The severity level (should be 5 for MiniImageNet-C).
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, `transforms.ToTensor()`.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, root, corruption, severity=5, transform=None, target_transform=None):
        self.root = Path(root)
        self.corruption = corruption
        self.severity = severity
        self.transform = transform if transform is not None else identity
        self.target_transform = target_transform if target_transform is not None else identity

        self.shard_path = self.root / self.corruption / str(self.severity)
        if not self.shard_path.exists():
            raise FileNotFoundError(f"Shards not found at: {self.shard_path}")

        shard_urls = [str(p) for p in sorted(self.shard_path.glob("*.tar"))]
        if not shard_urls:
            raise FileNotFoundError(f"No .tar shards found in {self.shard_path}")

        self.dataset = (
            wds.WebDataset(shard_urls, shardshuffle=True)
            .decode(pil_decoder, cls_decoder)
            .to_tuple("jpg", "cls")
            .map(self.apply_transforms)
        )

    def apply_transforms(self, sample):
        image, target = sample
        return self.transform(image), self.target_transform(target)

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        # The length of a WebDataset is not trivially known beforehand.
        # You can estimate it or, if needed, iterate through it once to count.
        # For Mini-ImageNet-C, each class has 50 images, and there are 1000 classes.
        return 50 * 1000

# Example Usage
if __name__ == '__main__':
    print("Example of how to use MiniImageNetCWebDataset")

    # This assumes you have a 'mini-imagenet-c-webdataset' directory
    # created by the convert_to_webdataset.py script.
    dataset_root = "../data/mini-imagenet-c-webdataset" 
    if not Path(dataset_root).exists():
        print(f"\nERROR: Example dataset root '{dataset_root}' not found.")
        print("Please run 'python data/scripts/convert_to_webdataset.py' first.")
        exit()

    # Define transformations for the images
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 1. Create a dataset for a specific corruption
    try:
        corruption_type = 'gaussian_noise'
        print(f"\nLoading dataset for corruption: '{corruption_type}'")
        
        dataset = MiniImageNetCWebDataset(
            root=dataset_root,
            corruption=corruption_type,
            transform=image_transform
        )

        # 2. Create a DataLoader
        # WebDataset is designed for streaming, so shuffling is handled differently.
        # For shuffling, you typically shuffle the shard URLs and the samples within each shard.
        # The loader here provides a basic sequential stream.
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4)

        # 3. Iterate through a few batches
        print("Iterating through a few batches...")
        for i, (images, labels) in enumerate(dataloader):
            if i >= 3:
                break
            print(f"  Batch {i+1}:")
            print(f"    Images shape: {images.shape}")
            print(f"    Labels shape: {labels.shape}")
            print(f"    Sample labels: {labels[:4].tolist()}")

        print("\nExample finished successfully!")

    except FileNotFoundError as e:
        print(f"\nERROR: Could not run example. {e}")
        print("Please ensure the WebDataset has been generated and the paths are correct.")

    except ImportError:
        print("\nERROR: 'webdataset' library not found.")
        print("Please install it by running: pip install webdataset")

