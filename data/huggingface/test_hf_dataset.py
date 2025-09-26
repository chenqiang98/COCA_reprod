"""
Example usage of MiniImageNet-C dataset after uploading to Hugging Face
"""

from datasets import load_dataset
import matplotlib.pyplot as plt

def test_hf_dataset(repo_name: str = "your-username/mini-imagenet-c"):
    """
    Test the uploaded MiniImageNet-C dataset
    
    Args:
        repo_name: Name of the repository on Hugging Face
    """
    
    print(f"Loading dataset from: {repo_name}")
    
    # Load dataset
    dataset = load_dataset(repo_name)
    
    print("Dataset info:")
    print(f"- Number of examples: {len(dataset['test'])}")
    print(f"- Features: {dataset['test'].features}")
    
    # Show some statistics
    print("\nDataset statistics:")
    corruption_counts = {}
    for example in dataset['test']:
        corruption = example['corruption_type']
        corruption_counts[corruption] = corruption_counts.get(corruption, 0) + 1
    
    print("Images per corruption type:")
    for corruption, count in sorted(corruption_counts.items()):
        print(f"  {corruption}: {count}")
    
    # Show a sample image
    print("\nSample data point:")
    sample = dataset['test'][0]
    print(f"- Label: {sample['label']}")
    print(f"- Class name: {sample['class_name']}")
    print(f"- Corruption: {sample['corruption_type']}")
    print(f"- Severity: {sample['severity']}")
    print(f"- Image shape: {sample['image'].size}")
    
    return dataset

def visualize_samples(dataset, num_samples=9):
    """
    Visualize random samples from the dataset
    """
    import random
    
    # Get random samples
    indices = random.sample(range(len(dataset['test'])), num_samples)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        sample = dataset['test'][idx]
        
        axes[i].imshow(sample['image'])
        axes[i].set_title(f"{sample['class_name']}\n{sample['corruption_type']}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test the dataset (replace with your actual repo name)
    dataset = test_hf_dataset("your-username/mini-imagenet-c")
    
    # Uncomment to visualize samples
    # visualize_samples(dataset)