
import os
import json
from pathlib import Path
import argparse
import subprocess
import sys

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError:
    print("nltk library not found. Attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
        import nltk
        from nltk.corpus import wordnet as wn
        print("nltk installed successfully.")
    except Exception as e:
        print(f"Error: Failed to install nltk. Please install it manually using 'pip install nltk'.")
        print(f"Details: {e}")
        sys.exit(1)

def download_wordnet():
    """Checks if WordNet is downloaded and downloads it if necessary."""
    try:
        # The quickest way to check for wordnet is to try to use it.
        wn.synsets('dog')
    except (LookupError, NameError):
        print("WordNet corpus not found. Attempting to download...")
        try:
            nltk.download('wordnet')
            # Re-check to ensure download was successful
            wn.synsets('dog')
            print("WordNet corpus downloaded successfully.")
        except Exception as e:
            print(f"Error: Failed to download WordNet corpus. Please run `python -m nltk.downloader wordnet` manually.")
            print(f"Details: {e}")
            sys.exit(1)

def get_wordnet_name(synset_id: str) -> str:
    """
    Retrieves the human-readable name for a given WordNet synset ID.

    Args:
        synset_id: The synset ID in the format 'n12345678'.

    Returns:
        The human-readable name or a placeholder if not found.
    """
    try:
        pos = synset_id[0]
        offset = int(synset_id[1:])
        synset = wn.synset_from_pos_and_offset(pos, offset)
        # Return the first lemma, which is usually the most common name
        return synset.lemmas()[0].name().replace('_', ' ')
    except Exception:
        return f"Unknown Class ({synset_id})"

def create_class_index(imagenet_c_root: str, output_file: str):
    """
    Scans the ImageNet-C directory to find all class synsets and generates a JSON index file.

    Args:
        imagenet_c_root: The root directory of the ImageNet-C dataset.
        output_file: The path to save the generated JSON file.
    """
    # Ensure WordNet is available before starting
    download_wordnet()

    source_path = Path(imagenet_c_root)
    output_path = Path(output_file)
    
    print(f"Scanning {source_path} for class directories...")

    # We only need to scan one corruption type as they should all have the same classes.
    # Let's pick 'gaussian_noise' and severity '5'.
    corruption_path = source_path / 'gaussian_noise' / '5'
    if not corruption_path.exists():
        raise FileNotFoundError(f"Could not find a sample corruption directory to scan: {corruption_path}")

    synset_dirs = sorted([d for d in corruption_path.iterdir() if d.is_dir()])
    
    if not synset_dirs:
        raise FileNotFoundError(f"No class directories found in {corruption_path}")

    class_index = {}
    missing_names = 0
    print("Looking up synset names from WordNet...")
    for i, synset_dir in enumerate(synset_dirs):
        synset_id = synset_dir.name
        class_name = get_wordnet_name(synset_id)
        if "Unknown" in class_name:
            missing_names += 1
        class_index[str(i)] = [synset_id, class_name]

    print(f"Found {len(class_index)} unique classes.")
    if missing_names > 0:
        print(f"Warning: Could not find human-readable names for {missing_names} synset IDs.")

    # Save the generated index to the specified file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(class_index, f, indent=4)

    print(f"Successfully created class index file at: {output_path}")
    print(f"Total classes: {len(class_index)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate ImageNet class index from dataset directory using WordNet.")
    parser.add_argument("source_root", type=str, help="Path to the original ImageNet-C dataset.")
    parser.add_argument("--output_file", type=str, default="data/imagenet_class_index.json",
                        help="Path to save the generated JSON file.")
    args = parser.parse_args()

    create_class_index(args.source_root, args.output_file)
