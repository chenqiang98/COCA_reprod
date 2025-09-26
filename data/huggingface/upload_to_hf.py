#!/usr/bin/env python3
"""
Upload script for MiniImageNet-C dataset to Hugging Face Hub
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder


def upload_dataset(
    dataset_path: str,
    repo_name: str,
    organization: str = None,
    private: bool = False,
    token: str = None
):
    """
    Upload MiniImageNet-C dataset to Hugging Face Hub
    
    Args:
        dataset_path: Path to the generated MiniImageNet-C dataset
        repo_name: Name of the repository on Hugging Face
        organization: Optional organization name
        private: Whether to make the repository private
        token: Hugging Face token (if not provided, will use HF_TOKEN env var)
    """
    
    # Initialize Hugging Face API
    api = HfApi(token=token)
    
    # Create repository name
    if organization:
        full_repo_name = f"{organization}/{repo_name}"
    else:
        full_repo_name = repo_name
    
    print(f"Creating repository: {full_repo_name}")
    
    try:
        # Create repository
        create_repo(
            repo_id=full_repo_name,
            repo_type="dataset",
            private=private,
            token=token,
            exist_ok=True
        )
        print(f"✓ Repository created/verified: {full_repo_name}")
        
        # Upload dataset files
        print("Uploading dataset files...")
        
        dataset_path = Path(dataset_path)
        
        # Upload the main dataset folder
        upload_folder(
            folder_path=dataset_path,
            repo_id=full_repo_name,
            repo_type="dataset",
            token=token,
            ignore_patterns=["*.pyc", "__pycache__", ".git", ".gitignore"]
        )
        
        print(f"✓ Dataset uploaded successfully!")
        print(f"Dataset URL: https://huggingface.co/datasets/{full_repo_name}")
        
    except Exception as e:
        print(f"✗ Error during upload: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Upload MiniImageNet-C to Hugging Face")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=True,
        help="Path to the MiniImageNet-C dataset directory"
    )
    parser.add_argument(
        "--repo_name", 
        type=str, 
        default="mini-imagenet-c",
        help="Repository name on Hugging Face (default: mini-imagenet-c)"
    )
    parser.add_argument(
        "--organization", 
        type=str, 
        default=None,
        help="Hugging Face organization name (optional)"
    )
    parser.add_argument(
        "--private", 
        action="store_true",
        help="Make the repository private"
    )
    parser.add_argument(
        "--token", 
        type=str, 
        default=None,
        help="Hugging Face token (if not provided, uses HF_TOKEN env var)"
    )
    
    args = parser.parse_args()
    
    # Get token from environment if not provided
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("Error: Please provide a Hugging Face token via --token or HF_TOKEN env var")
        return
    
    # Verify dataset path exists
    if not Path(args.dataset_path).exists():
        print(f"Error: Dataset path does not exist: {args.dataset_path}")
        return
    
    # Upload dataset
    upload_dataset(
        dataset_path=args.dataset_path,
        repo_name=args.repo_name,
        organization=args.organization,
        private=args.private,
        token=token
    )


if __name__ == "__main__":
    main()