import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import json
from utils.augmentations import get_transform

class ImageNetC(Dataset):
    def __init__(self, root, corruption_type, severity, transform_anchor=None, transform_aux=None, single_model=False, class_index_path: str | None = None, n_examples=None):
        """
        Robust ImageNet-C loader supporting two common layouts:
        1) Folder-per-class: <root>/<corruption>/<severity>/<synset>/*.JPEG
        2) Flat files:       <root>/<corruption>/<severity>/*.png and labels(.txt|.npy)
           or filenames that start with synset IDs (e.g., n01440764_615.JPEG)
        """
        self.root = os.path.join(root, corruption_type, str(severity))
        self.transform_anchor = transform_anchor
        self.transform_aux = transform_aux
        self.single_model = single_model
        self.image_paths = []
        self.labels = []

        # Select appropriate class index mapping
        if class_index_path is None:
            # Heuristic: Tiny-ImageNet-C has 200 classes and often present as Tiny-ImageNet-C in path
            tiny_idx = 'data/tiny-imagenet-c_class_index.json'
            imagenet_idx = 'data/imagenet_class_index.json'
            if 'Tiny-ImageNet-C' in root or os.path.exists(tiny_idx):
                class_index_path = tiny_idx
            else:
                class_index_path = imagenet_idx

        with open(class_index_path, 'r') as f:
            dictionary = json.load(f)
        self.synset_to_class = {v[0]: int(k) for k, v in dictionary.items()}

        # Detect layout
        entries = sorted(os.listdir(self.root)) if os.path.isdir(self.root) else []
        has_subdirs = any(os.path.isdir(os.path.join(self.root, e)) for e in entries)

        if has_subdirs:
            # Folder-per-class layout
            for class_folder in entries:
                class_path = os.path.join(self.root, class_folder)
                if not os.path.isdir(class_path):
                    continue
                class_label = self.synset_to_class.get(class_folder, -1)
                for img_name in sorted(os.listdir(class_path)):
                    img_path = os.path.join(class_path, img_name)
                    if not os.path.isfile(img_path):
                        continue
                    self.image_paths.append(img_path)
                    self.labels.append(class_label)
        else:
            # Flat layout: try filenames with synset prefix; otherwise use labels(.txt|.npy)
            all_files = [f for f in entries if os.path.isfile(os.path.join(self.root, f))]
            # Prefer images only
            img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.JPEG', '.PNG'}
            images = [f for f in all_files if os.path.splitext(f)[1] in img_exts]

            def parse_synset_from_name(name: str):
                base = os.path.basename(name)
                # common pattern: n01440764_615.JPEG or n01440764-615.png
                token = base.split('_')[0].split('-')[0]
                return token if token in self.synset_to_class else None

            # Try to extract labels from filenames
            fn_labels = []
            ok = True
            for fname in images:
                syn = parse_synset_from_name(fname)
                if syn is None:
                    ok = False
                    break
                fn_labels.append(self.synset_to_class[syn])

            if ok and images:
                for fname, lbl in zip(sorted(images), fn_labels):
                    self.image_paths.append(os.path.join(self.root, fname))
                    self.labels.append(lbl)
            else:
                # Fallback to labels file (txt or npy) following original ImageNet-C ordering
                labels_path_txt = os.path.join(os.path.dirname(self.root), 'labels.txt')
                labels_path_npy = os.path.join(os.path.dirname(self.root), 'labels.npy')
                labels = None
                if os.path.isfile(labels_path_txt):
                    with open(labels_path_txt, 'r') as f:
                        labels = [int(x.strip()) for x in f if x.strip()]
                elif os.path.isfile(labels_path_npy):
                    labels = list(np.load(labels_path_npy))

                if labels is None:
                    # As a last resort, assume alphabetical order corresponds to labels sequence across severities
                    # This may be inaccurate but avoids -1 labels leading to near-zero accuracy.
                    raise RuntimeError(f"Cannot determine labels for ImageNet-C at {self.root}. Ensure labels.txt or labels.npy exists, or filenames start with synset IDs.")

                # Sort images deterministically; ImageNet-C specifies order 1..50000.png
                images_sorted = sorted(images)
                if len(labels) < len(images_sorted):
                    raise RuntimeError(f"Labels length {len(labels)} < number of images {len(images_sorted)} for {self.root}")
                for idx, fname in enumerate(images_sorted):
                    self.image_paths.append(os.path.join(self.root, fname))
                    self.labels.append(int(labels[idx]))

        if n_examples:
            self.image_paths = self.image_paths[:n_examples]
            self.labels = self.labels[:n_examples]
                    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        img_anchor = self.transform_anchor(image) if self.transform_anchor else image
        
        if self.single_model:
            return img_anchor, label

        img_aux = self.transform_aux(image) if self.transform_aux else image
            
        return img_anchor, img_aux, label

def get_imagenet_c_loader(data_dir, corruption_type, severity, batch_size, num_workers=8, anchor_model_name='vit_base_patch16_224', aux_model_name=None, shuffle=False, n_examples=None):
    transform_anchor = get_transform(anchor_model_name)
    
    single_model = aux_model_name is None
    transform_aux = None
    if not single_model:
        transform_aux = get_transform(aux_model_name)

    dataset = ImageNetC(data_dir, corruption_type, severity, transform_anchor=transform_anchor, transform_aux=transform_aux, single_model=single_model, n_examples=n_examples)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader