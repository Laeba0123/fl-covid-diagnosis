import os
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse

def collect_images(root_dir):
    """Collect images from nested structure"""
    root_path = Path(root_dir)
    print(f"\nScanning: {root_path.absolute()}")
    
    # Expected class folders
    class_folders = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    found_classes = []
    items = []
    
    for cls in class_folders:
        cls_path = root_path / cls / "images"
        if not cls_path.exists():
            print(f"Warning: Missing folder {cls_path}")
            continue
            
        print(f"\nChecking {cls_path}:")
        image_count = 0
        for ext in ['.png', '.jpg', '.jpeg']:
            for img_path in cls_path.glob(f'*{ext}'):
                items.append((str(img_path.absolute()), cls))
                image_count += 1
                if image_count <= 3:  # Print first 3 files
                    print(f"  Found: {img_path.name}")
        
        print(f"Total {cls} images: {image_count}")
        if image_count > 0:
            found_classes.append(cls)
    
    print(f"\nTotal images collected: {len(items)}")
    if not items:
        raise ValueError("No images found in any class folder!")
    
    return items, found_classes

def simple_train_test_split(items, test_frac=0.2, seed=42):
    """Split data into train and test sets"""
    random.Random(seed).shuffle(items)
    n_test = int(len(items) * test_frac)
    return items[n_test:], items[:n_test]

def dirichlet_partition(train_items, classes, num_clients=5, alpha=0.5, seed=0):
    """Partition data using Dirichlet distribution"""
    idx_by_class = defaultdict(list)
    for idx, (_, cls) in enumerate(train_items):
        idx_by_class[cls].append(idx)
    
    client_idx = {i: [] for i in range(num_clients)}
    rng = np.random.default_rng(seed)
    
    for cls in classes:
        idxs = idx_by_class[cls]
        if not idxs:
            continue
            
        proportions = rng.dirichlet([alpha]*num_clients)
        portions = (proportions * len(idxs)).astype(int)
        while portions.sum() < len(idxs):
            portions[rng.integers(0, num_clients)] += 1
            
        start = 0
        for c in range(num_clients):
            client_idx[c].extend(idxs[start:start+portions[c]])
            start += portions[c]
    
    return {c: [train_items[i] for i in idxs] for c, idxs in client_idx.items()}

def save_partitions(client_partitions, test_items, out_dir):
    """Save partitioned data to files"""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(set([cls for _, cls in test_items])))}
    
    for c, items in client_partitions.items():
        with open(Path(out_dir, f"client_{c}_train.txt"), 'w') as f:
            for p, cls in items:
                f.write(f"{p}\t{class_to_idx[cls]}\n")
    
    with open(Path(out_dir, "test.txt"), 'w') as f:
        for p, cls in test_items:
            f.write(f"{p}\t{class_to_idx[cls]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--out_dir", default="data/partitions", help="Output directory")
    parser.add_argument("--num_clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet parameter")
    parser.add_argument("--test_frac", type=float, default=0.2, help="Test fraction")
    args = parser.parse_args()

    try:
        items, classes = collect_images(args.data_dir)
        train, test = simple_train_test_split(items, args.test_frac)
        parts = dirichlet_partition(train, classes, args.num_clients, args.alpha)
        save_partitions(parts, test, args.out_dir)
        print(f"Successfully created partitions in {args.out_dir}")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise