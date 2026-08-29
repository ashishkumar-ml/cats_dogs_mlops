import os
import shutil
import random
from pathlib import Path
from torchvision import transforms
from PIL import Image


IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_transforms(img_size: int = IMG_SIZE):
    """Return (train_transform, val_transform) for ImageFolder datasets."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return train_transform, val_transform


def preprocess_image(image: Image.Image, img_size: int = IMG_SIZE):
    """Preprocess a single PIL image for inference. Returns a (1, 3, H, W) tensor."""
    import torch
    _, val_tf = get_transforms(img_size)
    tensor = val_tf(image.convert("RGB"))
    return tensor.unsqueeze(0)


def validate_image(image: Image.Image) -> bool:
    """Return True if the image can be opened and has 3 colour channels."""
    try:
        img = image.convert("RGB")
        return img.size[0] > 0 and img.size[1] > 0
    except Exception:
        return False


def split_dataset(
    src_dir: str,
    dst_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Split a flat class-folder dataset into train/val/test sub-folders.

    src_dir layout:   src_dir/Cat/*.jpg   src_dir/Dog/*.jpg
    dst_dir layout:   dst_dir/{train,val,test}/{Cat,Dog}/...
    """
    random.seed(seed)
    classes = [d for d in os.listdir(src_dir)
               if os.path.isdir(os.path.join(src_dir, d))]

    for split in ("train", "val", "test"):
        for cls in classes:
            os.makedirs(os.path.join(dst_dir, split, cls), exist_ok=True)

    for cls in classes:
        files = sorted([
            f for f in os.listdir(os.path.join(src_dir, cls))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        splits = {
            "train": files[:n_train],
            "val":   files[n_train: n_train + n_val],
            "test":  files[n_train + n_val:],
        }
        for split, names in splits.items():
            for fname in names:
                src = os.path.join(src_dir, cls, fname)
                dst = os.path.join(dst_dir, split, cls, fname)
                shutil.copy2(src, dst)

    print(f"Dataset split complete -> {dst_dir}")
