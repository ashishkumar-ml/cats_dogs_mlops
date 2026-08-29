"""
Unit tests for src/preprocess.py
"""
import io
import os
import shutil
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from src.preprocess import (
    get_transforms,
    preprocess_image,
    split_dataset,
    validate_image,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rgb_image(w=64, h=64) -> Image.Image:
    """Create a small random RGB PIL image."""
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture()
def sample_pil_image():
    return _make_rgb_image()


@pytest.fixture()
def tmp_class_dir():
    """Creates a temporary dataset with two classes, 10 images each."""
    root = tempfile.mkdtemp()
    for cls in ("Cat", "Dog"):
        cls_dir = os.path.join(root, cls)
        os.makedirs(cls_dir)
        for i in range(10):
            img = _make_rgb_image()
            img.save(os.path.join(cls_dir, f"{i:04d}.jpg"))
    yield root
    shutil.rmtree(root)


# ---------------------------------------------------------------------------
# get_transforms
# ---------------------------------------------------------------------------

class TestGetTransforms:
    def test_returns_two_transforms(self):
        train_tf, val_tf = get_transforms(224)
        assert train_tf is not None
        assert val_tf is not None

    def test_train_transform_output_shape(self, sample_pil_image):
        train_tf, _ = get_transforms(224)
        tensor = train_tf(sample_pil_image)
        assert tensor.shape == (3, 224, 224), f"Unexpected shape: {tensor.shape}"

    def test_val_transform_output_shape(self, sample_pil_image):
        _, val_tf = get_transforms(224)
        tensor = val_tf(sample_pil_image)
        assert tensor.shape == (3, 224, 224)

    def test_custom_img_size(self, sample_pil_image):
        _, val_tf = get_transforms(128)
        tensor = val_tf(sample_pil_image)
        assert tensor.shape == (3, 128, 128)

    def test_output_is_float_tensor(self, sample_pil_image):
        _, val_tf = get_transforms(224)
        tensor = val_tf(sample_pil_image)
        assert tensor.dtype == torch.float32

    def test_normalised_range(self, sample_pil_image):
        """After ImageNet normalisation most pixels lie outside [0, 1]."""
        _, val_tf = get_transforms(224)
        tensor = val_tf(sample_pil_image)
        # Not all values should be in [0,1] once normalised
        assert not ((tensor >= 0) & (tensor <= 1)).all()


# ---------------------------------------------------------------------------
# preprocess_image
# ---------------------------------------------------------------------------

class TestPreprocessImage:
    def test_output_shape(self, sample_pil_image):
        tensor = preprocess_image(sample_pil_image, img_size=224)
        assert tensor.shape == (1, 3, 224, 224)

    def test_converts_grayscale_to_rgb(self):
        gray = Image.fromarray(
            np.random.randint(0, 255, (64, 64), dtype=np.uint8), mode="L"
        )
        tensor = preprocess_image(gray, img_size=224)
        assert tensor.shape == (1, 3, 224, 224)

    def test_batch_dim_is_one(self, sample_pil_image):
        tensor = preprocess_image(sample_pil_image)
        assert tensor.shape[0] == 1


# ---------------------------------------------------------------------------
# validate_image
# ---------------------------------------------------------------------------

class TestValidateImage:
    def test_valid_rgb(self, sample_pil_image):
        assert validate_image(sample_pil_image) is True

    def test_valid_grayscale_converted(self):
        gray = Image.fromarray(
            np.random.randint(0, 255, (32, 32), dtype=np.uint8), mode="L"
        )
        assert validate_image(gray) is True


# ---------------------------------------------------------------------------
# split_dataset
# ---------------------------------------------------------------------------

class TestSplitDataset:
    def test_split_creates_directories(self, tmp_class_dir):
        dst = tempfile.mkdtemp()
        try:
            split_dataset(tmp_class_dir, dst, train_ratio=0.8, val_ratio=0.1)
            for split in ("train", "val", "test"):
                for cls in ("Cat", "Dog"):
                    assert os.path.isdir(os.path.join(dst, split, cls))
        finally:
            shutil.rmtree(dst)

    def test_split_counts_sum_to_total(self, tmp_class_dir):
        dst = tempfile.mkdtemp()
        try:
            split_dataset(tmp_class_dir, dst, train_ratio=0.8, val_ratio=0.1)
            for cls in ("Cat", "Dog"):
                total = 0
                for split in ("train", "val", "test"):
                    total += len(os.listdir(os.path.join(dst, split, cls)))
                assert total == 10
        finally:
            shutil.rmtree(dst)

    def test_train_split_is_largest(self, tmp_class_dir):
        dst = tempfile.mkdtemp()
        try:
            split_dataset(tmp_class_dir, dst, train_ratio=0.8, val_ratio=0.1)
            train_n = len(os.listdir(os.path.join(dst, "train", "Cat")))
            val_n   = len(os.listdir(os.path.join(dst, "val",   "Cat")))
            test_n  = len(os.listdir(os.path.join(dst, "test",  "Cat")))
            assert train_n >= val_n and train_n >= test_n
        finally:
            shutil.rmtree(dst)
