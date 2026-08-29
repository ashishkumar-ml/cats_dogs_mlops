"""
predictor.py — Model loading and single-image inference logic.
"""
import os
import io
import logging
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from src.model import SimpleCNN, load_model
from src.preprocess import preprocess_image

log = logging.getLogger(__name__)

CLASS_NAMES = ["Cat", "Dog"]
_MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pt")


class Predictor:
    """Singleton wrapper around SimpleCNN for thread-safe inference."""

    def __init__(self, model_path: str = _MODEL_PATH, num_classes: int = 2):
        self.model_path  = model_path
        self.num_classes = num_classes
        self._model: SimpleCNN | None = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        if not os.path.exists(self.model_path):
            log.warning("Model weights not found at %s — using random weights.", self.model_path)
            self._model = SimpleCNN(num_classes=self.num_classes)
        else:
            self._model = load_model(self.model_path, num_classes=self.num_classes)
            log.info("Model loaded from %s", self.model_path)
        self._model.eval()
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @torch.no_grad()
    def predict(self, image_bytes: bytes) -> Tuple[str, float, Dict[str, float]]:
        """
        Run inference on raw image bytes.
        Returns (predicted_class, confidence, {class_name: probability}).
        """
        if not self._loaded:
            self.load()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = preprocess_image(image)                  # (1, 3, 224, 224)

        logits = self._model(tensor)                      # (1, num_classes)
        probs  = F.softmax(logits, dim=1).squeeze(0)      # (num_classes,)

        class_idx  = probs.argmax().item()
        confidence = probs[class_idx].item()
        probabilities = {CLASS_NAMES[i]: round(probs[i].item(), 4)
                         for i in range(len(CLASS_NAMES))}

        return CLASS_NAMES[class_idx], confidence, probabilities


# Module-level singleton — shared across all FastAPI workers in the same process
predictor = Predictor()
