"""
sunspot_cnn.py — CNN classifier for sunspot magnetic configurations.

Classifies active region patches from SUIT UV images into four magnetic
complexity classes (McIntosh/Mount Wilson scheme simplified):

    0 = alpha   (α)   — single polarity, lowest flare risk
    1 = beta    (β)   — bipolar, moderate risk
    2 = beta-gamma (βγ)  — complex bipolar with mixed polarities
    3 = beta-gamma-delta (βγδ) — most complex, highest flare risk

Architecture
------------
- ResNet-18 backbone (ImageNet-pretrained for transfer learning)
- Custom dual-head:
    • magnetic_class: 4-way softmax
    • complexity_score: sigmoid scalar ∈ [0, 1]

Falls back to a rule-based heuristic (intensity variance + gradient
magnitude) when PyTorch/torchvision is not available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.vision.suit_reader import SuitImage


# ── Output dataclass ──────────────────────────────────────────────────────────

MAGNETIC_CLASSES = ("alpha", "beta", "beta-gamma", "beta-gamma-delta")

@dataclass
class SunspotClassification:
    """Result of sunspot magnetic classification."""

    magnetic_class: str       # One of MAGNETIC_CLASSES
    class_index: int          # 0–3
    class_probabilities: np.ndarray  # shape (4,)
    complexity_score: float   # 0.0 (quiet) → 1.0 (extremely complex)
    method: str               # "cnn" or "heuristic"


# ── CNN classifier ────────────────────────────────────────────────────────────

class SunspotClassifier:
    """
    Classify sunspot magnetic configuration from a SUIT UV image.

    Automatically selects CNN (if PyTorch available) or heuristic fallback.

    Parameters
    ----------
    model_path : str or None
        Path to a saved .pt state dict.  If None, uses untrained ResNet-18
        (useful for architecture validation / synthetic testing).
    device : str
        PyTorch device string ('cpu', 'cuda', etc.).
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._pytorch_available = False
        self._model = None

        try:
            import torch
            import torchvision
            self._pytorch_available = True
            self._build_model()
        except ImportError:
            pass

    def _build_model(self):
        """Construct the ResNet-18 based classifier."""
        import torch
        import torch.nn as nn
        from torchvision.models import resnet18, ResNet18_Weights

        class SunspotResNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Use pretrained backbone
                backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
                # Replace first conv to accept 1-channel grayscale
                backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                # Remove the original FC layer
                self.features = nn.Sequential(*list(backbone.children())[:-1])
                self.flatten = nn.Flatten()

                # Dual-head
                self.magnetic_head = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 4),
                )
                self.complexity_head = nn.Sequential(
                    nn.Linear(512, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                feat = self.flatten(self.features(x))
                mag_logits = self.magnetic_head(feat)
                complexity = self.complexity_head(feat).squeeze(-1)
                return mag_logits, complexity

        self._model = SunspotResNet().to(self.device)

        if self.model_path:
            import torch
            state = torch.load(self.model_path, map_location=self.device, weights_only=True)
            self._model.load_state_dict(state)

        self._model.eval()

    def classify(self, suit_image: SuitImage) -> SunspotClassification:
        """
        Classify a SUIT image into a magnetic complexity class.

        Parameters
        ----------
        suit_image : SuitImage
            Normalised SUIT observation.

        Returns
        -------
        SunspotClassification
        """
        if self._pytorch_available and self._model is not None:
            return self._classify_cnn(suit_image)
        return self._classify_heuristic(suit_image)

    def _classify_cnn(self, suit_image: SuitImage) -> SunspotClassification:
        """Run the CNN forward pass."""
        import torch
        import torch.nn.functional as F

        img = self._preprocess(suit_image.image)
        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 224, 224)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            mag_logits, complexity = self._model(tensor)
            probs = F.softmax(mag_logits, dim=-1).cpu().numpy()[0]
            complexity_val = float(complexity.cpu().numpy()[0])

        class_idx = int(np.argmax(probs))
        return SunspotClassification(
            magnetic_class=MAGNETIC_CLASSES[class_idx],
            class_index=class_idx,
            class_probabilities=probs,
            complexity_score=complexity_val,
            method="cnn",
        )

    def _classify_heuristic(self, suit_image: SuitImage) -> SunspotClassification:
        """
        Rule-based fallback using image statistics.

        Physics rationale:
        - Higher intensity variance → more active regions / mixed polarities
        - Higher gradient magnitude → sharper boundaries → more complex topology
        """
        img = self._preprocess(suit_image.image)

        # Intensity variance (normalised)
        intensity_var = float(np.var(img))
        # Gradient magnitude (Sobel-like)
        gy = np.diff(img, axis=0)
        gx = np.diff(img, axis=1)
        min_h = min(gy.shape[0], gx.shape[0])
        min_w = min(gy.shape[1], gx.shape[1])
        grad_mag = float(np.mean(np.sqrt(gy[:min_h, :min_w] ** 2 + gx[:min_h, :min_w] ** 2)))

        # Combined complexity score
        complexity = float(np.clip(intensity_var * 10 + grad_mag * 5, 0, 1))

        # Map complexity to class
        if complexity < 0.2:
            class_idx = 0  # alpha
        elif complexity < 0.45:
            class_idx = 1  # beta
        elif complexity < 0.7:
            class_idx = 2  # beta-gamma
        else:
            class_idx = 3  # beta-gamma-delta

        # Approximate probabilities
        probs = np.zeros(4, dtype=np.float32)
        probs[class_idx] = 0.6
        for i in range(4):
            if i != class_idx:
                probs[i] = 0.4 / 3
        probs /= probs.sum()

        return SunspotClassification(
            magnetic_class=MAGNETIC_CLASSES[class_idx],
            class_index=class_idx,
            class_probabilities=probs,
            complexity_score=complexity,
            method="heuristic",
        )

    @staticmethod
    def _preprocess(image: np.ndarray, target_size: int = 224) -> np.ndarray:
        """Resize and normalise image to (target_size, target_size)."""
        h, w = image.shape[:2]
        if h != target_size or w != target_size:
            # Simple bilinear resize without OpenCV dependency
            row_idx = np.linspace(0, h - 1, target_size).astype(int)
            col_idx = np.linspace(0, w - 1, target_size).astype(int)
            image = image[np.ix_(row_idx, col_idx)]
        return image.astype(np.float32)


# ── Convenience function ──────────────────────────────────────────────────────

_default_classifier: Optional[SunspotClassifier] = None


def classify_sunspots(suit_image: SuitImage, model_path: Optional[str] = None) -> SunspotClassification:
    """
    Convenience function: classify a SUIT image using a module-level classifier.

    Creates the classifier lazily on first call.
    """
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = SunspotClassifier(model_path=model_path)
    return _default_classifier.classify(suit_image)
