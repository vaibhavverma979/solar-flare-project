"""
train_patchtst_nasa.py — PatchTST training on NOAA GOES 1-minute X-ray data.

Training data source: NOAA GOES satellite XRSB (1–8 Å) flux time-series
fetched via ``src.data.nasa_pipeline.build_goes_training_dataset()``.

Key differences from the legacy ``train_fits_model.py``:
- Single channel input (c_in=1): GOES log-scaled X-ray flux only
- Shorter windows (seq_len=60 steps = 60 minutes at 1-min cadence)
- FocalLoss replaces BCEWithLogitsLoss to handle severe quiet-sun class imbalance
- Binary classification only (flare / no-flare within 30-min horizon)
- Saves checkpoint → src/forecast/models/patchtst_nasa_goes.pt
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


# ===========================================================================
# FocalLoss — handles severe quiet-sun / flare class imbalance
# ===========================================================================

class FocalLoss(nn.Module):
    """Binary Focal Loss for class-imbalanced binary classification.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", 2017.
    https://arxiv.org/abs/1708.02002

    Parameters
    ----------
    gamma : float
        Focusing parameter.  Larger values down-weight easy negatives more.
        Recommended range: [0.5, 5].  Default 2 (from paper).
    alpha : float
        Weighting factor for the positive class in [0, 1].
        Set < 0.5 to penalise false negatives more.  Default 0.25.
    reduction : str
        ``'mean'`` or ``'sum'``.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : [B, 1] raw un-sigmoid-ed logits
        targets : [B, 1] binary float labels (0.0 or 1.0)
        """
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = torch.exp(-bce)  # predicted probability of the true class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ===========================================================================
# PatchTST model (single-channel, binary-classification variant)
# ===========================================================================

class PatchTSTSingleChannel(nn.Module):
    """Channel-independent PatchTST for binary flare classification.

    Adapted from ``src.forecast.train_fits_model.PatchTST`` but with:
    - c_in = 1 (GOES log-flux only; no HXR/SXR multi-channel)
    - Single binary classification head only
    - Configurable patch_size / stride optimised for 60-step windows

    Parameters
    ----------
    seq_len     : Input context length in steps (default 60).
    patch_size  : Patch length in time steps (default 10).
    stride      : Patch stride (default 5, giving 11 patches for seq_len=60).
    d_model     : Transformer hidden dimension.
    n_heads     : Number of attention heads.
    e_layers    : Number of Transformer encoder layers.
    dropout     : Dropout inside Transformer layers.
    head_dropout: Dropout in the classification head.
    """

    def __init__(
        self,
        seq_len: int = 60,
        patch_size: int = 10,
        stride: int = 5,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 3,
        dropout: float = 0.1,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.patch_num = int((seq_len - patch_size) / stride + 1)

        # Input embedding
        self.value_embedding = nn.Linear(patch_size, d_model)
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.patch_num, d_model)
        )
        nn.init.uniform_(self.position_embedding, -0.02, 0.02)

        # Shared Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.flatten = nn.Flatten(start_dim=1)

        # Binary classification head
        head_dim = self.patch_num * d_model
        self.flare_head = nn.Sequential(
            nn.Dropout(p=head_dropout),
            nn.Linear(head_dim, 1),  # raw logit — no sigmoid here
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, seq_len, 1] — single-channel input (log-scaled GOES flux)

        Returns
        -------
        [B, 1] — raw logit for binary flare classification
        """
        B, T, C = x.shape  # C == 1

        # Extract overlapping patches: [B, patch_num, patch_size]
        x_squeezed = x.squeeze(-1)  # [B, T]
        patches = x_squeezed.unfold(dimension=1, size=self.patch_size, step=self.stride)

        # Embed + positional encoding: [B, patch_num, d_model]
        x_emb = self.value_embedding(patches) + self.position_embedding

        # Transformer encoder: [B, patch_num, d_model]
        out = self.transformer_encoder(x_emb)

        # Flatten → classify
        out = self.flatten(out)  # [B, patch_num * d_model]
        return self.flare_head(out)  # [B, 1]


# ===========================================================================
# PyTorch Dataset for pre-built numpy arrays
# ===========================================================================

class GOESWindowDataset(Dataset):
    """Wraps pre-built GOES window arrays (X, y) as a PyTorch Dataset.

    Parameters
    ----------
    X : np.ndarray, shape (N, seq_len, 1)
        Log-scaled GOES X-ray windows.
    y : np.ndarray, shape (N,)
        Binary flare labels.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert len(X) == len(y), f"X ({len(X)}) and y ({len(y)}) must have same length."
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # [N, 1]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ===========================================================================
# Metric helpers
# ===========================================================================

def _binary_classification_metrics(
    logits: list[float],
    labels: list[float],
    threshold: float = 0.5,
) -> dict:
    """Compute TP/FP/FN/TN/Precision/Recall/F1 and AUC-PR from raw logits."""
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    y = np.array(labels)
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    # AUC-PR via sklearn if available, else approximate trapezoidal
    try:
        from sklearn.metrics import average_precision_score
        auc_pr = float(average_precision_score(y, probs)) if y.sum() > 0 else 0.0
    except ImportError:
        auc_pr = float(np.trapz(
            sorted(probs[y == 1]), dx=1.0 / max(y.sum(), 1)
        )) if y.sum() > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_pr": auc_pr,
    }


def _count_class_balance(dataset: Subset) -> tuple[int, int]:
    """Return (n_pos, n_neg) counts in a Subset without moving to GPU."""
    n_pos = n_neg = 0
    for _, label in dataset:
        if label.item() >= 0.5:
            n_pos += 1
        else:
            n_neg += 1
    return n_pos, n_neg


# ===========================================================================
# Training entry-point
# ===========================================================================

def train_patchtst_nasa(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    val_split: float = 0.2,
    device: Optional[str] = None,
    model_dir: str = "src/forecast/models",
    model_name: str = "patchtst_nasa_goes.pt",
) -> str:
    """Train a single-channel PatchTST on NOAA GOES sliding windows.

    Parameters
    ----------
    X : np.ndarray, shape (N, seq_len, 1)
        Log-scaled GOES X-ray windows from ``build_goes_training_dataset()``.
    y : np.ndarray, shape (N,)
        Binary flare labels (1 = M/X within 30-min forecast horizon).
    epochs : int
        Number of full training passes.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Initial AdamW learning rate.
    weight_decay : float
        AdamW L2 regularisation weight.
    focal_gamma : float
        FocalLoss focusing parameter.
    focal_alpha : float
        FocalLoss positive-class weighting factor.
    val_split : float
        Fraction of data reserved for validation (chronological split).
    device : str, optional
        ``'cuda'``, ``'mps'``, or ``'cpu'``.  Auto-detected if None.
    model_dir : str
        Directory where the best-checkpoint .pt file is saved.
    model_name : str
        Filename of the saved model checkpoint.

    Returns
    -------
    str
        Absolute (or relative) path to the saved best checkpoint.
    """
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    logger.info("=" * 70)
    logger.info("[train_patchtst_nasa] START  — NOAA GOES PatchTST Training")
    logger.info("  device     : %s", device)
    logger.info("  X shape    : %s  (N, seq_len, channels)", X.shape)
    logger.info("  y shape    : %s  (N,)  pos=%d  neg=%d",
                y.shape, int(y.sum()), int((y == 0).sum()))
    logger.info("  epochs     : %d", epochs)
    logger.info("  batch_size : %d", batch_size)
    logger.info("  FocalLoss  : gamma=%.1f  alpha=%.2f", focal_gamma, focal_alpha)
    logger.info("=" * 70)

    seq_len = X.shape[1]

    # ── Chronological train/val split ────────────────────────────────────────
    full_dataset = GOESWindowDataset(X, y)
    n_total  = len(full_dataset)
    n_train  = int(n_total * (1 - val_split))
    n_val    = n_total - n_train

    train_set = Subset(full_dataset, list(range(n_train)))
    val_set   = Subset(full_dataset, list(range(n_train, n_total)))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info("  train windows : %d", n_train)
    logger.info("  val windows   : %d", n_val)

    # Class balance in training split
    n_pos_train, n_neg_train = _count_class_balance(train_set)
    logger.info(
        "  train balance : pos=%d (%.1f%%)  neg=%d (%.1f%%)",
        n_pos_train, 100 * n_pos_train / max(n_train, 1),
        n_neg_train, 100 * n_neg_train / max(n_train, 1),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = PatchTSTSingleChannel(
        seq_len=seq_len,
        patch_size=10,
        stride=5,
        d_model=128,
        n_heads=4,
        e_layers=3,
        dropout=0.1,
        head_dropout=0.2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("  model params  : %s", f"{n_params:,}")

    # ── Optimizer + Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Loss ─────────────────────────────────────────────────────────────────
    criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)

    # ── Training loop ────────────────────────────────────────────────────────
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    best_val_loss = float("inf")

    logger.info("-" * 70)
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)
        scheduler.step()

        # Validate
        model.eval()
        val_loss   = 0.0
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                loss   = criterion(logits, batch_y)
                val_loss += loss.item()
                all_logits.extend(logits.squeeze(-1).cpu().tolist())
                all_labels.extend(batch_y.squeeze(-1).cpu().tolist())
        val_loss /= max(len(val_loader), 1)

        metrics = _binary_classification_metrics(all_logits, all_labels)

        logger.info(
            "Epoch [%3d/%d] | TrainLoss: %.4f | ValLoss: %.4f | "
            "Prec: %.3f | Rec: %.3f | F1: %.3f | AUC-PR: %.3f",
            epoch + 1, epochs,
            train_loss, val_loss,
            metrics["precision"], metrics["recall"],
            metrics["f1"], metrics["auc_pr"],
        )
        logger.info(
            "             | TP=%d FP=%d FN=%d TN=%d",
            metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            logger.info("             | ✓ Best model saved (ValLoss=%.4f) → %s", best_val_loss, model_path)

    logger.info("-" * 70)
    logger.info("[train_patchtst_nasa] Training complete.")
    logger.info("  Best validation loss : %.4f", best_val_loss)
    logger.info("  Model checkpoint     : %s", model_path)
    logger.info("=" * 70)

    return model_path
