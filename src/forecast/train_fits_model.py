"""
train_fits_model.py — PatchTST multi-task training loop for the Aditya-L1 pipeline.

.. deprecated::
    **TRAINING DEPENDENCY ON ISRO FITS FILES HAS BEEN REMOVED.**

    This module no longer calls ``load_fits_dataset`` or reads ISRO Aditya-L1
    FITS files for model training.  The ``train_patchtst()`` function below
    retains its DataFrame interface for backward compatibility with any callers
    that supply a pre-built feature DataFrame, but new training workflows should
    use the NASA/NOAA pipeline instead::

        # New training path (recommended)
        python nasa_train.py --start 2017-01-01 --end 2023-12-31

        # Or import directly:
        from src.forecast.train_patchtst_nasa import train_patchtst_nasa

    The ``PatchTST`` model class and metric helpers defined in this file are
    still active and are imported by ``train_patchtst_nasa.py``.

    ISRO SoLEXS/HEL1OS telemetry is now handled exclusively by::
        src/inference/aditya_live_predictor.py

Changelog (overfitting remediation):
  FIX-3  Class imbalance — dynamic pos_weight
         Counts positive (M/X-class) and negative (quiet/B-class) windows in
         the TRAINING split and passes pos_weight = n_neg/n_pos to
         BCEWithLogitsLoss.  Prevents the model from achieving high accuracy
         by predicting all-negative (which produces F1 = 0.0).

  FIX-5a Dropout on prediction heads (p=0.2)
         All three task heads (flare, SXR, HXR) now pass through a Dropout
         layer before the final Linear projection.  This forces the shared
         transformer representation to be more general and reduces co-adaptation
         between head units.

  FIX-5b Weight decay increased to 1e-3
         AdamW weight_decay raised from 1e-4 → 1e-3.  This is the most
         computationally free regularisation knob available and significantly
         penalises the large norm solutions that the transformer tends to find
         when overfitting on small astrophysical datasets.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from src.forecast.dataset import FlareSlidingWindowDataset
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Model definition
# ──────────────────────────────────────────────────────────────────────────────

class PatchTST(nn.Module):
    """Channel-independent Patch Time-Series Transformer with multi-task heads.

    Architecture overview:
      - Per-channel patching: each of the C input channels is independently
        divided into overlapping patches of length `patch_size` with step `stride`.
      - Shared Transformer encoder (weight-sharing across channels).
      - Three separate prediction heads:
          * flare_head  → logit for binary flare classification (BCEWithLogitsLoss)
          * sxr_head    → scalar peak SXR flux (MSELoss, log10 scale)
          * hxr_head    → scalar peak HXR counts (MSELoss)

    FIX-5a: Each head now includes Dropout(p=0.2) before the final Linear layer.

    Parameters
    ----------
    c_in        : Number of input feature channels.
    seq_len     : Input context length (number of time steps).
    patch_size  : Length of each patch in time steps.
    stride      : Step size between consecutive patches.
    d_model     : Transformer hidden dimension.
    n_heads     : Number of attention heads.
    e_layers    : Number of Transformer encoder layers.
    dropout     : Dropout probability used inside Transformer layers.
    head_dropout: Dropout probability applied in the task-specific heads (FIX-5a).
    """

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        patch_size: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 3,
        dropout: float = 0.1,
        head_dropout: float = 0.2,  # FIX-5a
    ):
        super().__init__()
        self.c_in = c_in
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model

        self.patch_num = int((seq_len - patch_size) / stride + 1)

        # ── Input embedding ────────────────────────────────────────────────
        self.value_embedding = nn.Linear(patch_size, d_model)
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.patch_num, d_model)
        )
        nn.init.uniform_(self.position_embedding, -0.02, 0.02)

        # ── Shared Transformer encoder ─────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=e_layers
        )

        self.flatten = nn.Flatten(start_dim=1)

        # ── Multi-task heads (FIX-5a: Dropout before Linear) ──────────────
        head_dim = self.patch_num * d_model * c_in

        self.flare_head = nn.Sequential(
            nn.Dropout(p=head_dropout),   # FIX-5a
            nn.Linear(head_dim, 1),       # outputs a raw logit (no sigmoid here)
        )
        self.sxr_head = nn.Sequential(
            nn.Dropout(p=head_dropout),   # FIX-5a
            nn.Linear(head_dim, 1),
        )
        self.hxr_head = nn.Sequential(
            nn.Dropout(p=head_dropout),   # FIX-5a
            nn.Linear(head_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [Batch, Seq, Channels]

        Returns:
            dict with keys 'flare_logits', 'sxr_pred', 'hxr_pred'
        """
        B, T, C = x.shape

        # Channel-independent patching: [B, C, T] → [B*C, T, 1]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B * C, T, 1)

        # Extract overlapping patches: [B*C, patch_num, patch_size]
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.stride).squeeze(2)

        # Embedding + positional encoding: [B*C, patch_num, d_model]
        x_emb = self.value_embedding(patches) + self.position_embedding

        # Transformer encoder: [B*C, patch_num, d_model]
        out = self.transformer_encoder(x_emb)

        # Reshape back and flatten: [B, C * patch_num * d_model]
        out = out.view(B, C, self.patch_num, self.d_model)
        out = self.flatten(out)

        return {
            'flare_logits': self.flare_head(out),   # [B, 1] — raw logit
            'sxr_pred':     self.sxr_head(out),     # [B, 1]
            'hxr_pred':     self.hxr_head(out),     # [B, 1]
        }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _count_pos_neg(dataset: Subset) -> tuple[int, int]:
    """Count positive (M/X-class) and negative (quiet/B-class) windows.

    Iterates the training subset without loading GPU tensors.  Used to
    compute pos_weight for BCEWithLogitsLoss (FIX-3).

    Returns
    -------
    (n_pos, n_neg) : counts of positive and negative samples
    """
    n_pos = 0
    n_neg = 0
    for _, targets in dataset:
        if targets['flare_flag'].item() >= 0.5:
            n_pos += 1
        else:
            n_neg += 1
    return n_pos, n_neg


def _compute_val_classification_metrics(
    all_logits: list[float],
    all_labels: list[float],
    threshold: float = 0.5,
) -> dict:
    """Compute TP/FP/FN/TN/Precision/Recall/F1 from raw logits."""
    probs  = torch.sigmoid(torch.tensor(all_logits)).numpy()
    labels = np.array(all_labels)
    preds  = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training entry-point
# ──────────────────────────────────────────────────────────────────────────────

def train_patchtst(
    df: pd.DataFrame,
    epochs: int = 35,
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> str:
    """Train a PatchTST model on the provided feature dataframe.

    .. deprecated::
        This function accepts a pre-built feature DataFrame for backward
        compatibility. For training on multi-year NASA/NOAA data, use::

            from src.forecast.train_patchtst_nasa import train_patchtst_nasa

        This function does **not** call ``load_fits_dataset`` — the caller is
        responsible for supplying the DataFrame (e.g. from feature engineering
        on GOES data). ISRO FITS files are no longer a valid training source.

    Parameters
    ----------
    df         : Feature-engineered dataframe (must contain 'timestamp' column
                 and numeric feature columns). Previously sourced from
                 AstrophysicalFeatureEngineer.process(); now expected to be
                 built from NASA data pipelines.
    epochs     : Number of training epochs.
    batch_size : Mini-batch size.
    device     : 'cuda' or 'cpu'.

    Returns
    -------
    best_model_path : Absolute path to the saved best-validation-loss checkpoint.
    """
    logger.info("=" * 60)
    logger.info("train_patchtst() START")
    logger.info(f"  device     : {device}")
    logger.info(f"  epochs     : {epochs}")
    logger.info(f"  batch_size : {batch_size}")

    # ── Dataset & chronological split ─────────────────────────────────────
    logger.info("Initialising FlareSlidingWindowDataset...")
    dataset = FlareSlidingWindowDataset(df, context_length=360, forecast_horizon=30)

    dataset_size = len(dataset)
    train_size   = int(0.8 * dataset_size)

    train_indices = list(range(0, train_size))
    val_indices   = list(range(train_size, dataset_size))

    train_dataset = Subset(dataset, train_indices)
    val_dataset   = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"  train windows : {len(train_dataset)}")
    logger.info(f"  val windows   : {len(val_dataset)}")

    # ── FIX-3: Dynamic pos_weight for class-imbalanced BCE ────────────────
    logger.info("FIX-3 | Counting class distribution in training set...")
    n_pos, n_neg = _count_pos_neg(train_dataset)
    pos_weight_value = n_neg / max(n_pos, 1)

    logger.info(
        f"FIX-3 | Training class distribution: "
        f"n_pos={n_pos} (M/X-class), n_neg={n_neg} (quiet/B-class), "
        f"pos_weight={pos_weight_value:.2f}"
    )

    if n_pos == 0:
        logger.warning(
            "FIX-3 | WARNING: Zero positive (M/X-class) samples in training split. "
            "pos_weight will be set to 1.0 — flare classification will not improve. "
            "Ensure the dataset covers an active solar period."
        )
        pos_weight_value = 1.0

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    # ── Model ─────────────────────────────────────────────────────────────
    c_in = len(dataset.feature_cols)
    model = PatchTST(
        c_in=c_in,
        seq_len=360,
        patch_size=16,
        stride=8,
        d_model=128,
        n_heads=4,
        e_layers=3,
        dropout=0.1,
        head_dropout=0.2,   # FIX-5a
    ).to(device)

    logger.info(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    # FIX-5b: weight_decay raised from 1e-4 → 1e-3
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-3,   # FIX-5b
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Loss functions ────────────────────────────────────────────────────
    # FIX-3: BCEWithLogitsLoss with dynamic pos_weight
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # FIX-3
    mse_loss = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss  = float('inf')
    os.makedirs('src/forecast/models', exist_ok=True)
    # NOTE: Checkpoint filename retained for backward compat.
    # New NASA-trained model saves to: patchtst_nasa_goes.pt
    best_model_path = 'src/forecast/models/patchtst_fits_trained.pt'

    logger.info(f"Starting training on {device}...")
    logger.info("-" * 60)

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x   = batch_x.to(device)
            flare_flag = batch_y['flare_flag'].to(device)
            peak_sxr   = batch_y['peak_sxr'].to(device)
            peak_hxr   = batch_y['peak_hxr'].to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)

            loss_flare = bce_loss(outputs['flare_logits'], flare_flag)
            loss_sxr   = mse_loss(outputs['sxr_pred'],     peak_sxr)
            loss_hxr   = mse_loss(outputs['hxr_pred'],     peak_hxr)

            # Weighted multi-task loss: flare classification is primary objective
            loss = loss_flare + 0.5 * loss_sxr + 0.5 * loss_hxr
            loss.backward()

            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        val_sxr_mse = 0.0
        val_hxr_mse = 0.0
        all_logits  = []
        all_labels  = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x    = batch_x.to(device)
                flare_flag = batch_y['flare_flag'].to(device)
                peak_sxr   = batch_y['peak_sxr'].to(device)
                peak_hxr   = batch_y['peak_hxr'].to(device)

                outputs = model(batch_x)

                loss_flare = bce_loss(outputs['flare_logits'], flare_flag)
                loss_sxr   = mse_loss(outputs['sxr_pred'],     peak_sxr)
                loss_hxr   = mse_loss(outputs['hxr_pred'],     peak_hxr)

                loss = loss_flare + 0.5 * loss_sxr + 0.5 * loss_hxr
                val_loss    += loss.item()
                val_sxr_mse += loss_sxr.item()
                val_hxr_mse += loss_hxr.item()

                # Accumulate for classification metrics
                all_logits.extend(
                    outputs['flare_logits'].squeeze(-1).cpu().tolist()
                )
                all_labels.extend(
                    flare_flag.squeeze(-1).cpu().tolist()
                )

        val_loss    /= len(val_loader)
        val_sxr_mse /= len(val_loader)
        val_hxr_mse /= len(val_loader)

        # Classification metrics at threshold=0.5
        clf_metrics = _compute_val_classification_metrics(all_logits, all_labels)

        # ── Epoch summary log ──────────────────────────────────────────────
        logger.info(
            f"Epoch [{epoch+1:>3}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"SXR-MSE: {val_sxr_mse:.4f} | "
            f"HXR-MSE: {val_hxr_mse:.4f}"
        )
        logger.info(
            f"             | "
            f"TP={clf_metrics['tp']} FP={clf_metrics['fp']} "
            f"FN={clf_metrics['fn']} TN={clf_metrics['tn']} | "
            f"Prec={clf_metrics['precision']:.3f} "
            f"Rec={clf_metrics['recall']:.3f} "
            f"F1={clf_metrics['f1']:.3f}"
        )

        # Checkpoint best model by total validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"             | ✓ New best model saved "
                f"(Val Loss: {best_val_loss:.4f}) → {best_model_path}"
            )

    logger.info("-" * 60)
    logger.info("Training complete.")
    logger.info(f"Best validation loss : {best_val_loss:.4f}")
    logger.info(f"Best model saved at  : {best_model_path}")
    logger.info("=" * 60)

    return best_model_path
