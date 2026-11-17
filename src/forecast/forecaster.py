import math
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    pass

class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, d_model, n_patches):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        T_eff = (T // self.patch_len) * self.patch_len
        x = x[:, -T_eff:, :]

        # (B, n_patches, patch_len, C)
        x = x.reshape(B, T_eff // self.patch_len, self.patch_len, C)
        
        # Channel independence: (B*C, n_patches, patch_len)
        x = x.permute(0, 3, 1, 2).reshape(B * C, T_eff // self.patch_len, self.patch_len)
        
        x = self.proj(x)
        actual_patches = x.shape[1]
        x = x + self.pos_enc[:, :actual_patches, :]
        return x, B, C

class MultiTaskPatchTSTNet(nn.Module):
    """
    PatchTST Transformer with a Multi-Task Output Head for 3 targets:
    1. Flare Probability (0-1)
    2. Dst Index (Linear)
    3. Kp Index (0-9)
    """
    def __init__(self, context_len=360, patch_len=16, d_model=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.n_patches = context_len // patch_len
        self.patch_embed = PatchEmbedding(patch_len, d_model, self.n_patches)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        
        # Multi-task heads
        self.flare_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1), nn.Sigmoid()
        )
        self.dst_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1) # Linear mapping for nT
        )
        self.kp_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1), nn.Sigmoid() # Will scale by 9.0 in forward pass
        )

    def forward(self, x):
        z, B, C = self.patch_embed(x)
        z = self.transformer(z)
        z = self.norm(z)
        
        # Global average pooling
        z = z.mean(dim=1) # (B*C, d_model)
        z = z.reshape(B, C, d_model)
        z = z.mean(dim=1) # (B, d_model)
        
        prob = self.flare_head(z).squeeze(-1)
        dst = self.dst_head(z).squeeze(-1)
        kp = self.kp_head(z).squeeze(-1) * 9.0
        
        # Return stacked output (B, 3)
        return torch.stack([prob, dst, kp], dim=-1)
