import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
    _PYTORCH_AVAILABLE = True
except ImportError:
    _PYTORCH_AVAILABLE = False
    class Dataset: pass

class MultiVariateTelemetryDataset(Dataset):
    """
    PyTorch Dataset for 6-channel multi-variate telemetry streams.
    Channels: [soft_xray_flux, hard_xray_flux, b_z, wind_velocity, e_y, p_dyn]
    """
    def __init__(self, df: pd.DataFrame, context_len: int = 360):
        self.context_len = context_len
        
        # 6 Expected channels
        self.channels = ['soft_xray_flux', 'hard_xray_flux', 'b_z', 'wind_velocity', 'e_y', 'p_dyn']
        
        # Ensure all columns exist (fill missing with 0 for safety)
        for col in self.channels:
            if col not in df.columns:
                df[col] = 0.0
                
        self.data = df[self.channels].values.astype(np.float32)
        
        # Generate target variables for multi-task
        # Probability (proxy via soft_xray_flux threshold)
        flare_prob = np.where(df['soft_xray_flux'] > 1e-5, 1.0, 0.0).astype(np.float32)
        # Estimated Dst (proxy via physics formula)
        dst_index = (7.26 * np.sqrt(np.maximum(df['p_dyn'], 0)) - 15.0 * df['e_y']).values.astype(np.float32)
        # Estimated Kp (proxy via physics formula)
        kp_index = np.clip(1.0 + (0.8 * df['e_y']) + (0.2 * np.sqrt(np.maximum(df['p_dyn'], 0))), 0.0, 9.0).values.astype(np.float32)
        
        self.targets = np.stack([flare_prob, dst_index, kp_index], axis=1)

    def __len__(self):
        return max(0, len(self.data) - self.context_len)

    def __getitem__(self, idx):
        if not _PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available.")
            
        x = self.data[idx : idx + self.context_len]
        
        # The target for the window is the target at the end of the window
        y = self.targets[idx + self.context_len - 1]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
