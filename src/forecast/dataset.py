import torch
from torch.utils.data import Dataset
import numpy as np

class FlareSlidingWindowDataset(Dataset):
    def __init__(self, df, context_length=360, forecast_horizon=30, m_class_threshold=5.0):
        """
        Args:
            df (pd.DataFrame): Dataframe containing the time series features.
            context_length (int): Number of steps in the input window.
            forecast_horizon (int): Number of steps in the forecast window.
            m_class_threshold (float): Threshold on 'solexs_total_sxr' to classify as M-class or above.
        """
        self.df = df
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.m_class_threshold = m_class_threshold
        
        # Select numeric columns for input features
        self.feature_cols = [c for c in df.columns if c not in ['timestamp']]
        
        self.data = df[self.feature_cols].values.astype(np.float32)
        
        # Pre-compute target indices for faster access
        try:
            self.sxr_idx = self.feature_cols.index('solexs_total_sxr')
        except ValueError:
            self.sxr_idx = 0
            
        try:
            self.hxr_idx = self.feature_cols.index('hel1os_total_hxr')
        except ValueError:
            self.hxr_idx = 0
            
        # Valid starting indices
        self.valid_indices = range(len(self.data) - self.context_length - self.forecast_horizon + 1)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.context_length
        forecast_end_idx = end_idx + self.forecast_horizon
        
        # Input context window shape: [context_length, num_features]
        x = self.data[start_idx:end_idx]
        
        # Forecast window targets
        y_window = self.data[end_idx:forecast_end_idx]
        
        y_sxr = y_window[:, self.sxr_idx]
        y_hxr = y_window[:, self.hxr_idx]
        
        # Multi-task targets
        # 1. Binary Flare Classification (>= M-Class threshold in forecast window)
        peak_sxr = np.max(y_sxr)
        flare_flag = 1.0 if peak_sxr >= self.m_class_threshold else 0.0
        
        # 2. Peak SXR Flux Regression
        # 3. Peak HXR Count Regression
        peak_hxr = np.max(y_hxr)
        
        targets = {
            'flare_flag': torch.tensor([flare_flag], dtype=torch.float32),
            'peak_sxr': torch.tensor([peak_sxr], dtype=torch.float32),
            'peak_hxr': torch.tensor([peak_hxr], dtype=torch.float32)
        }
        
        return torch.tensor(x, dtype=torch.float32), targets
