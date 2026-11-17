import numpy as np
from src.forecast.baseline_model import BaseForecaster, RandomForestBaseline


class LSTMForecaster(BaseForecaster):
    """Bidirectional LSTM sequence model for flare probability forecasting."""

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, epochs: int = 5, lr: float = 1e-3):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.pytorch_available = False
        try:
            import torch
            import torch.nn as nn
            self.pytorch_available = True
        except ImportError:
            self.fallback = RandomForestBaseline()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        if not self.pytorch_available:
            self.fallback.fit(X, y)
            self.is_fitted = True
            return self

        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)

        n_samples, seq_len, n_features = X.shape
        
        class PyTorchLSTM(nn.Module):
            def __init__(self, in_dim, h_dim, n_layers):
                super().__init__()
                self.lstm = nn.LSTM(in_dim, h_dim, num_layers=n_layers, batch_first=True, bidirectional=True)
                self.fc = nn.Linear(h_dim * 2, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return self.sigmoid(out).squeeze(-1)

        self.net = PyTorchLSTM(n_features, self.hidden_dim, self.num_layers)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        x_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=min(32, max(1, len(X))), shuffle=True)

        self.net.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self.net(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or len(X) == 0:
            return np.zeros(len(X), dtype=float)

        if not self.pytorch_available:
            return self.fallback.predict_proba(X)

        import torch
        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)

        self.net.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32)
            preds = self.net(x_tensor).numpy()
        return np.atleast_1d(preds)


def build_model(model_type: str = "lstm", **kwargs) -> BaseForecaster:
    """Factory function to build forecaster model by name."""
    m_type = model_type.lower()
    if m_type == "lstm":
        return LSTMForecaster(**kwargs)
    elif m_type == "patchtst" or m_type == "patch_tst":
        from src.forecast.patch_tst import PatchTSTForecaster
        return PatchTSTForecaster(**kwargs)
    elif m_type == "rf" or m_type == "random_forest":
        return RandomForestBaseline(**kwargs)
    else:
        from src.forecast.baseline_model import LogisticBaseline
        return LogisticBaseline(**kwargs)

