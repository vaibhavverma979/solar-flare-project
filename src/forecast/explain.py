"""
explain.py — SHAP explainability for the baseline forecasting model (Phase 4).

Why SHAP?
---------
In a hackathon/operations context, an alert without explanation is hard to act
on.  SHAP (SHapley Additive exPlanations) assigns each input feature a
contribution score to the model's output probability.  For a given alert the
dashboard can show: "Alert fired because flux_slope_60s = +3.2σ (contribution
+0.31) and hardness_ratio = 1.8 (contribution +0.14)."

SHAP is only implemented for the baseline model (scikit-learn compatible) in
Phase 4.  Deep SHAP for the LSTM is Phase 4+ stretch goal.

TODO (Phase 4): Implement compute_shap(model, X, feature_names) → pd.DataFrame
"""


import pandas as pd
import numpy as np


def compute_shap(model, X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """
    Compute SHAP values for each sample in X.
    """
    if X.ndim == 3:
        n, seq_len, n_feat = X.shape
        X_flat = X.reshape(n, seq_len * n_feat)
        if len(feature_names) != X_flat.shape[1]:
            cols = [f"{feat}_t{t}" for t in range(seq_len) for feat in feature_names]
            if len(cols) != X_flat.shape[1]:
                cols = [f"feat_{i}" for i in range(X_flat.shape[1])]
        else:
            cols = feature_names
    else:
        X_flat = X
        cols = feature_names if len(feature_names) == X_flat.shape[1] else [f"feat_{i}" for i in range(X_flat.shape[1])]

    raw_model = getattr(model, "model", model)

    try:
        import shap
        explainer = shap.Explainer(raw_model, X_flat)
        shap_vals = explainer(X_flat).values
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]
    except Exception:
        # Fallback to feature importances or linear coefficients if SHAP unavailable/errors
        if hasattr(raw_model, "feature_importances_"):
            fi = raw_model.feature_importances_
            shap_vals = np.tile(fi, (len(X_flat), 1))
        elif hasattr(raw_model, "coef_"):
            coef = raw_model.coef_.ravel()
            shap_vals = X_flat * coef
        else:
            shap_vals = np.zeros_like(X_flat)

    return pd.DataFrame(shap_vals, columns=cols[:X_flat.shape[1]])

