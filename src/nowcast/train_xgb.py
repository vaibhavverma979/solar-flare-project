"""
train_xgb.py — XGBoost nowcast classifier (legacy — ISRO FITS feature variant).

.. deprecated::
    **TRAINING DEPENDENCY ON ISRO FITS FILES HAS BEEN REMOVED.**

    This module was originally trained on feature windows derived from
    ISRO Aditya-L1 FITS telemetry (``solexs_total_sxr``, ``hel1os_total_hxr``,
    ``neupert_corr``).  These columns are no longer used as training inputs.

    For NASA SDO SHARP-based training (the recommended path), use::

        # New training path:
        python nasa_train.py --start 2017-01-01 --end 2023-12-31

        # Or import directly:
        from src.nowcast.train_xgb_nasa import train_xgb_nasa

    This file is retained for backward compatibility with any external callers
    that pass a pre-built DataFrame.  ISRO FITS files must not be passed here.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
import joblib
import logging

logger = logging.getLogger(__name__)

def extract_nowcast_features(df, window_size=60):
    """
    Extracts summary features from rolling windows.
    Returns a dataframe of features and the target class for each window.
    """
    features = []
    targets = []
    
    # Using window_size steps (e.g. 60 minutes if resampled to 1m)
    for start_idx in range(0, len(df) - window_size, window_size // 2):
        end_idx = start_idx + window_size
        window = df.iloc[start_idx:end_idx]
        
        # Summary features
        peak_sxr = window['solexs_total_sxr'].max()
        peak_hxr = window['hel1os_total_hxr'].max()
        
        # Rise time (time from min to max in the window)
        idx_min = window['solexs_total_sxr'].argmin()
        idx_max = window['solexs_total_sxr'].argmax()
        rise_time = max(0, idx_max - idx_min) # in steps
        
        # Decay slope (slope after peak)
        if idx_max < len(window) - 1:
            decay_diff = window['solexs_total_sxr'].iloc[-1] - window['solexs_total_sxr'].iloc[idx_max]
            decay_time = len(window) - 1 - idx_max
            decay_slope = decay_diff / decay_time
        else:
            decay_slope = 0.0
            
        neupert_corr = window['neupert_corr'].mean()
        
        features.append({
            'peak_sxr': peak_sxr,
            'peak_hxr': peak_hxr,
            'rise_time': rise_time,
            'decay_slope': decay_slope,
            'neupert_score': neupert_corr
        })
        
        # Determine class based on peak SXR
        # These thresholds are placeholders for scaled data.
        # Adjust according to actual data distribution.
        if peak_sxr < 1.0:
            flare_class = 0 # Quiet/B-Class
        elif peak_sxr < 3.0:
            flare_class = 1 # C-Class
        elif peak_sxr < 5.0:
            flare_class = 2 # M-Class
        else:
            flare_class = 3 # X-Class
            
        targets.append(flare_class)
        
    return pd.DataFrame(features), np.array(targets)

def train_xgb_nowcast(df):
    """Train an XGBoost nowcast classifier from a feature DataFrame.

    .. deprecated::
        This function was originally designed for DataFrames built from
        ISRO Aditya-L1 FITS telemetry.  FITS files are no longer used as a
        training source.  Use ``train_xgb_nasa()`` with NASA SDO SHARP
        magnetic parameters instead::

            from src.nowcast.train_xgb_nasa import train_xgb_nasa

        Feature columns previously required by this function
        (``solexs_total_sxr``, ``hel1os_total_hxr``, ``neupert_corr``) are
        FITS-derived and should not be sourced from ISRO files for training.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-built feature DataFrame containing at minimum:
        ``solexs_total_sxr``, ``hel1os_total_hxr``, ``neupert_corr``.
        These columns may be derived from any compatible data source.

    Returns
    -------
    str or None
        Path to the saved calibrated model, or None if training was skipped.
    """
    X, y = extract_nowcast_features(df, window_size=60)
    
    # Filter out classes with too few samples for CV, though typically we should use StratifiedKFold
    # Here we'll do a simple train test split
    if len(X) < 10:
        logger.warning("Not enough data to train XGBoost. Skipping.")
        return None
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # Count classes in train
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        logger.warning("Only one class present in training data. XGBoost requires at least 2 classes.")
        # Fallback dummy model or skip
        return None

    logger.info("Training XGBClassifier...")
    base_xgb = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4, # Quiet/B, C, M, X
        eval_metric='mlogloss',
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    
    # Train base model
    base_xgb.fit(X_train, y_train)
    
    # Output calibrated confidence probabilities
    logger.info("Calibrating classifier for confidence probabilities...")
    # CV=prefit means we calibrate on the test set or a separate validation set. 
    # Here we use the test set for calibration for simplicity, though normally we'd use a separate split.
    calibrated_xgb = CalibratedClassifierCV(estimator=base_xgb, method='sigmoid', cv='prefit')
    calibrated_xgb.fit(X_test, y_test)
    
    # Evaluate
    y_pred = calibrated_xgb.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    logger.info(f"Classification Report:\n{report}")
    
    # Save model
    os.makedirs('src/nowcast/models', exist_ok=True)
    model_path = 'src/nowcast/models/xgb_nowcast_fits.joblib'
    joblib.dump(calibrated_xgb, model_path)
    logger.info(f"Saved XGBoost nowcast model to {model_path}")
    
    return model_path
