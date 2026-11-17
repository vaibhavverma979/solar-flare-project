import argparse
import logging
import os
import warnings
import matplotlib.pyplot as plt

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_pipeline')

# ─────────────────────────────────────────────────────────────────────────────
# DEPRECATION NOTICE
# This script uses the legacy ISRO FITS-based training pipeline.
# FITS files are no longer used as a training data source.
# Use the NASA/NOAA pipeline instead:
#
#   python nasa_train.py --start 2017-01-01 --end 2023-12-31
#
# ISRO Aditya-L1 telemetry is now used exclusively for live inference via:
#   src/inference/aditya_live_predictor.py
# ─────────────────────────────────────────────────────────────────────────────
warnings.warn(
    "\n\ntrain.py is DEPRECATED for model training.\n"
    "ISRO FITS files are no longer a valid training source.\n"
    "Use: python nasa_train.py --start 2017-01-01 --end 2023-12-31\n",
    DeprecationWarning,
    stacklevel=1,
)

from src.data.isro_fits_parser import ISROFITSParser, load_fits_dataset
from src.data.flare_features import AstrophysicalFeatureEngineer
from src.forecast.train_fits_model import train_patchtst
from src.nowcast.train_xgb import train_xgb_nowcast

def plot_evaluation_curves(df, output_dir='output'):
    """Plots a simple summary of the data and features to represent evaluation curves."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['solexs_total_sxr'], label='SXR Total (Scaled)')
    plt.plot(df['timestamp'], df['hel1os_total_hxr'], label='HXR Total (Scaled)')
    plt.title("Aditya-L1 SXR and HXR over Time")
    plt.xlabel("Time")
    plt.ylabel("Scaled Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'evaluation_curve.png')
    plt.savefig(plot_path)
    logger.info(f"Saved evaluation curve to {plot_path}")

def main():
    parser = argparse.ArgumentParser(
        description=(
            "[DEPRECATED] ISRO Aditya-L1 FITS-based Training Pipeline.\n"
            "\nThis pipeline no longer uses FITS files for model training.\n"
            "Use 'python nasa_train.py' for the NASA/NOAA pipeline."
        )
    )
    parser.add_argument('--data_dir', type=str, default='data/raw_fits', help='Directory containing raw FITS files')
    parser.add_argument('--epochs', type=int, default=35, help='Number of epochs for PatchTST')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for PatchTST')
    
    args = parser.parse_args()
    
    logger.info("=== Starting Legacy FITS Training Pipeline ===")
    logger.warning(
        "DEPRECATED: This pipeline reads ISRO FITS files for training. "
        "For multi-year NASA/NOAA training, run: python nasa_train.py"
    )
    
    # 1. Parsing and Ingestion (bulk batch pipeline)
    logger.info(f"Scanning for FITS files under '{args.data_dir}' (recursive) ...")
    df_raw = load_fits_dataset(
        raw_fits_dir=args.data_dir,
        resample_freq='1min',
        ffill_limit_minutes=5,
    )

    if df_raw.empty:
        logger.error("Parsed dataframe is empty. Please check the data directory and FITS files.")
        return

    logger.info(f"Parsed {len(df_raw)} rows of data spanning "
                f"{df_raw['timestamp'].min()} → {df_raw['timestamp'].max()}.")
    
    # 2. Feature Engineering
    logger.info("Extracting Astrophysical Features...")
    engineer = AstrophysicalFeatureEngineer(df_raw)
    df_features = engineer.process(fit_scaler=True)
    
    # Optional: plot evaluation curves
    plot_evaluation_curves(df_features)
    
    # 3. PatchTST Training
    logger.info(f"Starting PatchTST Training (epochs={args.epochs}, batch_size={args.batch_size})...")
    best_patchtst_model = train_patchtst(df_features, epochs=args.epochs, batch_size=args.batch_size)
    logger.info(f"PatchTST training completed. Best model saved at {best_patchtst_model}")
    
    # 4. XGBoost Training
    logger.info("Starting XGBoost Nowcast Training...")
    best_xgb_model = train_xgb_nowcast(df_features)
    if best_xgb_model:
        logger.info(f"XGBoost training completed. Best model saved at {best_xgb_model}")
    else:
        logger.warning("XGBoost training skipped or failed.")
        
    logger.info("=== Pipeline Execution Complete ===")

if __name__ == "__main__":
    main()
