"""
nasa_train.py — Top-level CLI orchestrator for the NASA/NOAA training pipeline.

This is the replacement for the legacy ``train.py`` (which used ISRO FITS files).
All training data is sourced from NASA/NOAA; ISRO Aditya-L1 telemetry is not
touched here.

Usage
-----
    python nasa_train.py --start 2017-01-01 --end 2023-12-31
    python nasa_train.py --start 2019-01-01 --end 2023-12-31 --sharp_csv data/sharp_features.csv
    python nasa_train.py --skip_patchtst          # XGBoost only
    python nasa_train.py --skip_xgb               # PatchTST only
    python nasa_train.py --help

Models saved
------------
    src/forecast/models/patchtst_nasa_goes.pt     ← GOES PatchTST checkpoint
    src/nowcast/models/xgb_nasa_sharp.joblib      ← SDO SHARP XGBoost bundle
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("nasa_train")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nasa_train.py",
        description=(
            "NASA/NOAA Solar Flare Prediction Training Pipeline\n\n"
            "Trains two models:\n"
            "  1. GOES PatchTST  — 30-min X-ray flare forecasting\n"
            "  2. SDO XGBoost    — 6–24h magnetic early-warning classifier\n\n"
            "No ISRO FITS files are used.  For live ISRO inference, see:\n"
            "  src/inference/aditya_live_predictor.py\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Date range
    parser.add_argument(
        "--start", type=str, default="2017-01-01",
        help="Start date for data download (YYYY-MM-DD). Default: 2017-01-01",
    )
    parser.add_argument(
        "--end", type=str, default="2023-12-31",
        help="End date for data download (YYYY-MM-DD). Default: 2023-12-31",
    )

    # SHARP data
    parser.add_argument(
        "--sharp_csv", type=str, default="data/sharp_features.csv",
        help=(
            "Path to a pre-downloaded NASA SDO SHARP CSV/parquet file. "
            "Download from http://jsoc.stanford.edu/. "
            "Use 'harp://<number>' for live JSOC access. "
            "Default: data/sharp_features.csv"
        ),
    )

    # Training hyperparameters — PatchTST
    parser.add_argument("--patchtst_epochs",    type=int,   default=40,   help="PatchTST training epochs. Default: 40")
    parser.add_argument("--patchtst_batch",     type=int,   default=64,   help="PatchTST batch size. Default: 64")
    parser.add_argument("--window_size",        type=int,   default=60,   help="GOES sliding window length (minutes). Default: 60")
    parser.add_argument("--stride",             type=int,   default=30,   help="GOES window stride (minutes). Default: 30")
    parser.add_argument("--focal_gamma",        type=float, default=2.0,  help="FocalLoss gamma. Default: 2.0")
    parser.add_argument("--focal_alpha",        type=float, default=0.25, help="FocalLoss alpha. Default: 0.25")

    # Training hyperparameters — XGBoost
    parser.add_argument("--xgb_estimators",    type=int,   default=300,  help="XGBoost n_estimators. Default: 300")
    parser.add_argument("--xgb_depth",         type=int,   default=5,    help="XGBoost max_depth. Default: 5")
    parser.add_argument("--xgb_lr",            type=float, default=0.05, help="XGBoost learning rate. Default: 0.05")
    parser.add_argument("--lookahead_hours",   type=int,   default=24,   help="DONKI labelling horizon (hours). Default: 24")

    # Control flags
    parser.add_argument("--skip_patchtst",  action="store_true", help="Skip PatchTST training.")
    parser.add_argument("--skip_xgb",       action="store_true", help="Skip XGBoost training.")
    parser.add_argument("--no_cache",       action="store_true", help="Disable GOES local cache.")
    parser.add_argument("--nasa_api_key",   type=str, default=None,
                        help="NASA API key for DONKI. Falls back to NASA_API_KEY env var or DEMO_KEY.")

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args   = parser.parse_args()

    logger.info("=" * 70)
    logger.info("  NASA / NOAA Solar Flare Prediction — Training Pipeline")
    logger.info("=" * 70)
    logger.info("  Data range    : %s → %s", args.start, args.end)
    logger.info("  SHARP source  : %s", args.sharp_csv)
    logger.info("  Skip PatchTST : %s", args.skip_patchtst)
    logger.info("  Skip XGBoost  : %s", args.skip_xgb)
    logger.info("=" * 70)

    api_key = args.nasa_api_key or os.environ.get("NASA_API_KEY")
    wall_start = time.perf_counter()

    # ── Import pipeline modules ──────────────────────────────────────────────
    try:
        from src.data.nasa_pipeline import (
            build_goes_training_dataset,
            build_sharp_training_dataset,
            fetch_donki_flare_labels,
        )
    except ImportError as exc:
        logger.error("Failed to import src.data.nasa_pipeline: %s", exc)
        sys.exit(1)

    summary: dict = {}

    # =========================================================================
    # 1. PatchTST — GOES X-ray training
    # =========================================================================
    if not args.skip_patchtst:
        logger.info("")
        logger.info("─" * 70)
        logger.info("  [1/2]  PatchTST Training  —  NOAA GOES X-ray")
        logger.info("─" * 70)

        t0 = time.perf_counter()
        X_goes, y_goes, ts_goes = build_goes_training_dataset(
            start_date=args.start,
            end_date=args.end,
            window_size=args.window_size,
            stride=args.stride,
            use_cache=not args.no_cache,
            api_key=api_key,
        )

        if len(X_goes) == 0:
            logger.error("Empty GOES dataset — PatchTST training skipped.")
            summary["patchtst"] = "SKIPPED — empty dataset"
        else:
            try:
                from src.forecast.train_patchtst_nasa import train_patchtst_nasa

                patchtst_path = train_patchtst_nasa(
                    X=X_goes,
                    y=y_goes,
                    epochs=args.patchtst_epochs,
                    batch_size=args.patchtst_batch,
                    focal_gamma=args.focal_gamma,
                    focal_alpha=args.focal_alpha,
                )
                elapsed = time.perf_counter() - t0
                summary["patchtst"] = f"✓  {patchtst_path}  ({elapsed:.0f}s)"
                logger.info("PatchTST training complete in %.0f s → %s", elapsed, patchtst_path)
            except Exception as exc:
                logger.exception("PatchTST training failed: %s", exc)
                summary["patchtst"] = f"✗  FAILED: {exc}"
    else:
        logger.info("[1/2] PatchTST training skipped (--skip_patchtst).")
        summary["patchtst"] = "SKIPPED"

    # =========================================================================
    # 2. XGBoost — SDO SHARP training
    # =========================================================================
    if not args.skip_xgb:
        logger.info("")
        logger.info("─" * 70)
        logger.info("  [2/2]  XGBoost Training  —  NASA SDO SHARP Features")
        logger.info("─" * 70)

        t0 = time.perf_counter()

        # Fetch DONKI labels once and reuse for SHARP labelling
        try:
            donki_df = fetch_donki_flare_labels(
                start_date=args.start,
                end_date=args.end,
                api_key=api_key,
            )
        except Exception as exc:
            logger.warning("DONKI fetch failed (%s) — XGBoost will proceed without pre-fetched labels.", exc)
            donki_df = None

        X_sharp, y_sharp = build_sharp_training_dataset(
            sharp_source=args.sharp_csv,
            donki_df=donki_df,
            start_date=args.start,
            end_date=args.end,
            lookahead_hours=args.lookahead_hours,
            api_key=api_key,
        )

        if len(X_sharp) == 0:
            logger.error("Empty SHARP dataset — XGBoost training skipped.")
            summary["xgboost"] = "SKIPPED — empty dataset"
        else:
            try:
                from src.nowcast.train_xgb_nasa import train_xgb_nasa

                xgb_path = train_xgb_nasa(
                    X=X_sharp,
                    y=y_sharp,
                    n_estimators=args.xgb_estimators,
                    max_depth=args.xgb_depth,
                    learning_rate=args.xgb_lr,
                )
                elapsed = time.perf_counter() - t0
                if xgb_path:
                    summary["xgboost"] = f"✓  {xgb_path}  ({elapsed:.0f}s)"
                    logger.info("XGBoost training complete in %.0f s → %s", elapsed, xgb_path)
                else:
                    summary["xgboost"] = "✗  FAILED (returned None)"
            except Exception as exc:
                logger.exception("XGBoost training failed: %s", exc)
                summary["xgboost"] = f"✗  FAILED: {exc}"
    else:
        logger.info("[2/2] XGBoost training skipped (--skip_xgb).")
        summary["xgboost"] = "SKIPPED"

    # =========================================================================
    # Summary
    # =========================================================================
    total_elapsed = time.perf_counter() - wall_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("  TRAINING COMPLETE — total wall time: %.0f s (%.1f min)",
                total_elapsed, total_elapsed / 60)
    logger.info("─" * 70)
    logger.info("  PatchTST model : %s", summary.get("patchtst", "N/A"))
    logger.info("  XGBoost model  : %s", summary.get("xgboost",  "N/A"))
    logger.info("─" * 70)
    logger.info("  Next step for live inference:")
    logger.info("    from src.inference.aditya_live_predictor import AdityaLivePredictor")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
