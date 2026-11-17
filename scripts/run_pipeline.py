#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end pipeline runner: raw → processed → catalogue → forecast.

Usage
-----
    python scripts/run_pipeline.py --solexs data/raw/solexs_*.fits
                                   --hel1os data/raw/hel1os_*.fits
                                   [--cadence 1]
                                   [--horizon 30]

TODO: Implement full pipeline once Phases 2–4 are complete.
Phase 1 only: runs ingestion + alignment and saves the merged DataFrame.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from src.ingestion.solexs_reader import read_solexs
from src.ingestion.hel1os_reader import read_hel1os
from src.ingestion.align import align

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_pipeline")


def run(solexs_files: list[Path], hel1os_files: list[Path], cadence_s: int = 0) -> None:
    """Phase 1 pipeline: ingest, align, save."""
    settings.ensure_dirs()

    # ── Ingest SoLEXS ─────────────────────────────────────────────────────────
    import pandas as pd
    sx_frames = []
    for f in solexs_files:
        logger.info("Loading SoLEXS: %s", f)
        sx_frames.append(read_solexs(f))
    sx_df = pd.concat(sx_frames, ignore_index=True).sort_values("timestamp") if sx_frames else pd.DataFrame()

    # ── Ingest HEL1OS ─────────────────────────────────────────────────────────
    hx_frames = []
    for f in hel1os_files:
        logger.info("Loading HEL1OS: %s", f)
        hx_frames.append(read_hel1os(f))
    hx_df = pd.concat(hx_frames, ignore_index=True).sort_values("timestamp") if hx_frames else pd.DataFrame()

    if sx_df.empty and hx_df.empty:
        logger.error("No data loaded. Check your file paths.")
        sys.exit(1)

    # ── Align ─────────────────────────────────────────────────────────────────
    logger.info("Aligning instruments (cadence_s=%d)…", cadence_s)
    merged = align(sx_df, hx_df, cadence_s=cadence_s)

    out_path = settings.data_processed_dir / "aligned_lightcurve.parquet"
    merged.to_parquet(out_path, index=False)
    logger.info("Saved aligned light curve → %s  (%d rows)", out_path, len(merged))


def main():
    parser = argparse.ArgumentParser(description="Run the Aditya-L1 flare pipeline.")
    parser.add_argument("--solexs", nargs="*", type=Path, default=[], metavar="FILE")
    parser.add_argument("--hel1os", nargs="*", type=Path, default=[], metavar="FILE")
    parser.add_argument("--cadence", type=int, default=0, help="Resample cadence in seconds (0=auto)")
    args = parser.parse_args()
    run(args.solexs, args.hel1os, cadence_s=args.cadence)


if __name__ == "__main__":
    main()
