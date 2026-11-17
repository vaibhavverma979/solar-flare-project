#!/usr/bin/env python3
"""
inspect_fits.py — FITS file structure inspector for PRADAN-downloaded files.

Usage
-----
    python scripts/inspect_fits.py <path_to_file.fits>
    python scripts/inspect_fits.py <path_to_file.fits> --hdu 1   # inspect one HDU

Purpose
-------
Before writing instrument-specific readers we need to know the exact layout of
each PRADAN file:  how many HDUs, what the column names and types are, what the
time column is called, and what the data range looks like.

Run this on every new file format you encounter and paste the output here so
the readers can be adapted to the real schema.

This script makes NO assumptions about extension names or column layouts.  It
iterates over every HDU and prints all available metadata.
"""

import argparse
import sys
from pathlib import Path

try:
    from astropy.io import fits
    import numpy as np
except ImportError:
    sys.exit(
        "astropy and numpy are required.  Run:  pip install astropy numpy"
    )


# ─── Formatting helpers ───────────────────────────────────────────────────────

DIVIDER = "─" * 72
SECTION = "═" * 72


def _section(title: str) -> None:
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(SECTION)


def _sub(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ─── Per-HDU inspectors ───────────────────────────────────────────────────────

def _inspect_primary(hdu, idx: int) -> None:
    """Print header and shape for a PrimaryHDU or ImageHDU."""
    _sub(f"HDU {idx} — {type(hdu).__name__}  (name={hdu.name!r})")

    print("\n[Header — all cards]")
    for card in hdu.header.cards:
        print(f"  {card.keyword:<10} = {str(card.value):<30}  / {card.comment}")

    if hdu.data is not None:
        arr = hdu.data
        print(f"\n[Image data]  shape={arr.shape}  dtype={arr.dtype}")
        finite = arr[np.isfinite(arr)]
        if finite.size:
            print(
                f"  min={finite.min():.6g}  max={finite.max():.6g}  "
                f"mean={finite.mean():.6g}  NaN count={np.sum(~np.isfinite(arr))}"
            )
    else:
        print("\n[No image data in this HDU]")


def _inspect_table(hdu, idx: int, n_sample: int = 5) -> None:
    """Print header, column info, and sample rows for a BinTableHDU or TableHDU."""
    _sub(f"HDU {idx} — {type(hdu).__name__}  (name={hdu.name!r})")

    print("\n[Header — selected cards]")
    important = {
        "EXTNAME", "TELESCOP", "INSTRUME", "DATE-OBS", "DATE-END",
        "TSTART", "TSTOP", "TIMESYS", "TIMEUNIT", "MJDREFI",
        "MJDREFF", "TIMEDEL", "NAXIS2", "CREATOR", "ORIGIN",
    }
    for card in hdu.header.cards:
        if card.keyword in important or card.keyword.startswith(("TTYPE", "TFORM", "TUNIT", "TZERO", "TSCAL")):
            print(f"  {card.keyword:<10} = {str(card.value):<30}  / {card.comment}")

    tbl = hdu.data
    if tbl is None or len(tbl) == 0:
        print("\n[Table is empty]")
        return

    print(f"\n[Columns]  ({len(tbl.columns)} total, {len(tbl)} rows)")
    print(f"  {'#':<4} {'Name':<24} {'Format':<12} {'Unit':<16} {'Null/fill'}")
    print(f"  {'─'*4} {'─'*24} {'─'*12} {'─'*16} {'─'*10}")

    for i, col in enumerate(tbl.columns):
        unit = col.unit if col.unit else "—"
        null = str(col.null) if col.null is not None else "—"
        print(f"  {i:<4} {col.name:<24} {col.format:<12} {unit:<16} {null}")

    # Detect likely time columns
    time_candidates = [
        c.name for c in tbl.columns
        if any(kw in c.name.upper() for kw in ("TIME", "DATE", "MJD", "TT", "UTC"))
    ]
    if time_candidates:
        print(f"\n[Likely time columns] {time_candidates}")
        for tc in time_candidates:
            col_data = tbl[tc]
            valid = col_data[np.isfinite(col_data.astype(float))] if col_data.dtype.kind in "fiu" else col_data
            if len(valid):
                print(f"  {tc}: first={valid[0]}  last={valid[-1]}  n={len(valid)}")

    print(f"\n[First {min(n_sample, len(tbl))} rows]")
    # Print as aligned table
    names = [c.name for c in tbl.columns]
    header_row = "  " + "  ".join(f"{n[:16]:<16}" for n in names)
    print(header_row)
    print("  " + "  ".join("─" * 16 for _ in names))
    for row in tbl[:n_sample]:
        cells = []
        for n in names:
            val = row[n]
            # Shorten arrays
            if hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
                val = f"[{val[0]:.4g}…]" if len(val) else "[]"
            else:
                try:
                    val = f"{val:.6g}"
                except (TypeError, ValueError):
                    val = str(val)
            cells.append(f"{val[:16]:<16}")
        print("  " + "  ".join(cells))

    # Per-column stats for numeric columns
    print(f"\n[Numeric column statistics]")
    for col in tbl.columns:
        try:
            arr = tbl[col.name].astype(float)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            nan_count = int(np.sum(~np.isfinite(arr)))
            print(
                f"  {col.name:<24}  min={finite.min():<12.6g}  "
                f"max={finite.max():<12.6g}  mean={finite.mean():<12.6g}  "
                f"NaNs={nan_count}"
            )
        except (TypeError, ValueError):
            pass  # skip non-numeric


# ─── Main ─────────────────────────────────────────────────────────────────────

def inspect(filepath: Path, target_hdu: int | None = None, n_sample: int = 5) -> None:
    """Open a FITS file and print its full structural summary."""

    _section(f"FITS INSPECTION: {filepath.name}")
    print(f"  Full path : {filepath.resolve()}")
    print(f"  File size : {filepath.stat().st_size / 1024:.1f} KB")

    with fits.open(filepath, memmap=True, ignore_missing_simple=True) as hdul:
        print(f"\n[HDU Summary]  {len(hdul)} extension(s)")
        hdul.info()

        if target_hdu is not None:
            hdus_to_inspect = [hdul[target_hdu]]
            indices = [target_hdu]
        else:
            hdus_to_inspect = list(hdul)
            indices = list(range(len(hdul)))

        for idx, hdu in zip(indices, hdus_to_inspect):
            if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                _inspect_table(hdu, idx, n_sample=n_sample)
            else:
                _inspect_primary(hdu, idx)

    _section("END OF INSPECTION")
    print(
        "\nNext step: paste this output into the chat so the readers can be\n"
        "adapted to match your exact PRADAN file schema.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the full structure of a FITS file (PRADAN / Aditya-L1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("filepath", type=Path, help="Path to the .fits file")
    parser.add_argument(
        "--hdu",
        type=int,
        default=None,
        help="Index of a single HDU to inspect (default: all)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of sample rows to print per table (default: 5)",
    )
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    if not args.filepath.exists():
        sys.exit(f"ERROR: File not found: {args.filepath}")
    if not args.filepath.suffix.lower() in (".fits", ".fit", ".fits.gz", ".fz"):
        print(f"WARNING: File extension {args.filepath.suffix!r} is unusual for FITS — proceeding anyway.")

    inspect(args.filepath, target_hdu=args.hdu, n_sample=args.rows)


if __name__ == "__main__":
    main()
