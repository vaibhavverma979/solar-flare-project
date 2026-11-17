"""
synthetic_lc.py — Synthetic solar flare light-curve fixture generator.

Used by all unit tests so the test suite runs without any real PRADAN files.

Flare shape model
-----------------
A solar flare light curve has a characteristic fast-rise / slow-decay (FRED)
profile in soft X-rays.  We model it as:

    flux(t) = background(t) + A * G(t) * exp(-t / τ_decay)

where G(t) is a half-Gaussian rise (σ = rise_time_s / 2) for t < t_peak,
and a pure exponential for t > t_peak.  This approximates the impulsive and
gradual phases of a typical flare.

A Poisson noise floor is added: σ_noise = √(background + flux).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_flare_lc(
    duration_s: int = 3600,
    cadence_s: int = 1,
    background: float = 50.0,
    amplitude: float = 500.0,
    flare_start_s: int = 900,
    rise_time_s: int = 60,
    decay_time_s: int = 300,
    instrument: str = "SoLEXS",
    energy_low: float = 1.0,
    energy_high: float = 30.0,
    energy_channel: str = "1-30keV",
    seed: int = 42,
    quality_flag: int = 0,
) -> pd.DataFrame:
    """
    Generate a synthetic flare light curve in the standard schema.

    Parameters
    ----------
    duration_s : int
        Total observation length in seconds.
    cadence_s : int
        Time resolution in seconds.
    background : float
        Baseline count rate (counts/s).
    amplitude : float
        Peak flare excess above background (counts/s).
    flare_start_s : int
        Onset of the flare in seconds from the start of the observation.
    rise_time_s : int
        e-folding rise time in seconds (Gaussian σ).
    decay_time_s : int
        Exponential decay constant in seconds (τ).
    instrument : str
        Value for the `instrument` column.
    energy_low, energy_high : float
        Energy band bounds (keV).
    energy_channel : str
        Human-readable energy label.
    seed : int
        NumPy random seed for reproducibility.
    quality_flag : int
        Quality flag to assign uniformly (0 = good).

    Returns
    -------
    pd.DataFrame
        Standard schema with a synthetic flare injected into Poisson noise.
    """
    rng = np.random.default_rng(seed)
    n = duration_s // cadence_s
    t = np.arange(n) * cadence_s  # seconds from start

    # ── Flare profile ──────────────────────────────────────────────────────────
    t_peak = flare_start_s + rise_time_s
    flare = np.zeros(n)
    rise_mask = (t >= flare_start_s) & (t <= t_peak)
    decay_mask = t > t_peak

    # Gaussian rise
    flare[rise_mask] = amplitude * np.exp(
        -0.5 * ((t[rise_mask] - t_peak) / rise_time_s) ** 2
    )
    # Exponential decay
    flare[decay_mask] = amplitude * np.exp(-(t[decay_mask] - t_peak) / decay_time_s)

    signal = background + flare
    # Poisson noise (approximated with Gaussian for large counts)
    noise = rng.normal(0, np.sqrt(np.maximum(signal, 1.0)))
    flux = np.maximum(signal + noise, 0.0)
    flux_err = np.sqrt(np.maximum(flux, 0.0))

    timestamps = pd.date_range(
        start="2024-01-01T00:00:00Z",
        periods=n,
        freq=f"{cadence_s}s",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "flux": flux,
            "flux_err": flux_err,
            "energy_low": np.float32(energy_low),
            "energy_high": np.float32(energy_high),
            "energy_channel": energy_channel,
            "instrument": instrument,
            "quality_flag": np.int8(quality_flag),
        }
    )


def make_two_instrument_flares(
    hxr_lead_s: int = 60,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a matched SoLEXS (SXR) + HEL1OS (HXR) pair with Neupert-effect
    timing: the HXR peak leads the SXR peak by *hxr_lead_s* seconds.

    This is used to test the fusion.py Neupert-effect-consistency check.

    Parameters
    ----------
    hxr_lead_s : int
        How many seconds before the SXR peak the HXR peaks (physics: HXR
        emission from non-thermal electrons precedes the SXR thermal emission).
    **kwargs
        Forwarded to ``make_flare_lc``; applied to the SoLEXS light curve.

    Returns
    -------
    (solexs_df, hel1os_df) : tuple of pd.DataFrame
    """
    sx_kwargs = dict(
        instrument="SoLEXS",
        energy_low=1.0,
        energy_high=30.0,
        energy_channel="1-30keV",
        **kwargs,
    )
    hx_flare_start = kwargs.get("flare_start_s", 900) - hxr_lead_s
    hx_kwargs = dict(
        instrument="HEL1OS",
        energy_low=8.0,
        energy_high=150.0,
        energy_channel="8-150keV",
        flare_start_s=max(0, hx_flare_start),
        rise_time_s=kwargs.get("rise_time_s", 60) // 2,  # HXR rise is faster
        decay_time_s=kwargs.get("decay_time_s", 300) // 3,  # HXR decays faster
        amplitude=kwargs.get("amplitude", 500.0) * 0.3,  # lower absolute flux in HXR
        seed=kwargs.get("seed", 42) + 1,
    )
    solexs_df = make_flare_lc(**sx_kwargs)
    hel1os_df = make_flare_lc(**hx_kwargs)
    return solexs_df, hel1os_df
