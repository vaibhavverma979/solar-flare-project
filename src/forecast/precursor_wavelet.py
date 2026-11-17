"""
precursor_wavelet.py — CWT-based precursor pattern detector (Phase 4).

Physics context
---------------
Continuous Wavelet Transform (CWT) with a Morlet mother wavelet decomposes
the light curve into time-frequency space.  Flare precursors (microflare
bursts, pulsations) appear as elevated power at specific scales (periods
10–100 s) in the minutes before the main flare onset.

This module is a *complementary signal* to the ML forecaster — its output
(a precursor power index per time window) is added as an additional feature
column to the feature DataFrame, not used as a standalone predictor.

Rationale: wavelets capture oscillatory precursors that are not well-captured
by the rolling statistics in features.py (which are amplitude-based).

TODO (Phase 4): Implement compute_precursor_index(flux, timestamps) → np.ndarray
"""


def compute_precursor_index(flux, timestamps, scales=None):
    """
    Compute a scalar precursor index from the CWT power spectrum.

    TODO (Phase 4)
    """
    raise NotImplementedError("Phase 4 — not yet implemented")
