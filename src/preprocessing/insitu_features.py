import numpy as np
import pandas as pd

def calculate_l1_propagation_delay(wind_velocity: float | pd.Series) -> float | pd.Series:
    """
    Calculates L1 to Earth propagation delay in minutes.
    D_L1 = 1,500,000 km
    """
    # Protect against division by zero
    v = np.where(wind_velocity <= 0, 1e-5, wind_velocity)
    delay_seconds = 1_500_000 / v
    return delay_seconds / 60.0

def calculate_reconnection_field(wind_velocity: float | pd.Series, b_z: float | pd.Series) -> float | pd.Series:
    """
    Dawn-to-dusk electric field E_y in mV/m.
    Only active when b_z < 0 (Southward IMF).
    """
    # E_y = -V * B_z * 1e-3 (mV/m)
    e_y = -1 * wind_velocity * b_z * 1e-3
    # Relu: max(e_y, 0)
    return np.maximum(e_y, 0.0)

def calculate_dynamic_pressure(proton_density: float | pd.Series, wind_velocity: float | pd.Series) -> float | pd.Series:
    """
    Solar wind dynamic pressure P_dyn in nPa.
    Formula: (1.6726e-27 kg) * (density * 1e6 m^-3) * (velocity * 1000 m/s)^2 * 1e9 nPa/Pa
    """
    m_p = 1.6726e-27
    n_m3 = proton_density * 1e6
    v_ms = wind_velocity * 1000.0
    
    p_pa = m_p * n_m3 * (v_ms ** 2)
    return p_pa * 1e9

def estimate_geomagnetic_indices(b_z: pd.Series, wind_velocity: pd.Series, proton_density: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Approximates Dst and Kp indices, and maps to NOAA Storm Scale.
    Returns: (estimated_dst, estimated_kp, storm_category)
    """
    e_y = calculate_reconnection_field(wind_velocity, b_z)
    p_dyn = calculate_dynamic_pressure(proton_density, wind_velocity)
    
    # Burton et al. empirical approximation for instantaneous Dst
    # Dst* = 7.26 * sqrt(P_dyn) - 20 * E_y (Simplified proxy without time integration)
    est_dst = 7.26 * np.sqrt(np.maximum(p_dyn, 0)) - 15.0 * e_y
    
    # Newell coupling function approximation for Kp
    # Normally Kp is 3-hour, but we estimate an instantaneous driver proxy
    # Kp ~ baseline + Ey contribution + pressure compression
    est_kp = 1.0 + (0.8 * e_y) + (0.2 * np.sqrt(np.maximum(p_dyn, 0)))
    est_kp = np.clip(est_kp, 0.0, 9.0)
    
    # Map to NOAA Storm Scales
    # G1: Kp=5, G2: Kp=6, G3: Kp=7, G4: Kp=8, G5: Kp=9
    conditions = [
        est_kp >= 9.0,
        est_kp >= 8.0,
        est_kp >= 7.0,
        est_kp >= 6.0,
        est_kp >= 5.0
    ]
    choices = ["G5 Extreme", "G4 Severe", "G3 Strong", "G2 Moderate", "G1 Minor"]
    storm_category = pd.Series(np.select(conditions, choices, default="Quiet"), index=b_z.index)
    
    return est_dst, est_kp, storm_category

def process_insitu_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Batch preprocessing pipeline for merged MAG and ASPEX DataFrame.
    Expected columns: ['b_z', 'wind_velocity', 'proton_density']
    """
    out_df = df.copy()
    
    # Calculate physics features
    out_df["propagation_delay_mins"] = calculate_l1_propagation_delay(out_df["wind_velocity"])
    out_df["e_y"] = calculate_reconnection_field(out_df["wind_velocity"], out_df["b_z"])
    out_df["p_dyn"] = calculate_dynamic_pressure(out_df["proton_density"], out_df["wind_velocity"])
    
    # Estimate Indices
    dst, kp, storm = estimate_geomagnetic_indices(out_df["b_z"], out_df["wind_velocity"], out_df["proton_density"])
    out_df["estimated_dst"] = dst
    out_df["estimated_kp"] = kp
    out_df["storm_category"] = storm
    
    return out_df[['b_z', 'wind_velocity', 'proton_density', 'e_y', 'p_dyn', 'propagation_delay_mins', 'estimated_dst', 'estimated_kp', 'storm_category']]
