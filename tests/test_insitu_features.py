import pytest
import numpy as np
import pandas as pd
from src.preprocessing.insitu_features import (
    calculate_l1_propagation_delay,
    calculate_reconnection_field,
    calculate_dynamic_pressure,
    process_insitu_dataframe
)

def test_l1_propagation_delay():
    # At 800 km/s, 1,500,000 km should take ~1875 seconds = 31.25 minutes
    delay = calculate_l1_propagation_delay(800.0)
    assert np.isclose(delay, 31.25)

def test_reconnection_electric_field_southward():
    # Southward IMF (b_z < 0) should trigger positive Ey
    ey = calculate_reconnection_field(wind_velocity=500.0, b_z=-5.0)
    # Ey = -1 * 500 * -5 * 1e-3 = 2.5 mV/m
    assert np.isclose(ey, 2.5)

def test_reconnection_electric_field_northward():
    # Northward IMF (b_z >= 0) should trigger Ey = 0
    ey = calculate_reconnection_field(wind_velocity=500.0, b_z=5.0)
    assert ey == 0.0

def test_dynamic_pressure():
    # Density = 5 cm^-3, Vel = 400 km/s
    p = calculate_dynamic_pressure(proton_density=5.0, wind_velocity=400.0)
    # Expected: 1.6726e-27 * 5e6 * (4e5)^2 * 1e9 = ~1.338 nPa
    assert np.isclose(p, 1.33808)

def test_process_insitu_dataframe():
    # Create mock dataframe
    df = pd.DataFrame({
        "b_z": [5.0, -10.0, -20.0], # Northward, Southward, Extreme Southward
        "wind_velocity": [400.0, 600.0, 900.0],
        "proton_density": [5.0, 10.0, 20.0]
    })
    
    result = process_insitu_dataframe(df)
    
    # Check shape and columns
    assert len(result) == 3
    assert "e_y" in result.columns
    assert "p_dyn" in result.columns
    assert "estimated_kp" in result.columns
    assert "storm_category" in result.columns
    
    # Check Ey calculation in DataFrame
    assert result.loc[0, "e_y"] == 0.0  # Northward
    assert result.loc[1, "e_y"] > 0.0   # Southward
    
    # Check storm categories
    # The extreme case (bz=-20, v=900, p=20) should have high Kp
    kp_extreme = result.loc[2, "estimated_kp"]
    assert kp_extreme >= 5.0
    assert result.loc[2, "storm_category"] != "Quiet"
