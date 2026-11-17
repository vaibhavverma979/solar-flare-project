from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class MAGTelemetrySchema(BaseModel):
    timestamp: datetime
    b_x: Optional[float] = Field(None, description="B-field X component in nT")
    b_y: Optional[float] = Field(None, description="B-field Y component in nT")
    b_z: Optional[float] = Field(None, description="B-field Z component in nT")
    b_total: float = Field(..., description="Total magnetic field strength in nT")

    class Config:
        from_attributes = True

class ASPEXTelemetrySchema(BaseModel):
    timestamp: datetime
    proton_density: Optional[float] = Field(None, description="Proton density in particles/cm^3")
    wind_velocity: Optional[float] = Field(None, description="Solar wind velocity in km/s")
    dynamic_pressure: Optional[float] = Field(None, description="Dynamic pressure in nPa")

    class Config:
        from_attributes = True

class GeomagneticForecastSchema(BaseModel):
    id: Optional[int] = None
    forecast_target_time: datetime
    predicted_dst: Optional[float] = Field(None, description="Predicted Dst index in nT")
    predicted_kp: Optional[float] = Field(None, description="Predicted Kp index (0.0 to 9.0)")
    storm_category: Optional[str] = Field(None, description="Storm Category (e.g. 'G1 Minor')")
    l1_propagation_delay_mins: Optional[float] = Field(None, description="Estimated propagation delay from L1 to Earth in minutes")

    class Config:
        from_attributes = True

class SpaceWeatherImpactSchema(BaseModel):
    timestamp: datetime
    kp_index: Optional[float] = Field(None, description="Kp index")
    dst_index: Optional[float] = Field(None, description="Dst index")
    imf_b_z: Optional[float] = Field(None, description="IMF Bz component")
    earth_impact_eta: Optional[float] = Field(None, description="Earth impact ETA")
    spectral_hardness_ratio: Optional[float] = Field(None, description="Spectral hardness ratio")
    risk: Optional[str] = Field(None, description="Risk level")

    class Config:
        from_attributes = True
