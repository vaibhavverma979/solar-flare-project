from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class SolexsTelemetry(Base):
    __tablename__ = "solexs_telemetry"
    
    timestamp = Column(DateTime(timezone=True), primary_key=True, index=True)
    soft_xray_flux = Column(Float, nullable=False)
    quality_flag = Column(Integer, default=0)
    sdd1_flux = Column(Float, nullable=True)
    sdd2_flux = Column(Float, nullable=True)

class Hel1osTelemetry(Base):
    __tablename__ = "hel1os_telemetry"
    
    timestamp = Column(DateTime(timezone=True), primary_key=True, index=True)
    hard_xray_flux = Column(Float, nullable=False)
    cdte1_counts = Column(Float, nullable=True)
    cdte2_counts = Column(Float, nullable=True)

class FlareEvent(Base):
    __tablename__ = "flare_events"
    
    event_id = Column(String, primary_key=True, index=True)
    peak_timestamp = Column(DateTime(timezone=True), nullable=False)
    end_timestamp = Column(DateTime(timezone=True), nullable=False)
    flare_class = Column(String, index=True)
    confidence_score = Column(Float)
    neupert_flag = Column(Boolean, default=False)

class MAGTelemetry(Base):
    __tablename__ = "mag_telemetry"
    
    timestamp = Column(DateTime(timezone=True), primary_key=True, index=True)
    b_x = Column(Float, nullable=True)
    b_y = Column(Float, nullable=True)
    b_z = Column(Float, nullable=True)
    b_total = Column(Float, nullable=False)

class ASPEXTelemetry(Base):
    __tablename__ = "aspex_telemetry"
    
    timestamp = Column(DateTime(timezone=True), primary_key=True, index=True)
    proton_density = Column(Float, nullable=True)
    wind_velocity = Column(Float, nullable=True)
    dynamic_pressure = Column(Float, nullable=True)

class GeomagneticForecast(Base):
    __tablename__ = "geomagnetic_forecast"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    forecast_target_time = Column(DateTime(timezone=True), index=True, nullable=False)
    predicted_dst = Column(Float, nullable=True)
    predicted_kp = Column(Float, nullable=True)
    storm_category = Column(String, nullable=True)
    l1_propagation_delay_mins = Column(Float, nullable=True)

class SpaceWeatherImpact(Base):
    __tablename__ = "space_weather_impact"
    
    timestamp = Column(DateTime(timezone=True), primary_key=True, index=True)
    kp_index = Column(Float, nullable=True)
    dst_index = Column(Float, nullable=True)
    imf_b_z = Column(Float, nullable=True)
    earth_impact_eta = Column(Float, nullable=True)
    spectral_hardness_ratio = Column(Float, nullable=True)
    risk = Column(String, nullable=True)
