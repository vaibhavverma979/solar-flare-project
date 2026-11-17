"""
models.py — SQLAlchemy ORM models for the flare system database (Phase 5).

All tables share a common ``created_at`` audit column.  The schema is designed
to be swap-in-ready for TimescaleDB (by converting LightCurvePoint to a
hypertable on the timestamp column) or InfluxDB (by treating each row as a
measurement with tags = instrument/channel and field = flux).

Tables
------
LightCurvePoint  — one row per (timestamp, instrument, energy_channel)
FlareEvent       — master flare catalogue (output of fusion.py)
Alert            — triggered alerts (nowcast or forecast)

TODO (Phase 5): Implement full ORM models and migrations.
"""

from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Boolean, ForeignKey, Index
)
from sqlalchemy.orm import DeclarativeBase, relationship
import datetime


class Base(DeclarativeBase):
    pass


class LightCurvePoint(Base):
    """
    One time-series sample from either SoLEXS or HEL1OS.
    Indexed on (timestamp, instrument, energy_channel) for fast range queries.
    """
    __tablename__ = "lightcurve_points"
    __table_args__ = (
        Index("ix_lcp_ts_instr", "timestamp", "instrument", "energy_channel"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    instrument = Column(String(16), nullable=False)   # "SoLEXS" | "HEL1OS"
    energy_channel = Column(String(32), nullable=False)
    flux = Column(Float, nullable=True)
    flux_err = Column(Float, nullable=True)
    quality_flag = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class FlareEvent(Base):
    """
    A confirmed or candidate flare event from the master catalogue.
    One row per event regardless of which instruments detected it.
    """
    __tablename__ = "flare_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(64), unique=True, nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False)
    peak_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    flare_class = Column(String(4), nullable=True)   # "A", "B", "C", "M", "X"
    peak_flux_soft = Column(Float, nullable=True)     # SoLEXS peak counts/s
    peak_flux_hard = Column(Float, nullable=True)     # HEL1OS peak counts/s
    confidence = Column(Float, nullable=True)         # 0.0–1.0
    instruments_agreeing = Column(String(32), nullable=True)  # "SoLEXS,HEL1OS"
    neupert_consistent = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    alerts = relationship("Alert", back_populates="flare_event")


class Alert(Base):
    """
    An alert issued by the nowcast or forecast system.
    Multiple alerts can reference the same FlareEvent.
    """
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_type = Column(String(16), nullable=False)   # "nowcast" | "forecast"
    triggered_at = Column(DateTime(timezone=True), nullable=False)
    forecast_probability = Column(Float, nullable=True)   # 0.0–1.0 (forecast only)
    lead_time_s = Column(Float, nullable=True)            # seconds before flare peak
    flare_event_id = Column(Integer, ForeignKey("flare_events.id"), nullable=True)
    shap_summary = Column(String(2048), nullable=True)    # JSON-serialised SHAP values
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    flare_event = relationship("FlareEvent", back_populates="alerts")
