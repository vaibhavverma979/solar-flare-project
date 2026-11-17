-- init_insitu.sql
-- Extension should already be created, but just in case
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create MAG Telemetry Table
CREATE TABLE IF NOT EXISTS mag_telemetry (
    timestamp TIMESTAMPTZ NOT NULL,
    b_x DOUBLE PRECISION,
    b_y DOUBLE PRECISION,
    b_z DOUBLE PRECISION,
    b_total DOUBLE PRECISION NOT NULL
);

-- Convert to Hypertable
SELECT create_hypertable('mag_telemetry', 'timestamp', if_not_exists => TRUE);

-- Create ASPEX Telemetry Table
CREATE TABLE IF NOT EXISTS aspex_telemetry (
    timestamp TIMESTAMPTZ NOT NULL,
    proton_density DOUBLE PRECISION,
    wind_velocity DOUBLE PRECISION,
    dynamic_pressure DOUBLE PRECISION
);

-- Convert to Hypertable
SELECT create_hypertable('aspex_telemetry', 'timestamp', if_not_exists => TRUE);
