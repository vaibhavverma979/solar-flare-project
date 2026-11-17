-- init_timescale.sql
-- Ensure the TimescaleDB extension is enabled
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 1. solexs_telemetry
CREATE TABLE IF NOT EXISTS solexs_telemetry (
    timestamp TIMESTAMPTZ NOT NULL,
    soft_xray_flux DOUBLE PRECISION NOT NULL,
    quality_flag INTEGER DEFAULT 0,
    sdd1_flux DOUBLE PRECISION,
    sdd2_flux DOUBLE PRECISION,
    PRIMARY KEY (timestamp)
);
SELECT create_hypertable('solexs_telemetry', 'timestamp', if_not_exists => TRUE);

-- 2. hel1os_telemetry
CREATE TABLE IF NOT EXISTS hel1os_telemetry (
    timestamp TIMESTAMPTZ NOT NULL,
    hard_xray_flux DOUBLE PRECISION NOT NULL,
    cdte1_counts DOUBLE PRECISION,
    cdte2_counts DOUBLE PRECISION,
    PRIMARY KEY (timestamp)
);
SELECT create_hypertable('hel1os_telemetry', 'timestamp', if_not_exists => TRUE);

-- 3. mag_telemetry
CREATE TABLE IF NOT EXISTS mag_telemetry (
    timestamp TIMESTAMPTZ NOT NULL,
    b_x DOUBLE PRECISION,
    b_y DOUBLE PRECISION,
    b_z DOUBLE PRECISION,
    b_total DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (timestamp)
);
SELECT create_hypertable('mag_telemetry', 'timestamp', if_not_exists => TRUE);

-- 4. aspex_telemetry
CREATE TABLE IF NOT EXISTS aspex_telemetry (
    timestamp TIMESTAMPTZ NOT NULL,
    proton_density DOUBLE PRECISION,
    wind_velocity DOUBLE PRECISION,
    dynamic_pressure DOUBLE PRECISION,
    PRIMARY KEY (timestamp)
);
SELECT create_hypertable('aspex_telemetry', 'timestamp', if_not_exists => TRUE);

-- 5. space_weather_impact
CREATE TABLE IF NOT EXISTS space_weather_impact (
    timestamp TIMESTAMPTZ NOT NULL,
    kp_index DOUBLE PRECISION,
    dst_index DOUBLE PRECISION,
    imf_b_z DOUBLE PRECISION,
    earth_impact_eta DOUBLE PRECISION,
    spectral_hardness_ratio DOUBLE PRECISION,
    risk VARCHAR,
    PRIMARY KEY (timestamp)
);
SELECT create_hypertable('space_weather_impact', 'timestamp', if_not_exists => TRUE);

-- 6. flare_events (Standard table, not hypertable)
CREATE TABLE IF NOT EXISTS flare_events (
    event_id VARCHAR PRIMARY KEY,
    peak_timestamp TIMESTAMPTZ NOT NULL,
    end_timestamp TIMESTAMPTZ NOT NULL,
    flare_class VARCHAR,
    confidence_score DOUBLE PRECISION,
    neupert_flag BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS ix_flare_events_flare_class ON flare_events (flare_class);

-- 7. geomagnetic_forecast (Standard table, not hypertable)
CREATE TABLE IF NOT EXISTS geomagnetic_forecast (
    id SERIAL PRIMARY KEY,
    forecast_target_time TIMESTAMPTZ NOT NULL,
    predicted_dst DOUBLE PRECISION,
    predicted_kp DOUBLE PRECISION,
    storm_category VARCHAR,
    l1_propagation_delay_mins DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS ix_geomagnetic_forecast_target_time ON geomagnetic_forecast (forecast_target_time);


-- ==========================================
-- Continuous Aggregates
-- ==========================================

-- Create a continuous aggregate for 1-minute time buckets on SoLEXS
CREATE MATERIALIZED VIEW IF NOT EXISTS solexs_1min_agg
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 minute', timestamp) AS bucket,
       AVG(soft_xray_flux) as avg_flux,
       MAX(soft_xray_flux) as max_flux
FROM solexs_telemetry
GROUP BY bucket;

-- Create a continuous aggregate for 1-minute time buckets on HEL1OS
CREATE MATERIALIZED VIEW IF NOT EXISTS hel1os_1min_agg
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 minute', timestamp) AS bucket,
       AVG(hard_xray_flux) as avg_flux,
       MAX(hard_xray_flux) as max_flux
FROM hel1os_telemetry
GROUP BY bucket;
