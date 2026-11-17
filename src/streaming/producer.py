import asyncio
import random
import time
import uuid
from datetime import datetime, timezone, timedelta
from src.streaming.redis_client import redis_client


async def publish_telemetry(stream_name: str, payload: dict):
    """
    Publishes telemetry payload to the specified Redis stream.
    Ensures all dictionary values are strings before publishing.
    """
    if redis_client.redis:
        str_payload = {k: str(v) for k, v in payload.items()}
        await redis_client.redis.xadd(stream_name, str_payload)


async def telemetry_producer():
    """
    Simulates high-frequency ingestion of telemetry data from SoLEXS, HEL1OS,
    MAG, ASPEX, SpaceWeatherImpact, FlareEvent, and GeomagneticForecast
    by publishing to Redis Streams.
    """
    print("Starting Streaming Producer...")
    base_solexs = 38.5
    base_hel1os = 18.2
    base_bz = 5.0
    base_vel = 400.0

    # Track how many ticks have elapsed to reduce write frequency on slower tables
    tick = 0

    while True:
        try:
            if not redis_client.redis:
                await asyncio.sleep(5)
                continue

            # Simulate 10Hz stream
            await asyncio.sleep(0.1)
            tick += 1

            now_str = datetime.now(timezone.utc).isoformat()

            # Simulated drift
            solexs_val = base_solexs + random.uniform(-1, 1)
            hel1os_val = base_hel1os + random.uniform(-0.5, 0.5)

            # 5% chance of Southward IMF spike
            is_southward = random.random() < 0.05
            bz_val = random.uniform(-20, -5) if is_southward else base_bz + random.uniform(-2, 2)
            btotal_val = abs(bz_val) + random.uniform(2, 5)

            # Simulated velocity spike
            vel_val = random.uniform(600, 900) if is_southward else base_vel + random.uniform(-20, 20)
            density_val = random.uniform(10, 30) if is_southward else random.uniform(2, 10)
            pressure_val = (1.6726e-6) * density_val * (vel_val ** 2)

            # ── SoLEXS stream ─────────────────────────────────────────────
            await publish_telemetry("aditya.telemetry.solexs", {
                "timestamp": now_str,
                "soft_xray_flux": str(solexs_val),
                "quality_flag": "0",
            })

            # ── HEL1OS stream ─────────────────────────────────────────────
            await publish_telemetry("aditya.telemetry.hel1os", {
                "timestamp": now_str,
                "hard_xray_flux": str(hel1os_val),
            })

            # ── MAG stream ────────────────────────────────────────────────
            await publish_telemetry("aditya.telemetry.mag", {
                "timestamp": now_str,
                "b_z": str(bz_val),
                "b_total": str(btotal_val),
            })

            # ── ASPEX stream ──────────────────────────────────────────────
            await publish_telemetry("aditya.telemetry.aspex", {
                "timestamp": now_str,
                "wind_velocity": str(vel_val),
                "proton_density": str(density_val),
                "dynamic_pressure": str(pressure_val),
            })

            # ── Space Weather Impact stream (every 10 ticks ≈ every 1s) ──
            if tick % 10 == 0:
                kp_val = random.uniform(1.0, 5.0)
                risk_val = (
                    "High Risk (G3+)" if kp_val >= 5 else
                    "Moderate Risk (G1)" if kp_val >= 3 else
                    "Low Risk"
                )
                await publish_telemetry("aditya.telemetry.space_weather_impact", {
                    "timestamp": now_str,
                    "kp_index": str(kp_val),
                    "dst_index": str(random.uniform(-50.0, -10.0)),
                    "imf_b_z": str(bz_val),
                    "earth_impact_eta": str(random.uniform(12.0, 48.0)),
                    "spectral_hardness_ratio": str(random.uniform(0.5, 1.5)),
                    "risk": risk_val,
                })

            # ── Flare Event stream (every 100 ticks ≈ every 10s) ─────────
            if tick % 100 == 0:
                flare_classes = ["A", "B", "C", "M", "X"]
                weights = [40, 30, 20, 8, 2]
                fclass = random.choices(flare_classes, weights=weights)[0]
                dt_now = datetime.now(timezone.utc)
                peak_ts = (dt_now - timedelta(minutes=random.uniform(1, 30))).isoformat()
                end_ts  = (dt_now + timedelta(minutes=random.uniform(5, 20))).isoformat()
                await publish_telemetry("aditya.telemetry.flare_events", {
                    "event_id": f"FL-{dt_now.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}",
                    "peak_timestamp": peak_ts,
                    "end_timestamp": end_ts,
                    "flare_class": fclass,
                    "confidence_score": str(round(random.uniform(0.6, 0.99), 3)),
                    "neupert_flag": str(random.random() < 0.2),
                })

            # ── Geomagnetic Forecast stream (every 100 ticks ≈ every 10s) ─
            if tick % 100 == 0:
                dt_now = datetime.now(timezone.utc)
                forecast_t = (dt_now + timedelta(hours=random.uniform(1, 24))).isoformat()
                pred_kp = round(random.uniform(0.5, 6.0), 2)
                storm_cat = (
                    "Severe (G4)" if pred_kp >= 8 else
                    "Strong (G3)" if pred_kp >= 7 else
                    "Moderate (G2)" if pred_kp >= 6 else
                    "Minor (G1)" if pred_kp >= 5 else
                    "Quiet"
                )
                await publish_telemetry("aditya.telemetry.geomagnetic_forecast", {
                    "forecast_target_time": forecast_t,
                    "predicted_dst": str(round(random.uniform(-80.0, -10.0), 2)),
                    "predicted_kp": str(pred_kp),
                    "storm_category": storm_cat,
                    "l1_propagation_delay_mins": str(round(random.uniform(30, 90), 1)),
                })

        except asyncio.CancelledError:
            print("Producer task cancelled.")
            break
        except Exception as e:
            print(f"Producer error: {e}")
            await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(telemetry_producer())
    except KeyboardInterrupt:
        print("\nProducer stopped.")
