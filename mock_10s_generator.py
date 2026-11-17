import asyncio
import random
import uuid
from datetime import datetime, timezone, timedelta
from src.database.connection import AsyncSessionLocal
from src.database.models import (
    SolexsTelemetry,
    Hel1osTelemetry,
    MAGTelemetry,
    ASPEXTelemetry,
    SpaceWeatherImpact,
    FlareEvent,
    GeomagneticForecast,
)


async def generate_mock_data():
    print("Starting 10-second interval mock data generator for TimescaleDB...")
    while True:
        try:
            async with AsyncSessionLocal() as session:
                now = datetime.now(timezone.utc)

                # ── SoLEXS ────────────────────────────────────────────────
                solexs_flux = 38.5 + random.uniform(-2, 2)
                sdd1 = solexs_flux * 0.4
                sdd2 = solexs_flux * 0.6
                solexs = SolexsTelemetry(
                    timestamp=now, soft_xray_flux=solexs_flux, quality_flag=0,
                    sdd1_flux=sdd1, sdd2_flux=sdd2
                )

                # ── HEL1OS ────────────────────────────────────────────────
                hel1os_flux = 18.2 + random.uniform(-1, 1)
                hel1os = Hel1osTelemetry(
                    timestamp=now, hard_xray_flux=hel1os_flux,
                    cdte1_counts=hel1os_flux * 10, cdte2_counts=hel1os_flux * 8
                )

                # ── MAG ───────────────────────────────────────────────────
                bz = 5.0 + random.uniform(-2, 2)
                mag = MAGTelemetry(
                    timestamp=now, b_x=random.uniform(-1, 1), b_y=random.uniform(-1, 1),
                    b_z=bz, b_total=abs(bz) + random.uniform(2, 5)
                )

                # ── ASPEX ─────────────────────────────────────────────────
                vel = 400.0 + random.uniform(-20, 20)
                den = random.uniform(2, 10)
                aspex = ASPEXTelemetry(
                    timestamp=now, wind_velocity=vel, proton_density=den,
                    dynamic_pressure=(1.6726e-6) * den * (vel ** 2)
                )

                # ── Space Weather Impact ───────────────────────────────────
                kp = random.uniform(1.0, 5.0)
                risk_choice = (
                    "High Risk (G3+)" if kp >= 5 else
                    "Moderate Risk (G1)" if kp >= 3 else
                    "Low Risk"
                )
                impact = SpaceWeatherImpact(
                    timestamp=now,
                    kp_index=kp,
                    dst_index=random.uniform(-50.0, -10.0),
                    imf_b_z=bz,
                    earth_impact_eta=random.uniform(12.0, 48.0),
                    spectral_hardness_ratio=random.uniform(0.5, 1.5),
                    risk=risk_choice,
                )

                # ── Flare Event ───────────────────────────────────────────
                flare_classes = ["A", "B", "C", "M", "X"]
                weights = [40, 30, 20, 8, 2]
                fclass = random.choices(flare_classes, weights=weights)[0]
                peak_ts = now - timedelta(minutes=random.uniform(1, 30))
                end_ts  = peak_ts + timedelta(minutes=random.uniform(5, 20))
                flare_event = FlareEvent(
                    event_id=f"FL-{now.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}",
                    peak_timestamp=peak_ts,
                    end_timestamp=end_ts,
                    flare_class=fclass,
                    confidence_score=round(random.uniform(0.6, 0.99), 3),
                    neupert_flag=random.random() < 0.2,
                )

                # ── Geomagnetic Forecast ──────────────────────────────────
                forecast_t = now + timedelta(hours=random.uniform(1, 24))
                pred_kp = round(random.uniform(0.5, 6.0), 2)
                storm_cat = (
                    "Severe (G4)" if pred_kp >= 8 else
                    "Strong (G3)" if pred_kp >= 7 else
                    "Moderate (G2)" if pred_kp >= 6 else
                    "Minor (G1)" if pred_kp >= 5 else
                    "Quiet"
                )
                geo_forecast = GeomagneticForecast(
                    forecast_target_time=forecast_t,
                    predicted_dst=round(random.uniform(-80.0, -10.0), 2),
                    predicted_kp=pred_kp,
                    storm_category=storm_cat,
                    l1_propagation_delay_mins=round(random.uniform(30, 90), 1),
                )

                session.add_all([solexs, hel1os, mag, aspex, impact, flare_event, geo_forecast])

                try:
                    await session.commit()
                    print(f"[{now.isoformat()}] [OK] Inserted mock data into all 7 tables.")
                except Exception as commit_err:
                    await session.rollback()
                    print(f"[{now.isoformat()}] [FAIL] Commit failed: {commit_err}")

        except asyncio.CancelledError:
            print("Mock generator stopped.")
            break
        except Exception as e:
            print(f"Error inserting data: {e}")

        await asyncio.sleep(10)


if __name__ == "__main__":
    try:
        asyncio.run(generate_mock_data())
    except KeyboardInterrupt:
        print("\nExiting...")
