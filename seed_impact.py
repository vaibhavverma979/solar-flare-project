"""
seed_impact.py — Inserts one row of mock data into every table that was empty.
Run this after init_db.py to verify data insertion works end-to-end.

Usage:
    python seed_impact.py
"""
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from src.database.connection import AsyncSessionLocal
from src.database.models import (
    SpaceWeatherImpact,
    FlareEvent,
    GeomagneticForecast,
    MAGTelemetry,
    ASPEXTelemetry,
)


async def seed_data():
    async with AsyncSessionLocal() as session:
        now = datetime.now(timezone.utc)

        # 1. Space Weather Impact
        impact = SpaceWeatherImpact(
            timestamp=now,
            kp_index=4.5,
            dst_index=-50.2,
            imf_b_z=-7.1,
            earth_impact_eta=14.5,
            spectral_hardness_ratio=0.85,
            risk="Moderate Risk (G1)",
        )

        # 2. Flare Event
        flare_event = FlareEvent(
            event_id=f"FL-SEED-{uuid.uuid4().hex[:8].upper()}",
            peak_timestamp=now - timedelta(minutes=15),
            end_timestamp=now - timedelta(minutes=5),
            flare_class="M",
            confidence_score=0.92,
            neupert_flag=True,
        )

        # 3. Geomagnetic Forecast
        geo_forecast = GeomagneticForecast(
            forecast_target_time=now + timedelta(hours=6),
            predicted_dst=-42.5,
            predicted_kp=4.1,
            storm_category="Minor (G1)",
            l1_propagation_delay_mins=55.0,
        )

        # 4. MAG Telemetry (seed if empty)
        mag = MAGTelemetry(
            timestamp=now,
            b_x=-0.5,
            b_y=1.2,
            b_z=-7.1,
            b_total=7.4,
        )

        # 5. ASPEX Telemetry (seed if empty)
        aspex = ASPEXTelemetry(
            timestamp=now,
            proton_density=8.5,
            wind_velocity=430.0,
            dynamic_pressure=2.45,
        )

        session.add_all([impact, flare_event, geo_forecast, mag, aspex])

        try:
            await session.commit()
            print("[OK] Seed data inserted successfully into all 5 tables!")
            print(f"     space_weather_impact  @ {now.isoformat()}")
            print(f"     flare_events          - {flare_event.event_id}")
            print(f"     geomagnetic_forecast  @ {geo_forecast.forecast_target_time.isoformat()}")
            print(f"     mag_telemetry         @ {now.isoformat()}")
            print(f"     aspex_telemetry       @ {now.isoformat()}")
        except Exception as e:
            await session.rollback()
            print(f"[FAIL] Seed failed: {e}")


if __name__ == "__main__":
    asyncio.run(seed_data())
