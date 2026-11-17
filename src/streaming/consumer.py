import asyncio
from datetime import datetime
from src.streaming.redis_client import redis_client
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


def _parse_bool(value: str) -> bool:
    """Parse a string boolean value (True/False) from Redis."""
    return value.strip().lower() in ("true", "1", "yes")


async def process_batch(stream_name: str, messages: list):
    async with AsyncSessionLocal() as session:
        for msg_id, payload in messages:
            try:
                ts_raw = payload.get("timestamp")
                ts = datetime.fromisoformat(ts_raw) if ts_raw else None

                if stream_name == "aditya.telemetry.solexs":
                    obj = SolexsTelemetry(
                        timestamp=ts,
                        soft_xray_flux=float(payload["soft_xray_flux"]),
                        quality_flag=int(payload.get("quality_flag", 0)),
                        sdd1_flux=float(payload["sdd1_flux"]) if "sdd1_flux" in payload else None,
                        sdd2_flux=float(payload["sdd2_flux"]) if "sdd2_flux" in payload else None,
                    )

                elif stream_name == "aditya.telemetry.hel1os":
                    obj = Hel1osTelemetry(
                        timestamp=ts,
                        hard_xray_flux=float(payload["hard_xray_flux"]),
                        cdte1_counts=float(payload["cdte1_counts"]) if "cdte1_counts" in payload else None,
                        cdte2_counts=float(payload["cdte2_counts"]) if "cdte2_counts" in payload else None,
                    )

                elif stream_name == "aditya.telemetry.mag":
                    obj = MAGTelemetry(
                        timestamp=ts,
                        b_x=float(payload["b_x"]) if "b_x" in payload else None,
                        b_y=float(payload["b_y"]) if "b_y" in payload else None,
                        b_z=float(payload["b_z"]),
                        b_total=float(payload["b_total"]),
                    )

                elif stream_name == "aditya.telemetry.aspex":
                    obj = ASPEXTelemetry(
                        timestamp=ts,
                        wind_velocity=float(payload["wind_velocity"]),
                        proton_density=float(payload["proton_density"]),
                        dynamic_pressure=float(payload["dynamic_pressure"]),
                    )

                elif stream_name == "aditya.telemetry.space_weather_impact":
                    obj = SpaceWeatherImpact(
                        timestamp=ts,
                        kp_index=float(payload["kp_index"]) if "kp_index" in payload else None,
                        dst_index=float(payload["dst_index"]) if "dst_index" in payload else None,
                        imf_b_z=float(payload["imf_b_z"]) if "imf_b_z" in payload else None,
                        earth_impact_eta=float(payload["earth_impact_eta"]) if "earth_impact_eta" in payload else None,
                        spectral_hardness_ratio=float(payload["spectral_hardness_ratio"]) if "spectral_hardness_ratio" in payload else None,
                        risk=payload.get("risk"),
                    )

                elif stream_name == "aditya.telemetry.flare_events":
                    peak_ts = datetime.fromisoformat(payload["peak_timestamp"])
                    end_ts  = datetime.fromisoformat(payload["end_timestamp"])
                    obj = FlareEvent(
                        event_id=payload["event_id"],
                        peak_timestamp=peak_ts,
                        end_timestamp=end_ts,
                        flare_class=payload.get("flare_class"),
                        confidence_score=float(payload["confidence_score"]) if "confidence_score" in payload else None,
                        neupert_flag=_parse_bool(payload.get("neupert_flag", "False")),
                    )

                elif stream_name == "aditya.telemetry.geomagnetic_forecast":
                    forecast_ts = datetime.fromisoformat(payload["forecast_target_time"])
                    obj = GeomagneticForecast(
                        forecast_target_time=forecast_ts,
                        predicted_dst=float(payload["predicted_dst"]) if "predicted_dst" in payload else None,
                        predicted_kp=float(payload["predicted_kp"]) if "predicted_kp" in payload else None,
                        storm_category=payload.get("storm_category"),
                        l1_propagation_delay_mins=float(payload["l1_propagation_delay_mins"]) if "l1_propagation_delay_mins" in payload else None,
                    )

                else:
                    print(f"Unknown stream: {stream_name}, skipping.")
                    continue

                session.add(obj)

            except Exception as e:
                print(f"⚠️ Failed to parse payload from {stream_name}: {payload} — Error: {e}")

        try:
            await session.commit()
            print(f"✅ Inserted {len(messages)} records into DB for stream: {stream_name}")
        except Exception as e:
            print(f"❌ DB commit failed for {stream_name}: {e}")
            await session.rollback()


async def consume_telemetry():
    """
    Consumes telemetry from Redis Streams and inserts into TimescaleDB.
    Handles all 7 data streams:
      solexs, hel1os, mag, aspex, space_weather_impact, flare_events, geomagnetic_forecast
    """
    print("Starting Streaming Consumer (all streams)...")

    last_id = {
        "aditya.telemetry.solexs": "$",
        "aditya.telemetry.hel1os": "$",
        "aditya.telemetry.mag": "$",
        "aditya.telemetry.aspex": "$",
        "aditya.telemetry.space_weather_impact": "$",
        "aditya.telemetry.flare_events": "$",
        "aditya.telemetry.geomagnetic_forecast": "$",
    }

    while True:
        try:
            if not redis_client.redis:
                await asyncio.sleep(5)
                continue

            results = await redis_client.redis.xread(last_id, count=100, block=1000)

            if not results:
                continue

            for stream, messages in results:
                if messages:
                    await process_batch(stream, messages)
                    last_id[stream] = messages[-1][0]

        except asyncio.CancelledError:
            print("Consumer task cancelled.")
            break
        except Exception as e:
            print(f"Consumer error: {e}")
            await asyncio.sleep(1)
