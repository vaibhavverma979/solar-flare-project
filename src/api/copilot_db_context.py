from datetime import datetime, timezone, timedelta
from sqlalchemy import select, desc
from src.database.connection import AsyncSessionLocal
from src.database.models import (
    SolexsTelemetry,
    Hel1osTelemetry,
    MAGTelemetry,
    ASPEXTelemetry,
    FlareEvent,
    GeomagneticForecast,
    SpaceWeatherImpact
)

async def get_live_space_weather_context() -> tuple[str, str]:
    """
    Fetches the live context from the database and returns a markdown string and the snapshot timestamp.
    """
    snapshot_timestamp = datetime.now(timezone.utc).isoformat()
    
    async with AsyncSessionLocal() as session:
        # SoLEXS
        solexs_stmt = select(SolexsTelemetry).order_by(desc(SolexsTelemetry.timestamp)).limit(1)
        solexs_res = (await session.execute(solexs_stmt)).scalar_one_or_none()
        
        # HEL1OS
        hel1os_stmt = select(Hel1osTelemetry).order_by(desc(Hel1osTelemetry.timestamp)).limit(1)
        hel1os_res = (await session.execute(hel1os_stmt)).scalar_one_or_none()
        
        # MAG
        mag_stmt = select(MAGTelemetry).order_by(desc(MAGTelemetry.timestamp)).limit(1)
        mag_res = (await session.execute(mag_stmt)).scalar_one_or_none()
        
        # ASPEX
        aspex_stmt = select(ASPEXTelemetry).order_by(desc(ASPEXTelemetry.timestamp)).limit(1)
        aspex_res = (await session.execute(aspex_stmt)).scalar_one_or_none()
        
        # 2. Latest Nowcast Status
        two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
        flare_stmt = select(FlareEvent).where(FlareEvent.peak_timestamp >= two_hours_ago).order_by(desc(FlareEvent.peak_timestamp))
        flares_res = (await session.execute(flare_stmt)).scalars().all()
        
        # 3. Latest Forecast Predictions
        forecast_stmt = select(GeomagneticForecast).order_by(desc(GeomagneticForecast.forecast_target_time)).limit(1)
        forecast_res = (await session.execute(forecast_stmt)).scalar_one_or_none()
        
        # 4. Space Weather Impact Risks
        try:
            impact_stmt = select(SpaceWeatherImpact).order_by(desc(SpaceWeatherImpact.timestamp)).limit(1)
            impact_res = (await session.execute(impact_stmt)).scalar_one_or_none()
        except Exception as e:
            print(f"Warning: Could not fetch SpaceWeatherImpact data: {e}")
            impact_res = None
        
    # Formatting Context
    context = []
    
    context.append("### 1. LATEST TELEMETRY DATA")
    context.append(f"- **SoLEXS Soft X-ray flux**: {solexs_res.soft_xray_flux if solexs_res else 'N/A'}")
    context.append(f"- **SoLEXS SDD1 flux**: {solexs_res.sdd1_flux if solexs_res and solexs_res.sdd1_flux is not None else 'N/A'}")
    context.append(f"- **SoLEXS SDD2 flux**: {solexs_res.sdd2_flux if solexs_res and solexs_res.sdd2_flux is not None else 'N/A'}")
    context.append(f"- **HEL1OS Hard X-ray flux**: {hel1os_res.hard_xray_flux if hel1os_res else 'N/A'}")
    context.append(f"- **MAG IMF (Bz / Total B)**: {mag_res.b_z if mag_res else 'N/A'} / {mag_res.b_total if mag_res else 'N/A'}")
    context.append(f"- **ASPEX (Velocity / Density)**: {aspex_res.wind_velocity if aspex_res else 'N/A'} / {aspex_res.proton_density if aspex_res else 'N/A'}")
    
    context.append("\n### 2. NOWCAST STATUS (Active/Recent Flares - Last 2 Hours)")
    if flares_res:
        for flare in flares_res:
            context.append(f"- Flare ID: {flare.event_id} | Class: {flare.flare_class} | Peak: {flare.peak_timestamp} | Confidence: {flare.confidence_score} | Neupert: {flare.neupert_flag}")
    else:
        context.append("- No active flares detected in the last 2 hours.")
        
    context.append("\n### 3. FORECAST PREDICTIONS")
    if forecast_res:
        context.append(f"- **Predicted Dst Index**: {forecast_res.predicted_dst}")
        context.append(f"- **Predicted Kp Index**: {forecast_res.predicted_kp}")
        context.append(f"- **Earth ETA Delay (mins)**: {forecast_res.l1_propagation_delay_mins}")
        context.append(f"- **Storm Category**: {forecast_res.storm_category}")
    else:
        context.append("- No forecast predictions available.")
        
    context.append("\n### 4. SPACE WEATHER IMPACT / RISK")
    if impact_res:
        context.append(f"- **Kp Index**: {impact_res.kp_index}")
        context.append(f"- **Dst Index**: {impact_res.dst_index}")
        context.append(f"- **IMF Bz**: {impact_res.imf_b_z}")
        context.append(f"- **Spectral Hardness Ratio**: {impact_res.spectral_hardness_ratio}")
        context.append(f"- **Earth Impact ETA (hrs/mins)**: {impact_res.earth_impact_eta}")
        context.append(f"- **Risk Level**: {impact_res.risk}")
    else:
        context.append("- No impact risk data available.")
        
    return "\n".join(context), snapshot_timestamp
