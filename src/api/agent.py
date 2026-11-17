from functools import lru_cache
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from config import settings
from src.api.tools import (
    get_nowcast_status,
    get_forecast_predictions,
    get_current_telemetry
)
from src.api.copilot_tools import (
    query_historical_flare_events,
    get_lightcurve_timeseries
)


SYSTEM_PROMPT = """You are the ISRO Aditya-L1 Space Weather Co-Pilot. You assist scientists by analyzing real-time solar X-ray flux, nowcast events, and predictive forecasts. Always use your tools to fetch real data before answering."""

@lru_cache(maxsize=1)
def _get_agent_executor():
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment or config.")
        
    llm = ChatOpenAI(
        api_key=settings.openai_api_key, 
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4o-mini",
        temperature=0.0
    )
    
    tools = [
        get_nowcast_status,
        get_forecast_predictions,
        get_current_telemetry,
        query_historical_flare_events,
        get_lightcurve_timeseries
    ]
    
    agent_executor = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    return agent_executor

def get_agent():
    """FastAPI dependency to get the LangGraph agent."""
    return _get_agent_executor()
