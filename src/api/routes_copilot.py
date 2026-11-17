import os
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from src.api.agent import get_agent
from src.api.report_generator import generate_advisory_pdf
from src.api.tools import get_current_telemetry, get_forecast_predictions
from src.api.copilot_db_context import get_live_space_weather_context

router = APIRouter(prefix="/api/copilot", tags=["AI Co-Pilot"])

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's input message")
    history: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Optional chat history in the format [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]")

class ChatResponse(BaseModel):
    response: str
    metadata: Optional[Dict[str, Any]] = None
    
@router.post("/chat", response_model=ChatResponse)
async def chat_with_copilot(
    request: ChatRequest,
    agent_executor: Any = Depends(get_agent)
):
    try:
        db_snapshot, snapshot_timestamp = await get_live_space_weather_context()
        
        strict_system_prompt = f"""You are the ISRO Aditya-L1 Mission Control Co-Pilot.
You MUST answer the scientist's question using ONLY the real-time database snapshot provided below.
DO NOT use assumed or generic space weather data. If the data is not in the snapshot or tool output, explicitly reply: "Data not available in the current telemetry stream."

=== LIVE ADITYA-L1 DATABASE SNAPSHOT ===
{db_snapshot}
========================================="""
        
        # Format chat history for LangGraph
        formatted_messages = []
        formatted_messages.append(("system", strict_system_prompt))
        
        for msg in request.history:
            if msg.get("role") == "user":
                formatted_messages.append(("user", msg.get("content", "")))
            elif msg.get("role") == "assistant":
                formatted_messages.append(("assistant", msg.get("content", "")))
        
        formatted_messages.append(("user", request.message))
                
        # Invoke agent
        result = await agent_executor.ainvoke({"messages": formatted_messages})
        
        # Extract the last message from the assistant
        final_message = result["messages"][-1].content
        
        return ChatResponse(
            response=final_message,
            metadata={"timestamp": snapshot_timestamp}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def cleanup_temp_file(path: str):
    if os.path.exists(path):
        os.remove(path)

@router.post("/generate-report")
async def generate_report(
    background_tasks: BackgroundTasks,
    agent_executor: Any = Depends(get_agent)
):
    try:
        # Trigger the LangGraph agent for an executive summary
        prompt = "Generate a brief executive summary of the current space weather conditions and forecasts. Use the provided tools."
        result = await agent_executor.ainvoke({
            "messages": [("user", prompt)]
        })
        summary_text = result["messages"][-1].content
        
        # Get raw telemetry and forecast using the tools directly
        telemetry_result = await get_current_telemetry.ainvoke({})
        forecast_result = await get_forecast_predictions.ainvoke({})
        
        # Depending on LangChain version, ainvoke might return stringified dicts
        telemetry = telemetry_result if isinstance(telemetry_result, dict) else {}
        forecast = forecast_result if isinstance(forecast_result, dict) else {}
        
        if isinstance(telemetry_result, str):
            import ast
            try:
                telemetry = ast.literal_eval(telemetry_result)
            except Exception:
                pass
                
        if isinstance(forecast_result, str):
            import ast
            try:
                forecast = ast.literal_eval(forecast_result)
            except Exception:
                pass
                
        # Generate the PDF
        pdf_path = await generate_advisory_pdf(summary_text, telemetry, forecast)
        
        # Register background task to clean up the temporary file after it is served
        background_tasks.add_task(cleanup_temp_file, pdf_path)
        
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename="ISRO_Advisory.pdf"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
