"""
ws_stream.py — WebSocket endpoint for simulated real-time light-curve playback (Phase 5).

Design rationale
----------------
The dashboard needs to demo real-time streaming behavior during the hackathon
even though we are working with archival data.  This WebSocket replays
historical light-curve data at a configurable playback speed (1× = real time,
10× = 10 seconds of data per 1 second of wall time).

Messages are sent as JSON with the same schema as the REST light-curve endpoint
so the frontend can use the same parser for both historical and live data.

TODO (Phase 5): Implement the /ws/stream WebSocket endpoint.
"""
from fastapi import WebSocket


async def ws_stream_handler(websocket: WebSocket):
    """
    Accept a WebSocket connection and stream historical light-curve data
    at the configured playback speed.
    TODO (Phase 5)
    """
    await websocket.accept()
    await websocket.send_json({"status": "not_implemented", "phase": 5})
    await websocket.close()
