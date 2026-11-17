"""
websocket.py — High-performance 10Hz WebSocket streaming for Aditya-L1 Telemetry.
"""
import asyncio
import json
import time
import math
from typing import List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._is_streaming = False

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        # Convert message to string once to save serialization overhead per client
        payload = json.dumps(message)
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception:
                dead_connections.append(connection)
        
        # Cleanup disconnected clients gracefully
        for conn in dead_connections:
            self.disconnect(conn)

manager = ConnectionManager()

async def telemetry_stream_loop():
    """Background task to simulate and broadcast telemetry at 10Hz."""
    manager._is_streaming = True
    
    # We will simulate data drifting around base values for the demo
    base_solexs = 38.5
    base_hel1os = 18.2
    
    # Simulated flare state
    burst_phase = 0.0
    is_bursting = False
    
    while manager._is_streaming:
        try:
            # Sleep 100ms for 10Hz
            await asyncio.sleep(0.1)
            
            if not manager.active_connections:
                continue

            # Simulate random telemetry drift
            import random
            # Determine if a burst should happen (1% chance per tick if not already bursting)
            if not is_bursting and random.random() < 0.005:
                is_bursting = True
                burst_phase = 0.0
            
            # Burst profile
            if is_bursting:
                burst_phase += 0.02 # controls the length of the burst
                # Peak multiplier up to ~5-10x
                multiplier = 1.0 + (math.sin(burst_phase * math.pi) * 8.0 if burst_phase < 1.0 else 0)
                
                if burst_phase >= 1.0:
                    is_bursting = False
            else:
                multiplier = 1.0
            
            current_solexs = base_solexs * multiplier + random.uniform(-1, 1)
            current_hel1os = base_hel1os * (multiplier * 1.5 if multiplier > 1 else 1) + random.uniform(-0.5, 0.5)
            
            # MAG/ASPEX Simulation
            base_bz = 5.0
            bz_val = random.uniform(-20, -5) if is_bursting else base_bz + random.uniform(-2, 2)
            btotal_val = abs(bz_val) + random.uniform(2, 5)
            vel_val = random.uniform(600, 900) if is_bursting else 400.0 + random.uniform(-20, 20)
            density_val = random.uniform(10, 30) if is_bursting else random.uniform(2, 10)
            pressure_val = (1.6726e-6) * density_val * (vel_val ** 2)
            
            # Predictions
            e_y = max(-1 * vel_val * bz_val * 1e-3, 0.0)
            est_dst = 7.26 * math.sqrt(max(pressure_val, 0)) - 15.0 * e_y
            est_kp = min(max(1.0 + (0.8 * e_y) + (0.2 * math.sqrt(max(pressure_val, 0))), 0.0), 9.0)
            
            if est_kp >= 9: scale = "G5 Extreme"
            elif est_kp >= 8: scale = "G4 Severe"
            elif est_kp >= 7: scale = "G3 Strong"
            elif est_kp >= 6: scale = "G2 Moderate"
            elif est_kp >= 5: scale = "G1 Minor"
            else: scale = "Quiet"
            
            delay = (1_500_000 / vel_val) / 60.0
            
            from src.nowcast.classify import classify_peak
            from datetime import datetime, timezone
            
            telemetry_frame = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "solexs": { "flux": round(current_solexs * 1e-6, 7), "class": classify_peak(current_solexs) },
                "hel1os": { "flux": round(current_hel1os * 1e-6, 7) },
                "mag": { "b_z": round(bz_val, 2), "b_total": round(btotal_val, 2), "is_southward": bz_val < 0 },
                "aspex": { "velocity": round(vel_val, 2), "density": round(density_val, 2), "pressure": round(pressure_val, 2) },
                "geomagnetic_forecast": {
                    "predicted_dst": round(est_dst, 2),
                    "predicted_kp": round(est_kp, 2),
                    "storm_scale": scale,
                    "eta_earth_mins": round(delay, 2)
                }
            }
            
            await manager.broadcast(telemetry_frame)

        except asyncio.CancelledError:
            manager._is_streaming = False
            break
        except Exception as e:
            print(f"WS Streaming error: {e}")

@router.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # We just keep the connection open and respond to pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
