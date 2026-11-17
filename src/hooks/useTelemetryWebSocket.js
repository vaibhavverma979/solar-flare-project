import { useState, useEffect, useRef } from 'react';

export default function useTelemetryWebSocket(url, onMessageCallback) {
  const [isConnected, setIsConnected] = useState(false);
  const [latency, setLatency] = useState(0);
  
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const pingIntervalRef = useRef(null);
  const lastPingTimeRef = useRef(0);

  const connect = () => {
    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        console.log(`[WebSocket] Connected to ${url}`);
        
        // Setup ping interval to measure latency
        pingIntervalRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            lastPingTimeRef.current = performance.now();
            ws.send("ping");
          }
        }, 2000); // Ping every 2 seconds
      };

      ws.onmessage = (event) => {
        if (event.data === "pong") {
          const currentLatency = performance.now() - lastPingTimeRef.current;
          setLatency(Math.round(currentLatency));
          return;
        }

        try {
          const data = JSON.parse(event.data);
          if (onMessageCallback) {
            // Callback handles state directly to avoid generic hook re-renders
            onMessageCallback(data);
          }
        } catch (e) {
          console.warn("[WebSocket] Failed to parse message:", e);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        console.warn(`[WebSocket] Disconnected from ${url}. Reconnecting in 3s...`);
        cleanup();
        // Fallback reconnection logic
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, 3000);
      };

      ws.onerror = (err) => {
        console.error("[WebSocket] Error:", err);
        ws.close();
      };
    } catch (e) {
      console.error("[WebSocket] Connection attempt failed:", e);
    }
  };

  const cleanup = () => {
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
    }
  };

  useEffect(() => {
    connect();

    return () => {
      cleanup();
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [url]); // Reconnect only if URL changes

  return { isConnected, latency };
}
