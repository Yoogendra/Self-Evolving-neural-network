#!/usr/bin/env python3
import asyncio
import websockets
import json

async def test_websocket():
    try:
        uri = "ws://localhost:8000/ws"
        print(f"Connecting to {uri}...")
        
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket!")
            
            await websocket.send(json.dumps({"type": "ping"}))
            
            response = await websocket.recv()
            print(f"Received: {response}")
                
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
