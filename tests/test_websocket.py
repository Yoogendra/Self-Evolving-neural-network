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
            
            # Listen for messages
            async for message in websocket:
                data = json.loads(message)
                print(f"Received: {data['type']}")
                
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
