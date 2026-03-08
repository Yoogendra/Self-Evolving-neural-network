import asyncio
import json
import os
import re
import sys
import uuid
from datetime import datetime
from typing import Dict, List, Any
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

from asyncio.subprocess import PIPE, STDOUT

from config import Config
from data.CIFAR10 import get_cifar10_loaders
from evolution.evolve import evolve
from evolution.dna_builder import build_model_from_dna
from utils.utils import set_seed, get_device, ensure_dir

app = FastAPI(title="SENN API Server", version="1.0.0")

# Enable CORS for frontend
cors_origins_str = os.getenv("CORS_ORIGINS", '["http://localhost:5173", "http://127.0.0.1:5173"]')
try:
    allow_origins = json.loads(cors_origins_str)
except json.JSONDecodeError:
    allow_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for evolution
class EvolutionState:
    def __init__(self):
        self.is_running = False
        self.current_generation = 0
        self.best_fitness = 0.0
        self.active_species = 1
        self.fitness_history = []
        self.complexity_history = []
        self.events = []
        self.clients: List[WebSocket] = []
        self.evolution_task = None
        self.evolution_process = None

state = EvolutionState()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Connection closed, remove it
                self.active_connections.remove(connection)

manager = ConnectionManager()

@app.get("/")
async def get():
    return {"message": "SENN API Server is running"}

@app.get("/status")
async def get_status():
    return {
        "is_running": state.is_running,
        "generation": state.current_generation,
        "best_fitness": state.best_fitness,
        "active_species": state.active_species,
        "fitness_history": state.fitness_history,
        "complexity_history": state.complexity_history,
        "events": state.events[-200:]  # Last 200 events
    }

@app.post("/start")
async def start_evolution():
    if state.is_running:
        return {"message": "Evolution already running"}
    
    state.is_running = True
    state.evolution_task = asyncio.create_task(run_evolution())
    return {"message": "Evolution started"}

@app.post("/pause")
async def pause_evolution():
    state.is_running = False
    if state.evolution_process and state.evolution_process.returncode is None:
        state.evolution_process.terminate()
        try:
            await asyncio.wait_for(state.evolution_process.wait(), timeout=5)
        except asyncio.TimeoutError:
            state.evolution_process.kill()
    if state.evolution_task:
        state.evolution_task.cancel()
    state.evolution_process = None
    return {"message": "Evolution paused"}

@app.post("/reset")
async def reset_evolution():
    state.is_running = False
    if state.evolution_process and state.evolution_process.returncode is None:
        state.evolution_process.terminate()
        try:
            await asyncio.wait_for(state.evolution_process.wait(), timeout=5)
        except asyncio.TimeoutError:
            state.evolution_process.kill()
    if state.evolution_task:
        state.evolution_task.cancel()
    state.evolution_process = None
    
    state.current_generation = 0
    state.best_fitness = 0.0
    state.active_species = 1
    state.fitness_history = []
    state.complexity_history = []
    state.events = []
    
    await manager.broadcast(json.dumps({
        "type": "reset",
        "data": {
            "generation": 0,
            "best_fitness": 0.0,
            "active_species": 1,
            "fitness_history": [],
            "complexity_history": [],
            "events": []
        }
    }))
    
    return {"message": "Evolution reset"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial state
        await websocket.send_text(json.dumps({
            "type": "initial_state",
            "data": {
                "generation": state.current_generation,
                "best_fitness": state.best_fitness,
                "active_species": state.active_species,
                "fitness_history": state.fitness_history,
                "complexity_history": state.complexity_history,
                "events": state.events[-200:],
                "is_running": state.is_running
            }
        }))
        
        while True:
            # Keep connection alive and handle incoming messages
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                elif message.get("type") == "start":
                    if not state.is_running:
                        state.is_running = True
                        state.evolution_task = asyncio.create_task(run_evolution())
                        await websocket.send_text(json.dumps({"type": "status", "data": {"message": "Evolution started"}}))
                    else:
                        await websocket.send_text(json.dumps({"type": "status", "data": {"message": "Evolution already running"}}))
                elif message.get("type") == "pause":
                    state.is_running = False
                    if state.evolution_process and state.evolution_process.returncode is None:
                        state.evolution_process.terminate()
                        try:
                            await asyncio.wait_for(state.evolution_process.wait(), timeout=5)
                        except asyncio.TimeoutError:
                            state.evolution_process.kill()
                        state.evolution_process = None
                    if state.evolution_task:
                        state.evolution_task.cancel()
                    await websocket.send_text(json.dumps({"type": "status", "data": {"message": "Evolution paused"}}))
                elif message.get("type") == "reset":
                    state.is_running = False
                    if state.evolution_process and state.evolution_process.returncode is None:
                        state.evolution_process.terminate()
                        try:
                            await asyncio.wait_for(state.evolution_process.wait(), timeout=5)
                        except asyncio.TimeoutError:
                            state.evolution_process.kill()
                        state.evolution_process = None
                    if state.evolution_task:
                        state.evolution_task.cancel()
                    state.current_generation = 0
                    state.best_fitness = 0.0
                    state.active_species = 1
                    state.fitness_history = []
                    state.complexity_history = []
                    state.events = []
                    await websocket.send_text(json.dumps({"type": "status", "data": {"message": "Evolution reset"}}))
                else:
                    await websocket.send_text(json.dumps({"type": "error", "data": {"message": f"Unknown message type: {message.get('type')}"}}))
                    
            except WebSocketDisconnect:
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)

async def run_evolution():
    """Run the actual SENN evolution algorithm"""
    try:
        cfg = Config()
        set_seed(cfg.seed)
        get_device(cfg.device)
        ensure_dir("outputs")

        def add_event(message: str):
            state.events.append({"timestamp": datetime.now().isoformat(), "message": message})

        add_event("[INIT] Starting SENN (running main.py)")
        await manager.broadcast(json.dumps({
            "type": "event",
            "data": {"message": "[INIT] Starting SENN (running main.py)"}
        }))

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "main.py",
            stdout=PIPE,
            stderr=STDOUT,
        )
        state.evolution_process = proc

        gen_header_re = re.compile(r"^===\s*Generation\s*(\d+)/(\d+)\s*===")
        model_line_re = re.compile(r"fit=(?P<fit>[-+]?\d*\.?\d+)")
        params_re = re.compile(r"params=(?P<params>\d+)")

        current_gen = 0
        current_gen_best_fit = None
        current_gen_best_params = None

        while True:
            if not state.is_running:
                break

            line = await proc.stdout.readline()
            if not line:
                break

            text = line.decode("utf-8", errors="replace").rstrip("\n")
            if not text:
                continue

            add_event(text)

            m_gen = gen_header_re.match(text.strip())
            if m_gen:
                if current_gen and current_gen_best_fit is not None:
                    state.fitness_history.append({
                        "generation": current_gen,
                        "fitness": float(current_gen_best_fit),
                    })
                    if current_gen_best_params is not None:
                        state.complexity_history.append({
                            "generation": current_gen,
                            "complexity": int(current_gen_best_params),
                        })

                current_gen = int(m_gen.group(1))
                state.current_generation = current_gen
                current_gen_best_fit = None
                current_gen_best_params = None

                await manager.broadcast(json.dumps({
                    "type": "update",
                    "data": {
                        "generation": state.current_generation,
                        "best_fitness": state.best_fitness,
                        "active_species": state.active_species,
                        "complexity": state.complexity_history[-1]["complexity"] if state.complexity_history else 0,
                        "event": text,
                    }
                }))
                continue

            m_fit = model_line_re.search(text)
            if m_fit:
                fit_val = float(m_fit.group("fit"))
                if current_gen_best_fit is None or fit_val > current_gen_best_fit:
                    current_gen_best_fit = fit_val
                    m_params = params_re.search(text)
                    if m_params:
                        current_gen_best_params = int(m_params.group("params"))

                    if fit_val > float(state.best_fitness or 0.0):
                        state.best_fitness = float(fit_val)

            if len(state.events) > 500:
                state.events = state.events[-500:]

        if proc.returncode is None:
            try:
                await asyncio.wait_for(proc.wait(), timeout=1)
            except asyncio.TimeoutError:
                proc.terminate()

        if current_gen and current_gen_best_fit is not None:
            state.fitness_history.append({
                "generation": current_gen,
                "fitness": float(current_gen_best_fit),
            })
            if current_gen_best_params is not None:
                state.complexity_history.append({
                    "generation": current_gen,
                    "complexity": int(current_gen_best_params),
                })

        state.is_running = False
        state.evolution_process = None
        complete_msg = "[COMPLETE] main.py finished"
        add_event(complete_msg)
        await manager.broadcast(json.dumps({
            "type": "complete",
            "data": {"message": complete_msg}
        }))
    
    except Exception as e:
        error_event = f"[ERROR] Evolution failed: {str(e)}"
        state.events.append({"timestamp": datetime.now().isoformat(), "message": error_event})
        
        await manager.broadcast(json.dumps({
            "type": "error", 
            "data": {"message": error_event}
        }))
        
    finally:
        state.is_running = False

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
