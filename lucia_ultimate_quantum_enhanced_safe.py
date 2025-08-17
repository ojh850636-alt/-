#!/usr/bin/env python3
"""
Safe simulation wrapper for Lucia Ultimate Quantum Enhanced
- Lightweight FastAPI app exposing a /health and /simulate endpoint
- Does NOT perform hardware control, real MQTT/Kafka, Qiskit runs, or start background clients
- Loads API keys from .env (do NOT commit real keys into the repo)
Run: python lucia_ultimate_quantum_enhanced_safe.py
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from datetime import datetime, timezone

load_dotenv()

APP_NAME = "Lucia Enhanced (Safe Simulation)"
API_VERSION = "0.1"

app = FastAPI(title=APP_NAME, version=API_VERSION)

class SimRequest(BaseModel):
    mode: str = "quick"
    steps: int = 3

@app.get("/health")
async def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat(), "app": APP_NAME}

@app.post("/simulate")
async def simulate(req: SimRequest):
    # Simple deterministic simulation that mimics parts of the original
    steps = max(1, min(int(req.steps), 20))
    psi = 0.5
    ent = 0.0
    events = []
    for i in range(steps):
        psi = min(0.99, psi + 0.01 * (i+1))
        if psi > 0.9 and ent < 0.9:
            ent = round(0.85 + 0.01 * i, 3)
            events.append({"type": "entanglement", "psi": round(psi,3), "ent": ent})
        else:
            events.append({"type": "tick", "psi": round(psi,3)})
    return {"ok": True, "steps": steps, "psi": round(psi,3), "entanglement": ent, "events": events}

if __name__ == "__main__":
    port = int(os.getenv("SERVER_PORT", "8002"))
    print(f"Starting {APP_NAME} on http://127.0.0.1:{port}")
    uvicorn.run("lucia_ultimate_quantum_enhanced_safe:app", host="127.0.0.1", port=port, log_level="info")
