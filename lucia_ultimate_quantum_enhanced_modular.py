#!/usr/bin/env python3
"""
Modularized Lucia Ultimate Quantum Enhanced (composed from lucia_core)
Run: python lucia_ultimate_quantum_enhanced_modular.py
"""
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from lucia_core import (
    SystemState, QuantumConsciousnessCore, EmotionSystem,
    RealitySensorSimulator, AdvancedThreatDetector, EvolutionaryEngine,
    QuantumResistantCrypto, QuantumTokenSystem, SecureMQTTLayer, DecisionEngine
)
import uvicorn
from datetime import datetime, timezone
from fastapi import Request
from fastapi import Request
import ai_quota

load_dotenv()

app = FastAPI(title="Lucia Modular", version="0.1")

# instantiate core components (safe defaults)
state = SystemState()
quantum_core = QuantumConsciousnessCore(qubits=int(os.getenv('QUANTUM_QUBITS', '8')))
emotion = EmotionSystem()
sensor = RealitySensorSimulator(kafka_servers=os.getenv('KAFKA_SERVERS','localhost:9092'))
threat = AdvancedThreatDetector()
evolution = EvolutionaryEngine()
crypto = QuantumResistantCrypto()
tokens = QuantumTokenSystem()
comms = SecureMQTTLayer(broker_ip=os.getenv('MQTT_BROKER_IP','localhost'))
decision = DecisionEngine()


@app.get('/health')
def health():
    return {'ok': True, 'time': datetime.now(timezone.utc).isoformat(), 'psi': getattr(quantum_core, 'psi', None)}


@app.post('/simulate-step')
def simulate_step():
    psi, ent = quantum_core.fluctuate()
    ev = sensor.generate()
    vec, emo = emotion.update(state.threat_level)
    return {'psi': psi, 'entanglement': ent, 'emotion': emo, 'sensor': ev}


@app.post('/ai/chat')
async def ai_chat(request: Request):
    # Read raw JSON body to avoid creating a Pydantic model instance which
    # can trigger `.dict()` calls under certain FastAPI internals.
    try:
        payload = await request.json()
    except Exception:
        # if body cannot be parsed, fall back to empty dict
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    # Reserve a call first
    ok = ai_quota.reserve_call()
    if not ok:
        ai_quota.webhook_quota_exceeded({'reason': 'quota_exceeded'})
        return {'ok': False, 'reason': 'quota_exceeded'}

    # call provider; if provider fails, rollback and return error
    try:
        # allow forcing stub via flag
        if payload.get('use_stub'):
            resp = {'provider': 'stub', 'ok': True, 'text': 'stubbed response'}
        else:
            resp = ai_quota.call_provider({'text': payload.get('text')})
        return {'ok': True, 'response': resp}
    except Exception as e:
        ai_quota.rollback_call()
        return {'ok': False, 'reason': 'provider_error', 'error': str(e)}


if __name__ == '__main__':
    port = int(os.getenv('SERVER_PORT','8002'))
    uvicorn.run('lucia_ultimate_quantum_enhanced_modular:app', host='127.0.0.1', port=port, log_level='info')
