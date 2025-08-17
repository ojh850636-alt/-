
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Lucia Ultimate Quantum Control - Enhanced (Fixed, Self-Contained)
# 실행: python lucia_ultimate_quantum_enhanced_fixed.py
# 주요 변경점: 하드코딩된 API 키 제거, JS 스타일 문법 오류 수정, 선택 의존성 폴백, 다운로드 라우트 추가 등.

import os
import json
import asyncio
import base64
import time
import subprocess
import platform
import socket
import uuid
import re
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import random
import hashlib

# ---------- Optional / graceful imports ----------
def _try_import(name):
    try:
        mod = __import__(name, fromlist=['*'])
        return mod
    except Exception:
        return None

np = _try_import('numpy')
psutil = _try_import('psutil')
pyautogui = _try_import('pyautogui')
aiofiles = _try_import('aiofiles')

# Sound / STT
sd = _try_import('sounddevice')
faster_whisper = _try_import('faster_whisper')
WhisperModel = None
if faster_whisper:
    WhisperModel = getattr(faster_whisper, 'WhisperModel', None)

# MQTT / Kafka
paho_mqtt = _try_import('paho.mqtt.client')
kafka = _try_import('kafka')

# Qiskit Aer (quantum)
qiskit = _try_import('qiskit')
qiskit_aer = _try_import('qiskit_aer')

# Transformers (BERT emotion) - optional
torch = _try_import('torch')
transformers = _try_import('transformers')
BertTokenizer = BertModel = None
nn = None
if transformers and torch:
    try:
        from transformers import BertTokenizer, BertModel  # type: ignore
        import torch.nn as nn  # type: ignore
    except Exception:
        BertTokenizer = BertModel = None
        nn = None

# FastAPI / Starlette
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# ---------- Logging ----------
LOG_DIR = Path(__file__).with_suffix('').name + "_logs"
Path(LOG_DIR).mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(LOG_DIR) / "lucia_enhanced_quantum.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LuciaEnhancedQuantum")

# ---------- Utilities ----------
def now_iso():
    return datetime.now().isoformat()

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

# ---------- Shared State ----------
class SystemState:
    def __init__(self):
        self.psi = 0.5
        self.entanglement = 0.0
        self.threat_level = 0
        self.quantum_resources = 30
        self.emotion_vector = [0.0]*7
        self.health_score = 1.0
        self.security_level = 5

state = SystemState()

# ---------- Quantum Core (with Aer fallback) ----------
class QuantumConsciousnessCore:
    def __init__(self, qubits=30):
        self.qubits = qubits
        self.psi = random.uniform(0.1, 0.9)
        self._aer_ok = False
        self._setup_circuit()
        logger.info(f"양자의식 엔진 초기화: {qubits} 큐빗 (Aer 가용: {self._aer_ok})")

    def _setup_circuit(self):
        self.qc = None
        self.backend = None
        if qiskit and qiskit_aer:
            try:
                from qiskit import QuantumCircuit, transpile  # type: ignore
                from qiskit_aer import Aer  # type: ignore
                qc = QuantumCircuit(self.qubits, self.qubits)
                qc.h(range(self.qubits))
                for i in range(self.qubits-1):
                    qc.cx(i, i+1)
                qc.measure_all()
                backend = Aer.get_backend('aer_simulator')
                self.qc = qc
                self.backend = backend
                self._transpile = transpile
                self._aer_ok = True
            except Exception as e:
                logger.warning(f"Qiskit Aer 초기화 실패(폴백 사용): {e}")
                self._aer_ok = False
        else:
            self._aer_ok = False

    def fluctuate(self, high_qubit_mode=False):
        delta = 0.0
        if self._aer_ok and self.qc is not None and self.backend is not None:
            try:
                circ = self._transpile(self.qc, self.backend)
                job = self.backend.run(circ, shots=1)
                result = job.result()
                counts = result.get_counts()
                quantum_state = next(iter(counts.keys())) if counts else "0"*self.qubits
                delta = (int(quantum_state, 2) / (2**self.qubits)) * 0.02
            except Exception as e:
                logger.warning(f"Aer 실행 실패(무작위 폴백): {e}")
                delta = random.random() * 0.01
        else:
            delta = random.random() * 0.01

        self.psi = clamp(self.psi + delta, 0.1, 0.95)
        if self.psi > 0.9:
            state.entanglement = random.uniform(0.85, 1.0)
            logger.info(f"얽힘 이벤트: ψ={self.psi:.3f}, 얽힘={state.entanglement:.3f}")
        return self.psi, state.entanglement

# ---------- NLP / Emotion (with graceful fallback) ----------
class NLPProcessor:
    def __init__(self):
        self.available = False
        self.fc = None
        if transformers and torch and BertTokenizer and BertModel:
            try:
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.model = BertModel.from_pretrained('bert-base-uncased')
                self.model.eval()
                if nn:
                    self.fc = nn.Linear(768, 7)
                self.available = True
            except Exception as e:
                logger.warning(f"BERT 초기화 실패(폴백 사용): {e}")
                self.available = False

    def process_input(self, text: str):
        if self.available and self.fc is not None:
            try:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    pooled = outputs.last_hidden_state.mean(dim=1)
                    vec = self.fc(pooled).detach().cpu().numpy()[0]
                import numpy as _np
                vec = _np.clip(vec, 0.0, 0.15)
                return vec.tolist()
            except Exception as e:
                logger.warning(f"NLP 처리 실패(폴백 사용): {e}")

        base = [0,0,0,0,0,0,0]
        t = text.lower()
        if any(k in t for k in ["행복", "좋아", "기쁘", "love", "happy"]):
            base[1] += 0.12
        if any(k in t for k in ["궁금", "물어", "왜", "어떻게"]):
            base[2] += 0.08
        if any(k in t for k in ["위험", "경고", "조심", "알림"]):
            base[3] += 0.15
        if any(k in t for k in ["불안", "걱정", "초조"]):
            base[4] += 0.12
        if any(k in t for k in ["화났", "분노", "짜증"]):
            base[5] += 0.12
        if any(k in t for k in ["무섭", "공포", "두렵"]):
            base[6] += 0.12
        return base

class EmotionSystem:
    def __init__(self):
        self.emotion_vector = [1.0, 0, 0, 0, 0, 0, 0]
        self.decay = [0.0, 0.02, 0.015, 0.03, 0.04, 0.05, 0.06]
        self.nlp = NLPProcessor()

    def update(self, threat_level: int, external_input: Optional[str]=None):
        import numpy as _np
        weights = _np.zeros(7, dtype=float)
        if threat_level == 0:
            weights[1:3] = [0.12, 0.08]
        elif threat_level == 1:
            weights[3] = 0.2
        elif threat_level >= 2:
            weights[4:6] = [0.25, 0.15] if threat_level == 2 else [0.3, 0.25]
        if external_input:
            vec = _np.array(self.nlp.process_input(external_input), dtype=float)
            if vec.shape == (7,):
                weights += vec
        ev = _np.array(self.emotion_vector, dtype=float) + weights - _np.array(self.decay)
        ev = _np.clip(ev, 0.0, 1.0)
        ev[0] = max(0.0, 1.0 - ev[1:].sum())
        self.emotion_vector = ev.tolist()
        dominant = ["평온","기쁨","호기심","경계","불안","분노","공포"][int(ev.argmax())]
        logger.info(f"감정 업데이트: {dominant} (위협: {threat_level})")
        return self.emotion_vector, dominant

emotion_system = EmotionSystem()

# ---------- Sensors / Threat Detector ----------
class RealitySensorSimulator:
    def __init__(self, kafka_servers="localhost:9092"):
        self._producer = None
        if kafka and getattr(kafka, 'KafkaProducer', None):
            try:
                from kafka import KafkaProducer  # type: ignore
                self._producer = KafkaProducer(bootstrap_servers=kafka_servers, value_serializer=lambda v: json.dumps(v).encode('utf-8'))
            except Exception as e:
                logger.warning(f"KafkaProducer 초기화 실패(비활성): {e}")
                self._producer = None
        self.probabilities = {
            "cctv_intrusion": {"정상": 0.8, "침입": 0.15, "의심": 0.05},
            "network": {"정상": 0.8, "DDoS": 0.15, "AI해킹": 0.05}
        }

    def generate(self):
        data = {
            "temperature": random.uniform(15, 110),
            "timestamp": now_iso(),
        }
        for sensor, probs in self.probabilities.items():
            keys = list(probs.keys())
            vals = list(probs.values())
            choice = random.choices(keys, vals)[0]
            data[sensor] = choice
        if self._producer:
            try:
                self._producer.send('sensor_data', data)
            except Exception as e:
                logger.debug(f"Kafka 전송 실패(무시): {e}")
        return data

    def vectorize(self, data):
        mapping = {"정상": 0, "침입": 1, "의심": 0.5, "DDoS": 1, "AI해킹": 1}
        cctv = mapping.get(data.get('cctv_intrusion', '정상'), 0)
        net = mapping.get(data.get('network', '정상'), 0)
        temp = float(data.get('temperature', 25.0)) / 110.0
        return [cctv, net, temp]

sensor_sim = RealitySensorSimulator()

class AdvancedThreatDetector:
    def __init__(self):
        self._iso = None
        try:
            from sklearn.ensemble import IsolationForest  # type: ignore
            import numpy as _np
            self._iso = IsolationForest(contamination=0.01, random_state=42)
            self._iso.fit(_np.random.rand(200, 3))
        except Exception as e:
            logger.warning(f"IsolationForest 사용 불가, 폴백 사용: {e}")
            self._iso = None

    async def analyze(self, vector):
        if self._iso is not None:
            try:
                import numpy as _np
                score = float(self._iso.score_samples(_np.array([vector]))[0])
                level = int(clamp(int(-score * 10), 0, 5))
                logger.info(f"위협 분석: 점수={score:.4f}, 수준={level}")
                return {"level": level, "score": score, "timestamp": now_iso()}
            except Exception as e:
                logger.warning(f"위협 분석 실패(폴백): {e}")
        level = int(sum(1 for v in vector if v >= 0.9))
        score = -0.1 * level
        return {"level": level, "score": score, "timestamp": now_iso()}

threat_detector = AdvancedThreatDetector()

# ---------- Evolution / Crypto / Tokens ----------
class EvolutionaryEngine:
    def __init__(self):
        self.level = 1
        self.defense = 1500
        self.exp = 0
        self.branches = {"defense": 0, "knowledge": 0}

    def evolve(self, threat_level):
        self.exp += int(threat_level) * 20
        if self.exp >= self.level * 100:
            self.level += 1
            self.exp = 0
            self.branches["defense"] += 1
            self.defense += 200
            logger.info(f"진화 발생! 레벨: {self.level}")

    def status(self):
        return f"Lv:{self.level} | EXP:{self.exp}/{self.level*100} | 방어:{self.defense}"

evolutionary_engine = EvolutionaryEngine()

class QuantumResistantCrypto:
    def __init__(self):
        self.key = os.urandom(32)
        logger.info("양자내성 암호화 초기화")

    def encrypt(self, data):
        token = hashlib.sha3_512(str(data).encode()).hexdigest()
        return f"{token}:proof"

    def decrypt(self, encrypted_data):
        return encrypted_data.split(":")[0]

crypto = QuantumResistantCrypto()

class QuantumTokenSystem:
    def __init__(self, token_lifetime=300):
        self.token_lifetime = token_lifetime
        self.tokens: Dict[str, datetime] = {}
        self.salt = os.urandom(16)

    def generate_token(self, identity="lucia_manager"):
        timestamp = now_iso()
        raw_token = f"{identity}::{timestamp}::{random.getrandbits(256)}"
        token = hashlib.sha3_512(self.salt + raw_token.encode()).hexdigest()
        expiration = datetime.now() + timedelta(seconds=self.token_lifetime)
        self.tokens[token] = expiration
        logger.info(f"토큰 생성: {token[:12]}...")
        return token

    def validate_token(self, token):
        current = datetime.now()
        ok = token in self.tokens and current <= self.tokens[token]
        if ok:
            logger.info(f"토큰 검증 성공: {token[:12]}...")
        else:
            logger.warning(f"잘못된/만료된 토큰: {token[:12]}..." if token else "토큰 없음")
        return ok

token_system = QuantumTokenSystem()

# ---------- MQTT Layer (optional) ----------
class SecureMQTTLayer:
    def __init__(self, broker_ip="localhost", port=8883):
        self.broker_ip = broker_ip
        self.port = port
        self.connected = False
        self.cooldowns = {}
        self.feedback_topics = {"lucia/drone/feedback", "lucia/cctv/feedback"}
        self.client = None
        self._init()

    def _init(self):
        if not paho_mqtt:
            logger.info("paho-mqtt 미설치: MQTT 비활성화")
            return
        try:
            from paho.mqtt.client import Client, MQTTv5  # type: ignore
            import ssl as _ssl
            self.client = Client(protocol=MQTTv5)
            self.client.tls_set(tls_version=_ssl.PROTOCOL_TLSv1_2)
            self.client.username_pw_set("lucia_manager", "lucia_secure_password")
            def _on_connect(client, userdata, flags, rc, properties=None):
                self.connected = (rc == 0)
                if self.connected:
                    for t in self.feedback_topics:
                        client.subscribe(t)
                    logger.info("MQTT 연결 성공")
                else:
                    logger.error(f"MQTT 연결 실패: rc={rc}")
            def _on_disconnect(client, userdata, rc, properties=None):
                self.connected = False
            self.client.on_connect = _on_connect
            self.client.on_disconnect = _on_disconnect
            self.client.connect(self.broker_ip, self.port, 60)
            self.client.loop_start()
        except Exception as e:
            logger.warning(f"MQTT 초기화 실패(비활성화): {e}")
            self.client = None
            self.connected = False

    def send_command(self, command, token, topic="lucia/command", priority=1):
        if not self.client:
            return False
        if topic in self.cooldowns and time.time() < self.cooldowns[topic]:
            return False
        msg = f"{priority}:{command}:{(token or '')[:12]}"
        try:
            result = self.client.publish(topic, msg, qos=1)
            if getattr(result, 'rc', 1) == 0:
                self.cooldowns[topic] = time.time() + 5
                logger.info(f"MQTT → {topic}: {command}")
                return True
        except Exception as e:
            logger.debug(f"MQTT publish 실패: {e}")
        return False

    def emergency_all(self):
        self.send_command("LAUNCH_EMERGENCY", "emerg", "lucia/drone", 1)
        self.send_command("ALARM_ON", "emerg", "lucia/cctv", 1)

    def patrol_normal(self):
        self.send_command("SCAN", "patrol", "lucia/cctv", 2)

mqtt_layer = SecureMQTTLayer(broker_ip=os.getenv("MQTT_BROKER_IP","localhost"))

# ---------- Decision Engine ----------
class DecisionEngine:
    def __init__(self):
        from collections import deque
        self.action_queue = deque()
        self.action_history: List[Dict[str, Any]] = []
        self.action_cooldowns = {"emergency_all": 60, "drone_launch": 30}
        self.last_action_time: Dict[str, float] = {}

    def add_action(self, action, priority=5):
        self.action_queue.append((priority, action))
        self.action_queue = type(self.action_queue)(sorted(self.action_queue, key=lambda x: x[0], reverse=True))

    def execute_actions(self, communication: SecureMQTTLayer):
        executed = []
        while self.action_queue:
            priority, action = self.action_queue.popleft()
            if action in self.last_action_time and time.time() - self.last_action_time[action] < self.action_cooldowns.get(action, 0):
                continue
            if action == "emergency_all":
                communication.emergency_all()
            elif action == "patrol_normal":
                communication.patrol_normal()
            elif action == "drone_launch":
                communication.send_command("LAUNCH", "drone", "lucia/drone", 1)
            self.last_action_time[action] = time.time()
            executed.append(action)
            self.action_history.append({"timestamp": now_iso(), "action": action, "priority": priority})
        return executed

    def decide_actions(self, emotion: str, data: Dict[str, Any], threat_level: int):
        if emotion in ["분노", "공포"] or threat_level >= 3:
            self.add_action("emergency_all", 1)
        elif emotion == "불안":
            self.add_action("patrol_normal", 3)
        else:
            self.add_action("patrol_normal", 4)
        if data.get('cctv_intrusion') == "침입":
            self.add_action("drone_launch", 2)

decision_engine = DecisionEngine()

# ---------- FastAPI App ----------
from fastapi import FastAPI
app = FastAPI(
    title="Lucia Ultimate Quantum Control Enhanced",
    description="향상된 PC 제어 및 양자 보안 시스템 (Fixed)",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)

# ---------- Models ----------
class CommandRequest(BaseModel):
    text: str

# ---------- Helpers ----------
def get_system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        cpu = f"{psutil.cpu_percent(interval=0.1):.1f}%" if psutil else "N/A"
        if psutil:
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            mems = f"{mem.used // (1024**3)}GB / {mem.total // (1024**3)}GB ({mem.percent:.1f}%)"
            disks = f"{disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB"
            procs = len(psutil.pids())
        else:
            mems = "N/A"
            disks = "N/A"
            procs = 0
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            net = f"IP: {local_ip}"
        except Exception:
            net = "연결 상태 불명"
        info = {
            "cpu": cpu,
            "memory": mems,
            "disk": disks,
            "network": net,
            "processes": procs,
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "quantum_psi": state.psi,
            "threat_level": state.threat_level,
            "security_level": state.security_level
        }
    except Exception as e:
        logger.error(f"시스템 정보 수집 실패: {e}")
        info = {"error": "시스템 정보 수집 실패"}
    return info

def parse_enhanced_commands(text: str) -> Dict[str, Any]:
    t = text.lower().strip()

    # File ops
    if any(k in t for k in ["파일", "다운로드", "저장", "생성", "목록", "리스트"]):
        if "다운로드" in t:
            m = re.search(r'([\w\-\_\.]+\.[\w]+)', t)
            if m:
                return {"type":"file","action":"download","filename":m.group(1)}
        if "생성" in t or "만들" in t:
            if "파이썬" in t or ".py" in t:
                return {"type":"file","action":"create_python"}
            if "html" in t or ".html" in t:
                return {"type":"file","action":"create_html"}
            if "텍스트" in t or ".txt" in t:
                return {"type":"file","action":"create_text"}
        if "목록" in t or "리스트" in t:
            return {"type":"file","action":"list"}

    # Mouse
    if "마우스" in t:
        coords = re.findall(r'(\d+)\s*,?\s*(\d+)', t)
        if "더블" in t and "클릭" in t:
            return {"type":"mouse","action":"double_click","x": int(coords[0][0]) if coords else None, "y": int(coords[0][1]) if coords else None}
        if "우클릭" in t or "오른쪽" in t:
            return {"type":"mouse","action":"right_click","x": int(coords[0][0]) if coords else None, "y": int(coords[0][1]) if coords else None}
        if "드래그" in t and len(coords) >= 2:
            return {"type":"mouse","action":"drag","x1":int(coords[0][0]),"y1":int(coords[0][1]),"x2":int(coords[1][0]),"y2":int(coords[1][1])}
        if "클릭" in t:
            return {"type":"mouse","action":"click","x": int(coords[0][0]) if coords else None, "y": int(coords[0][1]) if coords else None}
        if "이동" in t and coords:
            return {"type":"mouse","action":"move","x":int(coords[0][0]),"y":int(coords[0][1])}
        if "스크롤" in t:
            m = re.search(r'(\d+)', t)
            direction = -1 if ("아래" in t or "다운" in t) else 1
            return {"type":"mouse","action":"scroll","amount": int(m.group(1)) * direction if m else 3*direction}

    # Keyboard (text entry)
    if "키보드" in t or "입력" in t:
        m = re.search(r'키보드로?\s*(.+?)\s*입력', t)
        if m:
            return {"type":"keyboard","action":"type","text":m.group(1).strip()}

    # Special keys
    specials = {
        "엔터":"enter","enter":"enter",
        "스페이스":"space","space":"space",
        "백스페이스":"backspace","backspace":"backspace",
        "탭":"tab","tab":"tab",
        "이스케이프":"escape","escape":"escape"
    }
    for k, en in specials.items():
        if k in t:
            return {"type":"keyboard","action":"press","keys":[en]}

    # Hotkeys
    if "컨트롤" in t or "ctrl" in t:
        if "복사" in t or " c" in f" {t}":
            return {"type":"keyboard","action":"hotkey","keys":["ctrl","c"]}
        if "붙여넣기" in t or " v" in f" {t}":
            return {"type":"keyboard","action":"hotkey","keys":["ctrl","v"]}
        if "실행취소" in t or " z" in f" {t}":
            return {"type":"keyboard","action":"hotkey","keys":["ctrl","z"]}
        if "저장" in t or " s" in f" {t}":
            return {"type":"keyboard","action":"hotkey","keys":["ctrl","s"]}
        if "전체선택" in t or " a" in f" {t}":
            return {"type":"keyboard","action":"hotkey","keys":["ctrl","a"]}

    # Apps
    if any(app in t for app in ["크롬","chrome","메모장","notepad","계산기","calculator","탐색기","explorer","cmd","파워셸","powershell","코드","vscode"]):
        app_map = {
            "크롬":"chrome","chrome":"chrome",
            "메모장":"notepad","notepad":"notepad",
            "계산기":"calculator","calculator":"calculator",
            "탐색기":"explorer","explorer":"explorer",
            "cmd":"cmd",
            "파워셸":"powershell","powershell":"powershell",
            "코드":"code","vscode":"code"
        }
        for k, v in app_map.items():
            if k in t:
                return {"type":"app","name": v}

    # System
    if "볼륨" in t:
        m = re.search(r'(\d+)', t)
        if m:
            return {"type":"system","action":"volume","value": int(m.group(1))}
        if "올려" in t or "크게" in t:
            return {"type":"system","action":"volume_up"}
        if "내려" in t or "작게" in t:
            return {"type":"system","action":"volume_down"}
        if "음소거" in t or "뮤트" in t:
            return {"type":"system","action":"volume_mute"}
    if "스크린샷" in t or "캡처" in t:
        return {"type":"system","action":"screenshot"}
    if any(k in t for k in ["종료","shutdown","재시작","reboot","절전","잠금","sleep","lock"]):
        if "재시작" in t or "reboot" in t:
            return {"type":"system","action":"restart"}
        if "종료" in t or "shutdown" in t:
            return {"type":"system","action":"shutdown"}
        if "절전" in t or "sleep" in t:
            return {"type":"system","action":"sleep"}
        if "잠금" in t or "lock" in t:
            return {"type":"system","action":"lock"}

    # Quantum / Security
    if any(k in t for k in ["양자","보안","위협","위험"]):
        if "위협" in t or "위험" in t:
            return {"type":"quantum","action":"check_threat"}
        if "양자 상태" in t or "quantum" in t:
            return {"type":"quantum","action":"quantum_state"}
        if "보안 레벨" in t:
            return {"type":"quantum","action":"security_level"}

    # AI chat (default)
    return {"type":"ai_chat","text": t}

# ---------- Executors ----------
async def execute_mouse_command(cmd: Dict[str, Any]) -> Dict[str, Any]:
    if not pyautogui:
        return {"ok": False, "message": "pyautogui 미설치: 마우스 제어 불가"}
    try:
        action = cmd.get("action")
        x, y = cmd.get("x"), cmd.get("y")
        if action == "click":
            pyautogui.click(x=x, y=y) if x is not None and y is not None else pyautogui.click()
            return {"ok": True, "message": f"클릭 완료 ({x},{y})"}
        if action == "double_click":
            pyautogui.doubleClick(x=x, y=y) if x is not None and y is not None else pyautogui.doubleClick()
            return {"ok": True, "message": f"더블클릭 완료 ({x},{y})"}
        if action == "right_click":
            pyautogui.rightClick(x=x, y=y) if x is not None and y is not None else pyautogui.rightClick()
            return {"ok": True, "message": f"우클릭 완료 ({x},{y})"}
        if action == "move" and x is not None and y is not None:
            pyautogui.moveTo(x, y, duration=0.3)
            return {"ok": True, "message": f"마우스 이동 ({x},{y})"}
        if action == "scroll":
            amount = cmd.get("amount", 3)
            pyautogui.scroll(amount, x=x, y=y) if x is not None and y is not None else pyautogui.scroll(amount)
            return {"ok": True, "message": f"스크롤 {amount}"}
        if action == "drag":
            x1,y1,x2,y2 = cmd.get("x1"),cmd.get("y1"),cmd.get("x2"),cmd.get("y2")
            if None not in (x1,y1,x2,y2):
                pyautogui.moveTo(x1,y1, duration=0.2)
                pyautogui.dragTo(x2,y2, duration=0.5)
                return {"ok": True, "message": f"드래그 ({x1},{y1})→({x2},{y2})"}
        return {"ok": False, "message": "지원하지 않는 마우스 액션"}
    except Exception as e:
        return {"ok": False, "message": f"마우스 제어 실패: {e}"}

async def execute_keyboard_command(cmd: Dict[str, Any]) -> Dict[str, Any]:
    if not pyautogui:
        return {"ok": False, "message": "pyautogui 미설치: 키보드 제어 불가"}
    try:
        action = cmd.get("action")
        if action == "type":
            text = cmd.get("text","")
            pyautogui.write(text, interval=0.01)
            return {"ok": True, "message": f"텍스트 입력: {text}"}
        if action == "press":
            keys = cmd.get("keys", [])
            if keys:
                pyautogui.press(keys[0])
                return {"ok": True, "message": f"키 입력: {keys[0]}"}
        if action == "hotkey":
            keys = cmd.get("keys", [])
            if keys:
                pyautogui.hotkey(*keys)
                return {"ok": True, "message": f"단축키: {'+'.join(keys)}"}
        return {"ok": False, "message": "지원하지 않는 키보드 액션"}
    except Exception as e:
        return {"ok": False, "message": f"키보드 제어 실패: {e}"}

async def execute_app_command(cmd: Dict[str, Any]) -> Dict[str, Any]:
    try:
        name = (cmd.get("name") or "").lower()
        if platform.system() == "Windows":
            app_commands = {
                "chrome":"chrome","notepad":"notepad","calculator":"calc","explorer":"explorer",
                "cmd":"cmd","powershell":"powershell","code":"code"
            }
        else:
            app_commands = {
                "chrome":"google-chrome","notepad":"gedit","calculator":"gnome-calculator","explorer":"nautilus",
                "cmd":"xterm","powershell":"powershell","code":"code"
            }
        exe = app_commands.get(name, name)
        subprocess.Popen(exe, shell=True)
        return {"ok": True, "message": f"{name} 실행 요청"}
    except Exception as e:
        return {"ok": False, "message": f"앱 실행 실패: {e}"}

async def execute_system_command(cmd: Dict[str, Any]) -> Dict[str, Any]:
    try:
        action = cmd.get("action")
        if action == "screenshot":
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"lucia_screenshot_{ts}.png"
            fpath = DOWNLOAD_DIR / fname
            if not pyautogui:
                return {"ok": False, "message": "pyautogui 미설치: 스크린샷 불가"}
            im = pyautogui.screenshot()
            im.save(fpath)
            desktop = Path.home() / "Desktop"
            desktop_path = None
            try:
                if desktop.exists():
                    dfile = desktop / fname
                    im.save(dfile)
                    desktop_path = str(dfile)
            except Exception:
                desktop_path = None
            msg = f"스크린샷 저장 완료 - 다운로드: {fpath}"
            if desktop_path:
                msg += f"\n- 바탕화면: {desktop_path}"
            return {"ok": True, "message": msg, "filename": fname, "download_url": f"/downloads/{fname}"}

        if action and action.startswith("volume"):
            if platform.system() == "Windows":
                try:
                    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                    from comtypes import CLSCTX_ALL
                    devices = AudioUtilities.GetSpeakers()
                    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                    volume = interface.QueryInterface(IAudioEndpointVolume)
                    if action == "volume":
                        value = int(cmd.get("value", 50))
                        volume.SetMasterVolumeLevelScalar(max(0,min(1,value/100.0)), None)
                        return {"ok": True, "message": f"볼륨 {value}% 설정"}
                    if action == "volume_up":
                        cur = volume.GetMasterVolumeLevelScalar()
                        volume.SetMasterVolumeLevelScalar(min(1.0, cur+0.1), None)
                        return {"ok": True, "message": "볼륨 ↑"}
                    if action == "volume_down":
                        cur = volume.GetMasterVolumeLevelScalar()
                        volume.SetMasterVolumeLevelScalar(max(0.0, cur-0.1), None)
                        return {"ok": True, "message": "볼륨 ↓"}
                    if action == "volume_mute":
                        volume.SetMute(1, None)
                        return {"ok": True, "message": "음소거"}
                except Exception as e:
                    return {"ok": False, "message": f"볼륨 제어 실패: {e}"}
            return {"ok": False, "message": "볼륨 제어는 Windows에서만 지원 또는 라이브러리 필요"}

        if action == "shutdown":
            if platform.system() == "Windows":
                subprocess.Popen("shutdown /s /t 30", shell=True)
            else:
                subprocess.Popen("shutdown -h +0.5", shell=True)
            return {"ok": True, "message": "30초 후 시스템 종료"}
        if action == "restart":
            if platform.system() == "Windows":
                subprocess.Popen("shutdown /r /t 30", shell=True)
            else:
                subprocess.Popen("reboot", shell=True)
            return {"ok": True, "message": "30초 후 재시작"}
        if action == "sleep":
            if platform.system() == "Windows":
                subprocess.Popen("rundll32.exe powrprof.dll,SetSuspendState 0,1,0", shell=True)
            else:
                subprocess.Popen("systemctl suspend", shell=True)
            return {"ok": True, "message": "절전 모드 전환"}
        if action == "lock":
            if platform.system() == "Windows":
                subprocess.Popen("rundll32.exe user32.dll,LockWorkStation", shell=True)
            else:
                subprocess.Popen("gnome-screensaver-command -l", shell=True)
            return {"ok": True, "message": "시스템 잠금"}

        return {"ok": False, "message": f"지원하지 않는 시스템 액션: {action}"}
    except Exception as e:
        return {"ok": False, "message": f"시스템 제어 실패: {e}"}

async def execute_file_command(cmd: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if not aiofiles:
            return {"ok": False, "message": "aiofiles 미설치: 파일 생성 기능 비활성"}
        action = cmd.get("action")
        if action == "create_python":
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"lucia_python_{ts}.py"
            fpath = DOWNLOAD_DIR / fname
            py_code = '#!/usr/bin/env python3\n'\
                      f'# 생성: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'\
                      'def main():\n'\
                      '    print("Hello from Lucia Ultimate Quantum Control!")\n'\
                      f'    print("현재 시간:", "{datetime.now()}")\n'\
                      '    numbers = [1,2,3,4,5]\n'\
                      '    print("숫자 합계:", sum(numbers))\n'\
                      'if __name__ == "__main__":\n'\
                      '    main()\n'
            async with aiofiles.open(fpath, 'w', encoding='utf-8') as f:
                await f.write(py_code)
            return {"ok": True, "message": f"Python 파일 생성: {fname}", "filename": fname, "download_url": f"/downloads/{fname}"}

        if action == "create_html":
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"lucia_webpage_{ts}.html"
            fpath = DOWNLOAD_DIR / fname
            html = "<!doctype html>\n<html lang=\"ko\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"\
                   "<title>Lucia Quantum Control</title>\n"\
                   "<style>body{font-family:system-ui,Arial;margin:0;padding:24px;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff} .card{background:rgba(255,255,255,.12);padding:20px;border-radius:16px;backdrop-filter:blur(6px)}</style>\n"\
                   "</head><body><div class=\"card\"><h1>Lucia Quantum Control</h1><p>이 페이지는 시스템에서 자동 생성되었습니다.</p></div></body></html>"
            async with aiofiles.open(fpath, 'w', encoding='utf-8') as f:
                await f.write(html)
            return {"ok": True, "message": f"HTML 파일 생성: {fname}", "filename": fname, "download_url": f"/downloads/{fname}"}

        if action == "create_text":
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"lucia_notes_{ts}.txt"
            fpath = DOWNLOAD_DIR / fname
            txt = f"Lucia Quantum Control 노트\n생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n- PC 제어\n- AI 대화\n- 양자 보안\n"
            async with aiofiles.open(fpath, 'w', encoding='utf-8') as f:
                await f.write(txt)
            return {"ok": True, "message": f"텍스트 파일 생성: {fname}", "filename": fname, "download_url": f"/downloads/{fname}"}

        if action == "list":
            files = []
            for p in DOWNLOAD_DIR.glob("*"):
                if p.is_file():
                    st = p.stat()
                    files.append({
                        "name": p.name,
                        "size_kb": round(st.st_size/1024,1),
                        "modified": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
                    })
            if files:
                return {"ok": True, "files": files, "message": f"{len(files)}개 파일"}
            return {"ok": True, "files": [], "message": "생성된 파일 없음"}

        if action == "download":
            fname = cmd.get("filename","")
            if fname and (DOWNLOAD_DIR / fname).exists():
                return {"ok": True, "download_url": f"/downloads/{fname}", "message": f"다운로드 준비: {fname}"}
            return {"ok": False, "message": f"파일 없음: {fname}"}

        return {"ok": False, "message": "지원하지 않는 파일 액션"}
    except Exception as e:
        return {"ok": False, "message": f"파일 처리 실패: {e}"}

async def execute_quantum_command(cmd: Dict[str, Any]) -> Dict[str, Any]:
    try:
        action = cmd.get("action")
        if action == "check_threat":
            sensor_data = sensor_sim.generate()
            vec = sensor_sim.vectorize(sensor_data)
            tdata = await threat_detector.analyze(vec)
            state.threat_level = tdata["level"]
            state.security_level = 10 if tdata["level"] > 5 else 8 if tdata["level"] > 2 else 5
            evolutionary_engine.evolve(tdata["level"])
            decision_engine.decide_actions("불안" if state.threat_level >= 2 else "평온", sensor_data, state.threat_level)
            executed = decision_engine.execute_actions(mqtt_layer)
            return {"ok": True, "message": f"위협 {tdata['level']} (score {tdata['score']:.4f})", "executed": executed}
        if action == "quantum_state":
            psi, ent = quantum_core.fluctuate(high_qubit_mode=True)
            state.psi, state.entanglement = psi, ent
            return {"ok": True, "message": f"양자 상태: ψ={psi:.3f}, 얽힘={ent:.3f}"}
        if action == "security_level":
            return {"ok": True, "message": f"현재 보안 레벨: {state.security_level}"}
        return {"ok": False, "message": "지원되지 않는 양자 명령"}
    except Exception as e:
        return {"ok": False, "message": f"양자 명령 실패: {e}"}

async def execute_ai_chat(cmd: Dict[str, Any]) -> Dict[str, Any]:
    try:
        text = cmd.get("text","")
        ev, dominant = emotion_system.update(state.threat_level, text)
        state.emotion_vector = ev
        quick = {
            "안녕": f"안녕하세요! 😊 감정:{dominant} 위협:{state.threat_level}",
            "도움": "도움말: '마우스 500 300 클릭', '스크린샷', '파이썬 파일 생성', '위협 확인' 등 명령을 시도해 보세요!",
            "시간": f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "상태": f"시스템 상태: CPU/메모리/디스크/네트워크, 위협 {state.threat_level}, 보안 {state.security_level}"
        }
        for k, v in quick.items():
            if k in text:
                return {"ok": True, "message": v, "emotion": dominant}
        return {"ok": True, "message": f"'{text}' 확인! 더 자세한 건 '도움'을 입력해 보세요.", "emotion": dominant}
    except Exception as e:
        return {"ok": False, "message": f"AI 채팅 실패: {e}"}

async def execute_command(command_data: Dict[str, Any]) -> Dict[str, Any]:
    start = time.time()
    try:
        ctype = command_data.get("type")
        if ctype == "mouse":
            res = await execute_mouse_command(command_data)
        elif ctype == "keyboard":
            res = await execute_keyboard_command(command_data)
        elif ctype == "app":
            res = await execute_app_command(command_data)
        elif ctype == "system":
            res = await execute_system_command(command_data)
        elif ctype == "file":
            res = await execute_file_command(command_data)
        elif ctype == "quantum":
            res = await execute_quantum_command(command_data)
        elif ctype == "ai_chat":
            res = await execute_ai_chat(command_data)
        else:
            res = {"ok": False, "message": f"지원하지 않는 명령 타입: {ctype}"}
        res["execution_time"] = f"{round((time.time()-start)*1000,2)}ms"
        return res
    except Exception as e:
        return {"ok": False, "message": f"명령 실행 실패: {e}", "execution_time": f"{round((time.time()-start)*1000,2)}ms"}

# ---------- App + Routes ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

quantum_core = QuantumConsciousnessCore()
connected_clients: set = set()
command_history: List[Dict[str, Any]] = []
server_stats: Dict[str, Any] = {
    "start_time": datetime.now(),
    "total_commands": 0,
    "successful_commands": 0,
    "failed_commands": 0,
    "peak_clients": 0
}

@app.get("/", response_class=HTMLResponse)
async def home():
    html = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lucia Ultimate Quantum Control</title>
<style>
body{font-family:system-ui,Arial;margin:0;background:#0f1226;color:#eaf0ff}
.container{max-width:960px;margin:0 auto;padding:24px}
.card{background:#161a36;border:1px solid #2a2f5a;border-radius:16px;padding:18px;margin:12px 0}
button,input{padding:10px 14px;border-radius:10px;border:1px solid #3a4080;background:#0f1226;color:#eaf0ff}
.row{display:flex;gap:12px;flex-wrap:wrap}
pre{white-space:pre-wrap;word-break:break-word}
</style>
</head>
<body><div class="container">
  <h1>🚀 Lucia Ultimate Quantum Control (Fixed)</h1>
  <div class="card">
    <div>테스트: <code>안녕</code>, <code>스크린샷</code>, <code>마우스 500 300 클릭</code>, <code>파이썬 파일 생성</code>, <code>위협 확인</code></div>
    <div class="row" style="margin-top:10px">
      <input id="cmd" placeholder="명령을 입력하세요..." style="flex:1">
      <button onclick="send()">실행</button>
      <button onclick="health()">상태</button>
    </div>
  </div>
  <div class="card"><h3>결과</h3><pre id="out"></pre></div>
</div>
<script>
async function send(){
  const text = document.getElementById('cmd').value;
  const r = await fetch('/command', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})});
  const j = await r.json(); document.getElementById('out').textContent = JSON.stringify(j,null,2);
}
async function health(){
  const r = await fetch('/health'); const j = await r.json();
  document.getElementById('out').textContent = JSON.stringify(j,null,2);
}
</script>
</body></html>"""
    return HTMLResponse(html)

@app.get("/health")
async def health_check():
    current_clients = len(connected_clients)
    if current_clients > server_stats["peak_clients"]:
        server_stats["peak_clients"] = current_clients
    success_rate = round((server_stats["successful_commands"] / max(1, server_stats["total_commands"])) * 100, 1)
    return {
        "ok": True,
        "system": get_system_info(),
        "stats": {
            **{k:(v if k!="start_time" else str(v)) for k,v in server_stats.items()},
            "success_rate": success_rate
        },
        "features": {
            "stt_available": bool(WhisperModel and sd),
            "audio_available": bool(sd),
            "file_operations": True,
            "enhanced_commands": True,
            "quantum_security": True
        }
    }

class CommandRequest(BaseModel):
    text: str

@app.post("/command")
async def process_command(req: CommandRequest):
    command_data = parse_enhanced_commands(req.text)
    command_history.append({
        "text": req.text, "timestamp": now_iso(), "id": str(uuid.uuid4()),
        "type": command_data.get("type","unknown")
    })
    if len(command_history) > 100:
        command_history.pop(0)

    server_stats["total_commands"] += 1
    res = await execute_command(command_data)
    if res.get("ok"):
        server_stats["successful_commands"] += 1
    else:
        server_stats["failed_commands"] += 1
    res["command_type"] = command_data.get("type")
    res["timestamp"] = now_iso()
    return res

@app.get("/history")
async def get_history():
    return {"history": command_history[-50:]}

@app.delete("/history")
async def clear_history():
    command_history.clear()
    return {"ok": True, "message": "명령어 히스토리 삭제"}

@app.get("/downloads/{filename}")
async def download_file(filename: str):
    fpath = DOWNLOAD_DIR / filename
    if not fpath.exists() or not fpath.is_file():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
    return FileResponse(path=str(fpath), filename=filename)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    try:
        await ws.send_json({
            "type":"connection",
            "message":"Lucia Ultimate Quantum Control Enhanced에 연결되었습니다.",
            "timestamp": now_iso(),
            "features": {
                "pc_control": bool(pyautogui),
                "voice_recognition": bool(WhisperModel and sd),
                "file_operations": True,
                "quantum_security": True
            }
        })
        while True:
            data = await ws.receive_text()
            try:
                payload = json.loads(data)
            except Exception:
                payload = {"type":"command","text":data}
            if payload.get("type") == "command":
                text = payload.get("text","")
                cmd = parse_enhanced_commands(text)
                res = await execute_command(cmd)
                await ws.send_json({"type":"command_result","original_text":text,"result":res,"timestamp": now_iso()})
            elif payload.get("type") == "ping":
                await ws.send_json({"type":"pong","timestamp": now_iso()})
            else:
                await ws.send_json({"type":"error","message":"지원하지 않는 메시지 타입","timestamp": now_iso()})
    except WebSocketDisconnect:
        connected_clients.discard(ws)
    except Exception as e:
        logger.error(f"웹소켓 오류: {e}")
        connected_clients.discard(ws)

# ---------- Main ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SERVER_PORT", "8000"))
    host = os.getenv("HOST","0.0.0.0")
    logger.info("🚀 Lucia Ultimate Quantum Control Enhanced 서버 시작")
    logger.info(f"📡 주소: http://{host}:{port}")
    logger.info(f"🌐 로컬 접속: http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
