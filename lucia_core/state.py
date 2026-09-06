class SystemState:
    def __init__(self):
        self.psi = 0.5
        self.threat_level = 0
        self.security_level = 5


state = SystemState()
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SystemState:
    psi: float = 0.5
    entanglement: float = 0.0
    threat_level: int = 0
    quantum_resources: int = 30
    emotion_vector: np.ndarray = field(default_factory=lambda: np.zeros(7))
    health_score: float = 1.0
    security_level: int = 5
