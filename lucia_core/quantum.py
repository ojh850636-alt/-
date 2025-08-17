import random
import logging
from typing import Tuple

logger = logging.getLogger('lucia_core.quantum')

try:
    from qiskit import QuantumCircuit, Aer, execute
    QISKIT_AVAILABLE = True
except Exception:
    QISKIT_AVAILABLE = False

class QuantumConsciousnessCore:
    def __init__(self, qubits: int = 8):
        self.qubits = qubits
        self.psi = random.uniform(0.1, 0.9)
        self.entanglement = 0.0
        # if qiskit is available, build a simple circuit lazily
        self._qc = None
        logger.info(f'QuantumConsciousnessCore initialized (qubits={qubits}, qiskit={QISKIT_AVAILABLE})')

    def _ensure_circuit(self):
        if QISKIT_AVAILABLE and self._qc is None:
            self._qc = QuantumCircuit(self.qubits, self.qubits)
            try:
                self._qc.h(range(self.qubits))
                for i in range(self.qubits-1):
                    self._qc.cx(i, i+1)
                self._qc.measure_all()
            except Exception:
                # fallback: mark as unavailable
                pass

    def fluctuate(self, high_qubit_mode: bool = False) -> Tuple[float, float]:
        # safe deterministic-ish fluctuation
        try:
            if QISKIT_AVAILABLE:
                self._ensure_circuit()
                backend = Aer.get_backend('qasm_simulator')
                result = execute(self._qc, backend, shots=1).result()
                counts = list(result.get_counts().keys())
                if counts:
                    quantum_state = counts[0]
                    delta = (int(quantum_state, 2) % 1024) / 1024 * 0.02
                    self.psi = min(max(self.psi + delta, 0.1), 0.99)
        except Exception:
            # simulation fallback
            self.psi = min(0.99, self.psi + random.uniform(0.0, 0.02))
        if self.psi > 0.9:
            self.entanglement = max(self.entanglement, random.uniform(0.85, 0.95))
        return self.psi, self.entanglement
