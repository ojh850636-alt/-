import logging
import numpy as np

try:
    from sklearn.ensemble import IsolationForest

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger("lucia_core.threat")


class AdvancedThreatDetector:
    def __init__(self):
        if SKLEARN_AVAILABLE:
            self.iso_forest = IsolationForest(contamination=0.01)
            try:
                self.iso_forest.fit(np.random.rand(100, 3))
            except Exception:
                self.iso_forest = None
        else:
            self.iso_forest = None

    async def analyze(self, vector):
        if self.iso_forest is None:
            # fallback heuristic
            score = -sum(vector) / len(vector)
            threat_level = min(5, max(0, int(-score * 10)))
            return {"level": threat_level, "score": score, "timestamp": None}
        score = self.iso_forest.score_samples([vector])[0]
        threat_level = min(5, max(0, int(-score * 10)))
        logger.info(f"위협 분석: 점수={score:.4f}, 수준={threat_level}")
        return {"level": threat_level, "score": score, "timestamp": None}
