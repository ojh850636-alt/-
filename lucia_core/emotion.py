import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger('lucia_core.emotion')

try:
    from transformers import BertTokenizer, BertModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

class NLPProcessor:
    def __init__(self):
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.model = BertModel.from_pretrained('bert-base-uncased')
            except Exception:
                self.tokenizer = None
                self.model = None
        else:
            self.tokenizer = None
            self.model = None

    def process_input(self, text: str):
        # lightweight heuristic fallback
        vec = np.zeros(7)
        if not text:
            return vec
        t = text.lower()
        if 'happy' in t or '기쁨' in t:
            vec[1] = 0.12
        if 'angry' in t or '분노' in t:
            vec[5] = 0.3
        return vec

class EmotionSystem:
    def __init__(self):
        self.emotion_vector = np.zeros(7)
        self.emotion_vector[0] = 1.0
        self.decay_rates = np.array([0.0, 0.02, 0.015, 0.03, 0.04, 0.05, 0.06])
        self.nlp_processor = NLPProcessor()

    def update(self, threat_level: int, external_input: str = None) -> Tuple[np.ndarray, str]:
        emotion_weights = np.zeros(7)
        if threat_level == 0:
            emotion_weights[1:3] = [0.12, 0.08]
        elif threat_level == 1:
            emotion_weights[3] = 0.2
        elif threat_level >= 2:
            emotion_weights[4:6] = [0.25, 0.15] if threat_level == 2 else [0.3, 0.25]
        if external_input:
            emotion_weights += self.nlp_processor.process_input(external_input)
        self.emotion_vector = np.clip(self.emotion_vector + emotion_weights - self.decay_rates, 0, 1)
        self.emotion_vector[0] = max(0, 1 - sum(self.emotion_vector[1:]))
        dominant_emotion = ["평온", "기쁨", "호기심", "경계", "불안", "분노", "공포"][int(np.argmax(self.emotion_vector))]
        logger.info(f"감정 업데이트: {dominant_emotion} (위협: {threat_level})")
        return self.emotion_vector, dominant_emotion
