import random
import logging
from datetime import datetime, timezone

logger = logging.getLogger("lucia_core.sensors")


class RealitySensorSimulator:
    def __init__(self, kafka_servers: str = "localhost:9092"):
        self.kafka_servers = kafka_servers
        self.probabilities = {
            "cctv_intrusion": {"정상": 0.8, "침입": 0.15, "의심": 0.05},
            "network": {"정상": 0.8, "DDoS": 0.15, "AI해킹": 0.05},
        }

    def generate(self):
        data = {
            "temperature": random.uniform(15, 110),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        for sensor, probs in self.probabilities.items():
            data[sensor] = random.choices(list(probs.keys()), list(probs.values()))[0]
        logger.debug(f"Generated sensor data: {data}")
        return data

    def vectorize(self, data):
        mapping = {"정상": 0, "침입": 1, "의심": 0.5, "DDoS": 1, "AI해킹": 1}
        return [
            mapping.get(data.get("cctv_intrusion", "정상"), 0),
            mapping.get(data.get("network", "정상"), 0),
            data["temperature"] / 110.0,
        ]
