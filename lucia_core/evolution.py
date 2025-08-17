import logging

logger = logging.getLogger("lucia_core.evolution")


class EvolutionaryEngine:
    def __init__(self):
        self.level = 1
        self.defense = 1500
        self.exp = 0
        self.branches = {"defense": 0, "knowledge": 0}

    def evolve(self, threat_level: int):
        self.exp += threat_level * 20
        if self.exp >= self.level * 100:
            self.level += 1
            self.exp = 0
            self.branches["defense"] += 1
            self.defense += 200
            logger.info(f"진화 발생! 레벨: {self.level}")

    def status(self):
        return (
            f"Lv:{self.level} | EXP:{self.exp}/{self.level * 100} | 방어:{self.defense}"
        )
