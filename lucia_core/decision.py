import logging
from collections import deque
import time
from datetime import datetime, timezone

logger = logging.getLogger('lucia_core.decision')

class DecisionEngine:
    def __init__(self):
        self.action_queue = deque()
        self.action_history = []
        self.action_cooldowns = {'emergency_all': 60, 'drone_launch': 30}
        self.last_action_time = {}

    def add_action(self, action, priority=5):
        self.action_queue.append((priority, action))
        self.action_queue = deque(sorted(self.action_queue, key=lambda x: x[0], reverse=True))
        logger.info(f"액션 추가: {action} (우선순위: {priority})")

    def execute_actions(self, communication):
        executed = []
        while self.action_queue:
            priority, action = self.action_queue.popleft()
            if action in self.last_action_time and time.time() - self.last_action_time[action] < self.action_cooldowns.get(action, 0):
                continue
            if action == 'emergency_all':
                communication.emergency_all()
            elif action == 'patrol_normal':
                communication.patrol_normal()
            elif action == 'drone_launch':
                communication.send_command('LAUNCH', 'simtoken', 'lucia/drone', 1)
            self.last_action_time[action] = time.time()
            executed.append(action)
            self.action_history.append({'timestamp': datetime.now(timezone.utc).isoformat(), 'action': action, 'priority': priority})
            logger.info(f"액션 실행: {action}")
        return executed

    def decide_actions(self, emotion, data, threat_level):
        if emotion in ["분노", "공포"]:
            self.add_action("emergency_all", 1)
        elif emotion == "불안":
            self.add_action("patrol_normal", 3)
        else:
            self.add_action("patrol_normal", 4)
        if data.get('cctv_intrusion') == "침입":
            self.add_action("drone_launch", 2)
        if threat_level >= 3:
            self.add_action("emergency_all", 1)
