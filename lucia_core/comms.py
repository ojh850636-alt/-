import logging
import ssl
import time
import warnings

logger = logging.getLogger('lucia_core.comms')

try:
    from paho.mqtt.client import Client, MQTTv5
    PAHO_AVAILABLE = True
except Exception:
    PAHO_AVAILABLE = False

class SecureMQTTLayer:
    def __init__(self, broker_ip='localhost', port=8883):
        self.broker_ip = broker_ip
        self.port = port
        self.connected = False
        if PAHO_AVAILABLE:
            try:
                # Recent paho-mqtt may emit a DeprecationWarning about the
                # callback API version during Client construction. This repo
                # doesn't rely on the old callback API; suppress that specific
                # warning to keep test output clean while preserving behavior.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=DeprecationWarning,
                        message=r".*Callback API version.*"
                    )
                    self.client = Client(protocol=MQTTv5)
            except Exception:
                self.client = None
        else:
            self.client = None

    def send_command(self, command, token, topic='lucia/command', priority=1):
        logger.info(f"(sim) MQTT send {topic}: {command} (token={token[:8]})")
        return True

    def emergency_all(self):
        self.send_command('LAUNCH_EMERGENCY', 'simtoken', 'lucia/drone')

    def patrol_normal(self):
        self.send_command('SCAN', 'simtoken', 'lucia/cctv')
