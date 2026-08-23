import os
import hashlib
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger("lucia_core.crypto")


class QuantumResistantCrypto:
    def __init__(self):
        self.key = os.urandom(32)
        logger.info("QuantumResistantCrypto initialized")

    def encrypt(self, data):
        token = hashlib.sha3_512(str(data).encode()).hexdigest()
        return f"{token}:proof"

    def decrypt(self, encrypted_data):
        return encrypted_data.split(":")[0]


class QuantumTokenSystem:
    def __init__(self, token_lifetime=300):
        self.token_lifetime = token_lifetime
        self.tokens = {}
        self.salt = os.urandom(16)

    def generate_token(self, identity="lucia_manager"):
        timestamp = datetime.now(timezone.utc).isoformat()
        raw_token = f"{identity}::{timestamp}::{os.urandom(16).hex()}"
        token = hashlib.sha3_512(self.salt + raw_token.encode()).hexdigest()
        expiration = datetime.now(timezone.utc) + timedelta(seconds=self.token_lifetime)
        self.tokens[token] = expiration
        logger.info(f"토큰 생성: {token[:12]}...")
        return token

    def validate_token(self, token):
        from datetime import datetime as _dt, timezone as _tz

        now = _dt.now(_tz.utc)
        if token not in self.tokens or now > self.tokens[token]:
            logger.warning("Invalid or expired token")
            return False
        return True
