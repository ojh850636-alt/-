"""Lightweight ASGI entrypoint that wires the modularized components.

This file intentionally re-uses the cleaned, test-friendly server in
`lucia_ultimate_quantum_integrated_fixed.py` and exposes `app` for
uvicorn/CI to import. Keeping a thin wrapper avoids duplicating endpoint
definitions and keeps heavy optional deps lazy.
"""

from pathlib import Path
import logging

# Ensure project root is on path if needed (tests import modules directly)
ROOT = Path(__file__).parent

# Re-export the test-friendly FastAPI app used by tests
from lucia_ultimate_quantum_integrated_fixed import app  # noqa: E402,F401

# Optional startup logging
logger = logging.getLogger("lucia_app")
logger.info("lucia_app: application imported and ready")

__all__ = ["app"]
