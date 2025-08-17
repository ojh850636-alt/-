"""Run the test-friendly FastAPI app locally via uvicorn.

Usage:
    python run_uvicorn.py

This script imports the lightweight `app` from `lucia_app` so the same app
instance used by tests is launched.
"""
import os

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("LUCIA_HOST", "127.0.0.1")
    port = int(os.getenv("LUCIA_PORT", "8000"))
    uvicorn.run("lucia_app:app", host=host, port=port, log_level="info", reload=False)
