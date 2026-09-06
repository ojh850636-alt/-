"""Small uvicorn runner that imports the test-friendly app and runs it.

This file intentionally keeps the runtime tiny: import `app` from
`lucia_app` (which re-exports the cleaned FastAPI app) and run with
uvicorn. Use this when you want a simple Python entrypoint instead of
calling `uvicorn` on the CLI.
"""

import os

if __name__ == "__main__":
    # Lazy import so module-level side-effects are minimized when the file
    # is imported by other tools or during tests.
    from lucia_app import app
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload_flag = os.getenv("DEV_RELOAD", "true").lower() in ("1", "true", "yes")

    uvicorn.run(app, host=host, port=port, reload=reload_flag)
