"""
Backward-compatible entry point.

The actual FastAPI app now lives in api/main.py.
This file exists so that ``uvicorn main:app`` still works if someone
hasn't updated their launch config.
"""

from api.main import app  # noqa: F401

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port)
