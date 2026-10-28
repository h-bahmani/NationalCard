# -*- coding: utf-8 -*-
"""
Thin production wrapper around the original v26 FastAPI app.

IMPORTANT:
- Does NOT replace detector/OCR/ROI/validation code.
- Adds request tracking, timing and privacy-safe JSONL request logs.
- Concurrency is provided by multiple Uvicorn worker PROCESSES, so the
  original OCR implementation remains untouched inside each process.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Request
from fastapi.responses import JSONResponse

from service_api import app  # the original v26 application

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "api_requests.jsonl"
PROCESS_STARTED = time.monotonic()


def _track_id(request: Request) -> str:
    supplied = (request.headers.get("X-Track-ID") or "").strip()
    if supplied and len(supplied) <= 128:
        return supplied
    return str(uuid.uuid4())


def _safe_log(record: dict) -> None:
    """Never log request bodies, OCR fields, image paths, API keys or tokens."""
    try:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # Logging must never break OCR/API execution.
        pass


@app.middleware("http")
async def company_runtime_middleware(request: Request, call_next):
    track_id = _track_id(request)
    started = time.monotonic()
    status_code = 500
    error_code = None

    try:
        response = await call_next(request)
        status_code = int(response.status_code)
    except Exception as exc:
        error_code = type(exc).__name__
        elapsed_ms = round((time.monotonic() - started) * 1000.0, 2)
        _safe_log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "trackId": track_id,
            "pid": os.getpid(),
            "method": request.method,
            "path": request.url.path,
            "status": 500,
            "durationMs": elapsed_ms,
            "errorCode": error_code,
        })
        return JSONResponse(
            status_code=500,
            content={
                "trackId": track_id,
                "responseCode": "INTERNAL_ERROR",
                "error": {
                    "code": error_code,
                    "message": "Request processing failed."
                }
            },
            headers={"X-Track-ID": track_id},
        )

    elapsed_ms = round((time.monotonic() - started) * 1000.0, 2)
    response.headers["X-Track-ID"] = track_id
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)

    _safe_log({
        "ts": datetime.now(timezone.utc).isoformat(),
        "trackId": track_id,
        "pid": os.getpid(),
        "method": request.method,
        "path": request.url.path,
        "status": status_code,
        "durationMs": elapsed_ms,
        "errorCode": error_code,
    })
    return response


_existing_paths = {getattr(route, "path", None) for route in app.routes}

if "/ready" not in _existing_paths:
    @app.get("/ready", include_in_schema=True)
    def runtime_ready():
        # This proves the current API process imported the original v26 app.
        # The original /health endpoint remains the authority for OCR/model state.
        return {
            "status": "ready",
            "pid": os.getpid(),
            "processUptimeSeconds": round(time.monotonic() - PROCESS_STARTED, 2),
            "note": "Use /health for original v26 model/OCR readiness."
        }

if "/runtime/info" not in _existing_paths:
    @app.get("/runtime/info", include_in_schema=True)
    def runtime_info():
        return {
            "pid": os.getpid(),
            "processUptimeSeconds": round(time.monotonic() - PROCESS_STARTED, 2),
            "trackIdHeader": "X-Track-ID",
            "concurrencyModel": "multi-process uvicorn workers",
            "ocrCore": "original-v26-unchanged"
        }
