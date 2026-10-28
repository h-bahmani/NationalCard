from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from cardreader_module import CardReader
from deep_adapter import analyze_deep
from ocr_ensemble import engine_status
from images import MAX_BYTES
from typing import Literal
import json
import threading
import sys

app = FastAPI(
    title='CardReader Company Service',
    version='2.0.0',
    description='Iranian national ID card detection + OCR with stable company-facing JSON',
)
reader = CardReader()
_processing_lock = threading.Lock()


def _analyze(data, *, deep=False, **options):
    # Bound model memory usage; a busy caller can retry without blocking health checks.
    if not _processing_lock.acquire(blocking=False):
        raise HTTPException(503, 'سرویس مشغول پردازش است؛ دوباره تلاش کنید.', headers={'Retry-After': '2'})
    try:
        if deep:
            return analyze_deep(data, **options)
        return reader.analyze(data, allow_review_ocr=True, **options)
    finally:
        _processing_lock.release()


@app.get('/health')
def health():
    engines = engine_status()
    important = [r for r in engines if r['engine'].startswith(('Bina','PP-OCRv5'))]
    ready = len(important) >= 2 and all(r.get('ready') and r.get('enabled') for r in important)
    return {
        'status': 'ok' if ready else 'degraded',
        'service_version': '2.0.0',
        'python': sys.version.split()[0],
        'engines': engines,
        'accuracy_ready': ready,
        'message': 'سرویس آماده است.' if ready else 'سرویس اجرا می‌شود، اما موتورهای دقت‌بالا کامل آماده نیستند.',
    }


@app.get('/ready')
def ready():
    body = health()
    return JSONResponse(body, status_code=200 if body['accuracy_ready'] else 503)


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, 'حجم فایل باید کمتر از ۳۰ مگابایت باشد.')
    if not data:
        raise HTTPException(400, 'فایل خالی است.')
    return data


@app.post('/v1/extract')
async def extract_v1(
    file: UploadFile = File(...),
    include_debug: bool = Query(False, description='Include the full internal result beside the stable company payload'),
    min_confidence: float = Query(0.90, ge=0.5, le=1.0, description='Minimum field confidence; at least two engines must agree'),
    selection_threshold: float = Query(0.85, ge=0.5, le=1.0, description='Below this card confidence, request region selection'),
    deep: bool = Query(False, description='Explicit slower recovery scan; fast Web Core remains the default'),
    card_box: str | None = Form(None, description='Optional JSON [left,top,right,bottom], normalized to [0,1]'),
    kind: Literal['smart', 'old'] = Form('smart'),
):
    try:
        box = None
        if card_box is not None:
            try:
                box = json.loads(card_box)
                if not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float) for v in box):
                    raise ValueError('Invalid card_box')
                from cardreader_module import _validate_box
                _validate_box(box)
            except (ValueError, TypeError):
                raise HTTPException(422, 'محدوده باید چهار عدد معتبر بین صفر و یک با عرض و ارتفاع مثبت باشد.') from None
        result = await run_in_threadpool(
            _analyze,
            await _read_upload(file),
            deep=deep,
            card_box=box,
            kind=kind,
            min_confidence=min_confidence,
            selection_threshold=selection_threshold,
        )
        payload = result.get('company') or {}
        if include_debug:
            payload = {**payload, 'debug': result}
        return JSONResponse(payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f'پردازش تصویر انجام نشد: {type(exc).__name__}')


@app.post('/extract')
async def extract_legacy(file: UploadFile = File(...)):
    """Backward-compatible endpoint returning the full result."""
    try:
        result = await run_in_threadpool(_analyze, await _read_upload(file))
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f'پردازش تصویر انجام نشد: {type(exc).__name__}')
