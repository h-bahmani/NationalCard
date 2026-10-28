"""Optional high-accuracy Persian OCR engines.

The optional engines are lazy, failure-isolated, and line-recognition only.  They
return raw model evidence; final acceptance is handled by local_worker using
engine-level consensus so multiple preprocessing variants from one model cannot
outvote other models.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import math
from pathlib import Path

os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK','True')


def _truthy(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def normalize_persian(text: object) -> str:
    return str(text or "").strip().translate(
        str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه"})
    )


def _json_payload(result):
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if payload is None and hasattr(result, "to_json"):
        payload = result.to_json()
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload if isinstance(payload, dict) else {}


def _bina_input_width(width, height, target_height=48):
    """Use the published 160..3200 dynamic-width range without clipping a field."""
    if min(width,height)<=0:
        raise ValueError('Empty OCR crop')
    return min(3200,max(160,32*math.ceil(target_height*width/height/32)))


class BinaReader:
    name = 'bina_rizeh_ppocrv6'

    def __init__(self):
        from static_ctc import StaticCTCReader
        root=Path(os.getenv('BINA_MODEL_DIR','models/Bina-0.2-Rizeh')).expanduser()
        if not root.is_absolute(): root=Path(__file__).resolve().parent/root
        # Use the model's RecResizeImg width (1536), as Paddle's inference
        # preprocessor does. 160 is a supported tensor bound, not its padding
        # width; shortening it causes repeated/missing characters on short names.
        self.model=StaticCTCReader(root/'inference',device=os.getenv('BINA_DEVICE','cpu'))

    def read_scored(self,image):
        text,score=self.model.read_scored(image)
        return normalize_persian(text),score


class PaddleV5Reader:
    name = 'paddleocr_v5_arabic'

    def __init__(self):
        from static_ctc import StaticCTCReader
        root=Path(os.getenv('PADDLEOCR_V5_MODEL_DIR','models/arabic_PP-OCRv5_mobile_rec')).expanduser()
        if not root.is_absolute(): root=Path(__file__).resolve().parent/root
        self.model=StaticCTCReader(root,device=os.getenv('PADDLE_DEVICE','cpu'),logical_mode='bidi')

    def read_scored(self,image):
        text,score=self.model.read_scored(image)
        return normalize_persian(text),score


def build_optional_readers():
    """Build enabled readers without allowing one broken optional dependency to stop OCR."""
    readers = []
    errors = []
    specs = []
    if _truthy("BINA_ENABLED", "1"):
        specs.append(("bina", BinaReader))
    if _truthy("PADDLEOCR_V5_ENABLED", "1"):
        specs.append(("paddle_v5", PaddleV5Reader))
    for name, cls in specs:
        try:
            readers.append(cls())
        except Exception as exc:  # optional engine failures are evidence, not fatal
            errors.append({"engine": name, "error": type(exc).__name__, "message": str(exc)[:320]})
    return readers, errors


def engine_status():
    """UI preflight with exact package/model/environment diagnostics."""
    import importlib.util
    import sys

    rows = []
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    supported_py = (3, 9) <= sys.version_info[:2] <= (3, 13)
    paddle_pkg = importlib.util.find_spec("paddleocr") is not None and importlib.util.find_spec("paddle") is not None
    marker = {}
    marker_path = Path(__file__).resolve().parent / "ACCURACY_READY.json"
    if marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:
            marker = {}
    same_python = str(marker.get("executable", "")).casefold() == str(sys.executable).casefold()
    bina_verified = bool(same_python and marker.get("bina_inference"))
    paddle_verified = bool(same_python and marker.get("paddle_v5_inference"))

    bina_on = _truthy("BINA_ENABLED", "1")
    bina_dir = os.getenv("BINA_MODEL_DIR", "models/Bina-0.2-Rizeh").strip()
    bina_root = Path(bina_dir).expanduser()
    if not bina_root.is_absolute():
        bina_root = (Path(__file__).resolve().parent / bina_root).resolve()
    bina_files = [
        bina_root / "bina_text_recognition.py",
        bina_root / "inference" / "inference.json",
        bina_root / "inference" / "inference.pdiparams",
        bina_root / "inference" / "inference.yml",
    ]
    bina_model_ready = all(path.is_file() for path in bina_files)
    bina_ready = supported_py and paddle_pkg and bina_model_ready and bina_verified
    if not supported_py:
        bina_detail = f"Python {py} برای این نصب پشتیبانی نمی‌شود؛ START_WINDOWS.bat را اجرا کن"
    elif not paddle_pkg:
        bina_detail = "Paddle نصب نیست؛ INSTALL_ACCURACY_NOW.bat را اجرا کن"
    elif not bina_model_ready:
        bina_detail = "وزن Bina ناقص است؛ INSTALL_ACCURACY_NOW.bat را اجرا کن"
    elif not bina_verified:
        bina_detail = "مدل هنوز در همین محیط با inference واقعی تأیید نشده؛ Setup را اجرا کن"
    else:
        bina_detail = "آماده و تست‌شده · مدل محلی Bina/PP-OCRv6"
    rows.append({"engine":"Bina 0.2 Rizeh / PP-OCRv6","enabled":bina_on,"ready":bina_ready,"detail":bina_detail})

    paddle_on = _truthy("PADDLEOCR_V5_ENABLED", "1")
    pp_dir = os.getenv("PADDLEOCR_V5_MODEL_DIR", "models/arabic_PP-OCRv5_mobile_rec").strip()
    pp_root = Path(pp_dir).expanduser()
    if not pp_root.is_absolute():
        pp_root = (Path(__file__).resolve().parent / pp_root).resolve()
    pp_files = [pp_root / "inference.json", pp_root / "inference.pdiparams", pp_root / "inference.yml"]
    pp_model_ready = all(path.is_file() for path in pp_files)
    pp_ready = supported_py and paddle_pkg and pp_model_ready and paddle_verified
    if not supported_py:
        pp_detail = f"Python {py} برای Paddle مناسب نیست؛ START_WINDOWS.bat را اجرا کن"
    elif not paddle_pkg:
        pp_detail = "paddlepaddle/paddleocr نصب نیست؛ INSTALL_ACCURACY_NOW.bat را اجرا کن"
    elif not pp_model_ready:
        pp_detail = "مدل محلی PP-OCRv5 ناقص است؛ INSTALL_ACCURACY_NOW.bat را اجرا کن"
    elif not paddle_verified:
        pp_detail = "مدل هنوز در همین محیط با inference واقعی تأیید نشده؛ Setup را اجرا کن"
    else:
        pp_detail = "آماده و تست‌شده · مدل فارسی/عربی محلی"
    rows.append({"engine":"PP-OCRv5 Persian/Arabic","enabled":paddle_on,"ready":pp_ready,"detail":pp_detail})
    rows.append({"engine":"Persian CRNN (bundled)","enabled":True,"ready":True,"detail":"همراه پروژه · fallback"})
    rows.append({"engine":"Paddle names (bundled)","enabled":True,"ready":True,"detail":"همراه پروژه · fallback نام‌ها"})
    return rows
