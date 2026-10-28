"""Project settings and runtime environment helpers.

The app reads `.env` on every rerun and may also apply temporary values entered in
Streamlit.  Secrets are never included in exported OCR results.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import dotenv_values, load_dotenv

DEFAULTS = {
    "GROQ_API_KEY": "",
    "OPENROUTER_API_KEY": "",
    "GROQ_VISION_MODEL": "qwen/qwen3.8-27b",
    "OPENROUTER_VISION_MODEL": "openrouter/free",
    "CLOUD_ASSIST_ENABLED": "0",
    "NAME_CORRECTION_ENABLED": "0",
    "OCR_PROVIDER": "local",
    "ROBOFLOW_ENABLED": "0",
    "ROBOFLOW_API_KEY": "",
    "ROBOFLOW_API_URL": "https://serverless.roboflow.com",
    "ROBOFLOW_MODEL_IDS": "id-card-detection-xtiwy/3,ichi-hhy0r/screen-idcard-detection-3-yolo26n-t1,idcard_2_0/7,idcard-detector/1",
    "ROBOFLOW_MIN_CONFIDENCE": "0.35",
    "ROBOFLOW_EARLY_STOP_CONFIDENCE": "0.78",
    "BINA_ENABLED": "1",
    "BINA_MODEL_DIR": "models/Bina-0.2-Rizeh",
    "BINA_DEVICE": "cpu",
    "PADDLEOCR_V5_ENABLED": "1",
    "PADDLEOCR_V5_MODEL": "arabic_PP-OCRv5_mobile_rec",
    "PADDLEOCR_V5_MODEL_DIR": "models/arabic_PP-OCRv5_mobile_rec",
    "PADDLE_DEVICE": "cpu",
}


def read_settings(root=None, environ=None):
    root = Path(root or Path(__file__).resolve().parent)
    file_values = dotenv_values(root / ".env", encoding="utf-8-sig", interpolate=False)
    env = os.environ if environ is None else environ
    result = {}
    for name, default in DEFAULTS.items():
        if name in env and str(env.get(name, "")).strip() != "":
            value = env.get(name)
        else:
            value = file_values.get(name)
            if value is None or str(value).strip() == "":
                value = default
        result[name] = str(value).strip()
    return result


def apply_runtime_env(root=None):
    """Load project `.env` into os.environ without overwriting explicit process values."""
    root = Path(root or Path(__file__).resolve().parent)
    load_dotenv(root / ".env", encoding="utf-8-sig", override=False)
    return root


def apply_overrides(values: dict):
    """Apply Streamlit settings to the current process and OCR subprocesses."""
    for name in DEFAULTS:
        if name in values and values[name] is not None:
            os.environ[name] = str(values[name])


def save_settings(root, updates: dict):
    """Persist selected settings in `.env` while preserving unrelated lines/comments."""
    root = Path(root or Path(__file__).resolve().parent)
    path = root / ".env"
    existing = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    lines = existing.splitlines()
    wanted = {str(k): str(v) for k, v in updates.items() if k in DEFAULTS}
    seen = set()
    output = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            output.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in wanted:
            output.append(f"{key}={wanted[key]}")
            seen.add(key)
        else:
            output.append(line)
    if output and output[-1].strip():
        output.append("")
    for key in DEFAULTS:
        if key in wanted and key not in seen:
            output.append(f"{key}={wanted[key]}")
    path.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    apply_overrides(wanted)
    return path
