"""Optional Roboflow proposal detector for ID cards.

Roboflow is proposal-only: a remote bounding box is never accepted as an Iranian
national ID card by itself. `detector.py` must still verify the crop against the
local Iranian-card layout templates. Remote use is opt-in because identity images
are sensitive.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from PIL import Image

DEFAULT_MODELS = (
    "ichi-hhy0r/screen-idcard-detection-3-yolo26n-t1",
    "idcard_2_0/7",
    "id-card-detection-xtiwy/3",
    "idcard-detector/1",
)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, fallback: float) -> float:
    try:
        raw = os.getenv(name)
        return float(raw) if raw is not None and raw.strip() else float(fallback)
    except (TypeError, ValueError):
        return float(fallback)


def configured() -> bool:
    return bool(os.getenv("ROBOFLOW_API_KEY", "").strip()) and _truthy(os.getenv("ROBOFLOW_ENABLED", "0"))


def configuration_status() -> dict:
    enabled = _truthy(os.getenv("ROBOFLOW_ENABLED", "0"))
    key = bool(os.getenv("ROBOFLOW_API_KEY", "").strip())
    try:
        import inference_sdk  # noqa: F401
        sdk = True
    except Exception:
        sdk = False
    if not enabled and key:
        reason = "کلید ثبت شده است، اما Roboflow خاموش است. سوییچ «فعال‌کردن Roboflow» را روشن کن."
    elif not enabled:
        reason = "Roboflow خاموش است."
    elif not key:
        reason = "API Key وارد نشده است."
    elif not sdk:
        reason = "بسته inference-sdk نصب نیست."
    else:
        reason = "تنظیمات اولیه کامل است؛ برای اطمینان دکمه «تست اتصال» را بزن."
    return {"enabled": enabled, "has_key": key, "sdk_ready": sdk, "configured": enabled and key and sdk, "message": reason}


def model_ids():
    raw = os.getenv("ROBOFLOW_MODEL_IDS", "").strip()
    return [item.strip() for item in raw.split(",") if item.strip()] if raw else list(DEFAULT_MODELS)


def _client(api_url: str, key: str):
    from inference_sdk import InferenceHTTPClient
    client = InferenceHTTPClient(api_url=api_url, api_key=key)
    try:
        from inference_sdk import InferenceConfiguration
        if hasattr(client, "configure"):
            client = client.configure(InferenceConfiguration(api_key_transport="header"))
    except Exception:
        pass
    return client


def test_connection(api_key: str | None = None, model_id: str | None = None, api_url: str | None = None) -> dict:
    """Make one tiny real hosted inference request to validate key + model access."""
    key = (api_key if api_key is not None else os.getenv("ROBOFLOW_API_KEY", "")).strip()
    model = (model_id or (model_ids()[0] if model_ids() else "")).strip()
    url = (api_url or os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")).strip()
    if not key:
        return {"ok": False, "code": "missing_key", "message": "API Key وارد نشده است."}
    if not model:
        return {"ok": False, "code": "missing_model", "message": "Model ID مشخص نشده است."}
    try:
        client = _client(url, key)
    except Exception as exc:
        return {"ok": False, "code": "sdk_error", "message": f"inference-sdk آماده نیست: {exc}"}

    handle = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    path = Path(handle.name)
    handle.close()
    try:
        Image.new("RGB", (160, 100), "white").save(path, quality=85)
        client.infer(str(path), model_id=model)
        return {"ok": True, "model_id": model, "message": "اتصال Roboflow و دسترسی به مدل برقرار است."}
    except Exception as exc:
        text = str(exc).strip()
        low = text.lower()
        if "401" in low or "unauthorized" in low or "api key" in low and "invalid" in low:
            message = "API Key توسط Roboflow پذیرفته نشد. کلید را دوباره از Roboflow کپی کن."
            code = "unauthorized"
        elif "403" in low or "forbidden" in low:
            message = "کلید معتبر است اما به این مدل/Workspace دسترسی ندارد."
            code = "forbidden"
        elif "404" in low or "not found" in low:
            message = "Model ID پیدا نشد. شناسه مدل را بررسی کن."
            code = "model_not_found"
        else:
            message = "اتصال Roboflow کامل نشد: " + (text[:260] or type(exc).__name__)
            code = type(exc).__name__
        return {"ok": False, "code": code, "message": message}
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _prediction_rows(payload):
    if not isinstance(payload, dict):
        return []
    rows = payload.get("predictions")
    if isinstance(rows, list):
        return rows
    for value in payload.values():
        if isinstance(value, dict) and isinstance(value.get("predictions"), list):
            return value["predictions"]
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and isinstance(item.get("predictions"), list):
                    return item["predictions"]
    return []


def detect_candidates(image: Image.Image, *, min_confidence: float | None = None, max_candidates: int = 6, progress=None):
    """Return card proposals in original-image coordinates; failures stay diagnostic-only."""
    status = configuration_status()
    if not status["enabled"] or not status["has_key"]:
        return [], {"enabled": status["enabled"], "reason": "not_configured", "message": status["message"]}
    if not status["sdk_ready"]:
        return [], {"enabled": True, "error": "inference_sdk_missing", "message": status["message"]}

    key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    api_url = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com").strip()
    threshold = max(0.0, min(1.0, _float_env("ROBOFLOW_MIN_CONFIDENCE", min_confidence if min_confidence is not None else 0.35)))
    allowed = {"id-card", "id_card", "idcard", "national-id-card", "national_id_card", "card", "id card", "id-card-front", "idcard-front", "id card front"}
    try:
        client = _client(api_url, key)
    except Exception as exc:
        return [], {"enabled": True, "error": "client_init_failed", "message": str(exc)[:220]}

    candidates = []
    errors = []
    tested_models = []
    if progress:
        try:
            progress("yolo_start", {"models": model_ids()})
        except Exception:
            pass
    early_stop = max(threshold, _float_env("ROBOFLOW_EARLY_STOP_CONFIDENCE", 0.78))

    handle = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    path = Path(handle.name)
    handle.close()
    try:
        image.convert("RGB").save(path, quality=94)
        for model_id in model_ids():
            tested_models.append(model_id)
            model_best = 0.0
            if progress:
                try:
                    progress("yolo_model_start", {"model_id": model_id})
                except Exception:
                    pass
            try:
                payload = client.infer(str(path), model_id=model_id)
                for row in _prediction_rows(payload):
                    label = str(row.get("class") or row.get("class_name") or "").strip().lower()
                    confidence = float(row.get("confidence", 0) or 0)
                    if confidence < threshold:
                        continue
                    normalized_label = label.replace(" ", "-")
                    if label and normalized_label not in allowed and not ("card" in label and "lanyard" not in label):
                        continue
                    x = float(row.get("x", 0)); y = float(row.get("y", 0))
                    w = float(row.get("width", 0)); h = float(row.get("height", 0))
                    if w <= 0 or h <= 0:
                        continue
                    l = max(0.0, x - w / 2); t = max(0.0, y - h / 2)
                    r = min(float(image.width), x + w / 2); b = min(float(image.height), y + h / 2)
                    if r - l < 35 or b - t < 22:
                        continue
                    candidates.append({"box": [l, t, r, b], "confidence": confidence,
                                       "class": label or "card", "model_id": model_id})
                    model_best = max(model_best, confidence)
            except Exception as exc:
                errors.append({"model_id": model_id, "error": type(exc).__name__, "message": str(exc)[:260]})
                if progress:
                    try:
                        progress("yolo_model_error", {"model_id": model_id, "error": type(exc).__name__})
                    except Exception:
                        pass
            if progress:
                try:
                    progress("yolo_model_done", {"model_id": model_id, "confidence": model_best, "found": model_best >= threshold})
                except Exception:
                    pass
            if model_best >= early_stop:
                break
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    kept = []
    for candidate in candidates:
        l, t, r, b = candidate["box"]
        area = max(1.0, (r - l) * (b - t))
        duplicate = False
        for other in kept:
            L, T, R, B = other["box"]
            inter = max(0.0, min(r, R) - max(l, L)) * max(0.0, min(b, B) - max(t, T))
            union = area + max(1.0, (R - L) * (B - T)) - inter
            if union and inter / union > 0.65:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
        if len(kept) >= max_candidates:
            break
    best = kept[0] if kept else None
    return kept, {"enabled": True, "models": model_ids(), "tested_models": tested_models, "errors": errors,
                  "count": len(kept), "card_present": bool(best), "best": best}
