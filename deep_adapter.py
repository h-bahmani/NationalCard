from __future__ import annotations

from typing import Any

from cardreader_module import (
    PreparedCard,
    prepare,
    prepare_manual,
    read_prepared,
)
from images import load_image


def _field_text(row: dict[str, Any]) -> str:
    return str(
        row.get("accepted_text")
        or row.get("raw_selected_text")
        or row.get("text")
        or ""
    ).strip()


def _score_result(result: dict[str, Any]) -> tuple[float, int]:
    """Rank only real OCR evidence. Never invent or repair field text."""
    raw = ((result.get("raw_result") or {}).get("fields") or {})
    exported = result.get("fields") or {}
    company = result.get("company") or {}
    cfields = company.get("fields") or {}

    weights = {
        "national_id": 4.0,
        "first_name": 1.5,
        "last_name": 1.5,
        "father_name": 1.0,
        "birth_date": 1.5,
        "expiry_date": 1.5,
    }

    score = 0.0
    readable = 0

    for name, weight in weights.items():
        text = _field_text(raw.get(name) or {})
        if not text:
            crow = cfields.get(name) or {}
            text = str(crow.get("display_value") or crow.get("value") or "").strip()
        if not text:
            erow = exported.get(name) or {}
            text = str(erow.get("value") or erow.get("suggested_value") or "").strip()

        if text:
            readable += 1
            score += weight

        conf = ((cfields.get(name) or {}).get("confidence") or {}).get("score")
        if isinstance(conf, (int, float)):
            score += max(0.0, min(1.0, float(conf))) * 0.20

    # Prefer passes without a local OCR crash.
    errors = result.get("errors") or []
    if any(isinstance(e, dict) and e.get("stage") == "local" for e in errors):
        score -= 3.0

    return round(score, 4), readable


def analyze_deep(
    image_bytes: bytes,
    *,
    frame: int = 0,
    number_box=None,
    field_boxes=None,
    card_box=None,
    kind: str = "smart",
    min_confidence: float = 0.90,
    selection_threshold: float = 0.85,
) -> dict[str, Any]:
    """Explicit slow recovery. Fast/Web Core behavior is not modified."""
    image = load_image(image_bytes, frame=frame)
    prepared = (
        prepare_manual(image, card_box, kind)
        if card_box is not None
        else prepare(image)
    )

    # First pass is exactly the same RAW read as the Web Core.
    base = read_prepared(
        prepared,
        number_box=number_box,
        field_boxes=field_boxes,
        cloud=False,
        key="",
        min_confidence=min_confidence,
        selection_threshold=selection_threshold,
        allow_review_ocr=True,
    )

    best = base
    best_score, best_readable = _score_result(base)
    attempts = [{
        "profile": "raw",
        "score": best_score,
        "readable_fields": best_readable,
    }]
    selected = "raw"

    # If the card was not allowed/detected, enhancement cannot safely fix identity.
    if prepared.aligned is None or not prepared.decision.get("allowed"):
        best["deep_scan"] = {
            "attempted": True,
            "selected_profile": "raw",
            "reason": "card_not_eligible_for_ocr",
            "attempts": attempts,
        }
        return best

    from quality_recovery import enhance_card

    # Deep is opt-in, so it may spend more time trying deterministic recovery views.
    for profile in ("auto", "clahe", "sharp", "denoise_sharp", "bw"):
        try:
            enhanced = enhance_card(prepared.aligned, profile)
            retry = PreparedCard(
                prepared.image,
                prepared.detection,
                prepared.decision,
                enhanced,
                prepared.roi,
            )
            candidate = read_prepared(
                retry,
                number_box=number_box,
                field_boxes=field_boxes,
                cloud=False,
                key="",
                min_confidence=min_confidence,
                selection_threshold=selection_threshold,
                allow_review_ocr=True,
            )
            score, readable = _score_result(candidate)
            attempts.append({
                "profile": profile,
                "score": score,
                "readable_fields": readable,
            })

            # Never replace the fast/raw result with a worse deep result.
            if score > best_score:
                best = candidate
                best_score = score
                best_readable = readable
                selected = profile

            if best_readable >= 6:
                break

        except Exception as exc:
            attempts.append({
                "profile": profile,
                "score": -999.0,
                "readable_fields": 0,
                "error": f"{type(exc).__name__}: {str(exc)[:250]}",
            })

    best["deep_scan"] = {
        "attempted": True,
        "selected_profile": selected,
        "selected_score": best_score,
        "selected_readable_fields": best_readable,
        "attempts": attempts,
        "kept_raw_if_deep_worse": selected == "raw",
    }
    if isinstance(best.get("company"), dict):
        best["company"]["deep_scan"] = best["deep_scan"]

    return best
