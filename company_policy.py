"""Stable company-facing confidence and decision policy.

The scores in this module are *operational confidence scores*, not calibrated
probabilities.  The JSON explicitly carries ``calibrated: false`` until the
customer validates/calibrates thresholds on a held-out production dataset.
"""
from __future__ import annotations

import math
import re
from typing import Any

from presentation import field_display, number_display, FIELD_NAMES
from validation import checksum_valid, normalize

DEFAULT_FIELD_THRESHOLD = 0.90
DEFAULT_SELECTION_THRESHOLD = 0.85


def _threshold(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.5 <= value <= 1.0:
        raise ValueError('Confidence threshold must be finite and between 0.5 and 1.')
    return value


def selection_policy(detection: dict, threshold=DEFAULT_SELECTION_THRESHOLD) -> dict:
    threshold = _threshold(threshold)
    confidence = detection_confidence(detection)
    multiple = bool(detection.get('multiple_cards'))
    needed = multiple or ((not detection.get('detected') or confidence['score'] < threshold) and not detection.get('user_selected'))
    if detection.get('intake_rejected') and not detection.get('detected') and not multiple:
        needed=False
    return {'required': needed, 'threshold_percent': _pct(threshold),
            'confidence_percent': confidence['percent'],
            'reason': 'multiple_cards' if multiple else 'card_confidence_low' if needed else None}


def _clip(value: float) -> float:
    try:
        value = float(value)
        return max(0.0, min(1.0, value)) if math.isfinite(value) else 0.0
    except Exception:
        return 0.0


def _pct(value: float) -> int:
    return int(round(100 * _clip(value)))


def detection_confidence(detection: dict[str, Any]) -> dict[str, Any]:
    """Operational score derived from independent evidence already produced by detector."""
    d = detection or {}
    method = str(d.get("method") or "unknown")
    detected = bool(d.get("detected"))
    user_selected = bool(d.get("user_selected"))
    ref = d.get("reference_verification") or {}
    ref_score = _clip(ref.get("score") or 0.0)

    if detected:
        evidence = d.get('selection_detection') or d
        inliers = float(evidence.get("inliers") or 0)
        matches = float(evidence.get("matches") or 0)
        ratio = inliers / matches if matches else 0.0
        coverage = _clip((evidence.get("reference_coverage") or 0.0) / 0.35)
        geom = 0.42 * _clip(inliers / 18.0) + 0.33 * _clip(ratio / 0.72) + 0.25 * coverage
        if "reference_verify" in method:
            geom = max(geom, 0.55 + 0.40 * ref_score)
        if "roboflow_proposal" in method:
            proposal = _clip(d.get("proposal_confidence") or d.get("yolo_confidence") or 0.0)
            geom = max(geom, 0.55 * geom + 0.45 * proposal)
        if user_selected:
            # Human selected the location; confidence here is about card type, not presence.
            geom = max(geom, 0.70 if d.get("manual_verification") == "verified" else 0.58)
        score = max(0.62, geom)
        state = "verified"
    elif d.get('intake_rejected'):
        score=0.05
        state='not_detected'
    elif d.get("yolo_presence"):
        score = 0.35 + 0.45 * _clip(d.get("yolo_confidence") or 0.0)
        state = "card_present_unverified"
    elif d.get("ocr_candidate") or d.get("reference_probable") or d.get("partial_card"):
        score = max(0.35, 0.35 + 0.40 * ref_score)
        state = "candidate"
    elif user_selected:
        score = 0.40
        state = "manual_unverified"
    else:
        score = 0.05
        state = "not_detected"

    return {
        "score": round(_clip(score), 4),
        "percent": _pct(score),
        "state": state,
        "method": method,
        "calibrated": False,
        "score_type": "operational_confidence",
    }


def image_quality_confidence(decision: dict[str, Any]) -> dict[str, Any]:
    q = (decision or {}).get("quality") or {}
    sharp = float(q.get("laplacian_variance") or 0.0)
    contrast = float(q.get("contrast_std") or 0.0)
    brightness = float(q.get("brightness", 128.0))
    sharp_score = _clip(sharp / 110.0)
    contrast_score = _clip(contrast / 45.0)
    exposure_score = _clip(1.0 - abs(brightness - 145.0) / 145.0)
    score = 0.50 * sharp_score + 0.30 * contrast_score + 0.20 * exposure_score
    source = (decision or {}).get('source_quality') or q
    native_size = q.get('native_card_size') or source.get('original_size') or source.get('processed_size') or []
    low_resolution = bool(native_size and min(native_size) < 180)
    if low_resolution:
        score = min(score, 0.40)
    return {
        "score": round(score, 4),
        "percent": _pct(score),
        "calibrated": False,
        "possibly_blurry": bool(q.get("possibly_blurry")),
        "recovery_recommended": bool(q.get("ocr_recovery_recommended")),
        "low_source_resolution": low_resolution,
    }


def _engine_agreement(field: dict[str, Any]) -> tuple[int, int, float]:
    evidence = field.get("engine_evidence") or {}
    selected = str(field.get("accepted_text") or field.get("raw_selected_text") or field.get("text") or "").strip()
    if not evidence or not selected:
        return 0, 0, 0.0
    from local_worker import _canonical
    # field name is attached by caller; this helper receives temporary key if present.
    name = field.get("__name__", "")
    selected_key = _canonical(selected, name) if name else selected
    agreeing = 0
    ready = 0
    scores = []
    for _, row in evidence.items():
        value = row.get("value")
        if value is None:
            continue
        ready += 1
        if value == selected_key and row.get('stable'):
            agreeing += 1
            scores.append(_clip(row.get("mean_score") or 0.0))
    mean = sum(scores) / len(scores) if scores else 0.0
    return agreeing, ready, mean


def field_confidence(name: str, field: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    f = dict(field or {})
    f["__name__"] = name
    shown = field_display(name, f)
    raw = str(f.get("raw_selected_text") or f.get("text") or "").strip()
    agreeing, ready, mean_score = _engine_agreement(f)

    if name == "national_id":
        value = normalize(shown.get("value") or raw or "")
        valid = bool(value and checksum_valid(value))
        if valid and agreeing >= 2:
            score = 0.98
        elif valid and agreeing == 1:
            score = 0.86
        elif valid:
            score = 0.78
        elif raw:
            score = 0.42
        else:
            score = 0.05
    else:
        status = shown.get("status")
        decision = f.get("decision")
        if status == "read" and decision in (None, "accepted") and agreeing >= 2:
            score = max(0.90, min(0.98, 0.90 + 0.08 * mean_score))
        elif status == "read" and decision in (None, "accepted"):
            score = 0.80
        elif raw and agreeing >= 2:
            score = 0.74
        elif raw:
            score = 0.52
        else:
            score = 0.05

    # Agreement cannot override an explicit ambiguity/rejection from the OCR gate.
    if f.get('ambiguous') or f.get('decision') in ('review', 'rejected'):
        # A stable single-model transcription is still review-only, but its
        # operational score should not be identical to arbitrary OCR garbage.
        # Independent disagreement between strong models still caps at 49%.
        evidence = f.get('engine_evidence') or {}
        strong_values = {row.get('value') for row in evidence.values()
                         if row.get('stable') and row.get('mean_score', 0) >= .90 and row.get('value')}
        single_numeric = name == 'national_id' and agreeing == 1 and valid and len(strong_values) == 1
        score = min(score, 0.86 if single_numeric else 0.49)
    elif agreeing >= 2:
        # Two equally weak engines are not strong evidence.
        score = min(score, mean_score)

    return {
        "score": round(_clip(score), 4),
        "percent": _pct(score),
        "calibrated": False,
        "engine_agreement": agreeing,
        "engines_with_candidate": ready,
    }


def build_company_result(result: dict[str, Any], *, min_confidence=DEFAULT_FIELD_THRESHOLD,
                         selection_threshold=DEFAULT_SELECTION_THRESHOLD) -> dict[str, Any]:
    """Return the small, stable payload intended for company integration."""
    detection = result.get("detection") or {}
    decision = result.get("decision") or {}
    det_conf = detection_confidence(detection)
    quality = image_quality_confidence(decision)
    min_confidence = _threshold(min_confidence)
    selection = selection_policy(detection, selection_threshold)

    raw_fields = ((result.get("raw_result") or {}).get("fields") or {})
    exported_fields = result.get("fields") or {}
    fields: dict[str, Any] = {}

    # National ID is represented in raw_result fields in modern builds; still support legacy.
    all_names = ["national_id", *FIELD_NAMES.keys()]
    for name in all_names:
        raw_field = raw_fields.get(name) or {}
        if name == "national_id":
            shown_num = number_display(result.get("raw_result") or result)
            raw_value = shown_num.get("value") if shown_num.get("value") != "—" else None
            accepted_value = result.get("national_id")
            tmp = dict(raw_field)
            if raw_value and not tmp.get("text"):
                tmp["text"] = str(raw_value)
            conf = field_confidence(name, tmp, result)
            fields[name] = {
                "value": accepted_value if accepted_value and checksum_valid(accepted_value) and conf['score'] >= min_confidence and conf['engine_agreement'] >= 2 else None,
                "display_value": raw_value,
                "status": "accepted" if accepted_value else ("review" if raw_value else "unread"),
                "confidence": conf,
                "checksum_valid": bool(accepted_value and checksum_valid(accepted_value)),
            }
            continue

        exported = exported_fields.get(name) or {}
        shown = field_display(name, raw_field) if raw_field else {"value": "—", "status": "unread"}
        display_value = shown.get("value")
        if display_value == "—":
            display_value = None
        conf = field_confidence(name, raw_field, result)
        fields[name] = {
            "value": exported.get("value") if conf['score'] >= min_confidence and conf['engine_agreement'] >= 2 else None,
            "display_value": display_value,
            "status": "accepted" if exported.get("value") else ("review" if display_value else "unread"),
            "confidence": conf,
            "suggested_value": exported.get("suggested_value"),
        }

    for field in fields.values():
        field['status'] = 'accepted' if field['value'] else ('review' if field['display_value'] else 'unread')

    kind = result.get('card_type') or detection.get('kind')
    required = ["national_id", "first_name", "last_name", "father_name"]
    required.append('document_number' if kind == 'old' else 'birth_date')
    # An attempted expiry date is part of a complete card result and must be
    # offered in the correction selector, even when the required birth date fails.
    if kind == 'smart' and 'expiry_date' in raw_fields:
        required.append('expiry_date')
    from presentation import date_value
    birth=date_value(fields.get('birth_date',{}).get('display_value'))
    expiry=date_value(fields.get('expiry_date',{}).get('display_value'))
    date_conflict=bool(birth and expiry and birth>=expiry)
    if date_conflict:
        for name in ('birth_date','expiry_date'):
            row=fields[name];row['value']=None;row['status']='review'
            row['confidence']={**row['confidence'],'score':min(.49,row['confidence']['score']),
                               'percent':min(49,row['confidence']['percent'])}
    accepted_required = [name for name in required if fields.get(name, {}).get("value")]
    review_required = len(accepted_required) != len(required)

    if det_conf["state"] == "not_detected" and not detection.get("user_selected") and not detection.get('search_limited'):
        status = "rejected"
        reasons = ["card_not_detected"]
    elif detection.get("detected") and not review_required and not selection['required'] and quality['score'] >= 0.45:
        status = "accepted"
        reasons = []
    else:
        status = "review"
        reasons = []
        if not detection.get("detected"):
            reasons.append("card_type_not_verified")
        missing = [name for name in required if not fields.get(name, {}).get("value")]
        if missing:
            reasons.append("fields_need_review:" + ",".join(missing))
        if date_conflict:
            reasons.append('date_order_conflict')
        if quality["score"] < 0.45:
            reasons.append("image_quality_low")
        if selection['required']:
            reasons.append(selection['reason'])
        if detection.get('search_limited'):
            reasons.append('detection_time_budget_exceeded')

    low_fields = [name for name in required if not fields[name]['value']]
    overall = min(det_conf['score'], *(fields[name]['confidence']['score'] for name in required))
    if status != 'accepted':
        overall = min(overall, 0.89)
    action = ('select_card_region' if selection['required'] else
              'replace_image' if detection.get('user_selected') and not detection.get('detected') else
              'replace_image' if quality['score'] < 0.45 else
              'review_fields' if low_fields else 'none')
    errors = (result.get('raw_result') or {}).get('errors') or []
    processing_error = any(e.get('stage') == 'local' for e in errors)
    engine_errors = {e.get('engine') for f in raw_fields.values() for e in f.get('ocr_engine_errors', [])}
    if processing_error or engine_errors:
        status = 'review'
        overall = min(overall, 0.49)
        for field in fields.values():
            field['value'] = None
            field['status'] = 'review' if field['display_value'] else 'unread'
        low_fields = required.copy()
        reasons.append('ocr_processing_failed' if processing_error else 'ocr_engines_unavailable')
        action = 'retry_processing' if processing_error else 'service_setup_required'
    if detection.get('multiple_cards') or not detection.get('detected'):
        for field in fields.values():
            field['value'] = None
            field['status'] = 'review' if field['display_value'] else 'unread'
        low_fields = required.copy()
        if detection.get('multiple_cards'):
            action = 'select_card_region'
        elif not (processing_error or engine_errors):
            action = 'replace_image' if detection.get('user_selected') else 'select_card_region'
    if detection.get('intake_rejected') and not detection.get('detected') and not detection.get('multiple_cards'):
        status='rejected'
        reasons=['card_not_detected']
        action='replace_image'
        low_fields=[]

    return {
        "schema_version": "2.0-company",
        "status": status,
        "review_required": status != "accepted",
        "reasons": reasons,
        "confidence": {'score': round(overall, 4), 'percent': _pct(overall), 'calibrated': False,
                       'score_type': 'operational_confidence'},
        "policy": {'field_threshold_percent': _pct(min_confidence),
                   'selection_threshold_percent': selection['threshold_percent'],
                   'minimum_agreeing_engines': 2, 'required_fields': required},
        "next_action": {'type': action, 'fields': low_fields,
                        'selection_required': selection['required'],
                        'card_box_format': 'normalized_xyxy',
                        'suggested_card_box': detection.get('geometric_candidate'),
                        'card_candidates': [{k:row.get(k) for k in ('kind','box')} for row in detection.get('card_candidates') or []],
                        'quality_tips': (decision.get('quality') or {}).get('tips') or []},
        "document": {
            "type": (result.get("card_type") or detection.get("kind")) if detection.get('detected') else None,
            "suggested_type": (result.get('card_type') or detection.get('kind')) if not detection.get('detected') else None,
            "detected": bool(detection.get("detected")),
            "user_selected": bool(detection.get("user_selected")),
            "horizontal_mirror_corrected": bool(detection.get('horizontal_mirror_corrected')),
            "confidence": det_conf,
        },
        "image_quality": quality,
        "fields": fields,
        "confidence_notice": "Operational scores are not calibrated probabilities. Calibrate thresholds on a held-out company dataset before treating percentages as statistical probabilities.",
    }
