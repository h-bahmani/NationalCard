from PIL import Image

import cardreader_module
from exporting import export_result
from presentation import number_display


def test_invalid_checksum_number_is_visible_as_review_only():
    result = {
        "national_id": None,
        "candidate": None,
        "fields": {"national_id": {
            "text": "۱۲۳۴۵۶۷۸۹۰",
            "raw_selected_text": "۱۲۳۴۵۶۷۸۹۰",
            "ambiguous": True,
        }},
    }
    shown = number_display(result)
    assert shown == {"value": "1234567890", "status": "review"}
    exported = export_result(result, "smart")
    assert exported["national_id"] is None
    assert exported["reading_status"] == "review"


def test_partial_number_is_visible_without_promoting_it():
    result = {
        "national_id": None,
        "candidate": None,
        "fields": {"national_id": {
            "text": "۱۲۳۴۵۶۷",
            "raw_selected_text": "۱۲۳۴۵۶۷",
            "ambiguous": True,
        }},
    }
    shown = number_display(result)
    assert shown["value"] == "1234567"
    assert shown["status"] == "review"


def test_weak_tight_rectangle_is_not_promoted_to_card(monkeypatch):
    import detector
    import card_verifier

    monkeypatch.setattr(cardreader_module, "detect_card", lambda image, progress=None: (
        {"detected": False, "kind": None, "method": "layout_match"}, None, None
    ))
    monkeypatch.setattr(detector, "suggest_card_crop", lambda image: (None, None))
    monkeypatch.setattr(card_verifier, "verify_reference_layout", lambda image, context="manual": (
        {"verified": False, "probable": False, "kind": "old", "score": 0.20,
         "inliers": 8, "alternatives": [{"valid_geometry": False}]}, None
    ))

    image = Image.new("RGB", (625, 392), (120, 160, 180))
    prepared = cardreader_module.prepare(image)
    assert prepared.decision["allowed"] is False
    assert prepared.detection["detected"] is False
    assert prepared.detection["ocr_candidate"] is True
    assert prepared.decision["status"] == "not_detected"
    assert prepared.detection['intake_rejected']
    assert prepared.decision['classification']['status']=='rejected'
    assert prepared.aligned is not None
    assert prepared.aligned.size == (800, 500)


def test_legacy_invalid_candidate_is_review_only():
    result = {"national_id": None, "candidate": "1234567890", "fields": {}}
    shown = number_display(result)
    assert shown == {"value": "1234567890", "status": "review"}
    exported = export_result(result, "smart")
    assert exported["national_id"] is None
