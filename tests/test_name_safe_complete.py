from anti_hallucination import assess_text_field
from exporting import export_result
from presentation import field_display, number_display
from settings import DEFAULTS


def _evidence(a, b):
    return {
        "bina_rizeh_ppocrv6": {"value": a},
        "paddleocr_v5_arabic": {"value": b},
    }


def test_raw_name_is_visible_but_not_exported_as_accepted():
    raw = {
        "text": "ثریا", "raw_selected_text": "ثریا", "accepted_text": None,
        "suggested_text": None, "decision": "review", "ambiguous": True,
    }
    shown = field_display("first_name", raw)
    assert shown["value"] == "ثریا"
    assert shown["status"] == "review"

    exported = export_result({"fields": {"first_name": raw}}, "smart")
    assert exported["fields"]["first_name"]["value"] is None
    assert exported["fields"]["first_name"]["raw_text"] == "ثریا"


def test_spelling_suggestion_is_off_by_default_and_never_auto_applied():
    assert DEFAULTS["NAME_CORRECTION_ENABLED"] == "0"
    off = assess_text_field("first_name", "عللی", _evidence("عللی", "عللی"), False, False)
    assert off["accepted_text"] == "عللی"
    assert off["suggested_text"] is None
    assert off["auto_corrected"] is False

    on = assess_text_field("first_name", "عللی", _evidence("عللی", "عللی"), False, True)
    assert on["accepted_text"] is None
    assert on["suggested_text"] == "علی"
    assert on["decision"] == "review"
    assert on["auto_corrected"] is False


def test_known_name_is_not_changed_to_neighbouring_dictionary_name():
    result = assess_text_field("first_name", "ثریا", _evidence("ثریا", "ثریا"), False, True)
    assert result["accepted_text"] == "ثریا"
    assert result["suggested_text"] is None
    assert result["auto_corrected"] is False


def test_raw_number_remains_visible_as_review_only():
    result = {
        "national_id": None, "candidate": None,
        "fields": {"national_id": {"text": "۱۲۳۴۵۶۷۸۹۰۱۲", "raw_selected_text": "۱۲۳۴۵۶۷۸۹۰۱۲", "ambiguous": True}},
    }
    shown = number_display(result)
    assert shown == {"value": "123456789012", "status": "review"}
    exported = export_result(result, "smart")
    assert exported["national_id"] is None
