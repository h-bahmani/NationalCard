"""Conservative Persian identity-field gate.

This module never invents identity data. It may normalize Arabic/Persian glyphs and,
for name fields only, attach a tightly-bounded dictionary correction. Corrections are
accepted automatically only when BOTH primary OCR engines are present and agree on the
same observed token and the dictionary match is unique at edit distance 1. Otherwise a
candidate is review-only and is hidden from the normal result screen.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

PRIMARY_ENGINES = ("bina_rizeh_ppocrv6", "paddleocr_v5_arabic")

_TRANSLATE = str.maketrans({
    "ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه",
    "ؤ": "و", "إ": "ا", "أ": "ا", "ٱ": "ا",
})
_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed]")
_ALLOWED = re.compile(r"^[\u0600-\u06ff\s\u200c-]+$")

# Curated, intentionally conservative. This is not used to fabricate a name: it is
# only used to validate/correct an OCR token that was actually observed in the image.
GIVEN_NAMES = {
    "ثریا",
    "آتنا","آرمان","آرمین","آرش","آریا","آیدا","ابراهیم","ابوالفضل","احسان","احمد","اکبر","امیر",
    "امیرحسین","امیررضا","امین","انسیه","الهام","الهه","ایمان","بابک","بهاره","بهرام","بهنام","پریسا",
    "پویا","پیمان","ترانه","حامد","حانیه","حبیب","حسن","حسین","حمید","حمیده","داود","داریوش","رضا",
    "رضوان","ریحانه","زهرا","زینب","سامان","سارا","سجاد","سعید","سمانه","سمیرا","سینا","شادی","شایان",
    "شبنم","شهرام","شهریار","صادق","طاهره","عباس","عبدالله","عاطفه","عرفان","علی","علیرضا","عمار","فاطمه",
    "فرهاد","فرزانه","فرزاد","فرشاد","فرناز","کامران","کیان","کیانا","کیوان","گلناز","لیلا","مائده","مبین",
    "مجتبی","محسن","محمد","محمدحسن","محمدحسین","محمدرضا","محمدمهدی","محمود","مرضیه","مریم","مسعود","مصطفی",
    "معصومه","مهدی","مهرداد","مهران","مهسا","میلاد","مینا","ناصر","ناهید","ندا","نرگس","نسترن","نعیم",
    "نوید","نیما","هادی","هانیه","وحید","یاسر","یاسمین","یوسف","کاظم","کریم","جواد","جلیل","جمال",
    "جعفر","خدیجه","رقیه","سحر","سلمان","سردار","شهاب","طیبه","عذرا","فرشته","مرتضی","مژگان","منصور",
    "نادر","نسرین","نوشین","پروین","پگاه","کوروش","داریوش","فرخ","شیرین","رویا","رعنا","شکوفه","افسانه",
}

COMMON_SURNAMES = {
    "احمدی","اکبری","امیری","ابراهیمی","اسدی","باقری","جعفری","حسینی","حیدری","خانی","رحیمی","رضایی",
    "زارعی","سعیدی","سلیمانی","صادقی","صالحی","عباسی","عزیزی","علیزاده","قاسمی","کریمی","کاظمی","محمدی",
    "مرادی","موسوی","نادری","نجفی","نوری","هاشمی","یوسفی","شعبانی","رستمی","رمضانی","میرزایی","مهدوی",
    "طاهری","فراهانی","قربانی","صفری","زندی","گلستانی","پورمحمدی","حاجی زاده","حاجی‌زاده","نصیری","زمانی",
    "سلطانی","شریفی","خسروی","فرهادی","بهشتی","پورحسینی","آقایی","آقازاده","دهقانی","شفیعی","معصومی",
}


def normalize_persian(value: str) -> str:
    text = str(value or "").translate(_TRANSLATE)
    text = _DIACRITICS.sub("", text)
    text = re.sub(r"[\u200d\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -\u200c")
    return text


def _compact(value: str) -> str:
    return re.sub(r"[\s\u200c-]", "", normalize_persian(value))


def _edit_distance(a: str, b: str) -> int:
    a, b = _compact(a), _compact(b)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _valid_shape(text: str) -> bool:
    if not text or len(_compact(text)) < 3 or len(_compact(text)) > 30:
        return False
    if not _ALLOWED.fullmatch(text):
        return False
    # OCR garbage often contains pathological repeated runs.
    if re.search(r"(.)\1\1", _compact(text)):
        return False
    return True


def _nearest_unique(raw: str, lexicon: set[str]):
    target = _compact(raw)
    ranked = sorted((_edit_distance(target, item), item) for item in lexicon)
    if not ranked:
        return None
    best_d, best = ranked[0]
    second_d = ranked[1][0] if len(ranked) > 1 else 99
    # Only a single-character correction. Require a unique nearest candidate.
    if best_d == 1 and second_d >= 2 and len(target) >= 3:
        return best
    return None


def _primary_values(engine_evidence: dict) -> dict[str, str]:
    out = {}
    for name in PRIMARY_ENGINES:
        row = (engine_evidence or {}).get(name) or {}
        value = normalize_persian(row.get("value") or row.get("text") or "")
        if value:
            out[name] = value
    return out


def assess_text_field(field_name: str, selected_text: str, engine_evidence: dict, ambiguous: bool,
                      suggest_corrections: bool = False) -> dict:
    """Preserve the literal OCR reading; spelling suggestions are optional and never auto-applied."""
    raw = normalize_persian(selected_text)
    primary = _primary_values(engine_evidence)
    both_primary = all(name in primary for name in PRIMARY_ENGINES)
    primary_agreement = both_primary and len({_compact(v) for v in primary.values()}) == 1
    base = {
        "raw": raw, "accepted_text": None, "suggested_text": None,
        "decision": "review", "reason": "insufficient_primary_evidence",
        "primary_agreement": primary_agreement,
        "primary_engines_present": list(primary), "auto_corrected": False,
    }
    if not _valid_shape(raw):
        return {**base, "reason": "invalid_text_shape"}

    # A cheap path requires three separate recognizers to repeatedly read the
    # identical literal at high model confidence. Never use this exception to
    # silence a conflicting heavy-model observation already present.
    fast_engines = ('paddleocr_v5_arabic', 'persian_crnn', 'paddle_names')
    fast_rows = [(engine_evidence or {}).get(engine) or {} for engine in fast_engines]
    if ('bina_rizeh_ppocrv6' not in (engine_evidence or {}) and not ambiguous
            and all(row.get('stable') and row.get('mean_score', 0) >= .80
                    and _compact(row.get('value') or '') == _compact(raw) for row in fast_rows)
            and sum(row.get('mean_score', 0) >= .90 for row in fast_rows) >= 2):
        return {**base, 'accepted_text': raw, 'decision': 'accepted',
                'reason': 'three_engine_literal_agreement'}

    if field_name in ("first_name", "father_name"):
        lexicon = GIVEN_NAMES
    elif field_name == "last_name":
        lexicon = COMMON_SURNAMES
    else:
        # Free-form fields such as place of birth are never dictionary-corrected.
        if primary_agreement and not ambiguous:
            return {**base, "accepted_text": raw, "decision": "accepted", "reason": "primary_agreement_literal_preserved"}
        return base

    correction = _nearest_unique(raw, lexicon) if suggest_corrections else None
    if correction and _compact(correction) == _compact(raw):
        correction = None

    # Even when both engines agree, keep exactly what OCR observed. A dictionary match
    # can be shown as a suggestion but must never silently replace a person's name.
    if primary_agreement and not ambiguous:
        if correction:
            return {**base, "suggested_text": correction, "decision": "review",
                    "reason": "optional_spelling_suggestion_only"}
        return {
            **base,
            "accepted_text": raw,
            "decision": "accepted",
            "reason": "primary_agreement_literal_preserved",
            "auto_corrected": False,
        }

    if correction:
        return {**base, "suggested_text": correction, "decision": "review",
                "reason": "optional_spelling_suggestion_only"}
    return base
