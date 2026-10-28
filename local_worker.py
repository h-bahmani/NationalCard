"""Isolated local OCR worker.

Important safety rule for identity data: literal OCR evidence is preserved and a
value is marked confident only when there is stable within-engine evidence and/or
agreement across independent engines.  Checksum validation never invents digits.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

NUMERIC_FIELDS = {"national_id", "birth_date", "expiry_date", "document_number"}
TEXT_FIELDS = {"first_name", "last_name", "father_name", "birth_place"}


def _canonical(raw, name):
    from validation import normalize, DIGIT_MAP
    from presentation import date_value

    raw = str(raw or "").strip()
    if name == "national_id":
        return normalize(raw)
    if name in ("birth_date", "expiry_date"):
        return date_value(raw)
    if name == "document_number":
        value = re.sub(r"[\s\u200c-\u200f\-–]", "", raw.translate(DIGIT_MAP))
        return value if value.isdecimal() and 1 <= len(value) <= 12 else None
    value = raw.translate(str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه"}))
    if name in TEXT_FIELDS:
        if sum(c.isalpha() for c in value) < 2:
            return None
        if any(not (c.isalpha() or c.isspace() or c in "\u200c-") for c in value):
            return None
        value = re.sub(r"\s+", " ", value).strip()
    return value or None


def select_reading(readings, name):
    """Legacy flat-view selector retained for API/tests; new worker uses engine consensus."""
    values = [_canonical(v, name) for v in readings]
    counts = Counter(v for v in values if v is not None)
    if counts:
        best, support = counts.most_common(1)[0]
        text = next(raw for raw, value in zip(readings, values) if value == best)
        ambiguous = len(counts) > 1 or (name in NUMERIC_FIELDS and support < 2)
        return text, ambiguous, dict(counts)
    return ("" if name in TEXT_FIELDS else readings[0] if readings else ""), bool(any(readings)), {}


def _engine_candidate(readings, scores, name):
    """Reduce many preprocessing views to one vote from one OCR engine.

    Choose repeated literal observations before validity checks. Checksums
    never replace a repeated reading, repair a digit or invent a character.
    """
    rows = []
    from validation import checksum_valid
    for index, raw in enumerate(readings):
        value = _canonical(raw, name)
        if value is None:
            continue
        score = float(scores[index]) if index < len(scores) else 0.0
        rows.append((value, str(raw or "").strip(), score, checksum_valid(value) if name == "national_id" else None))
    if not rows:
        return {"value": None, "text": "", "support": 0, "variants": 0, "mean_score": 0.0,
                "stable": False, "checksum_valid": False}

    all_counts = Counter(value for value, _, _, _ in rows)
    selection_rows = rows
    # Count literal observations first. A rare, accidentally checksum-valid
    # misread must not replace the number seen consistently by this model.
    counts = Counter(value for value, _, _, _ in selection_rows)
    top_count = max(counts.values())
    winners = [value for value, count in counts.items() if count == top_count]

    def within_score(value):
        vals = [score for v, _, score, _ in selection_rows if v == value]
        return sum(vals) / len(vals) if vals else 0.0
    best_value = max(winners, key=lambda value: (within_score(value), checksum_valid(value) if name == 'national_id' else False))
    best_rows = [(raw, score, valid) for value, raw, score, valid in selection_rows if value == best_value]
    representative = max(best_rows, key=lambda item: item[1])[0] if best_rows else ""
    mean_score = within_score(best_value)
    support = counts[best_value]
    valid_checksum = bool(name == "national_id" and best_rows and best_rows[0][2])
    variants = len(readings) if name == 'national_id' else len(rows)
    stable = support >= 2 and (name != 'national_id' or support / max(1, variants) > .5)
    return {"value": best_value, "text": representative, "support": support,
            "variants": variants, "mean_score": round(mean_score, 6), "stable": stable,
            "checksum_valid": valid_checksum, "alternatives": dict(all_counts),
            "checksum_filtered": bool(name == "national_id" and selection_rows is not rows)}

def _number_consensus(evidence):
    """Rank literal numeric evidence; checksum only controls acceptance.

    Repeated views are one model's stability, never independent votes. Weak
    Arabic-model guesses cannot outvote strong Persian digit observations.
    No input filename, expected identity or digit repair is used here.
    """
    available = {engine: row for engine, row in evidence.items() if row.get('value')}
    if not available:
        return '', True, {}, evidence, None
    votes = Counter(row['value'] for row in available.values())
    strong = {engine: row for engine, row in available.items()
              if row.get('stable') and row.get('mean_score', 0) >= .90}
    partial = {engine: row for engine, row in available.items() if row.get('mean_score', 0) >= .90}
    voting = strong or partial or available
    counts = Counter(row['value'] for row in voting.values())
    preference = {'persian_crnn': 0, 'paddleocr_v5_arabic': 1, 'bina_rizeh_ppocrv6': 2}
    def rank(value):
        rows = [(engine, row) for engine, row in voting.items() if row['value'] == value]
        stability = max(row['support'] / max(1, row['variants']) for _, row in rows)
        return (-counts[value], -stability, min(preference.get(engine, 99) for engine, _ in rows), value)
    best = min(counts, key=rank)
    source = min((engine for engine, row in voting.items() if row['value'] == best),
                 key=lambda engine: preference.get(engine, 99))
    from validation import checksum_valid
    strong_count = sum(row['value'] == best for row in strong.values())
    ambiguous = not (checksum_valid(best) and strong_count >= 2 and len({r['value'] for r in strong.values()}) == 1)
    return available[source]['text'], ambiguous, dict(votes), evidence, source

def select_engine_consensus(engine_readings, engine_scores, name):
    """Choose one literal reading while keeping uncertainty explicit.

    Each engine has one vote. Names/dates retain the tested field-specific
    ranking. National IDs use stable numeric evidence before checksum checks.
    """
    evidence = {}
    for engine, readings in engine_readings.items():
        evidence[engine] = _engine_candidate(readings, engine_scores.get(engine, []), name)

    if name == 'national_id':
        return _number_consensus(evidence)

    available = {engine: row for engine, row in evidence.items() if row["value"] is not None}
    if not available:
        return "", True, {}, evidence, None

    voting = available

    all_votes = Counter(row["value"] for row in available.values())
    votes = Counter(row["value"] for row in voting.values())
    best_votes = max(votes.values())
    tied_values = [value for value, count in votes.items() if count == best_votes]

    if name in ('birth_date','expiry_date'):
        # Break a vote tie using the recognizer trained for Persian glyphs.
        # An Arabic-script model can confuse printed Persian zero with five.
        # This never repairs digits or removes the disagreement flag.
        preference = ["persian_crnn", "bina_rizeh_ppocrv6", "paddleocr_v5_arabic", "paddle_names"]
    elif name in TEXT_FIELDS:
        preference = ["bina_rizeh_ppocrv6", "paddleocr_v5_arabic", "paddle_names", "persian_crnn"]
    else:
        preference = ["bina_rizeh_ppocrv6", "paddleocr_v5_arabic", "persian_crnn", "paddle_names"]
    rank = {engine: index for index, engine in enumerate(preference)}

    def value_rank(value):
        engines = [engine for engine, row in voting.items() if row["value"] == value]
        best_engine_rank = min((rank.get(engine, 999) for engine in engines), default=999)
        support = sum(int(voting[engine].get("support", 0)) for engine in engines)
        # A weak primary reading must not displace a stable, high-scoring
        # reading from the other primary just because of a fixed model order.
        # This only ranks review text; acceptance still requires engine agreement.
        strong_primary = sum(engine in {'bina_rizeh_ppocrv6','paddleocr_v5_arabic'}
                             and voting[engine].get('stable')
                             and voting[engine].get('mean_score',0) >= .90 for engine in engines)
        strong_date = sum(voting[engine].get('stable') and voting[engine].get('mean_score',0)>=.90
                          for engine in engines)
        quality_rank = strong_primary if name in TEXT_FIELDS else strong_date if name in ('birth_date','expiry_date') else 0
        return (-quality_rank,best_engine_rank,-support,str(value))

    best_value = min(tied_values, key=value_rank)
    agreeing = [engine for engine, row in voting.items() if row["value"] == best_value]
    chosen_engine = min(agreeing, key=lambda engine: rank.get(engine, 999))
    text = voting[chosen_engine]["text"]

    if len(voting) >= 2:
        ambiguous = best_votes < 2
    else:
        ambiguous = not next(iter(voting.values()))["stable"]

    return text, ambiguous, dict(all_votes), evidence, chosen_engine

def _numeric_variants(crop):
    import cv2
    import numpy as np
    from PIL import Image, ImageOps, ImageEnhance

    gray = np.asarray(crop.convert("L"))
    # Upscale tiny ID-card text before local contrast operations. This interpolates
    # existing pixels only; it does not synthesize characters.
    scale = max(1.0, 64.0 / max(1, gray.shape[0]))
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    gaussian = cv2.GaussianBlur(clahe, (0, 0), 1.0)
    unsharp = cv2.addWeighted(clahe, 1.8, gaussian, -0.8, 0)
    denoised = cv2.fastNlMeansDenoising(clahe, None, 5, 7, 21)
    denoise_blur = cv2.GaussianBlur(denoised, (0, 0), 0.85)
    denoise_sharp = cv2.addWeighted(denoised, 1.65, denoise_blur, -0.65, 0)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 21, 7)
    return [crop, Image.fromarray(gray), Image.fromarray(clahe), Image.fromarray(unsharp),
            Image.fromarray(denoise_sharp), Image.fromarray(otsu), Image.fromarray(adaptive),
            ImageOps.autocontrast(crop.convert("L"), cutoff=1), ImageEnhance.Contrast(crop).enhance(2)]


def _text_variants(crop):
    from PIL import ImageOps, ImageEnhance
    return [crop, ImageOps.autocontrast(crop.convert("L"), cutoff=1), ImageEnhance.Contrast(crop).enhance(2)]

def _numeric_border_recovery(crop, reader, readings, scores):
    """Retry weak digit transcription with context borders, never digit repair.

    Compare two preprocessing families within the SAME model. A replacement
    needs repeated literal observations and a stronger model score. It still
    remains review-only when the two families read different numbers.
    """
    from PIL import ImageOps
    prior = _engine_candidate(readings, scores, 'national_id')
    if prior.get('stable') and prior.get('mean_score', 0) >= .95:
        return readings, scores, None
    pads = sorted({max(2, round(crop.height * ratio)) for ratio in (.09, .18)})
    observations = _safe_views(reader, [ImageOps.expand(crop, border=pad, fill='white') for pad in pads])
    texts, confidence = [v[0] for v in observations], [v[1] for v in observations]
    retry = _engine_candidate(texts, confidence, 'national_id')
    same = retry.get('value') == prior.get('value')
    use_retry = bool(retry.get('stable') and retry.get('mean_score', 0) >= .90
                     and (same or retry['mean_score'] >= prior.get('mean_score', 0) + .015))
    detail = {'original_readings': readings, 'original_scores': scores,
              'border_readings': texts, 'border_scores': confidence,
              'selected_family': 'border' if use_retry else 'original',
              'changed_value': bool(use_retry and not same)}
    return (texts, confidence, detail) if use_retry else (readings, scores, detail)


def _safe_views(reader, views):
    observations = []
    for view in views:
        try:
            observations.append(reader.read_scored(view))
        except Exception:
            observations.append(('', 0.0))
    return observations


def run(path, readers=None, progress=None):
    from PIL import Image
    from persian_reader import PersianReader
    # from names_reader import NamesReader
    # from ocr_ensemble import build_optional_readers

    if readers is None:
        # reader = PersianReader()
        # names_reader = NamesReader()
        # optional_readers, optional_errors = build_optional_readers()
        reader = PersianReader()
        names_reader = None
        optional_readers = []
        optional_errors = {}
    else:
        reader,names_reader,optional_readers,optional_errors=readers
    image = Image.open(path).convert("RGB")
    spec = json.loads(Path(path + ".json").read_text(encoding="utf-8"))
    spec.pop("__kind__", "smart")
    fields = {}

    for name, box in spec.items():
        field_start = time.monotonic()
        w, h = image.size
        pixels = tuple(round(v * (w if i % 2 == 0 else h)) for i, v in enumerate(box))
        crop = image.crop(pixels)
        variants = _numeric_variants(crop) if name in NUMERIC_FIELDS else _text_variants(crop)

        engine_readings = {}
        engine_scores = {}
        numeric_recovery = None
        prepared_legacy = prepared_names = None
        fast_text = False
        if name in TEXT_FIELDS:
            mobile = next((r for r in optional_readers if r.name == 'paddleocr_v5_arabic'), None)
            if mobile is not None:
                primary = _safe_views(mobile, variants)
                prepared_legacy = _safe_views(reader, variants)
                prepared_names = _safe_views(names_reader, variants[:2])
                triples = {'paddleocr_v5_arabic': primary, 'persian_crnn': prepared_legacy,
                           'paddle_names': prepared_names}
                reduced = {key: _engine_candidate([v[0] for v in values], [v[1] for v in values], name)
                           for key, values in triples.items()}
                fast_text = (all(row.get('value') and row.get('stable') and row.get('mean_score', 0) >= .80
                                 for row in reduced.values())
                             and sum(row.get('mean_score', 0) >= .90 for row in reduced.values()) >= 2
                             and len({row['value'] for row in reduced.values()}) == 1)
                for key, values in triples.items():
                    if key == 'paddleocr_v5_arabic' or fast_text:
                        engine_readings[key] = [v[0] for v in values]
                        engine_scores[key] = [v[1] for v in values]

        # The Persian recognizer is cheap and materially better on printed
        # Persian digits. Keep its literal evidence even when primaries agree.
        if name == 'national_id':
            legacy = [reader.read_scored(v) for v in variants]
            texts, scores, numeric_recovery = _numeric_border_recovery(
                crop, reader, [text for text, _ in legacy], [score for _, score in legacy])
            engine_readings['persian_crnn'] = texts
            engine_scores['persian_crnn'] = scores
        numeric_primary = _engine_candidate(engine_readings.get('persian_crnn', []),
                                             engine_scores.get('persian_crnn', []), name)

        # High-accuracy engines vote first. Legacy engines are true fallback and are
        # not allowed to disturb a stable Bina + PP-OCR agreement.
        for extra_reader in optional_readers:
            if fast_text or extra_reader.name in engine_readings:
                continue
            if (name == 'national_id' and extra_reader.name == 'bina_rizeh_ppocrv6'
                    and numeric_primary.get('stable') and numeric_primary.get('mean_score', 0) >= .90):
                continue
            texts = []
            scores = []
            # The medium Bina model uses its native 1536-wide tensor. Keep its
            # views bounded to native/gray/local contrast; running all nine
            # numeric filters on this model exceeded the worker's deadline.
            # The mobile model and legacy fallback retain all numeric views.
            engine_variants = variants[:3] if extra_reader.name == 'bina_rizeh_ppocrv6' else variants
            for variant in engine_variants:
                try:
                    text, score = extra_reader.read_scored(variant)
                except Exception:
                    text, score = "", 0.0
                texts.append(text)
                scores.append(float(score or 0.0))
            engine_readings[extra_reader.name] = texts
            engine_scores[extra_reader.name] = scores

        primary_names = {"bina_rizeh_ppocrv6", "paddleocr_v5_arabic"}
        available_primary = [n for n in engine_readings if n in primary_names]
        primary_text, primary_ambiguous, _, primary_evidence, _ = select_engine_consensus(
            {k: engine_readings[k] for k in available_primary},
            {k: engine_scores[k] for k in available_primary},
            name,
        ) if available_primary else ("", True, {}, {}, None)

        primary_value=_canonical(primary_text,name)
        primary_support=sum(row.get('value')==primary_value and row.get('stable') and row.get('mean_score',0)>=.90
                            for row in primary_evidence.values()) if primary_value else 0
        fallback_needed = len(available_primary) < 2 or not primary_text or primary_ambiguous or primary_support < 2
        if name == "national_id":
            from validation import normalize, checksum_valid
            fallback_needed = fallback_needed or not checksum_valid(normalize(primary_text or ""))

        if fallback_needed and 'persian_crnn' not in engine_readings:
            legacy = prepared_legacy if prepared_legacy is not None else [reader.read_scored(v) for v in variants]
            engine_readings["persian_crnn"] = [text for text, _ in legacy]
            engine_scores["persian_crnn"] = [score for _, score in legacy]
            if name in TEXT_FIELDS:
                # secondary = prepared_names if prepared_names is not None else [names_reader.read_scored(v) for v in variants[:2]]
                secondary = []
                # engine_readings["paddle_names"] = [text for text, _ in secondary]
                # engine_scores["paddle_names"] = [score for _, score in secondary]

        text, ambiguous, votes, engine_evidence, source = select_engine_consensus(
            engine_readings, engine_scores, name
        )
        if numeric_recovery and numeric_recovery.get('changed_value'):
            ambiguous = True
        # For production display, one missing primary engine means the automatic OCR
        # result remains review-only. This prevents legacy fallback from looking certain.
        if name != 'national_id' and not fast_text and len(available_primary) < 2:
            ambiguous = True

        gate = None
        accepted_text = None
        suggested_text = None
        decision = "review" if ambiguous else "accepted"
        if name in TEXT_FIELDS:
            from anti_hallucination import assess_text_field
            gate = assess_text_field(
                name, text, engine_evidence, ambiguous,
                suggest_corrections=os.environ.get("NAME_CORRECTION_ENABLED", "0").lower() in {"1", "true", "yes", "on"},
            )
            accepted_text = gate.get("accepted_text")
            suggested_text = gate.get("suggested_text")
            decision = gate.get("decision", "rejected")
            # The UI/export must never treat rejected/review-only text as accepted.
            ambiguous = decision != "accepted"

        flat_readings = [value for rows in engine_readings.values() for value in rows]
        disagreement = len(votes) > 1

        fields[name] = {
            "text": text,
            "raw_selected_text": text,
            "accepted_text": accepted_text,
            "suggested_text": suggested_text,
            "decision": decision,
            "correction": gate or None,
            "readings": flat_readings,
            "ambiguous": ambiguous,
            "candidate_support": votes,
            "source": source or "none",
            "engine_readings": engine_readings,
            "engine_scores": engine_scores,
            "engine_evidence": engine_evidence,
            "engine_disagreement": disagreement,
            "ocr_engine_errors": optional_errors,
            "numeric_recovery": numeric_recovery,
            "recognition_path": 'three_model_fast_consensus' if fast_text else 'numeric_evidence' if name == 'national_id' else 'full_primary_consensus',
            "note": "Literal engine evidence is preserved. Numeric stability precedes checksum validation. Text uses strong three-model agreement or the full primary path; no automatic spelling or digit repair.",
        }
        if progress:
            payload = {'field': name, 'seconds': round(time.monotonic() - field_start, 2)}
            if name == 'national_id' and _canonical(text, name):
                from company_policy import field_confidence
                confidence = field_confidence(name, fields[name], {'candidate': _canonical(text, name)})
                payload['preview'] = {'value': _canonical(text, name), 'confidence_percent': confidence['percent'],
                                      'provisional': True}
            progress(payload)
    return fields


if __name__ == "__main__":
    try:
        result = {"ok": True, "fields": run(sys.argv[1])}
    except Exception as exc:
        result = {"ok": False, "error": type(exc).__name__, "message": str(exc)[:700]}
    Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
