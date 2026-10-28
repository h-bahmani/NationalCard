# -*- coding: utf-8 -*-
"""CardReader terminal client. Uses a fixed local config, uploads images, prints a concise English report."""
from __future__ import annotations

import configparser
import getpass
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    print("ERROR: Python package 'requests' is missing.")
    print("Install it in the Python environment used by this launcher:")
    print("    python -m pip install requests")
    input("Press Enter to exit...")
    raise SystemExit(1)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "cardreader_client.ini"
DEFAULT_API_URL = "http://127.0.0.1:8510"

FIELDS = [
    ("national_id", "National ID"),
    ("first_name", "First name"),
    ("last_name", "Last name"),
    ("father_name", "Father's name"),
    ("birth_date", "Date of birth"),
    ("expiry_date", "Expiry date"),
]
COMMON_NAMES = {
    "محمدحسن": "Mohammad Hasan", "محمد حسن": "Mohammad Hasan",
    "محمدرضا": "Mohammad Reza", "محمد رضا": "Mohammad Reza",
    "محمد": "Mohammad", "حسن": "Hasan", "علی": "Ali",
    "آروین فر": "Arvin Far", "آروین": "Arvin", "فر": "Far",
    "میر علی اصغر": "Mir Ali Asghar", "میرعلی اصغر": "Mir Ali Asghar",
    "سیدکاظم": "Seyyed Kazem", "سید کاظم": "Seyyed Kazem",
    "مرتضوی اسکوئی": "Mortazavi Oskouei", "مرتضوی": "Mortazavi",
    "اسکوئی": "Oskouei", "مریم": "Maryam", "زهرا": "Zahra",
    "فاطمه": "Fatemeh", "حسین": "Hossein", "رضا": "Reza",
    "اکبر": "Akbar", "اصغر": "Asghar", "احمد": "Ahmad",
    "ابراهیم": "Ebrahim", "جعفر": "Jafar", "مجید": "Majid",
    "مهدی": "Mehdi", "امیر": "Amir", "امیرحسین": "Amir Hossein",
    "فاطمه زهرا": "Fatemeh Zahra",
}

CHAR_MAP = {
    "ا": "a", "آ": "a", "ب": "b", "پ": "p", "ت": "t", "ث": "s",
    "ج": "j", "چ": "ch", "ح": "h", "خ": "kh", "د": "d", "ذ": "z",
    "ر": "r", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh", "ص": "s",
    "ض": "z", "ط": "t", "ظ": "z", "ع": "", "غ": "gh", "ف": "f",
    "ق": "gh", "ک": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n",
    "و": "v", "ه": "h", "ی": "y", "ئ": "y", "ء": "", "ؤ": "v",
    " ": " ", "-": "-", "_": "_",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
}

def transliterate(value: Any) -> Any:
    """Presentation-only transliteration. Does not alter API JSON saved to disk."""
    if not isinstance(value, str) or not value:
        return value
    normalized = value.strip().replace("\u200c", " ").replace("ي", "ی").replace("ك", "ک")
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in COMMON_NAMES:
        return COMMON_NAMES[normalized]
    result = "".join(CHAR_MAP.get(ch, ch if ch.isascii() else "") for ch in normalized)
    return re.sub(r"\s+", " ", result).strip().title() or value

def read_config() -> tuple[str, str]:
    config = configparser.ConfigParser()
    if CONFIG_FILE.exists():
        try:
            config.read(CONFIG_FILE, encoding="utf-8")
        except configparser.Error as exc:
            print(f"WARNING: Could not read {CONFIG_FILE.name}: {exc}")

    api_url = os.getenv("CARDREADER_API_URL", "").strip()
    api_key = os.getenv("CARDREADER_API_KEY", "").strip()
    if not api_url and config.has_option("api", "base_url"):
        api_url = config.get("api", "base_url").strip()
    if not api_key and config.has_option("api", "api_key"):
        api_key = config.get("api", "api_key").strip()

    api_url = api_url or DEFAULT_API_URL
    api_key = api_key.strip('"').strip("'")
    if api_key.upper().startswith("REPLACE_") or api_key.upper() in ("YOUR_API_KEY", "PASTE_API_KEY_HERE"):
        api_key = ""

    parsed = urlparse(api_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        print(f"ERROR: Invalid API URL in {CONFIG_FILE.name}: {api_url}")
        print("Use a full URL, for example http://127.0.0.1:8510")
        raise SystemExit(2)

    if not api_key:
        print("No API key is configured. Enter it once in this run, or save it in cardreader_client.ini.")
        api_key = getpass.getpass("API key (press Enter only if the API does not require one): ").strip()

    return api_url.rstrip("/"), api_key

def percent(conf: Any) -> str:
    if not isinstance(conf, dict):
        return "N/A"
    if conf.get("percent") is not None:
        return f"{conf['percent']}%"
    score = conf.get("score")
    if isinstance(score, (int, float)):
        return f"{round(score * 100)}%"
    return "N/A"

def get_candidate(item: Any) -> Any:
    if not isinstance(item, dict):
        return None
    for key in ("value", "display_value", "suggested_value"):
        value = item.get(key)
        if value not in (None, ""):
            return value
    return None

def print_report(payload: dict[str, Any], health: dict[str, Any] | None) -> None:
    doc = payload.get("document") or {}
    fields = payload.get("fields") or {}
    status = str(payload.get("status") or "unknown").lower()
    detected = doc.get("detected")
    detected_text = "YES" if detected is True else "NO" if detected is False else "UNKNOWN"

    print("\n" + "=" * 62)
    print("                 CARDREADER EXTRACTION REPORT")
    print("=" * 62)
    print(f"{'Card detected':27}: {detected_text}")
    print(f"{'Card detection confidence':27}: {percent(doc.get('confidence'))}")
    print(f"{'Extraction status':27}: {status.replace('_', ' ').upper()}")
    print("-" * 62)
    print("EXTRACTED FIELDS")
    print("-" * 62)

    review_fields = []
    checksum_warning = False
    readable_count = 0

    # for key, label in FIELDS:
    #     item = fields.get(key)
    #     candidate = get_candidate(item)
    #     if candidate is None:
    #         display = "Not read"
    #         review_fields.append(label)
    #     else:
    #         display = str(transliterate(candidate))
    #         readable_count += 1
    #         field_status = str(item.get("status") or "unread").lower() if isinstance(item, dict) else "unread"
    #         accepted = isinstance(item, dict) and item.get("value") not in (None, "") and field_status in ("accepted", "verified", "valid")
    #         if not accepted:
    #             review_fields.append(label)
    #     print(f"{label:27}: {display}")
    #     if key == "national_id" and isinstance(item, dict) and item.get("checksum_valid") is False:
    #         checksum_warning = True

    for key, label in FIELDS:
        item = fields.get(key)
        candidate = get_candidate(item)

        not_read_reason = None

        if candidate is None:
            display = "Not read"
            review_fields.append(label)

            reasons = payload.get("reasons") or []

            # 1) Whole OCR worker failed
            if "ocr_processing_failed" in reasons:
                not_read_reason = (
                    "OCR worker failed before this field could be read"
                )

            # 2) OCR engines are not available
            elif "ocr_engines_unavailable" in reasons:
                not_read_reason = (
                    "Required OCR engine is unavailable"
                )

            # 3) We have field-level information
            elif isinstance(item, dict):
                confidence = item.get("confidence") or {}

                engines_with_candidate = confidence.get(
                    "engines_with_candidate",
                    0,
                )

                engine_agreement = confidence.get(
                    "engine_agreement",
                    0,
                )

                # No engine produced anything at all
                if engines_with_candidate == 0:
                    not_read_reason = (
                        "Reason: OCR ran, but all recognition attempts returned empty or invalid text"
                    )

                # Special explanation for dates
                elif key in ("birth_date", "expiry_date"):
                    not_read_reason = (
                        "OCR returned text, but no valid complete date was confirmed"
                    )

                # Engines returned conflicting candidates
                elif engine_agreement == 0:
                    not_read_reason = (
                        "OCR engines could not agree on a reliable value"
                    )

                # Candidate exists internally but was rejected by confidence/policy
                else:
                    not_read_reason = (
                        "OCR confidence was too low to accept the value"
                    )

            else:
                not_read_reason = (
                    "No readable OCR result was returned"
                )

        else:
            display = str(transliterate(candidate))
            readable_count += 1

            field_status = (
                str(item.get("status") or "unread").lower()
                if isinstance(item, dict)
                else "unread"
            )

            accepted = (
                isinstance(item, dict)
                and item.get("value") not in (None, "")
                and field_status in (
                    "accepted",
                    "verified",
                    "valid",
                )
            )

            if not accepted:
                review_fields.append(label)

        # Print field value
        print(f"{label:27}: {display}")

        # Print reason directly below Not read
        if candidate is None and not_read_reason:
            print(
                f"{'':27}  Reason: {not_read_reason}"
            )

        # National ID checksum warning
        if (
            key == "national_id"
            and isinstance(item, dict)
            and item.get("checksum_valid") is False
        ):
            checksum_warning = True







    reasons = payload.get("reasons") or []
    multiple_cards = "multiple_cards" in reasons
    engine_warnings = []
    if health and isinstance(health.get("engines"), list):
        for engine in health["engines"]:
            if isinstance(engine, dict) and engine.get("enabled") and not engine.get("ready"):
                engine_warnings.append(engine.get("engine") or "Unknown OCR engine")

    warning_lines = []
    if engine_warnings or "ocr_engines_unavailable" in reasons:
        warning_lines.append("OCR setup incomplete; extracted values may be inaccurate.")
    if multiple_cards:
        warning_lines.append("Multiple card candidates were detected.")
    if checksum_warning:
        warning_lines.append("National ID checksum is INVALID.")
    other_reasons = [
        str(r) for r in reasons
        if r not in ("ocr_engines_unavailable", "multiple_cards")
        and not (isinstance(r, str) and r.startswith("fields_need_review:"))
    ]
    warning_lines.extend(other_reasons)

    if warning_lines:
        print("-" * 62)
        print("WARNINGS")
        print("-" * 62)
        for line in warning_lines:
            print(f"- {line}")

    overall = percent(payload.get("confidence"))
    final_status = "REVIEW REQUIRED" if status == "review" or review_fields or warning_lines else status.upper()
    review_summary = ", ".join(dict.fromkeys(review_fields)) if review_fields else "None"
    print("-" * 62)
    print(f"Final assessment: {final_status}")
    print(f"Fields requiring review: {review_summary}")
    print(f"Overall confidence: {overall} (operational score, not calibrated accuracy)")
    print("=" * 62)

def request_image(api_url: str, api_key: str, image_path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    headers = {"X-API-Key": api_key} if api_key else {}
    health_data = None
    try:
        response = requests.get(f"{api_url}/health", headers=headers, timeout=8)
        if response.ok:
            try:
                health_data = response.json()
            except ValueError:
                pass
    except requests.RequestException:
        pass

    mime = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".webp": "image/webp", ".bmp": "image/bmp", ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(image_path.suffix.lower(), "application/octet-stream")

    with image_path.open("rb") as image:
        response = requests.post(
            f"{api_url}/v1/extract",
            headers=headers,
            files={"file": (image_path.name, image, mime)},
            timeout=(10, 300),
        )
    if response.status_code in (401, 403):
        raise PermissionError("API key rejected or missing. Check cardreader_client.ini.")
    if not response.ok:
        try:
            detail = json.dumps(response.json(), ensure_ascii=False, indent=2)
        except ValueError:
            detail = response.text[:3000]
        raise RuntimeError(f"API returned HTTP {response.status_code}:\n{detail}")
    try:
        return response.json(), health_data
    except ValueError as exc:
        raise RuntimeError("API did not return valid JSON.") from exc

def save_json_if_requested(payload: dict[str, Any], image_path: Path) -> None:
    answer = input("Save full raw API JSON response? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_path = image_path.parent / f"cardreader_result_{stamp}.json"
    typed = input(f"JSON output path [Enter = {default_path}]: ").strip().strip('"')
    output_path = Path(typed).expanduser() if typed else default_path
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON saved: {output_path}")
    except OSError as exc:
        print(f"Could not save JSON: {exc}")

def main() -> int:
    api_url, api_key = read_config()
    print("\n" + "=" * 62)
    print("CardReader | Terminal Image Client v2 — concise English report")
    print(f"API service: {api_url}")
    print("Enter an image path. Type exit to quit.")
    print("=" * 62)

    while True:
        raw = input("\nImage path (or exit): ").strip().strip('"').strip("'")
        if raw.lower() in ("exit", "quit", "q"):
            print("Finished.")
            return 0
        image_path = Path(raw).expanduser()
        if not image_path.is_file():
            print("ERROR: File not found. Check the path and try again.")
            continue

        print("Uploading image and processing. Please wait...")
        try:
            payload, health = request_image(api_url, api_key, image_path)
            print_report(payload, health)
            save_json_if_requested(payload, image_path)
        except PermissionError as exc:
            print(f"AUTHENTICATION ERROR: {exc}")
            retry_key = getpass.getpass("Enter a valid API key (blank to skip): ").strip()
            if retry_key:
                api_key = retry_key
                try:
                    payload, health = request_image(api_url, api_key, image_path)
                    print_report(payload, health)
                    save_json_if_requested(payload, image_path)
                except Exception as retry_exc:
                    print(f"ERROR: {retry_exc}")
        except requests.ConnectionError:
            print(f"ERROR: Cannot connect to {api_url}. Make sure the API service is running and the URL is correct.")
        except requests.Timeout:
            print("ERROR: API request timed out. Check whether the service is still processing.")
        except Exception as exc:
            print(f"ERROR: {exc}")

        print("\nEnter the next image path to continue, or type exit to quit.")

if __name__ == "__main__":
    raise SystemExit(main())
