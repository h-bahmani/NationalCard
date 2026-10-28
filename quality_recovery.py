"""Visible, deterministic card-image enhancement for OCR review.

These transforms never synthesize text. They operate only on the already selected /
aligned card so the user can see exactly which pixels are sent to OCR.
"""
from __future__ import annotations
from PIL import Image
import cv2
import numpy as np

PROFILES = {
    "raw": "بدون اصلاح",
    "auto": "خودکار برای OCR",
    "clahe": "کنتراست موضعی",
    "sharp": "شارپ ملایم",
    "denoise_sharp": "کاهش نویز + شارپ",
    "bw": "سیاه‌وسفید تطبیقی",
}


def _gray(image: Image.Image):
    return np.asarray(image.convert("L"))


def _clahe(gray):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _sharp(gray):
    blur = cv2.GaussianBlur(gray, (0, 0), 0.9)
    return cv2.addWeighted(gray, 1.65, blur, -0.65, 0)


def _denoise_sharp(gray):
    den = cv2.fastNlMeansDenoising(gray, None, 4, 7, 21)
    blur = cv2.GaussianBlur(den, (0, 0), 0.85)
    return cv2.addWeighted(den, 1.6, blur, -0.6, 0)


def quality_metrics(image: Image.Image):
    gray = _gray(image)
    factor = min(1.0, 900 / max(image.size))
    if factor < 1:
        gray = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    return {
        "brightness": round(float(gray.mean()), 1),
        "contrast": round(float(gray.std()), 1),
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_32F).var()), 1),
    }


def recommend_profile(image: Image.Image) -> str:
    m = quality_metrics(image)
    if m["contrast"] < 24:
        return "clahe"
    if m["sharpness"] < 45:
        return "denoise_sharp"
    if m["sharpness"] < 90:
        return "sharp"
    return "raw"


def enhance_card(image: Image.Image, profile: str = "auto") -> Image.Image:
    profile = profile if profile in PROFILES else "auto"
    if profile == "raw":
        return image.convert("RGB")
    if profile == "auto":
        profile = recommend_profile(image)
        if profile == "raw":
            return image.convert("RGB")
    gray = _gray(image)
    if profile == "clahe":
        out = _clahe(gray)
    elif profile == "sharp":
        out = _sharp(_clahe(gray))
    elif profile == "denoise_sharp":
        out = _denoise_sharp(_clahe(gray))
    elif profile == "bw":
        work = _clahe(gray)
        out = cv2.adaptiveThreshold(work, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 25, 7)
    else:
        out = gray
    return Image.fromarray(out).convert("RGB")


def preview_profiles(image: Image.Image):
    """Return a compact set for UI review; all use the same aligned coordinates."""
    names = ["raw", "auto", "clahe", "sharp", "denoise_sharp", "bw"]
    result=[]
    for name in names:
        out=enhance_card(image,name)
        result.append((name,PROFILES[name],out,quality_metrics(out)))
    return result
