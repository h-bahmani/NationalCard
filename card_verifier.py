"""Reference-assisted Iranian national-card verifier.

This verifier is intentionally used only after a card-sized region has been proposed
(by the user, geometry, or YOLO).  It is more tolerant of blur than the strict SIFT
layout matcher because it matches against masked reference card images and then
checks the warped static appearance.  It never authenticates a document.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent / "templates"
REFERENCE_FILES = {"smart": ROOT / "reference_smart.jpg", "old": ROOT / "reference_old.jpg"}
REFERENCE_SIZE = (800, 500)


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))


@lru_cache(maxsize=2)
def _reference(kind: str):
    path = REFERENCE_FILES[kind]
    rgb = np.asarray(Image.open(path).convert("RGB").resize(REFERENCE_SIZE, Image.Resampling.LANCZOS))
    sift = cv2.SIFT_create(nfeatures=5500, contrastThreshold=.008, edgeThreshold=16)
    keys, desc = sift.detectAndCompute(_gray(rgb), None)
    # White rectangles in the runtime references intentionally mask all personal data.
    mask = (rgb.min(axis=2) < 242).astype(np.uint8)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    mask[:8] = 0; mask[-8:] = 0; mask[:, :8] = 0; mask[:, -8:] = 0
    return rgb, keys, desc, mask


def _appearance_similarity(aligned_rgb: np.ndarray, kind: str) -> float:
    ref, _, _, mask = _reference(kind)
    cand = cv2.resize(aligned_rgb, REFERENCE_SIZE, interpolation=cv2.INTER_AREA)
    idx = mask.astype(bool)
    if not np.any(idx):
        return 0.0

    rg = cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), (0, 0), 4.0)
    cg = cv2.GaussianBlur(cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY), (0, 0), 4.0)
    x = cg[idx].astype(np.float32); y = rg[idx].astype(np.float32)
    if x.std() < 1e-6 or y.std() < 1e-6:
        corr01 = 0.0
    else:
        x = (x - x.mean()) / (x.std() + 1e-6)
        y = (y - y.mean()) / (y.std() + 1e-6)
        corr = float(np.mean(x * y))
        corr01 = max(0.0, min(1.0, (corr + .15) / 1.15))

    ce = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY), (3, 3), 0), 45, 120) > 0
    re = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), (3, 3), 0), 45, 120) > 0
    ce &= idx; re &= idx
    kernel = np.ones((5, 5), np.uint8)
    ced = cv2.dilate(ce.astype(np.uint8), kernel) > 0
    red = cv2.dilate(re.astype(np.uint8), kernel) > 0
    precision = float((ce & red).sum()) / max(1, int(ce.sum()))
    recall = float((re & ced).sum()) / max(1, int(re.sum()))
    edge_f1 = 2 * precision * recall / max(1e-6, precision + recall)

    ch = cv2.cvtColor(cand, cv2.COLOR_RGB2HSV)
    rh = cv2.cvtColor(ref, cv2.COLOR_RGB2HSV)
    hc = cv2.calcHist([ch], [0, 1], mask, [18, 16], [0, 180, 0, 256])
    hr = cv2.calcHist([rh], [0, 1], mask, [18, 16], [0, 180, 0, 256])
    cv2.normalize(hc, hc); cv2.normalize(hr, hr)
    hist = max(0.0, min(1.0, float(cv2.compareHist(hc, hr, cv2.HISTCMP_CORREL))))
    return float(.50 * corr01 + .30 * edge_f1 + .20 * hist)


def _thresholds(context: str):
    if context == "manual":
        return dict(score=.66, inliers=12, ratio=.10, coverage=.05, margin=.06, very_strong=.86)
    if context == "proposal":
        return dict(score=.71, inliers=14, ratio=.11, coverage=.05, margin=.07, very_strong=.88)
    return dict(score=.72, inliers=15, ratio=.12, coverage=.05, margin=.08, very_strong=.88)


def verify_reference_layout(image: Image.Image, *, context: str = "proposal"):
    """Return ``(diagnostic, aligned)`` for a card-sized candidate.

    ``verified`` means the selected/proposed region matches the static Iranian card
    reference strongly enough for this context. ``probable`` is exposed separately
    so the UI can permit cautious OCR without claiming certainty.
    """
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    if min(h, w) < 35:
        return {"verified": False, "probable": False, "kind": None, "score": 0.0, "reason": "candidate_too_small"}, None

    scale = min(1.0, 1600.0 / max(h, w))
    work = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    sift = cv2.SIFT_create(nfeatures=5500, contrastThreshold=.008, edgeThreshold=16)
    work_keys, work_desc = sift.detectAndCompute(_gray(work), None)
    if work_desc is None or len(work_keys) < 6:
        return {"verified": False, "probable": False, "kind": None, "score": 0.0, "reason": "not_enough_features"}, None

    rows = []
    for kind in ("smart", "old"):
        ref_rgb, ref_keys, ref_desc, _ = _reference(kind)
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(ref_desc, work_desc, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .82 * pair[1].distance]
        row = {"kind": kind, "good_matches": len(good), "inliers": 0, "inlier_ratio": 0.0,
               "coverage": 0.0, "appearance": 0.0, "score": 0.0, "quad": None, "valid_geometry": False, "appearance_only": False}
        if len(good) < 4:
            rows.append(row); continue
        src = np.float32([ref_keys[m.queryIdx].pt for m in good])
        dst = np.float32([work_keys[m.trainIdx].pt for m in good])
        matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.5)
        if matrix is None or mask is None:
            rows.append(row); continue
        inlier_mask = mask.ravel().astype(bool)
        inliers = int(inlier_mask.sum())
        ratio = inliers / max(1, len(good))
        hull = cv2.convexHull(src[inlier_mask]) if inliers >= 3 else None
        coverage = float(cv2.contourArea(hull) / (REFERENCE_SIZE[0] * REFERENCE_SIZE[1])) if hull is not None else 0.0
        corners = np.float32([[0, 0], [799, 0], [799, 499], [0, 499]]).reshape(-1, 1, 2)
        quad = cv2.perspectiveTransform(corners, matrix).reshape(4, 2)
        area = abs(float(cv2.contourArea(quad.astype(np.float32)))) if np.isfinite(quad).all() else 0.0
        edges = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1) if np.isfinite(quad).all() else np.zeros(4)
        valid_geometry = bool(np.isfinite(quad).all() and cv2.isContourConvex(quad.astype(np.float32)) and
                              area > .015 * work.shape[0] * work.shape[1] and edges.min() > 18 and
                              edges.max() / max(1.0, edges.min()) < 6.0)
        appearance = 0.0
        aligned = None
        if valid_geometry:
            try:
                aligned = cv2.warpPerspective(work, np.linalg.inv(matrix), REFERENCE_SIZE,
                                              flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
                appearance = _appearance_similarity(aligned, kind)
            except Exception:
                aligned = None
        score = (.50 * min(1.0, inliers / 35.0) +
                 .20 * min(1.0, ratio / .25) +
                 .10 * min(1.0, coverage / .35) +
                 .20 * appearance) if valid_geometry else 0.0
        row.update(inliers=inliers, inlier_ratio=round(ratio, 4), coverage=round(coverage, 4),
                   appearance=round(float(appearance), 4), score=round(float(score), 4),
                   quad=(quad / max(scale, 1e-6)).tolist() if valid_geometry else None,
                   valid_geometry=valid_geometry, _aligned=aligned)
        # On a tight user/YOLO crop, extreme blur can make homography unstable even
        # though the low-frequency static card design is still obvious.  Only use this
        # fallback when the crop itself has a card-like landscape ratio; never on a
        # full person/photo scene.  The crop is resized, not generatively restored.
        crop_ratio = max(work.shape[1], work.shape[0]) / max(1.0, min(work.shape[1], work.shape[0]))
        if not valid_geometry and context in {'manual','proposal'} and 1.35 <= crop_ratio <= 1.90:
            raw_aligned = cv2.resize(work, REFERENCE_SIZE, interpolation=cv2.INTER_CUBIC)
            raw_app = _appearance_similarity(raw_aligned, kind)
            if raw_app > row['score']:
                row.update(appearance=round(float(raw_app),4), score=round(float(raw_app),4),
                           appearance_only=True, _aligned=raw_aligned,
                           quad=[[0.0,0.0],[float(image.width-1),0.0],[float(image.width-1),float(image.height-1)],[0.0,float(image.height-1)]])
        rows.append(row)

    rows.sort(key=lambda item: item["score"], reverse=True)
    top = rows[0]
    second = rows[1] if len(rows) > 1 else {"score": 0.0}
    margin = float(top["score"] - second["score"])
    limits = _thresholds(context)
    geometric_verified = bool(top["valid_geometry"] and top["score"] >= limits["score"] and
                    top["inliers"] >= limits["inliers"] and top["inlier_ratio"] >= limits["ratio"] and
                    top["coverage"] >= limits["coverage"] and
                    (margin >= limits["margin"] or top["score"] >= limits["very_strong"]))
    appearance_threshold = .68 if context == 'manual' else .72
    appearance_verified = bool(top.get('appearance_only') and top['score'] >= appearance_threshold and margin >= .15)
    verified = geometric_verified or appearance_verified
    probable = bool(not verified and ((top["valid_geometry"] and top["score"] >= .46 and top["inliers"] >= 7) or
                                      (top.get('appearance_only') and top['score'] >= .58 and margin >= .12)))
    aligned_arr = top.pop("_aligned", None)
    for row in rows[1:]:
        row.pop("_aligned", None)
    diagnostic = {"verified": verified, "probable": probable, "kind": top.get("kind"),
                  "score": top.get("score", 0.0), "margin": round(margin, 4),
                  "inliers": top.get("inliers", 0), "inlier_ratio": top.get("inlier_ratio", 0.0),
                  "coverage": top.get("coverage", 0.0), "appearance": top.get("appearance", 0.0),
                  "quad": top.get("quad"), "context": context,
                  "alternatives": [{k: v for k, v in row.items() if k != "quad"} for row in rows]}
    aligned_img = Image.fromarray(aligned_arr) if aligned_arr is not None else None
    return diagnostic, aligned_img
