#!/usr/bin/env python3
"""Portable logical-order Persian line recognition for the Bina 0.2 family."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterator
from paddleocr import TextRecognition

_LTR_RUN = re.compile(r"[a-zA-Z0-9 :*./%+-]")

def pred_reverse(text: str) -> str:
    """Convert PaddleOCR visual-order Arabic output to logical reading order."""
    segments: list[str] = []
    current_ltr = ""
    for character in text:
        if _LTR_RUN.search(character):
            current_ltr += character
            continue
        if current_ltr:
            segments.append(current_ltr)
            current_ltr = ""
        segments.append(character)
    if current_ltr:
        segments.append(current_ltr)
    return "".join(reversed(segments))

class BinaTextRecognition:
    def __init__(self, model_dir: str | Path | None = None, device: str | None = None) -> None:
        model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent / "inference"
        options: dict[str, Any] = {"model_dir": str(model_dir)}
        if device:
            options["device"] = device
        self._model = TextRecognition(**options)

    def predict(self, inputs: str | Path | list[str] | list[Path], batch_size: int = 1) -> Iterator[dict[str, Any]]:
        for result in self._model.predict(input=inputs, batch_size=batch_size):
            payload = result.json() if callable(result.json) else result.json
            raw = payload["res"]
            visual = str(raw["rec_text"])
            yield {
                "input_path": raw.get("input_path"),
                "text": pred_reverse(visual),
                "score": float(raw["rec_score"]),
                "raw_visual_text": visual,
            }

def main() -> int:
    parser = argparse.ArgumentParser(description="Recognize Persian text-line crops with Bina 0.2")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--model-dir", default=str(Path(__file__).resolve().parent / "inference"))
    parser.add_argument("--device", default="cpu", help="cpu, gpu:0, ...")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    model = BinaTextRecognition(args.model_dir, device=args.device)
    for prediction in model.predict(args.images, batch_size=args.batch_size):
        print(json.dumps(prediction, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
