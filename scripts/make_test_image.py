"""Generate a small JPEG fixture for JMeter at tests/fixtures/tennis.jpg.

JMeter needs a real file on disk to POST to /predict.  This script
generates a 224x224 placeholder JPEG so the test plan works out of the
box.  You can replace the file with any real sports photo afterwards.

Run from the repo root:
    python scripts/make_test_image.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "tennis.jpg"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # 224×224 — same size as the model's input.  We draw a faint synthetic
    # "tennis-court-ish" pattern so the image is at least non-trivial.
    img = Image.new("RGB", (224, 224), color=(60, 110, 60))  # court green
    d = ImageDraw.Draw(img)
    # white court lines
    d.rectangle([12, 22, 211, 201], outline=(245, 245, 245), width=2)
    d.line([12, 111, 211, 111], fill=(245, 245, 245), width=2)  # net
    d.line([111, 22, 111, 201], fill=(245, 245, 245), width=1)  # centre
    # ball
    d.ellipse([100, 100, 124, 124], fill=(212, 230, 90), outline=(40, 40, 40))

    img.save(OUT, format="JPEG", quality=85, optimize=True)
    print(f"Wrote {OUT}  ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
