"""Read a photographed book page.

Phone photos of book pages are the hard case: skew, page curl, uneven lamp
light, shadow from the reader's own hand. The preprocessing chain here is
tuned for that rather than for clean scans.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from PIL import Image

from .config import DEFAULT_OCR_LANGS, TESSERACT_CMD

try:
    import pytesseract

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False


@dataclass
class OcrResult:
    text: str
    confidence: float
    word_count: int
    language: str
    preview: np.ndarray | None = None


class OcrUnavailable(RuntimeError):
    pass


def available_languages() -> List[str]:
    if not _HAS_TESS:
        return []
    try:
        return sorted(l for l in pytesseract.get_languages(config="") if l != "osd")
    except Exception:
        return ["eng"]


def preprocess(image: Image.Image, deskew: bool = True) -> np.ndarray:
    """Grayscale -> denoise -> flatten lighting -> binarise -> straighten."""
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Cap the long edge; Tesseract gains nothing above ~2000px and slows down.
    h, w = gray.shape
    if max(h, w) > 2000:
        scale = 2000 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Flatten the lamp gradient: divide by a heavily blurred copy of itself.
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
    flat = cv2.divide(gray, background, scale=255)

    denoised = cv2.bilateralFilter(flat, 7, 50, 50)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )

    if deskew:
        binary = _deskew(binary)
    return binary


def _deskew(binary: np.ndarray) -> np.ndarray:
    inverted = cv2.bitwise_not(binary)
    coords = cv2.findNonZero(inverted)
    if coords is None or len(coords) < 50:
        return binary

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    if abs(angle) < 0.3 or abs(angle) > 20:
        return binary

    h, w = binary.shape
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        binary, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def clean_text(raw: str) -> str:
    """Strip the artefacts OCR leaves on book pages."""
    text = raw.replace("|", "I").replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"-\n\s*", "", text)                  # de-hyphenate wraps
    text = re.sub(r"\n(?![\n])", " ", text)             # unwrap soft breaks
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # Drop lines that are mostly page furniture (numbers, running heads).
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        letters = sum(c.isalpha() for c in stripped)
        if letters < max(3, len(stripped) * 0.4):
            continue
        lines.append(stripped)
    return "\n".join(lines).strip()


def read_page(
    image: Image.Image,
    langs: str | None = None,
    psm: int = 6,
    do_preprocess: bool = True,
) -> OcrResult:
    if not _HAS_TESS:
        raise OcrUnavailable(
            "Tesseract is not on this machine. Install it, then set "
            "TESSERACT_CMD in .env if it is not on your PATH."
        )

    langs = langs or DEFAULT_OCR_LANGS
    source = preprocess(image) if do_preprocess else np.array(image.convert("L"))
    config = f"--oem 3 --psm {psm}"

    data = pytesseract.image_to_data(
        source, lang=langs, config=config, output_type=pytesseract.Output.DICT
    )
    confidences = [float(c) for c in data["conf"] if str(c) not in ("-1", "")]
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

    raw = pytesseract.image_to_string(source, lang=langs, config=config)
    text = clean_text(raw)

    return OcrResult(
        text=text,
        confidence=round(mean_conf, 1),
        word_count=len(re.findall(r"\w+", text)),
        language=langs,
        preview=source,
    )
