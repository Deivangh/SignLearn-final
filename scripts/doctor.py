#!/usr/bin/env python
"""Check the machine is ready. Run this when something will not start.

    python scripts/doctor.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, WARN, BAD = "  ok ", " warn", " fail"


def line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label}" + (f" — {detail}" if detail else ""))


def main() -> int:
    problems = 0
    print("SignBridge environment check\n" + "-" * 46)

    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 9) or (major, minor) > (3, 12):
        line(WARN, "Python", f"{major}.{minor} — mediapipe supports 3.9-3.12")
    else:
        line(OK, "Python", f"{major}.{minor}")

    for mod, note in [
        ("streamlit", "the app itself"),
        ("cv2", "image handling"),
        ("numpy", "maths"),
        ("PIL", "image loading"),
        ("requests", "dictionary fetching"),
    ]:
        try:
            __import__(mod)
            line(OK, mod, note)
        except ImportError:
            line(BAD, mod, f"missing — {note}")
            problems += 1

    try:
        import mediapipe  # noqa: F401
        line(OK, "mediapipe", "avatar motion extraction")
    except ImportError:
        line(WARN, "mediapipe", "missing — video mode works, avatar mode will not")

    try:
        import pytesseract
        binary = shutil.which("tesseract")
        from signbridge.config import TESSERACT_CMD

        if binary or TESSERACT_CMD:
            langs = pytesseract.get_languages(config="")
            line(OK, "tesseract", f"{len(langs)} languages: {', '.join(sorted(langs)[:8])}")
        else:
            line(BAD, "tesseract", "Python package found but the binary is not on PATH")
            problems += 1
    except Exception as exc:
        line(BAD, "tesseract", str(exc))
        problems += 1

    try:
        from signbridge.nlp.gloss import spacy_available

        line(OK if spacy_available() else WARN, "spaCy model",
             "en_core_web_sm loaded" if spacy_available()
             else "not installed — using the built-in tagger")
    except Exception:
        line(WARN, "spaCy model", "not installed")

    print("-" * 46)

    from signbridge.config import DATAGOV_API_KEY, GOOGLE_API_KEY, INDEX_FILE
    from signbridge.isl.index import SignIndex

    line(OK if DATAGOV_API_KEY else BAD, "DATAGOV_API_KEY",
         "set" if DATAGOV_API_KEY else "missing — add it to .env")
    if not DATAGOV_API_KEY:
        problems += 1
    line(OK if GOOGLE_API_KEY else WARN, "GOOGLE_API_KEY",
         "set" if GOOGLE_API_KEY else "not set — will use the key-free fallback")

    if INDEX_FILE.exists():
        idx = SignIndex.load()
        cached = sum(1 for e in idx.entries if e.is_cached)
        line(OK, "dictionary index", f"{len(idx):,} signs, {cached:,} clips cached")
        groups = sum(1 for e in idx.entries if e.groups)
        line(OK if groups else WARN, "synonym groups",
             f"{groups:,} found" if groups
             else "none — run scripts/build_index.py --refresh (needs nltk + wordnet)")
        letters = idx.letters_available
        line(OK if letters >= 26 else WARN, "fingerspelling",
             f"{letters}/26 letters" if letters >= 26
             else f"{letters}/26 letters — the dictionary has no alphabet clips; "
                  "add a.mp4 ... z.mp4 to static/alphabet")
    else:
        line(BAD, "dictionary index", "not built — run scripts/build_index.py")
        problems += 1

    print("-" * 46)
    print("Ready to run." if not problems else f"{problems} thing(s) need attention.")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
