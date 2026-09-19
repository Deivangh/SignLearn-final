"""Optional: a scene illustration for context before a passage.

Off by default (``ENABLE_SCENE_ILLUSTRATION`` in config).

This is deliberately *not* used to generate signs. A text-to-video model can
produce hands that look plausible and are linguistically wrong — wrong
handshape, wrong location, wrong movement path — with no signal to the reader
that anything is off. For someone who cannot check the output against the
text, a confidently wrong sign is worse than a missing one.

What generation *is* good for here is the layer above the signs: a single
still that establishes the setting of a passage before the signing starts, the
way an illustrated reader works. That is what this module does.
"""
from __future__ import annotations

from typing import Optional

from ..config import ENABLE_SCENE_ILLUSTRATION, GEMINI_API_KEY


def scene_prompt(summary: str, keywords: list[str]) -> str:
    return (
        "A clear, warm children's-book illustration establishing the setting of "
        f"this passage: {summary[:300]}. "
        f"Key elements: {', '.join(keywords[:6])}. "
        "No text, no writing, no hands signing. Simple composition, high contrast."
    )


def illustrate(summary: str, keywords: list[str]) -> Optional[bytes]:
    """Return image bytes, or None if the feature is off or unavailable."""
    if not ENABLE_SCENE_ILLUSTRATION or not GEMINI_API_KEY:
        return None
    try:
        import requests

        resp = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.0-flash-exp:generateContent",
            params={"key": GEMINI_API_KEY},
            json={
                "contents": [{"parts": [{"text": scene_prompt(summary, keywords)}]}],
                "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
            },
            timeout=60,
        )
        resp.raise_for_status()
        import base64

        for part in resp.json()["candidates"][0]["content"]["parts"]:
            if "inlineData" in part:
                return base64.b64decode(part["inlineData"]["data"])
    except Exception:
        return None
    return None
