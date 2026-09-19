"""Decide *what on the page is worth signing*.

A full book page is 250-400 words. Signed one word at a time that is roughly
eight minutes of video for something a hearing reader skims in ninety seconds,
and most of it is grammar the reader does not need. So before anything gets
signed, sentences are scored and the page is reduced to its load-bearing ones.

Scoring is a compact TextRank-flavoured heuristic — no model download, no
network call, runs in milliseconds:

* term frequency of content words, damped so one repeated noun cannot
  dominate the page
* overlap with the page's own title-case terms (usually the topic)
* position (opening and closing sentences carry the thesis)
* a mild penalty for very short or very long sentences
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List

from . import lexicon as lx
from .gloss import GlossedSentence

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")

_FUNCTION = lx.DROP | lx.DROP_PREPS | lx.PLACE_PREPS | lx.CONJUNCTIONS | lx.MODALS


def _content_words(text: str) -> List[str]:
    return [
        w.lower()
        for w in _WORD.findall(text)
        if len(w) > 2 and w.lower() not in _FUNCTION
    ]


def page_keywords(text: str, limit: int = 12) -> List[str]:
    counts = Counter(_content_words(text))
    return [w for w, _ in counts.most_common(limit)]


def score_sentences(
    sentences: List[GlossedSentence], full_text: str
) -> List[GlossedSentence]:
    if not sentences:
        return sentences

    corpus = Counter(_content_words(full_text))
    total = sum(corpus.values()) or 1
    # Damped weight: log scaling stops a single repeated noun from owning the page.
    weights = {w: math.log(1 + c) / math.log(1 + total) for w, c in corpus.items()}

    # Title-case terms outside sentence-initial position tend to be the topic.
    topic_terms = {
        m.group(0).lower()
        for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b", full_text)
    }

    n = len(sentences)
    for s in sentences:
        words = _content_words(s.source)
        if not words:
            s.score = 0.0
            continue

        base = sum(weights.get(w, 0.0) for w in words) / math.sqrt(len(words))
        topic_bonus = 0.15 * sum(1 for w in words if w in topic_terms) / len(words)

        pos = s.index / max(n - 1, 1)
        position_bonus = 0.12 * (1 - pos) + 0.06 * pos   # front-loaded, tail kept

        length = len(words)
        if length < 3:
            length_penalty = 0.45
        elif length > 32:
            length_penalty = 0.8
        else:
            length_penalty = 1.0

        s.score = (base + topic_bonus + position_bonus) * length_penalty

    return sentences


def select_key_sentences(
    sentences: List[GlossedSentence],
    full_text: str,
    coverage: float = 0.6,
    min_keep: int = 2,
) -> List[GlossedSentence]:
    """Mark the sentences that carry the page. ``coverage`` is 0.0-1.0."""
    score_sentences(sentences, full_text)
    if not sentences:
        return sentences

    keep_n = max(min_keep, int(round(len(sentences) * coverage)))
    keep_n = min(keep_n, len(sentences))

    ranked = sorted(sentences, key=lambda s: s.score, reverse=True)[:keep_n]
    keep_ids = {s.index for s in ranked}
    for s in sentences:
        s.is_key = s.index in keep_ids
    return sentences


def summarise(sentences: List[GlossedSentence], limit: int = 3) -> str:
    """A short plain-English gist, shown above the signing so the reader
    knows where the page is going before it starts."""
    ranked = sorted(sentences, key=lambda s: s.score, reverse=True)[:limit]
    ranked.sort(key=lambda s: s.index)
    return " ".join(s.source for s in ranked)


def estimate_duration(clip_count: int, seconds_per_clip: float = 2.4) -> str:
    total = clip_count * seconds_per_clip
    minutes, seconds = divmod(int(total), 60)
    return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"
