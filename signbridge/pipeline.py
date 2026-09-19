"""The spine: page text in, playable sign sequence out.

    text -> sentences -> ranked by importance -> ISL glosses
         -> dictionary lookup -> download on demand -> clip queue
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import MAX_CLIPS_PER_PAGE, MISS_LOG, STATIC_URL
from .avatar.framing import load_crops
from .avatar.pose_extract import has_pose, pose_path_for
from .isl import drive
from .isl.index import Resolution, SignIndex
from .nlp.context import select_key_sentences, summarise
from .nlp.gloss import GlossedSentence, gloss_text


@dataclass
class Clip:
    gloss: str
    src: str = ""
    pose: Optional[str] = None
    tag: str = ""
    sentence: int = 0
    source_word: str = ""      # the page's word, when gloss is a stand-in
    crop: Optional[dict] = None

    def as_video(self) -> dict:
        return {"gloss": self.gloss, "src": self.src, "tag": self.tag,
                "was": self.source_word if self.source_word != self.gloss else "",
                "crop": self.crop}

    def as_avatar(self) -> dict:
        return {"gloss": self.gloss, "pose": self.pose, "tag": self.tag}


@dataclass
class PagePlan:
    sentences: List[GlossedSentence] = field(default_factory=list)
    clips: List[Clip] = field(default_factory=list)
    resolutions: Dict[str, Resolution] = field(default_factory=dict)
    summary: str = ""
    missing: List[str] = field(default_factory=list)

    @property
    def total_glosses(self) -> int:
        return sum(len(s.glosses) for s in self.sentences if s.is_key)

    @property
    def coverage(self) -> float:
        total = self.total_glosses
        return len(self.clips) / total if total else 0.0


_TAGS = {"exact": "", "alias": "", "group": "", "fuzzy": "near",
         "related": "related", "fingerspell": "spelled"}


def plan_page(
    text: str,
    index: SignIndex,
    coverage: float = 0.6,
    keep_all: bool = False,
    allow_fingerspell: bool = True,
    max_sentences: Optional[int] = None,
) -> PagePlan:
    """Work out what to sign. No network, no downloads — fast enough to run
    on every slider nudge."""
    plan = PagePlan()
    plan.sentences = gloss_text(text, keep_all=keep_all, max_sentences=max_sentences)
    select_key_sentences(plan.sentences, text, coverage=coverage)
    plan.summary = summarise(plan.sentences)

    seen_miss: List[str] = []

    for sent in plan.sentences:
        if not sent.is_key:
            continue
        for gloss in sent.glosses:
            if len(plan.clips) >= MAX_CLIPS_PER_PAGE:
                break
            res = plan.resolutions.get(gloss) or index.resolve(gloss, allow_fingerspell)
            plan.resolutions[gloss] = res

            if res.entry is not None:
                # For a stand-in, show the sign actually being made. Labelling
                # the chip with the page's word would tell the reader they are
                # watching "accessibility" when the signer is signing "access".
                shown = res.entry.word if res.method in ("related", "fuzzy") else gloss
                plan.clips.append(
                    Clip(gloss=shown, tag=_TAGS.get(res.method, ""),
                         sentence=sent.index, source_word=gloss)
                )
            elif res.spelled:
                for i, letter in enumerate(res.spelled):
                    plan.clips.append(
                        Clip(
                            gloss=gloss if i == 0 else letter.query,
                            tag="spelled",
                            sentence=sent.index,
                        )
                    )
            else:
                if gloss not in seen_miss:
                    seen_miss.append(gloss)

    plan.missing = seen_miss
    _log_misses(seen_miss)
    return plan


def materialise(
    plan: PagePlan,
    index: SignIndex,
    fetch_missing: bool = True,
    progress=None,
) -> PagePlan:
    """Make sure every clip in the plan has a file on disk and a URL."""
    queue: List[Clip] = []
    total = len(plan.clips)
    crops = load_crops()          # read once, not once per clip

    for i, clip in enumerate(plan.clips):
        res = plan.resolutions.get(clip.gloss)
        entry = None
        if res and res.entry:
            entry = res.entry
        elif res and res.spelled:
            match = next((s for s in res.spelled if s.query == clip.gloss), None)
            entry = (match or res.spelled[0]).entry
        if entry is None:
            entry = index.get(clip.gloss).entry
        if entry is None:
            continue

        if not entry.is_cached and fetch_missing:
            if progress:
                progress(i / max(total, 1), f"Fetching “{clip.gloss}”")
            try:
                drive.download(entry.file_id, entry.local_path)
            except Exception:
                continue

        if not entry.is_cached:
            continue

        clip.src = entry.local_url
        clip.crop = crops.get(entry.word)
        if has_pose(entry.word):
            clip.pose = f"{STATIC_URL}/poses/{pose_path_for(entry.word).name}"
        queue.append(clip)

    if progress:
        progress(1.0, "Ready")
    plan.clips = queue
    return plan


def _log_misses(words: List[str]) -> None:
    """Keep a running tally of what the dictionary could not cover. This is
    the shopping list for the next dictionary release, and the fastest way to
    see where the app is weakest."""
    if not words:
        return
    try:
        existing = (
            json.loads(MISS_LOG.read_text(encoding="utf-8"))
            if MISS_LOG.exists()
            else {}
        )
    except Exception:
        existing = {}
    counter = Counter(existing)
    counter.update(words)
    MISS_LOG.write_text(json.dumps(dict(counter.most_common()), indent=1), encoding="utf-8")
