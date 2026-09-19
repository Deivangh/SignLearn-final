"""Resolve an English word to a clip from the ISL dictionary.

Dictionary filenames are messy in the wild: "12. Apple.mp4",
"Apple (fruit).MP4", "GOOD_MORNING.mp4". :func:`normalise` flattens all of
that into a lookup key, and every entry keeps its aliases so plurals and
compound spellings still land.

Two quirks of the official dictionary shape the lookup:

* Many everyday signs are filed under a *group* of synonyms —
  "House_Home.mp4", "Talk_Chat.mp4", "No_Not.mp4" — so the word "home" has
  no key of its own. The index builder marks which multi-word names are
  synonym groups (see :mod:`synonyms`) and each member word points at the
  clip. Phrases such as "Home_Automation.mp4" are not groups and never do.
* Numbers are filed as "100_Hundred.mp4". The digits are kept as an alias so
  a page that prints "100" still finds the sign.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from ..config import (
    ALPHABET_DIR,
    FUZZY_CUTOFF,
    INDEX_FILE,
    STATIC_DIR,
    STATIC_URL,
    VIDEO_DIR,
)
from ..nlp.lexicon import NUMBER_WORDS

_LEAD_NUM = re.compile(r"^\s*\d+\s*[.)\-_]\s*")
_PARENS = re.compile(r"\([^)]*\)")
_NON_WORD = re.compile(r"[^a-z0-9]+")
_NUMBER_FILE = re.compile(r"^\s*(\d+)[\s_]+(.+?)\s*$")
_DIGITS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine".split())}

# Agent-noun endings. "officer" is not a kind of "office", and "broker" is not
# a kind of "broke", so a fuzzy match that only adds or removes one of these
# has changed the meaning, not the spelling.
_AGENT_TAILS = {"r", "er", "or", "ist"}


def normalise(name: str) -> str:
    """Filename or word -> canonical lookup key."""
    text = name.lower().strip()
    for ext in (".mp4", ".mov", ".webm", ".m4v", ".avi", ".mkv"):
        if text.endswith(ext):
            text = text[: -len(ext)]
    text = _LEAD_NUM.sub("", text)
    text = text.replace("+", " plus ")    # "C++" must not collapse to the letter "c"
    text = _PARENS.sub(" ", text)
    text = _NON_WORD.sub(" ", text)
    return " ".join(text.split())


def _plural_forms(key: str) -> List[str]:
    """Singular forms of a plural key, so the entry "animals" also serves
    "animal".

    Only plurals. Stripping -ing or -ed from a *dictionary* word invents a
    different word: "noting" became an alias for "not", which put the wrong
    sign on every negated sentence.
    """
    forms = []
    if len(key) > 3 and " " not in key and not key.endswith(("ss", "us", "is")):
        if key.endswith("ies"):
            forms.append(key[:-3] + "y")
        elif key.endswith(("ches", "shes", "xes", "sses", "zes")):
            forms.append(key[:-2])
        elif key.endswith("s"):
            forms.append(key[:-1])
    return forms


def index_aliases(word: str, source_name: str = "", letter: str = "",
                  groups: Optional[List[str]] = None) -> List[str]:
    """Every alias an entry registers, derived from its word and file name."""
    forms = set(_plural_forms(word))
    # "good morning" -> "goodmorning", for pages that print it joined. Not when
    # a part is a lone letter: "B_Ed.mp4" (the degree) became "bed".
    if " " in word and all(len(t) > 1 for t in word.split()):
        forms.add(word.replace(" ", ""))
    # "100_Hundred.mp4" in the Numbers folder -> "100". Only when the rest of
    # the name is a number word, so "120_Volt.mp4" does not claim "120".
    m = _NUMBER_FILE.match(Path(source_name).stem.replace("_", " "))
    if m and letter.lower() == "numbers":
        rest = normalise(m.group(2)).split()
        if rest and all(t in NUMBER_WORDS for t in rest):
            forms.add(m.group(1))
    if word in _DIGITS:                   # "Zero.mp4" has no digits in its name
        forms.add(_DIGITS[word])
    forms.discard(word)
    return sorted(f for f in forms if f)


def query_forms(key: str) -> List[str]:
    """Base-form guesses for a word on the page, tried against the index.

    These run on the page's word, never on dictionary words, so a guess can
    only ever reach a sign that genuinely exists.
    """
    forms = set(_plural_forms(key))
    if len(key) > 4 and " " not in key:
        if key.endswith("ing"):
            forms.update({key[:-3], key[:-3] + "e"})
        if key.endswith("ied"):
            forms.add(key[:-3] + "y")
        elif key.endswith("ed"):
            forms.update({key[:-2], key[:-1]})
    forms.discard(key)
    return sorted(f for f in forms if len(f) >= 3)


def _shares_stem(query: str, candidate: str) -> bool:
    """A fuzzy match is only trusted when the two words are the same stem
    with different endings — "independence"/"independent",
    "healthy"/"health".

    Plain string similarity is not meaning: at 0.86 it turned "money" into
    MONKEY, "river" into DRIVER and "world" into WORD. Short words fail most,
    because one letter is often the whole difference, so they get no fuzzy
    matching at all.
    """
    if len(query) < 6 or " " in query or " " in candidate:
        return False
    cp = 0
    for a, b in zip(query, candidate):
        if a != b:
            break
        cp += 1
    tail_q, tail_c = query[cp:], candidate[cp:]
    if cp < 5 or len(tail_q) > 3 or len(tail_c) > 3:
        return False
    if {tail_q, tail_c} & _AGENT_TAILS and "" in (tail_q, tail_c):
        return False
    return True


@dataclass
class SignEntry:
    """One dictionary sign."""

    word: str                       # canonical key, e.g. "good morning"
    file_id: str                    # Google Drive file id
    source_name: str                # original filename
    letter: str = ""                # which folder it came from
    aliases: List[str] = field(default_factory=list)
    groups: List[str] = field(default_factory=list)   # words this synonym group serves
    path: str = ""                  # set for local files, relative to static/

    @property
    def filename(self) -> str:
        return f"{self.word.replace(' ', '_')}__{self.file_id[:8]}.mp4"

    @property
    def local_path(self) -> Path:
        return STATIC_DIR / self.path if self.path else VIDEO_DIR / self.filename

    @property
    def is_cached(self) -> bool:
        return self.local_path.exists() and self.local_path.stat().st_size > 1024

    @property
    def local_url(self) -> str:
        if self.path:
            return f"{STATIC_URL}/{self.path}"
        return f"{STATIC_URL}/videos/{self.filename}"

    @property
    def remote_url(self) -> str:
        return f"https://drive.google.com/uc?export=download&id={self.file_id}"

    @property
    def is_letter_clip(self) -> bool:
        """True only for a genuine manual-alphabet clip. The dictionary's own
        bare "c" key is "C++.mp4", which must never stand in for the letter."""
        if self.letter == "Alphabet":
            return True
        stem = Path(self.source_name).stem.strip().lower()
        return len(self.word) == 1 and stem == self.word


@dataclass
class Resolution:
    """What we found (or did not) for one requested word."""

    query: str
    entry: Optional[SignEntry] = None
    method: str = "miss"            # exact | alias | group | fuzzy | related | fingerspell | miss
    confidence: float = 0.0
    spelled: List["Resolution"] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.entry is not None or bool(self.spelled)


class SignIndex:
    """In-memory view of the built dictionary index."""

    def __init__(self, entries: List[SignEntry]):
        self.entries = entries
        self._by_key: Dict[str, SignEntry] = {}
        self._by_alias: Dict[str, SignEntry] = {}
        self._by_group: Dict[str, List[SignEntry]] = {}
        self._letters: Dict[str, SignEntry] = {}
        for e in entries:
            self._by_key.setdefault(e.word, e)
            for a in e.aliases:
                self._by_alias.setdefault(a, e)
            for g in e.groups:
                self._by_group.setdefault(g, []).append(e)
            if e.is_letter_clip:
                ch = e.word[-1]
                if e.letter == "Alphabet" or ch not in self._letters:
                    self._letters[ch] = e
        # Smallest group first: "house home" says more about "home" than a
        # four-word group does.
        for g in self._by_group.values():
            g.sort(key=lambda e: (len(e.word.split()), e.word))
        self._keys = list(self._by_key)

    def __len__(self) -> int:
        return len(self._by_key)

    # -- construction ---------------------------------------------------
    @classmethod
    def load(cls, path: Path | None = None) -> "SignIndex":
        path = path or INDEX_FILE
        entries: List[SignEntry] = []
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            for item in raw.get("entries", []):
                e = SignEntry(**item)
                # Recomputed rather than trusted: older index files carry
                # aliases from rules that have since been fixed.
                e.aliases = index_aliases(e.word, e.source_name, e.letter)
                entries.append(e)
        entries += alphabet_entries()
        return cls(entries)

    @staticmethod
    def save(entries: List[SignEntry], path: Path | None = None) -> None:
        path = path or INDEX_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        keep = [e for e in entries if not e.path]        # local clips are rescanned
        payload = {"count": len(keep), "entries": [asdict(e) for e in keep]}
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    # -- lookup ---------------------------------------------------------
    def _lookup(self, key: str) -> Optional[SignEntry]:
        return self._by_key.get(key) or self._by_alias.get(key)

    def get(self, word: str) -> Resolution:
        key = normalise(word)
        if not key:
            return Resolution(query=word)

        if key in self._by_key:
            return Resolution(word, self._by_key[key], "exact", 1.0)
        if key in self._by_alias:
            return Resolution(word, self._by_alias[key], "alias", 0.95)
        if key in self._by_group:
            return Resolution(word, self._by_group[key][0], "group", 0.92)

        for variant in query_forms(key):
            hit = self._lookup(variant)
            if hit is not None:
                return Resolution(word, hit, "alias", 0.9)
            if variant in self._by_group:
                return Resolution(word, self._by_group[variant][0], "group", 0.88)

        for cand in difflib.get_close_matches(key, self._keys, n=5, cutoff=FUZZY_CUTOFF):
            if _shares_stem(key, cand):
                ratio = difflib.SequenceMatcher(None, key, cand).ratio()
                return Resolution(word, self._by_key[cand], "fuzzy", ratio)

        return Resolution(query=word)

    def fingerspell(self, word: str) -> Resolution:
        """Spell an unknown word out of manual-alphabet and number signs.

        Letters come only from genuine alphabet clips (``static/alphabet``);
        digits come from the dictionary's number signs.
        """
        res = Resolution(query=word, method="fingerspell")
        chars = re.sub(r"[^a-z0-9]", "", word.lower())
        if not chars:
            return Resolution(query=word)

        spelled: List[Resolution] = []
        for ch in chars:
            hit = self._letters.get(ch) if ch.isalpha() else self._lookup(ch)
            if hit is None:
                # A word spelled half way is misinformation, not a partial
                # answer — the reader has no way to tell which letters were
                # dropped. Better to report the word as unavailable.
                return Resolution(query=word)
            spelled.append(Resolution(ch, hit, "fingerspell", 1.0))

        res.spelled = spelled
        res.confidence = 1.0
        return res

    def related(self, word: str) -> Resolution:
        """Fall back to a sign that means nearly the same thing.

        Tried before fingerspelling: signing "access" for "accessibility"
        carries the sentence, where spelling out twelve letters mostly tests
        the reader's patience. The substitution is always reported back so
        the UI can label it rather than passing it off as the real word.
        """
        from ..nlp.related import candidates

        for cand in candidates(word):
            key = normalise(cand)
            hit = self._lookup(key)
            if hit is None and key in self._by_group:
                hit = self._by_group[key][0]
            if hit is not None:
                return Resolution(word, hit, "related", 0.6)
        return Resolution(query=word)

    def resolve(self, word: str, allow_fingerspell: bool = True) -> Resolution:
        res = self.get(word)
        if res.found:
            return res
        near = self.related(word)
        if near.found:
            return near
        if not allow_fingerspell:
            return res
        return self.fingerspell(word)

    def suggest(self, prefix: str, limit: int = 12) -> List[str]:
        p = normalise(prefix)
        if not p:
            return []
        return [k for k in self._keys if k.startswith(p)][:limit]

    @property
    def letters_available(self) -> int:
        return len(self._letters)


def alphabet_entries(folder: Path | None = None) -> List[SignEntry]:
    """Manual-alphabet clips dropped into ``static/alphabet`` as a.mp4 ... z.mp4.

    The official dictionary release has no per-letter clips, so fingerspelling
    needs these supplied separately. Nothing is invented when they are absent:
    an unknown word is reported as missing instead.
    """
    folder = folder or ALPHABET_DIR
    out: List[SignEntry] = []
    if not folder.exists():
        return out
    for f in sorted(folder.iterdir()):
        stem = f.stem.strip().lower()
        if f.suffix.lower() in (".mp4", ".webm", ".mov") and len(stem) == 1 and stem.isalpha():
            out.append(SignEntry(
                word=f"letter {stem}", file_id=f"alpha-{stem}", source_name=f.name,
                letter="Alphabet", path=f"{folder.name}/{f.name}",
            ))
    return out
