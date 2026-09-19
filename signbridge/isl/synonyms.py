"""Tell a synonym group from a phrase, at index-build time.

The dictionary files many signs under several words at once —
"House_Home.mp4", "Clean_Tidy.mp4", "Watch_Wristwatch_Timepiece.mp4" —
because one sign covers them all. It also has genuine phrases with the same
shape — "Home_Automation.mp4", "Identity_Card.mp4" — where the sign means
the whole phrase and would be wrong for either word alone. Nothing in the
file name separates the two, so this asks WordNet whether the words are
related in meaning: same synset, one directly above or below the other,
marked similar, or derived forms. A name is a group when its
words form one connected cluster under that test.

WordNet does not cover pronouns or negation, so those closed classes are
listed outright. NLTK is needed only to *build* the index; the running app
reads the stored result and has no dependency on it.
"""
from __future__ import annotations

from typing import List, Optional

_CLOSED = [
    {"i", "me", "my", "mine", "myself"},
    {"we", "us", "our", "ours"},
    {"you", "your", "yours"},
    {"he", "him", "his"},
    {"she", "her", "hers"},
    {"they", "them", "their", "theirs"},
    {"no", "not", "never", "none", "nothing"},
    {"yes", "ok", "okay"},
]

_wn = None


def available() -> bool:
    global _wn
    if _wn is not None:
        return True
    try:
        from nltk.corpus import wordnet

        wordnet.synsets("home")        # raises if the corpus is not downloaded
        _wn = wordnet
        return True
    except Exception:
        return False


def _near(s) -> set:
    near = (set(s.hypernyms()) | set(s.hyponyms()) | set(s.similar_tos())
            | set(s.also_sees()) | set(s.verb_groups()))
    for lemma in s.lemmas():
        near |= {d.synset() for d in lemma.derivationally_related_forms()}
    # Siblings are deliberately not counted: "cluck" and "crow" are both kinds
    # of "utter", which would let the bird borrow a farmyard noise's sign.
    return near


def related(a: str, b: str) -> bool:
    if any(a in c and b in c for c in _CLOSED):
        return True
    A, B = set(_wn.synsets(a)), set(_wn.synsets(b))
    if not A or not B:
        return False
    if A & B:
        return True
    return any(_near(s) & B for s in A) or any(_near(s) & A for s in B)


def group_words(key: str) -> List[str]:
    """The words a multi-word dictionary name serves, or [] for a phrase."""
    words = list(dict.fromkeys(key.split()))         # "week one week" -> week, one
    if len(words) < 2 or len(words) > 4 or any(w.isdigit() for w in words):
        return []
    if not available():
        return []
    seen, todo = {words[0]}, [words[0]]
    while todo:
        x = todo.pop()
        for y in words:
            if y not in seen and related(x, y):
                seen.add(y)
                todo.append(y)
    return words if len(seen) == len(words) else []


def annotate(entries, progress: Optional[callable] = None) -> int:
    """Fill ``entry.groups`` in place. Returns how many groups were found."""
    found = 0
    for i, e in enumerate(entries):
        e.groups = group_words(e.word) if " " in e.word else []
        found += bool(e.groups)
        if progress and i % 500 == 0:
            progress(i, len(entries))
    return found
