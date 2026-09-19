"""English sentence -> Indian Sign Language gloss sequence.

The pipeline, per sentence:

  tokenise -> tag -> drop function words -> lemmatise -> simplify
           -> detect tense/negation/question -> reorder to ISL word order

ISL clause order is roughly ``TIME  PLACE  SUBJECT  OBJECT  VERB``, with
negation and any question word pushed to the end. Tense is marked once at
the front rather than on the verb.

spaCy is used when installed (better tagging, better lemmas). Without it the
module falls back to a closed-class + suffix tagger, which is noticeably
rougher but keeps the app runnable with no model download.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from . import lexicon as lx

_NLP = None
_SPACY_TRIED = False


def _spacy():
    global _NLP, _SPACY_TRIED
    if _SPACY_TRIED:
        return _NLP
    _SPACY_TRIED = True
    try:
        import spacy

        _NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        _NLP.add_pipe("sentencizer")
    except Exception:
        _NLP = None
    return _NLP


def spacy_available() -> bool:
    return _spacy() is not None


# ----------------------------------------------------------------------
@dataclass
class Token:
    text: str          # what the reader sees in the source
    lemma: str         # what we look up in the dictionary
    pos: str           # NOUN VERB ADJ ADV PRON NUM PROPN OTHER
    role: str = ""     # time | place | subject | object | verb | marker
    dropped: bool = False
    reason: str = ""


@dataclass
class GlossedSentence:
    index: int
    source: str
    glosses: List[str] = field(default_factory=list)
    tokens: List[Token] = field(default_factory=list)
    tense: str = "present"
    negated: bool = False
    question: bool = False
    is_key: bool = False        # survived context filtering
    score: float = 0.0

    @property
    def gloss_line(self) -> str:
        return "  ".join(g.upper() for g in self.glosses)


# ----------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?।॥])\s+|\n{2,}")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+")


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"-\n", "", text)          # rejoin hyphenated line breaks
    text = text.replace("\n", " ")
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    return [p for p in parts if len(_WORD.findall(p)) >= 1]


def _fallback_tag(word: str) -> str:
    w = word.lower()
    if w in lx.REWRITE:
        return "PRON"
    if w in lx.TIME_WORDS or w in lx.ING_NOUNS:
        return "NOUN"
    if w.isdigit() or w in lx.NUMBER_WORDS:
        return "NUM"
    if w in lx.MODALS or w in lx.IRREGULAR:
        return "VERB"
    if w in lx.COMMON_VERBS:
        return "VERB"
    if w.endswith("s") and not w.endswith("ss") and w[:-1] in lx.COMMON_VERBS:
        return "VERB"
    if w.endswith("es") and w[:-2] in lx.COMMON_VERBS:
        return "VERB"
    if w.endswith(lx.ADV_SUFFIXES) and len(w) > 4:
        return "ADV"
    if w.endswith(lx.NOUN_SUFFIXES):
        return "NOUN"
    if w.endswith(lx.VERB_SUFFIXES):
        return "VERB"
    if w.endswith(lx.ADJ_SUFFIXES) and len(w) > 4:
        return "ADJ"
    if w.endswith("ing") or w.endswith("ed"):
        return "VERB"
    return "NOUN"


def _strip_s(w: str) -> str:
    """Undo an -s / -es inflection.

    The awkward case is that "-es" attaches to sibilant stems (wash-es,
    box-es) while plain "-s" attaches to stems that already end in silent e
    (rise-s). Both surface as "...es", so check what the stem would look like:
    a stem ending in ss/x/z/ch/sh/o really took "-es"; anything else took "-s".
    """
    if len(w) < 4 or w.endswith(("ss", "us", "is")):     # bus, virus, photosynthesis
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("es"):
        stem = w[:-2]
        if stem in lx.COMMON_VERBS:
            return stem
        if w[:-1] in lx.COMMON_VERBS:
            return w[:-1]
        if stem.endswith(("ss", "x", "z", "ch", "sh", "o")):
            return stem
        return w[:-1]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


_VOWELS = set("aeiou")


def _base_from_stem(stem: str) -> str:
    """Repair a stem left behind by cutting -ed / -ing.

    Two opposite mistakes to avoid. A doubled consonant is usually the
    inflection's doing (stopped -> stop) but not after l, s, f or z, where the
    base already ends that way (called -> call, missed -> miss). And a base
    that ended in a silent e loses it (lived -> liv, closed -> clos,
    danced -> danc), which is put back when the stem could not end an English
    word without one.
    """
    if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "lsfz" + "".join(_VOWELS):
        return stem[:-1]
    if len(stem) >= 6 and stem.endswith("ell"):
        return stem[:-1]                           # travelled -> travel
    if stem.endswith(("v", "c", "iz", "bl", "os", "dg", "aus", "ais", "vis", "eas")):
        return stem + "e"                          # live, dance, close, cause, please
    if len(stem) == 2 and stem[0] in _VOWELS:
        return stem + "e"                          # us-ing -> use
    if (len(stem) == 3 and stem[0] not in _VOWELS and stem[1] in _VOWELS
            and stem[2] not in _VOWELS and stem[2] not in "wxy"):
        return stem + "e"                          # hop-ed, nam-ed, mak-ing
    return stem


def _fallback_lemma(word: str, pos: str) -> str:
    w = word.lower()
    if w in lx.IRREGULAR:
        return lx.IRREGULAR[w]
    if w in lx.TIME_WORDS or w in lx.ING_NOUNS or w in lx.NUMBER_WORDS:
        return w
    if pos == "VERB":
        if w.endswith("ing") and len(w) > 4:
            return _base_from_stem(w[:-3])
        if w.endswith("ied") and len(w) > 4:
            return w[:-3] + "y"                    # tried, studied
        if w.endswith("ed") and len(w) > 4:
            return _base_from_stem(w[:-2])
        return _strip_s(w)
    if pos in ("NOUN", "PROPN"):
        return _strip_s(w)
    return w


def _tokenise(sentence: str) -> List[Token]:
    nlp = _spacy()
    tokens: List[Token] = []

    if nlp is not None:
        for t in nlp(sentence):
            if not t.is_alpha and not t.like_num:
                continue
            pos = t.pos_ if t.pos_ in {
                "NOUN", "VERB", "ADJ", "ADV", "PRON", "NUM", "PROPN"
            } else "OTHER"
            lemma = t.lemma_.lower()
            if t.text.lower() in lx.ING_NOUNS or t.text.lower() in lx.TIME_WORDS:
                lemma, pos = t.text.lower(), "NOUN"
            tokens.append(Token(text=t.text, lemma=lemma, pos=pos))
    else:
        for w in _WORD.findall(sentence):
            pos = _fallback_tag(w)
            tokens.append(Token(text=w, lemma=_fallback_lemma(w, pos), pos=pos))
        _rescue_verb(tokens)
    return tokens


def _rescue_verb(tokens: List[Token]) -> None:
    """No verb in a clause means one was mis-tagged. Promote the best
    candidate: the first inflected word after the first noun or pronoun."""
    if any(t.pos == "VERB" for t in tokens):
        return
    seen_subject = False
    for tok in tokens:
        w = tok.text.lower()
        if not seen_subject:
            if tok.pos in ("NOUN", "PROPN", "PRON"):
                seen_subject = True
            continue
        if w.endswith(("s", "ed", "ing")) and w not in lx.DROP and len(w) > 3:
            tok.pos = "VERB"
            tok.lemma = _fallback_lemma(w, "VERB")
            return


def _regular_past(tokens: List[Token]) -> bool:
    """Past tense from a regular verb ("played", "planted") or a phrase like
    "last month". The hint list only knows irregular verbs, so without this
    "The children played in the garden" was signed as present."""
    words = [t.text.lower() for t in tokens]
    for i, tok in enumerate(tokens):
        w = words[i]
        if w == "last" and i + 1 < len(words) and words[i + 1] in lx.TIME_WORDS:
            return True
        if (tok.pos == "VERB" and w.endswith("ed") and len(w) > 4
                and (i == 0 or words[i - 1] not in lx.STATE_VERBS)):
            return True
    return False


# ----------------------------------------------------------------------
def gloss_sentence(sentence: str, index: int = 0, keep_all: bool = False) -> GlossedSentence:
    """Turn one English sentence into ordered ISL glosses."""
    out = GlossedSentence(index=index, source=sentence.strip())
    tokens = _tokenise(sentence)
    out.tokens = tokens

    lowered = [t.text.lower() for t in tokens]

    # --- clause-level features ----------------------------------------
    if any(w in lx.NEGATION or w in lx.REWRITE and lx.REWRITE[w] == "not"
           for w in lowered):
        out.negated = True
    asks = sentence.strip().endswith("?")
    if lowered and (lowered[0] in lx.QUESTION or asks):
        # Wh-questions carry a sign, moved to the end. Yes/no questions carry
        # none in ISL — raised eyebrows do the work — but are still flagged.
        out.question = (any(w in lx.QUESTION for w in lowered)
                        or (asks and lowered[0] in lx.AUX_START))
    if any(w in lx.FUTURE_HINTS for w in lowered):
        out.tense = "future"
    elif any(w in lx.PAST_TENSE_HINTS for w in lowered) or _regular_past(tokens):
        out.tense = "past"

    time_slot: List[str] = []
    place_slot: List[str] = []
    subject_slot: List[str] = []
    object_slot: List[str] = []
    verb_slot: List[str] = []
    tail_slot: List[str] = []

    seen_verb = False
    pending_place = False

    for tok in tokens:
        w = tok.text.lower()
        lemma = tok.lemma if tok.lemma and tok.lemma != "-pron-" else w

        # negation / question are handled as clause features
        if w in lx.NEGATION or lx.REWRITE.get(w) == "not":
            tok.role, tok.dropped, tok.reason = "marker", True, "clause negation"
            continue
        if w in lx.QUESTION:
            tok.role = "marker"
            tail_slot.append(w)
            continue

        if w in lx.PLACE_PREPS:
            pending_place = True
            tok.dropped, tok.reason = True, "spatial, shown not signed"
            continue
        if w in lx.DROP_PREPS or w in lx.CONJUNCTIONS:
            tok.dropped, tok.reason = True, "function word"
            continue
        if w in lx.DROP and not keep_all:
            tok.dropped, tok.reason = True, "no ISL equivalent"
            continue
        if w in lx.MODALS:
            tok.dropped, tok.reason = True, "modal folded into tense"
            continue

        lemma = lx.REWRITE.get(w, lemma)
        lemma = lx.IRREGULAR.get(lemma, lemma)
        lemma = lx.SIMPLIFY.get(lemma, lemma)

        if lemma in lx.DROP and not keep_all:
            tok.dropped, tok.reason = True, "no ISL equivalent"
            continue

        tok.lemma = lemma

        if lemma in lx.TIME_WORDS:
            tok.role = "time"
            time_slot.append(lemma)
        elif pending_place:
            tok.role = "place"
            place_slot.append(lemma)
            pending_place = False
        elif tok.pos == "VERB":
            tok.role = "verb"
            verb_slot.append(lemma)
            seen_verb = True
        elif tok.pos in ("PRON",) and not seen_verb:
            tok.role = "subject"
            subject_slot.append(lemma)
        elif tok.pos in ("NOUN", "PROPN", "NUM", "ADJ", "ADV"):
            tok.role = "subject" if not seen_verb and not subject_slot else "object"
            (subject_slot if tok.role == "subject" else object_slot).append(lemma)
        else:
            tok.role = "object"
            object_slot.append(lemma)

    # --- assemble in ISL order ----------------------------------------
    glosses: List[str] = []
    marker = lx.TENSE_MARKERS.get(out.tense, "")
    if marker:
        glosses.append(marker)
    glosses += time_slot + place_slot + subject_slot + object_slot + verb_slot
    if out.negated:
        glosses.append("not")
    glosses += tail_slot

    # de-duplicate neighbours ("me me go" -> "me go")
    deduped: List[str] = []
    for g in glosses:
        if not deduped or deduped[-1] != g:
            deduped.append(g)

    out.glosses = deduped
    return out


# ----------------------------------------------------------------------
def gloss_text(
    text: str,
    keep_all: bool = False,
    max_sentences: Optional[int] = None,
) -> List[GlossedSentence]:
    sentences = split_sentences(text)
    if max_sentences:
        sentences = sentences[:max_sentences]
    return [gloss_sentence(s, i, keep_all) for i, s in enumerate(sentences)]
