"""Find a sign that means *nearly* the right thing.

The dictionary has roughly nine thousand signs, and an ordinary page will
still walk off the end of it. When that happens the choices are: drop the
word, fingerspell it letter by letter, or sign something close and say so.
Fingerspelling a long word is slow and hard to read, and dropping it loses
the sentence — so a labelled near-miss is usually the kindest option.

Three strategies, cheapest first:

1. **Derivation.** "accessibility" is "access" with two suffixes bolted on.
   Peeling them back finds the sign that is already in the dictionary.
2. **Synonyms.** A curated map, because WordNet is a heavy dependency for a
   job that a few hundred common pairs handle.
3. **Head word.** For a compound like "school bus", the last noun usually
   carries the meaning, so "bus" beats nothing.

Everything found here is tagged so the reader always sees that the sign is
an approximation rather than the word on the page. Guessing silently would
be worse than not guessing at all.
"""
from __future__ import annotations

import re
from typing import Iterator, List

# Suffixes peeled back to a base form, longest first so "-ability" is tried
# before "-ity". Each entry is (suffix, replacements to try).
_DERIVATIONS = [
    ("abilities", ["able", ""]),
    ("ability", ["able", ""]),
    ("ibility", ["ible", ""]),
    ("iness", ["y", ""]),
    ("ness", [""]),
    ("ments", ["", "e"]),
    ("ment", ["", "e"]),
    ("ations", ["ate", "e", ""]),
    ("ation", ["ate", "e", ""]),
    ("itions", ["ite", ""]),
    ("ition", ["ite", ""]),
    ("tions", ["te", "t", ""]),
    ("tion", ["te", "t", ""]),
    ("sions", ["se", "d", ""]),
    ("sion", ["se", "d", ""]),
    ("ically", ["ic", "y", ""]),
    ("ically", ["ic", ""]),
    ("fully", ["ful", ""]),
    ("ingly", ["ing", ""]),
    ("ally", ["al", ""]),
    ("ily", ["y"]),
    ("ly", [""]),
    ("iful", ["y", ""]),
    ("ful", [""]),
    ("less", [""]),
    ("able", ["", "e"]),
    ("ible", ["", "e"]),
    ("ists", ["", "e"]),
    ("ist", ["", "e"]),
    ("isms", ["", "e"]),
    ("ism", ["", "e"]),
    ("ives", ["e", ""]),
    ("ive", ["e", ""]),
    ("ities", ["e", ""]),
    ("ity", ["e", ""]),
    ("ous", ["", "e", "y"]),
    ("ance", ["", "e"]),
    ("ence", ["", "e"]),
    ("ers", ["", "e"]),
    ("er", ["", "e"]),
    ("ors", ["", "e"]),
    ("or", ["", "e"]),
    ("ships", [""]),
    ("ship", [""]),
    ("hoods", [""]),
    ("hood", [""]),
    ("ings", ["", "e"]),
    ("ing", ["", "e"]),
    ("ies", ["y"]),
    ("es", ["", "e"]),
    ("ed", ["", "e"]),
    ("s", [""]),
]

# Near-synonyms pointing at the everyday word the dictionary is likely to
# hold. Deliberately one-directional: the value is always the commoner word.
RELATED = {
    # people and places
    "individual": "person", "human": "person", "citizen": "person",
    "youngster": "child", "kid": "child", "youth": "child",
    "parent": "mother", "guardian": "mother", "spouse": "wife",
    "relative": "family", "household": "family", "clan": "family",
    "educator": "teacher", "lecturer": "teacher", "scholar": "student",
    "surgeon": "doctor", "nurse": "doctor", "medic": "doctor",
    "officer": "police", "constable": "police", "cop": "police",
    "labourer": "worker", "laborer": "worker", "employee": "worker",
    "colleague": "friend", "companion": "friend", "mate": "friend",
    "village": "village", "township": "city", "town": "city",
    "nation": "country", "state": "country", "region": "country",
    "dwelling": "house", "apartment": "house", "flat": "house",
    "institute": "school", "academy": "school", "university": "college",
    "clinic": "hospital", "store": "shop", "bazaar": "market",
    "workplace": "office", "street": "road", "highway": "road",
    "path": "road", "lane": "road",
    # things
    "meal": "food", "dish": "food", "snack": "food", "nourishment": "food",
    "water": "water", "liquid": "water", "juice": "drink",
    "garment": "clothes", "clothing": "clothes", "attire": "clothes",
    "novel": "book", "textbook": "book", "notebook": "book",
    "pencil": "pen", "document": "paper", "sheet": "paper",
    "sack": "bag", "purse": "bag", "luggage": "bag",
    "mobile": "phone", "telephone": "phone", "cellphone": "phone",
    "vehicle": "car", "motorcar": "car", "lorry": "truck", "coach": "bus",
    "bicycle": "cycle", "bike": "cycle",
    "cash": "money", "rupee": "money", "payment": "money", "fee": "money",
    "creature": "animal", "beast": "animal", "cattle": "cow",
    "blossom": "flower", "plant": "tree", "forest": "tree",
    # qualities
    "enormous": "big", "huge": "big", "massive": "big", "giant": "big",
    "little": "small", "petite": "small", "compact": "small",
    "rapid": "fast", "swift": "fast", "quick": "fast", "speedy": "fast",
    "slow": "slow", "gradual": "slow",
    "joyful": "happy", "glad": "happy", "cheerful": "happy",
    "pleased": "happy", "content": "happy",
    "unhappy": "sad", "miserable": "sad", "gloomy": "sad",
    "furious": "angry", "annoyed": "angry", "cross": "angry",
    "weary": "tired", "exhausted": "tired", "fatigued": "tired",
    "unwell": "sick", "ill": "sick", "diseased": "sick",
    "clever": "smart", "wise": "smart", "bright": "smart",
    "tough": "hard", "complex": "hard", "complicated": "hard",
    "straightforward": "easy", "effortless": "easy",
    "lovely": "pretty", "gorgeous": "pretty", "handsome": "pretty",
    "accurate": "right", "correct": "right", "proper": "right",
    "incorrect": "wrong", "mistaken": "wrong", "false": "wrong",
    "chilly": "cold", "freezing": "cold", "warm": "hot", "boiling": "hot",
    "elderly": "old", "aged": "old", "ancient": "old",
    "fresh": "new", "modern": "new", "recent": "new",
    # actions
    "consume": "eat", "devour": "eat", "dine": "eat",
    "sip": "drink", "swallow": "drink",
    "slumber": "sleep", "rest": "sleep", "nap": "sleep",
    "stroll": "walk", "march": "walk", "wander": "walk",
    "sprint": "run", "dash": "run", "jog": "run",
    "purchase": "buy", "acquire": "buy",
    "assist": "help", "aid": "help", "support": "help",
    "inquire": "ask", "question": "ask", "query": "ask",
    "reply": "answer", "respond": "answer",
    "comprehend": "understand", "grasp": "understand",
    "recall": "remember", "recollect": "remember",
    "desire": "want", "wish": "want", "crave": "want",
    "require": "need", "demand": "need",
    "observe": "see", "notice": "see", "spot": "see", "view": "see",
    "gaze": "look", "stare": "look", "glance": "look",
    "converse": "talk", "chat": "talk", "discuss": "talk",
    "communicate": "talk", "speak": "talk",
    "instruct": "teach", "train": "teach", "educate": "teach",
    "study": "learn", "master": "learn",
    "construct": "build", "create": "make", "produce": "make",
    "commence": "start", "initiate": "start", "launch": "start",
    "terminate": "end", "conclude": "end", "cease": "stop", "halt": "stop",
    "depart": "leave", "exit": "leave", "arrive": "come", "reach": "come",
    "labour": "work", "labor": "work", "toil": "work",
    "reside": "live", "inhabit": "live", "dwell": "live",
    "perish": "die", "expire": "die",
    "transport": "carry", "convey": "carry",
    "repair": "fix", "mend": "fix",
    "select": "choose", "pick": "choose",
    "permit": "allow", "authorise": "allow", "authorize": "allow",
    "prohibit": "stop", "forbid": "stop",
    # abstract
    "employment": "job", "occupation": "job", "profession": "job",
    "difficulty": "problem", "trouble": "problem", "issue": "problem",
    "solution": "answer", "response": "answer",
    "information": "news", "data": "news", "detail": "news",
    "instruction": "rule", "regulation": "rule", "law": "rule",
    "objective": "goal", "aim": "goal", "purpose": "goal",
    "duration": "time", "period": "time", "moment": "time",
    "location": "place", "position": "place", "spot": "place",
    "quantity": "number", "amount": "number", "total": "number",
    "portion": "part", "section": "part", "segment": "part",
    "beginning": "start", "conclusion": "end", "finish": "end",
    "journey": "travel", "trip": "travel", "voyage": "travel",
    "celebration": "festival", "ceremony": "festival",
    "narrative": "story", "tale": "story", "account": "story",
    "language": "language", "speech": "talk", "conversation": "talk",
    "accessibility": "access", "availability": "available",
    # grammar markers and set phrases the dictionary files under another name
    "will": "future",           # its only "will" signs are the legal testament
    "thank": "thank you", "thanks": "thank you",
}

_WORD = re.compile(r"[a-z]+")


def derivations(word: str) -> Iterator[str]:
    """Peel derivational suffixes back toward a base form."""
    w = word.lower().strip()
    if len(w) < 4:
        return
    for suffix, replacements in _DERIVATIONS:
        if not w.endswith(suffix) or len(w) - len(suffix) < 3:
            continue
        stem = w[: -len(suffix)]
        for rep in replacements:
            cand = stem + rep
            if len(cand) >= 3 and cand != w:
                yield cand
        # "running" -> "runing" -> undo the doubled consonant
        if len(stem) >= 4 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            yield stem[:-1]


def synonyms(word: str) -> Iterator[str]:
    """RELATED first, then the bookish-word map the glosser already carries —
    no reason to maintain two lists of the same idea."""
    from .lexicon import SIMPLIFY

    key = word.lower().strip()
    seen = set()
    for table in (RELATED, SIMPLIFY):
        hit = table.get(key)
        if hit and hit not in seen:
            seen.add(hit)
            yield hit


def head_words(phrase: str) -> Iterator[str]:
    """For a compound, try the head noun first, then the modifier."""
    parts = _WORD.findall(phrase.lower())
    if len(parts) < 2:
        return
    for part in reversed(parts):
        if len(part) >= 3:
            yield part


def candidates(word: str) -> List[str]:
    """Every related form worth trying, best guess first, de-duplicated."""
    out: List[str] = []
    seen = {word.lower().strip()}
    for source in (synonyms(word), derivations(word), head_words(word)):
        for cand in source:
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
    # Stacked suffixes need a second pass: "carelessness" sheds -ness to reach
    # "careless", and only then -less to reach "care". Also pick up synonyms of
    # whatever the first pass uncovered ("accessibility" -> "access" -> ...).
    for base in list(out):
        for cand in list(derivations(base)) + list(synonyms(base)):
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
    return out
