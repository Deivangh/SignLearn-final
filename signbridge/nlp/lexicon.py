"""Word lists that drive the English -> ISL gloss rules.

ISL is not signed English. It drops function words, marks tense once at the
front of a clause, puts the verb last, and parks negation and question words
at the end. These tables encode the closed-class vocabulary those rules need.
"""
from __future__ import annotations

# Dropped entirely — ISL has no articles, copula, or auxiliary chain.
DROP = {
    "a", "an", "the",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done",
    "have", "has", "had",
    "shall", "of", "as", "than", "that", "this", "these", "those",
    "there", "here", "it", "its", "so", "such", "very", "just", "quite",
    "however", "therefore", "thus", "moreover", "indeed", "rather",
    "some", "any", "each", "every", "both", "either", "neither",
}

# Kept but rewritten — these carry meaning ISL expresses differently.
REWRITE = {
    "i": "me", "me": "me", "my": "my", "mine": "my", "myself": "me",
    "we": "we", "us": "we", "our": "our", "ourselves": "we",
    "you": "you", "your": "your", "yours": "your", "yourself": "you",
    "he": "he", "him": "he", "his": "his", "himself": "he",
    "she": "she", "her": "she", "hers": "her", "herself": "she",
    "they": "they", "them": "they", "their": "their", "themselves": "they",
    "cannot": "cannot", "can't": "cannot", "won't": "not", "don't": "not",
    "doesn't": "not", "didn't": "not", "isn't": "not", "aren't": "not",
    "wasn't": "not", "weren't": "not", "haven't": "not", "hasn't": "not",
    "shouldn't": "not", "wouldn't": "not", "couldn't": "not",
}

NEGATION = {"not", "no", "never", "none", "nothing", "nobody", "nowhere", "cannot"}

# Number words are never verbs, however they end: "hundred" is not "hundr-ed".
NUMBER_WORDS = set("""
zero one two three four five six seven eight nine ten eleven twelve thirteen
fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty
sixty seventy eighty ninety hundred thousand lakh lakhs crore crores
""".split())

QUESTION = {"what", "where", "when", "why", "who", "whom", "whose", "which", "how"}

# Tense is signalled once, at the head of the clause.
TENSE_MARKERS = {"past": "before", "future": "will", "present": ""}

TIME_WORDS = {
    "today", "tomorrow", "yesterday", "now", "later", "soon", "then",
    "morning", "afternoon", "evening", "night", "day", "week", "month",
    "year", "always", "never", "sometimes", "often", "daily", "weekly",
    "monthly", "yearly", "before", "after", "early", "late", "again",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}

PLACE_PREPS = {"in", "on", "at", "under", "over", "near", "inside", "outside",
               "above", "below", "between", "behind", "beside", "into", "onto"}

# Prepositions that vanish; the spatial relationship is shown, not spelled.
DROP_PREPS = {"to", "for", "from", "with", "by", "about", "into", "onto",
              "upon", "off", "out", "up", "down", "through", "during",
              "against", "among", "across", "along", "toward", "towards"}

MODALS = {"can", "could", "will", "would", "should", "must", "may", "might", "need"}

CONJUNCTIONS = {"and", "or", "but", "because", "if", "while", "although",
                "though", "unless", "since", "whereas", "yet"}

# Irregular past/participle -> base form.
IRREGULAR = {
    "was": "be", "were": "be", "been": "be", "am": "be", "is": "be", "are": "be",
    "went": "go", "gone": "go", "goes": "go",
    "ate": "eat", "eaten": "eat", "eats": "eat",
    "saw": "see", "seen": "see", "sees": "see",
    "took": "take", "taken": "take", "takes": "take",
    "gave": "give", "given": "give", "gives": "give",
    "came": "come", "come": "come", "comes": "come",
    "made": "make", "makes": "make",
    "said": "say", "says": "say",
    "told": "tell", "tells": "tell",
    "knew": "know", "known": "know", "knows": "know",
    "thought": "think", "thinks": "think",
    "got": "get", "gotten": "get", "gets": "get",
    "found": "find", "finds": "find",
    "ran": "run", "runs": "run", "running": "run",
    "wrote": "write", "written": "write", "writes": "write",
    "read": "read", "reads": "read",
    "brought": "bring", "bought": "buy", "taught": "teach", "caught": "catch",
    "sat": "sit", "stood": "stand", "slept": "sleep", "kept": "keep",
    "left": "leave", "felt": "feel", "held": "hold", "heard": "hear",
    "used": "use", "died": "die", "flew": "fly", "drank": "drink", "sent": "send",
    "sold": "sell", "spoke": "speak", "began": "begin", "grew": "grow", "fell": "fall",
    "children": "child", "men": "man", "women": "woman", "feet": "foot",
    "teeth": "tooth", "people": "person", "mice": "mouse", "geese": "goose",
    "leaves": "leaf", "lives": "life", "knives": "knife", "wives": "wife",
}

PAST_TENSE_HINTS = {
    "was", "were", "had", "did", "went", "ate", "saw", "took", "gave",
    "came", "made", "said", "told", "knew", "thought", "got", "found",
    "ran", "wrote", "brought", "bought", "taught", "caught", "sat",
    "stood", "slept", "kept", "left", "felt", "held", "heard", "yesterday",
    "died", "flew", "drank", "sent", "sold", "spoke", "began", "grew", "fell", "ago",
}

# An -ed word after one of these is a state, not a past event:
# "the door is closed", "I am tired".
STATE_VERBS = {"am", "is", "are", "was", "were", "be", "been", "being", "get",
               "gets", "got", "feel", "feels", "felt", "seem", "seems", "look",
               "looks", "become", "becomes", "very", "so", "too"}

# A sentence that ends in "?" and opens with one of these is a yes/no question.
AUX_START = {"is", "are", "am", "was", "were", "do", "does", "did", "can",
             "could", "will", "would", "shall", "should", "may", "might", "must",
             "have", "has", "had"}

FUTURE_HINTS = {"will", "shall", "tomorrow", "soon", "later", "going to"}

# Formal or bookish words swapped for the everyday word the dictionary is
# far more likely to have a sign for.
SIMPLIFY = {
    "purchase": "buy", "automobile": "car", "physician": "doctor",
    "residence": "house", "commence": "start", "terminate": "end",
    "assist": "help", "inquire": "ask", "reply": "answer", "respond": "answer",
    "obtain": "get", "receive": "get", "require": "need", "desire": "want",
    "observe": "see", "comprehend": "understand", "utilise": "use",
    "utilize": "use", "construct": "build", "depart": "leave",
    "arrive": "come", "converse": "talk", "communicate": "talk",
    "consume": "eat", "beverage": "drink", "infant": "baby",
    "spouse": "wife", "sibling": "brother", "offspring": "child",
    "gigantic": "big", "enormous": "big", "tiny": "small", "minute": "small",
    "rapid": "fast", "swift": "fast", "sluggish": "slow",
    "intelligent": "smart", "difficult": "hard", "simple": "easy",
    "beautiful": "pretty", "attractive": "pretty", "delighted": "happy",
    "sorrowful": "sad", "furious": "angry", "frightened": "afraid",
    "exhausted": "tired", "unwell": "sick", "ill": "sick",
    "instructor": "teacher", "pupil": "student", "employment": "job",
    "occupation": "job", "currency": "money", "cuisine": "food",
    "vehicle": "car", "aircraft": "plane", "locomotive": "train",
}

# Suffix hints for the fallback tagger when spaCy is not installed.
NOUN_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ance", "ence",
                 "ship", "hood", "dom", "ist", "ism", "er", "or")
VERB_SUFFIXES = ("ise", "ize", "ify", "ate")
ADJ_SUFFIXES = ("ous", "ful", "less", "able", "ible", "ive", "al", "ic",
                "ish", "ary", "y")
ADV_SUFFIXES = ("ly",)


# High-frequency verbs. Without spaCy the suffix tagger cannot tell "heats"
# (verb) from "sports" (noun), and mis-tagging the verb wrecks the SOV
# reordering — so the common cases are listed outright.
COMMON_VERBS = set("""
be have do go come get make take give put keep let say tell ask answer speak
talk call shout whisper read write draw paint sing dance play work study learn
teach show teach help try want need like love hate feel think know understand
remember forget believe hope wish mean guess decide choose plan agree refuse
see look watch hear listen smell taste touch hold carry lift push pull throw
catch drop break fix build cut open close lock start stop finish end begin
continue wait stay leave arrive enter exit return move walk run jump climb fall
sit stand lie sleep wake rise set eat drink cook wash clean wear buy sell pay
cost spend save earn win lose find lose search meet visit live die grow change
happen become seem appear cause bring send receive accept reject follow lead
join share join count measure weigh fill empty add remove join cross turn
travel drive ride fly swim rain snow shine blow burn freeze melt heat cool
smile laugh cry worry fear enjoy rest hurry protect save serve use practise
practice repeat explain describe compare check test pass fail rule govern
""".split())


# Nouns that end in -ing and must never be stemmed to a verb.
ING_NOUNS = {
    "morning", "evening", "king", "thing", "ring", "spring", "string",
    "building", "meaning", "feeling", "painting", "drawing", "writing",
    "reading", "clothing", "wedding", "meeting", "shopping", "swimming",
    "everything", "something", "nothing", "anything", "during", "wing",
    "sing", "bring", "young", "song", "long", "along", "being",
}
