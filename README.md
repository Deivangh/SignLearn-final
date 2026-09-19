# SignBridge

Photograph a page from any book. SignBridge reads it, works out what the page
is actually saying, and signs it back in Indian Sign Language — from the
official ISL Dictionary, on a 3D signing avatar.

Built for readers whose first language is a sign language and for whom printed
text is a second language, not a first one.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Strongly recommended — much better sentence structure
pip install spacy && python -m spacy download en_core_web_sm

cp .env.example .env             # then add your keys
python scripts/doctor.py         # tells you what is still missing
python scripts/build_index.py    # one-time, builds the dictionary index
streamlit run app.py
```

You also need **Tesseract OCR** installed as a program, not just the Python
package:

| Platform | Command |
| --- | --- |
| Windows | [download the installer](https://github.com/UB-Mannheim/tesseract/wiki), then set `TESSERACT_CMD` in `.env` |
| macOS | `brew install tesseract tesseract-lang` |
| Ubuntu/Debian | `sudo apt install tesseract-ocr tesseract-ocr-hin tesseract-ocr-tel` |

### Keys

`DATAGOV_API_KEY` — required. Yours is already working; put it in `.env`.

`GOOGLE_API_KEY` — optional but worth ten minutes. A Google Cloud API key with
the Drive API enabled makes dictionary indexing reliable. Without it SignBridge
falls back to Drive's public folder listing, which works but is undocumented
and can change without warning.

> Your data.gov.in key appeared in a chat transcript, so treat it as public.
> Regenerating it on data.gov.in takes about a minute and is worth doing.

### Warming the cache

The first read of a page downloads clips as it needs them. Pre-loading the
common vocabulary makes it feel instant:

```bash
python scripts/warm_cache.py --common 400        # download frequent words
python scripts/warm_cache.py --poses             # extract avatar motion
```

---

## How it works

```
  page photo
      │
      ├─ ocr.py ─────────── flatten lamp shadow → deskew → Tesseract
      │
      ├─ nlp/context.py ─── score sentences, keep the ones that carry the page
      │
      ├─ nlp/gloss.py ───── English → ISL gloss
      │                     drop articles and copula, lemmatise, mark tense
      │                     once, reorder to TIME · PLACE · SUBJECT · OBJECT ·
      │                     VERB, push negation and question words to the end
      │
      ├─ isl/index.py ───── gloss → dictionary clip
      │                     exact → alias → fuzzy → fingerspell → report miss
      │
      ├─ isl/drive.py ───── download on demand, cache in static/videos
      │
      ├─ avatar/pose_extract.py ── MediaPipe Pose + Hands → skeleton JSON
      │
      └─ two views, one transport
            • Dictionary video — the original clips, played in sequence
            • Signing avatar   — one consistent signer, any speed, offline
```

### Why the page is not signed word for word

A book page is 250–400 words. Signed one word at a time that is roughly eight
minutes of video for something a hearing reader skims in ninety seconds, and
most of it is grammar ISL does not use. So sentences are ranked by how much of
the page's meaning they carry, and the **Depth** control decides how many make
the cut — from the gist to every sentence.

### Why a skeleton avatar, not a rigged character

The dictionary clips are filmed by different signers, at different distances,
against different backgrounds. Thirty of them in a row is thirty jump cuts.
The avatar re-signs everything as one figure at one scale, which is far easier
to follow, and it works offline at any speed once motion is extracted.

The avatar is drawn procedurally rather than driven through a rigged GLB. A
GLB approach means every user must first source a rig, and it breaks the
moment the bone names differ. Swapping one in later is a contained change:
keep the frame data, retarget it in `buildFigure()` in
`avatar/renderer.py`.

Hands are drawn marigold and the body cool grey, because in a sign language
the hands are the words and the body is the grammar. The trail behind each
index finger keeps the *shape* of a fast sign readable, which is the hardest
part of learning a sign from flat video.

---

## What this does not do

Worth being straight about, because it shapes what you build next.

**This is a reading aid, not a translator.** English → ISL gloss is
rule-based. It gets word order, tense marking and negation right most of the
time, but ISL has spatial grammar, classifiers, non-manual markers and
role-shift that no word-by-word system reproduces. A Deaf reader will find it
useful. A Deaf reader will also spot where it is wrong. Get one to try it
early — that feedback is worth more than any amount of extra code.

**Generative video (Veo 3 and similar) will not work for the signs.** It was
a reasonable thing to reach for, but text-to-video models produce plausible
*looking* hands, not linguistically correct ones — wrong handshape, wrong
location, wrong movement path, and no way for the model to know it is wrong.
For a reader who cannot check the output against text, a confidently wrong
sign is worse than a missing one. The dictionary clips are ground truth and
the avatar replays them faithfully; that is the right foundation. Generative
video is genuinely useful one layer out — a scene illustration that gives
context before a passage — and the hook for that is in `config.py`
(`ENABLE_SCENE_ILLUSTRATION`), off by default.

**Coverage depends on the dictionary.** The index holds 12,007 English-named
signs, including the collections that sit only under the dictionary's root
folder. English has far more words. Unresolved words are logged to
`data/unresolved_words.json` — that file is your roadmap for what to add.

Many everyday signs are filed under a group of synonyms (`House_Home.mp4`,
`No_Not.mp4`). `build_index.py` uses WordNet to tell those groups from real
phrases (`Home_Automation.mp4`), so building the index needs
`pip install nltk` and `python -c "import nltk; nltk.download('wordnet')"`.
The app itself does not. `python scripts/build_index.py --refresh` redoes
this on an existing index without crawling Drive again.

**Fingerspelling needs alphabet clips.** The official release has no
per-letter clips (`Alphabet.mp4` is the sign for the word). Put `a.mp4` …
`z.mp4` in `static/alphabet/` and unknown words will be spelled; until then
they are reported as missing. Digits already spell from the number signs.

---

## Adding another sign language

The pipeline is language-agnostic; only the source adapter is ISL-specific.

1. Write a module exposing `SignIndex`-compatible `resolve(word)`.
2. Register it in `SIGN_LANGUAGES` in `config.py` with `"enabled": True`.
3. Adjust the gloss rules in `nlp/lexicon.py` — ASL and BSL order clauses
   differently from ISL.

The sidebar selector picks it up automatically.

---

## Layout

```
app.py                      Streamlit UI
signbridge/
  config.py                 all settings and paths
  ocr.py                    page photo → clean text
  pipeline.py               text → playable clip queue
  nlp/lexicon.py            ISL grammar tables and word lists
  nlp/gloss.py              English → ISL gloss
  nlp/context.py            sentence ranking, page gist
  isl/datagov.py            data.gov.in dictionary client
  isl/drive.py              Drive folder listing and download
  isl/index.py              word → clip resolution
  avatar/pose_extract.py    MediaPipe Pose + Hands → skeleton JSON
  avatar/renderer.py        Three.js signing stage
  ui/theme.py               design tokens and CSS
  ui/player.py              sequential clip player
scripts/
  build_index.py            one-time dictionary index
  warm_cache.py             pre-download clips, pre-extract motion
  doctor.py                 environment check
static/videos               cached clips (served, not inlined)
static/poses                extracted motion
data/                       index and the unresolved-word log
```

Clips are used under the terms of the Indian Sign Language Dictionary
published by the Department of Empowerment of Persons with Disabilities,
Ministry of Social Justice and Empowerment, via data.gov.in.
