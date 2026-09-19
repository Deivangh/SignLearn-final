"""SignBridge — read any printed page in Indian Sign Language.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import html
import re

import streamlit as st
from PIL import Image

from signbridge import ocr, pipeline
from signbridge.avatar.pose_extract import has_pose
from signbridge.avatar.renderer import build_avatar_html
from signbridge.avatar.rig import build_rig_html
from signbridge.config import (
    AVATAR_MODEL,
    AVATAR_MODEL_URL,
    DATAGOV_API_KEY,
    DEFAULT_SIGN_LANGUAGE,
    OCR_LANGUAGES,
    SIGN_LANGUAGES,
)
from signbridge.isl.index import SignIndex
from signbridge.nlp.context import estimate_duration, page_keywords
from signbridge.nlp.gloss import spacy_available
from signbridge.ui.player import build_video_html
from signbridge.ui.theme import APP_CSS

st.set_page_config(
    page_title="SignBridge — read in Indian Sign Language",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_index() -> SignIndex:
    return SignIndex.load()


@st.cache_data(show_spinner=False)
def run_ocr(image_bytes: bytes, langs: str, psm: int, preprocess: bool):
    import io

    image = Image.open(io.BytesIO(image_bytes))
    return ocr.read_page(image, langs=langs, psm=psm, do_preprocess=preprocess)


def digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:12]


# ----------------------------------------------------------------------
# Masthead
# ----------------------------------------------------------------------
st.markdown(
    """
<div class="sb-mast">
  <span class="sb-eyebrow">Indian Sign Language · reading aid</span>
  <h1>Every printed page,<br><em>in the language you already think in.</em></h1>
  <p>Photograph a page from any book. SignBridge reads it, works out what the page
     is actually saying, and signs it back to you from the official Indian Sign
     Language Dictionary — sentence by sentence, at your speed.</p>
</div>
""",
    unsafe_allow_html=True,
)

index = get_index()

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Reading setup")

    enabled = [k for k, v in SIGN_LANGUAGES.items() if v["enabled"]]
    sign_lang = st.selectbox(
        "Sign language",
        enabled,
        index=enabled.index(DEFAULT_SIGN_LANGUAGE),
        format_func=lambda k: SIGN_LANGUAGES[k]["name"],
    )
    st.caption(SIGN_LANGUAGES[sign_lang]["note"])

    installed = ocr.available_languages()
    options = [c for c in OCR_LANGUAGES if c in installed] or ["eng"]
    picked = st.multiselect(
        "Page is printed in",
        options,
        default=["eng"] if "eng" in options else options[:1],
        format_func=lambda c: OCR_LANGUAGES.get(c, c),
        help="Install more Tesseract language packs to unlock Hindi, Telugu, Tamil and others.",
    )
    ocr_langs = "+".join(picked) if picked else "eng"

    st.markdown("---")
    st.markdown("## How much to sign")
    depth = st.select_slider(
        "Depth",
        options=["Just the gist", "Main points", "Most of the page", "Every sentence"],
        value="Main points",
        label_visibility="collapsed",
    )
    coverage = {"Just the gist": 0.3, "Main points": 0.6,
                "Most of the page": 0.85, "Every sentence": 1.0}[depth]

    keep_all = st.toggle(
        "Sign grammar words too",
        value=False,
        help="Off by default. ISL does not use articles or the verb 'to be', so "
             "signing them adds length without adding meaning.",
    )
    fingerspell = st.toggle("Fingerspell unknown words", value=True)

    st.markdown("---")
    st.markdown("## Dictionary")

    if len(index):
        st.markdown(
            f'<div class="sb-stats"><div class="sb-stat"><div class="n">{len(index):,}</div>'
            '<div class="l">signs indexed</div></div></div>',
            unsafe_allow_html=True,
        )
        cached = sum(1 for e in index.entries if e.is_cached)
        posed = sum(1 for e in index.entries if has_pose(e.word))
        st.caption(f"{cached:,} clips downloaded · {posed:,} with avatar motion")
    else:
        st.warning("No dictionary yet.")
        if not DATAGOV_API_KEY:
            st.caption("Add DATAGOV_API_KEY to your .env file, then build the index.")
        st.code("python scripts/build_index.py", language="bash")

    with st.expander("Prepare more signs"):
        st.caption(
            "**Frame the signer** — finds the signer in each downloaded clip so "
            "the real-signer view can zoom to their hands. Quick, do this first."
        )
        st.code("python scripts/warm_cache.py --crops", language="bash")
        st.caption(
            "**Avatar motion** — turns clips into skeletons for the avatar view. "
            "A few seconds per sign."
        )
        st.code("python scripts/warm_cache.py --poses", language="bash")
        st.caption("**More words** — download and prepare the common ones.")
        st.code("python scripts/warm_cache.py --common 300 --crops --poses",
                language="bash")

    if not spacy_available():
        st.markdown(
            '<div class="sb-note">Running on the built-in tagger. For sharper '
            "sentence structure, install the spaCy model:<br>"
            "<code>python -m spacy download en_core_web_sm</code></div>",
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------
tab_upload, tab_camera, tab_type = st.tabs(["Upload a photo", "Use camera", "Type text"])

image_bytes = None
typed = ""

with tab_upload:
    up = st.file_uploader(
        "Page photo", type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )
    if up:
        image_bytes = up.getvalue()

with tab_camera:
    st.caption(
        "Point this at **printed text** — a book, a sign, a form. SignBridge "
        "reads writing and signs it back; it does not read hand gestures."
    )
    shot = st.camera_input("Point at the page", label_visibility="collapsed")
    if shot:
        image_bytes = shot.getvalue()

with tab_type:
    typed = st.text_area(
        "Text", height=150, label_visibility="collapsed",
        placeholder="Paste or type a passage to sign it directly.",
    )
    st.caption("Click outside the box (or press Ctrl+Enter) to sign what you typed.")

with st.expander("Reading adjustments", expanded=False):
    c1, c2 = st.columns(2)
    psm = c1.selectbox(
        "Page layout", [6, 3, 4, 11],
        format_func=lambda p: {
            6: "Single block of text (default)", 3: "Full page, mixed",
            4: "Columns", 11: "Sparse / scattered text",
        }[p],
    )
    do_pre = c2.toggle("Clean up the photo first", value=True,
                       help="Flattens lamp shadow, sharpens, straightens a tilted page.")

# ----------------------------------------------------------------------
# Read
# ----------------------------------------------------------------------
text, result = "", None

# Typed text wins over a photo. A file_uploader keeps its value across reruns,
# so without this the first photo of the session would silently block every
# passage typed afterwards.
if typed.strip():
    text = typed.strip()
    source = "the text you typed"
elif image_bytes:
    source = "your photo"
    try:
        with st.spinner("Reading the page…"):
            result = run_ocr(image_bytes, ocr_langs, psm, do_pre)
        text = result.text
    except ocr.OcrUnavailable as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Could not read that image: {exc}")
        st.stop()

    if not text.strip():
        st.warning(
            "**No readable text in that image.** SignBridge reads printed or "
            "written words — it does not recognise hand signs or objects. "
            "Photograph a page of text, or use the **Type text** tab."
        )
        st.stop()
else:
    source = ""

if not text:
    st.markdown(
        '<div class="sb-note">Start by photographing a page of text, or type a '
        "passage in the <b>Type text</b> tab. Good light and a flat page beat a "
        "high-resolution camera every time.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

if typed.strip() and image_bytes:
    st.caption(f"Signing {source}. Clear the text box to go back to your photo.")

if not len(index):
    st.error(
        "The dictionary index is empty, so there is nothing to sign with yet. "
        "Run `python scripts/build_index.py` once and reload."
    )
    st.stop()

# ----------------------------------------------------------------------
# Plan
# ----------------------------------------------------------------------
plan = pipeline.plan_page(
    text, index, coverage=coverage, keep_all=keep_all, allow_fingerspell=fingerspell
)

left, right = st.columns([1, 1.25], gap="large")

with left:
    st.markdown("#### What the page says")
    keywords = set(page_keywords(text, 10))
    shown = html.escape(text[:2600])
    for kw in sorted(keywords, key=len, reverse=True):
        shown = re.sub(rf"\b({re.escape(kw)})\b", r"<mark>\1</mark>", shown, flags=re.I)
    devanagari = "indic" if re.search(r"[\u0900-\u0D7F]", text) else ""
    st.markdown(
        f'<div class="sb-paper {devanagari}">{shown.replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )

    if result:
        st.markdown(
            f'<div class="sb-stats">'
            f'<div class="sb-stat"><div class="n">{result.word_count}</div><div class="l">words read</div></div>'
            f'<div class="sb-stat"><div class="n">{result.confidence:.0f}%</div><div class="l">read confidence</div></div>'
            f'<div class="sb-stat"><div class="n">{len(plan.sentences)}</div><div class="l">sentences</div></div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if result.confidence < 62:
            st.markdown(
                '<div class="sb-note">The photo is hard to read. Flatten the page, '
                "move the lamp behind you, and shoot straight down.</div>",
                unsafe_allow_html=True,
            )

with right:
    st.markdown("#### What will be signed")
    st.markdown(
        f'<div class="sb-stats">'
        f'<div class="sb-stat"><div class="n">{len(plan.clips)}</div><div class="l">signs queued</div></div>'
        f'<div class="sb-stat"><div class="n">{plan.coverage*100:.0f}%</div><div class="l">of glosses covered</div></div>'
        f'<div class="sb-stat"><div class="n">{estimate_duration(len(plan.clips))}</div><div class="l">watch time</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Sentence by sentence", expanded=True):
        rows = []
        for sent in plan.sentences:
            if not sent.is_key and coverage < 1.0:
                continue
            chips = []
            for g in sent.glosses:
                res = plan.resolutions.get(g)
                shown, note, title = g, "", ""
                if res is None:
                    cls = "miss"
                elif res.method in ("exact", "alias"):
                    cls = "ok"
                elif res.method == "group":
                    # The dictionary files this sign under a group of
                    # synonyms ("house home"); say so on hover.
                    cls = "ok"
                    title = f"Dictionary sign: {res.entry.word}"
                elif res.method in ("fuzzy", "related"):
                    cls = "near"
                    # Name the sign that stood in, so a substitution is never
                    # mistaken for a match on the page's own word.
                    if res.entry:
                        shown, note = res.entry.word, f" (for “{g}”)"
                elif res.spelled:
                    cls = "spell"
                else:
                    cls = "miss"
                chips.append(
                    f'<span class="sb-chip {cls}" title="{html.escape(title)}">'
                    f"{html.escape(shown)}{html.escape(note)}</span>"
                )
            label = sent.tense.upper() if sent.tense != "present" else ""
            if sent.question:
                label = f"{label} · QUESTION" if label else "QUESTION"
            rows.append(
                f'<div class="sb-sentence {"key" if sent.is_key else ""}">'
                f'<span class="no">SENTENCE {sent.index + 1}'
                f'{" · " + label if label else ""}</span>'
                f'<div class="src">{html.escape(sent.source[:260])}</div>'
                f'<div class="sb-gloss-row">{"".join(chips)}</div></div>'
            )
        st.markdown("".join(rows) or "<p>Nothing selected.</p>", unsafe_allow_html=True)

if not plan.clips:
    st.warning(
        "None of the words on this page are in the dictionary yet. Try a simpler "
        "passage, or turn on fingerspelling in the sidebar."
    )
    st.stop()

# ----------------------------------------------------------------------
# Fetch + play
# ----------------------------------------------------------------------
st.markdown("### Watch it signed")

# Real footage first. The avatar is a useful abstraction — steady framing, any
# angle, any speed — but a real signer's hands are what a reader can actually
# read, so that is what they should meet first.
view = st.radio(
    "View", ["Real signer", "Signing avatar"],
    horizontal=True, label_visibility="collapsed",
)

key = digest((text + str(coverage) + str(keep_all) + str(fingerspell)).encode())
bar = st.progress(0.0, text="Preparing signs…")
plan = pipeline.materialise(plan, index, progress=lambda p, m: bar.progress(min(p, 1.0), text=m))
bar.empty()

if view == "Signing avatar":
    clips = [c.as_avatar() for c in plan.clips if c.pose]
    # The rigged character when one is installed, the procedural figure
    # otherwise — the app stays usable with no model file present.
    if AVATAR_MODEL.exists():
        st.iframe(build_rig_html(clips, uid=key, model_url=AVATAR_MODEL_URL), height=690)
    else:
        st.iframe(build_avatar_html(clips, uid=key), height=690)
    if clips:
        st.caption(
            "Drag to walk around the signer. Space plays and pauses, arrow keys "
            "step between signs. The corner insets are close-ups of each hand."
        )
    st.caption(f"{len(clips)} of {len(plan.clips)} signs have avatar motion extracted.")
else:
    video_clips = [c.as_video() for c in plan.clips]
    st.iframe(build_video_html(video_clips, uid=key), height=690)
    framed = sum(1 for c in video_clips if c.get("crop"))
    st.caption(
        "Real signers from the Indian Sign Language Dictionary, Department of "
        "Empowerment of Persons with Disabilities — zoomed to the signer so the "
        "handshapes are readable."
    )
    if framed < len(video_clips):
        st.caption(
            f"{framed} of {len(video_clips)} clips are auto-framed. Run "
            "`python scripts/warm_cache.py --crops` to frame the rest."
        )

if plan.missing:
    st.markdown(
        '<div class="sb-note"><b>Not in the dictionary:</b> '
        + ", ".join(html.escape(w) for w in plan.missing[:18])
        + ("…" if len(plan.missing) > 18 else "")
        + "<br>These are logged to <code>data/unresolved_words.json</code> so you "
          "can see what the dictionary is missing most often.</div>",
        unsafe_allow_html=True,
    )
