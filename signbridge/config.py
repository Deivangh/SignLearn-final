"""Central configuration. Everything tunable lives here or in .env."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent

# --- Paths -------------------------------------------------------------
DATA_DIR = ROOT / "data"
STATIC_DIR = ROOT / "static"
VIDEO_DIR = STATIC_DIR / "videos"          # served at /app/static/videos/<file>
POSE_DIR = STATIC_DIR / "poses"            # served at /app/static/poses/<file>
ALPHABET_DIR = STATIC_DIR / "alphabet"     # optional a.mp4 ... z.mp4 for fingerspelling
INDEX_FILE = DATA_DIR / "isl_index.json"
MANIFEST_FILE = DATA_DIR / "isl_folders.json"
MISS_LOG = DATA_DIR / "unresolved_words.json"

for _d in (DATA_DIR, VIDEO_DIR, POSE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

STATIC_URL = "/app/static"                 # Streamlit static serving mount point

# --- data.gov.in (Indian Sign Language Dictionary) ---------------------
DATAGOV_API_KEY = os.getenv("DATAGOV_API_KEY", "")
DATAGOV_RESOURCE_ID = os.getenv(
    "DATAGOV_RESOURCE_ID", "fb44b68b-babb-4ba7-b2c7-ef334162c0ac"
)
DATAGOV_BASE = "https://api.data.gov.in/resource"

# Optional: a Google Cloud API key with the Drive API enabled. Without it we
# fall back to Drive's public embedded folder view, which needs no key but is
# more fragile.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# --- OCR ---------------------------------------------------------------
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")   # e.g. C:\Program Files\Tesseract-OCR\tesseract.exe
DEFAULT_OCR_LANGS = os.getenv("OCR_LANGS", "eng")

# Tesseract language code -> friendly name. Install the matching traineddata
# to unlock each one.
OCR_LANGUAGES = {
    "eng": "English",
    "hin": "Hindi",
    "tel": "Telugu",
    "tam": "Tamil",
    "ben": "Bengali",
    "mar": "Marathi",
    "guj": "Gujarati",
    "kan": "Kannada",
    "mal": "Malayalam",
    "pan": "Punjabi",
    "urd": "Urdu",
    "ori": "Odia",
}

# --- Sign language target ---------------------------------------------
# Only ISL for now. Adding another is a matter of adding a source adapter
# and a new entry here — the rest of the pipeline is language-agnostic.
SIGN_LANGUAGES = {
    "ISL": {
        "name": "Indian Sign Language",
        "adapter": "signbridge.isl.index",
        "enabled": True,
        "note": "Official dictionary, Dept. of Empowerment of Persons with Disabilities",
    },
    "ASL": {"name": "American Sign Language", "adapter": None, "enabled": False},
    "BSL": {"name": "British Sign Language", "adapter": None, "enabled": False},
}
DEFAULT_SIGN_LANGUAGE = "ISL"

# --- Pose extraction ---------------------------------------------------
POSE_TARGET_FPS = 20          # downsample; ISL signs read fine at 20fps
POSE_MAX_SECONDS = 8          # guard against a runaway dictionary clip
POSE_DECIMALS = 4             # rounding keeps JSON small
POSE_SMOOTH_WINDOW = 5        # odd; centred moving average that kills tracker tremor
POSE_HOLD_FRAMES = 6          # bridge dropouts up to this long; longer ones are real rest

# --- Framing the real signer -------------------------------------------
CROP_FILE = POSE_DIR / "_crops.json"
CROP_URL = f"{STATIC_URL}/poses/_crops.json"
CROP_SAMPLES = 16             # frames sampled per clip when finding the signer
CROP_PAD = 0.16               # margin around the signer, as a fraction of the box
TARGET_ASPECT = 4 / 3         # every crop is grown to this, so framing never jumps

# --- Rigged avatar -----------------------------------------------------
# A Mixamo-named glTF character. Present -> the avatar view drives it;
# absent -> the app falls back to the procedural figure, which needs no asset.
AVATAR_MODEL = STATIC_DIR / "avatar.glb"
AVATAR_MODEL_URL = f"{STATIC_URL}/avatar.glb"

# --- Matching ----------------------------------------------------------
# Fuzzy matches must also share a stem with the word (see isl/index.py);
# similarity alone turned "money" into MONKEY.
FUZZY_CUTOFF = 0.86
MAX_CLIPS_PER_PAGE = 120

# --- Optional generative illustration (off by default) -----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ENABLE_SCENE_ILLUSTRATION = os.getenv("ENABLE_SCENE_ILLUSTRATION", "0") == "1"

USER_AGENT = "SignBridge/1.0 (+education; ISL reading aid)"
