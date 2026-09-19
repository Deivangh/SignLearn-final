"""Work out where the signer actually is inside a dictionary clip.

The dictionary films at 1920x1080 with the signer standing well back, so the
hands — the only part that carries meaning — end up a few hundred pixels
across in a very large frame. Played raw, a handshape is too small to read.

This scans a clip, finds the box that contains the signer's head, arms and
hands across the whole sign, and stores it. The player then zooms each clip
into its own box, which does two things at once: the hands get the whole
screen, and clips shot by different signers at different distances all come
out framed the same way, so a page of thirty signs stops looking like thirty
jump cuts.

Boxes are normalised (0-1) against the source frame and pre-expanded so the
crop's *pixel* aspect matches :data:`TARGET_ASPECT`. That lets the browser do
the zoom with plain CSS and no distortion, whatever the source resolution.

Written to ``static/poses/_crops.json`` as ``{word: {x, y, w, h}}`` — one
small file the player fetches once, rather than a sidecar per clip.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from ..config import CROP_FILE, CROP_PAD, CROP_SAMPLES, TARGET_ASPECT

# Head, shoulders, elbows, wrists. Hips and legs are deliberately excluded:
# including them pulls the frame down and shrinks the hands again.
UPPER_BODY = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

_pose = None
_hands = None


def _get_pose():
    global _pose
    if _pose is None:
        import mediapipe as mp

        _pose = mp.solutions.pose.Pose(
            static_image_mode=True,          # sampled frames are not contiguous
            model_complexity=1,
            min_detection_confidence=0.4,
        )
    return _pose


def _get_hands():
    global _hands
    if _hands is None:
        import mediapipe as mp

        _hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.35,
        )
    return _hands


def _fit_aspect(x0: float, y0: float, x1: float, y1: float,
                width: int, height: int) -> Dict[str, float]:
    """Grow the box until its pixel aspect is TARGET_ASPECT, then clamp."""
    # Work in pixels; normalised coordinates hide the source's own aspect.
    px0, px1 = x0 * width, x1 * width
    py0, py1 = y0 * height, y1 * height
    bw, bh = max(px1 - px0, 1.0), max(py1 - py0, 1.0)
    cx, cy = (px0 + px1) / 2, (py0 + py1) / 2

    if bw / bh < TARGET_ASPECT:
        bw = bh * TARGET_ASPECT
    else:
        bh = bw / TARGET_ASPECT

    # Clamping a side would break the aspect, so shrink to whatever fits.
    scale = min(1.0, width / bw, height / bh)
    bw, bh = bw * scale, bh * scale
    cx = min(max(cx, bw / 2), width - bw / 2)
    cy = min(max(cy, bh / 2), height - bh / 2)

    return {
        "x": round((cx - bw / 2) / width, 5),
        "y": round((cy - bh / 2) / height, 5),
        "w": round(bw / width, 5),
        "h": round(bh / height, 5),
    }


def scan_clip(video_path: Path, samples: int = CROP_SAMPLES) -> Optional[Dict[str, float]]:
    """Sample a clip and return the normalised box holding the signer."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    if total <= 0:
        cap.release()
        return None

    pose, hands = _get_pose(), _get_hands()
    xs, ys = [], []          # signing frames only
    fallback_x, fallback_y = [], []
    step = max(1, total // max(samples, 1))

    try:
        for n in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = pose.process(rgb)
            if res.pose_landmarks is None:
                continue

            here_x, here_y = [], []
            for i in UPPER_BODY:
                lm = res.pose_landmarks.landmark[i]
                if lm.visibility < 0.35:
                    continue
                here_x.append(float(lm.x))
                here_y.append(float(lm.y))
            fallback_x += here_x
            fallback_y += here_y

            # Only frames with a visible hand define the framing. Frames where
            # the signer stands with arms down drag the box to the full height
            # of the shot, which zooms the hands back to postage-stamp size.
            hres = hands.process(rgb)
            if not hres.multi_hand_landmarks:
                continue
            for hand in hres.multi_hand_landmarks:
                for lm in hand.landmark:
                    here_x.append(float(lm.x))
                    here_y.append(float(lm.y))
            xs += here_x
            ys += here_y
    finally:
        cap.release()

    if len(xs) < 8:                      # no hand ever found: frame the body
        xs, ys = fallback_x, fallback_y
    if len(xs) < 8:
        return None

    # Trim outliers before padding: one bad frame with a landmark flung off
    # screen would otherwise zoom the whole clip back out to nothing.
    x0, x1 = float(np.percentile(xs, 2)), float(np.percentile(xs, 98))
    y0, y1 = float(np.percentile(ys, 2)), float(np.percentile(ys, 98))

    pad_x = (x1 - x0) * CROP_PAD
    span_y = y1 - y0
    # Extra headroom only at the top: landmark 0 is the nose, so the crown of
    # the head sits outside the raw box and would otherwise be sliced off.
    return _fit_aspect(
        max(0.0, x0 - pad_x), max(0.0, y0 - span_y * (CROP_PAD + 0.10)),
        min(1.0, x1 + pad_x), min(1.0, y1 + span_y * CROP_PAD),
        width, height,
    )


# ----------------------------------------------------------------------
def load_crops(path: Path | None = None) -> Dict[str, Dict[str, float]]:
    path = path or CROP_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_crop(word: str, box: Dict[str, float], path: Path | None = None) -> None:
    path = path or CROP_FILE
    crops = load_crops(path)
    crops[word] = box
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(crops, separators=(",", ":"), sort_keys=True),
                    encoding="utf-8")


def crop_for(word: str, crops: Dict | None = None) -> Optional[Dict[str, float]]:
    return (crops if crops is not None else load_crops()).get(word)
