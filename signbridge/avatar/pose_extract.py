"""Turn a dictionary clip into skeleton motion the avatar can replay.

Why not just play the video? Two reasons. The dictionary clips are filmed
against different backgrounds by different signers at different distances,
so a page of thirty signs looks like thirty jump cuts. And a skeleton is a
few hundred KB of JSON where the video is megabytes, so once a sign is
extracted the app can replay it instantly, offline, at any speed.

Two models run per frame, because neither alone is enough:

* **Pose** gives ``pose_world_landmarks`` — the body in metres. It carries a
  wrist but no handshape, so on its own a sign is an arm waving.
* **Hands** gives ``multi_hand_world_landmarks`` — each hand in real metres,
  wrist at the origin, with usable depth. We translate that onto the body's
  wrist, so the handshape keeps its true 3D form. Image-space hand landmarks
  would not do: a handshape pointing at the camera collapses to a flat smear
  in them, and that is the single thing a reader cannot afford to lose.

Any hand the full-frame pass drops gets a second look in close-up, cropped to
where the body model says the wrist is. It is the cheapest frames-recovered
per second available, and every frame it misses is a frame the avatar has to
show a hand it does not have.

Detected hands are matched to a wrist by image-space distance rather than by
MediaPipe's handedness label: the label assumes a mirrored selfie view, which
these clips are not, and getting it backwards swaps every two-handed sign.

Output format (one file per sign, written to ``static/poses``)::

    {
      "word": "apple",
      "fps": 20,
      "duration": 2.13,
      "scale": 0.31,
      "frames": [
        {"t": 0.0, "pose": [[x,y,z], ...25], "lh": [[x,y,z], ...21] | null,
         "rh": [...] | null},
        ...
      ]
    }

Coordinates are metres, Y-up, origin at the hip midpoint, so the browser can
drop them straight into a Three.js scene with no further fiddling.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ..config import (
    POSE_DECIMALS,
    POSE_DIR,
    POSE_HOLD_FRAMES,
    POSE_MAX_SECONDS,
    POSE_SMOOTH_WINDOW,
    POSE_TARGET_FPS,
)

# Upper body only — legs never carry meaning in a dictionary sign and drop
# a third of the file size.
POSE_KEEP = list(range(25))

# Wrist landmarks in the full pose model.
L_WRIST, R_WRIST = 15, 16

# Hand landmarks in the hand model, used to build a palm frame.
H_WRIST, H_INDEX_MCP, H_MIDDLE_MCP, H_PINKY_MCP = 0, 5, 9, 17

# Pose landmarks around the hand, used to aim the close-up pass.
_HAND_LOOK = {"lh": (15, 17, 19), "rh": (16, 18, 20)}   # wrist, pinky, index

_pose = None
_hands = None
_roi_hands = None


def _get_pose():
    """The body model, on its own.

    This used to be Holistic, for no benefit: its hand output is image-space
    only and was never read here, and in current MediaPipe builds its graph
    fails to parse on Windows, which took the whole extractor down with it.
    Pose is the half that was actually being used.
    """
    global _pose
    if _pose is None:
        import mediapipe as mp

        _pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _pose


def _get_hands():
    """A dedicated 2-hand tracker. Lower thresholds than the body model —
    a hand crossing the torso is worth keeping even at middling confidence,
    because the alternative is the hand vanishing mid-sign."""
    global _hands
    if _hands is None:
        import mediapipe as mp

        _hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )
    return _hands


def _get_roi_hands():
    """A second look, at one hand, in close-up.

    The dictionary clips frame a whole signer, so a hand is a small part of a
    small frame and the full-frame pass drops it more often than it should.
    Where the body model still knows where the wrist is, cropping to it and
    re-running recovers some of those frames. Fresh detection every call —
    there is no track to continue, and the crop moves every frame.
    """
    global _roi_hands
    if _roi_hands is None:
        import mediapipe as mp

        _roi_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.3,
        )
    return _roi_hands


def _round(value: float) -> float:
    return round(float(value), POSE_DECIMALS)


def _smooth(seq: List[Optional[np.ndarray]], window: int) -> List[Optional[np.ndarray]]:
    """Centred moving average over the frames that exist.

    MediaPipe jitters by a few millimetres per frame. At 20fps that reads as
    a permanent tremor, which is exactly the noise band a fingerspelled letter
    lives in. Averaging over a short odd window kills it without visibly
    softening the fast transitions between handshapes.
    """
    if window < 3:
        return seq
    half = window // 2
    out: List[Optional[np.ndarray]] = []
    for i, cur in enumerate(seq):
        if cur is None:
            out.append(None)
            continue
        stack = [
            seq[j]
            for j in range(max(0, i - half), min(len(seq), i + half + 1))
            if seq[j] is not None
        ]
        out.append(np.mean(stack, axis=0) if stack else cur)
    return out


def _fill_gaps(seq: List[Optional[np.ndarray]], max_gap: int) -> List[Optional[np.ndarray]]:
    """Bridge short detection dropouts by interpolating across them.

    A hand that blinks out for three frames and back is a tracker failure, not
    a sign. Left alone it makes the avatar's hand strobe. Longer gaps are left
    as None — those are genuine rest positions, and inventing a hand there
    would be worse than showing none.
    """
    out = list(seq)
    n = len(out)
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        start = i
        while i < n and out[i] is None:
            i += 1
        end = i                                  # first present frame after gap
        if start == 0 or end >= n:               # gap at either end: leave it
            continue
        if end - start > max_gap:
            continue
        a, b = out[start - 1], out[end]
        span = end - start + 1
        for k in range(start, end):
            t = (k - start + 1) / span
            out[k] = a + (b - a) * t
        # index i already sits past the gap
    return out


def _trim_idle(
    frames: List[dict], hands: List[bool]
) -> List[dict]:
    """Drop the lead-in and tail where the signer stands with hands down.

    The dictionary clips top and tail every sign with a second or two of rest.
    Across thirty signs that is a minute of a person standing still, and it
    buries the sign itself. Keep a few frames of run-up so the movement still
    has a starting point.
    """
    idx = [i for i, h in enumerate(hands) if h]
    if not idx:
        return frames
    pad = max(2, POSE_TARGET_FPS // 6)
    lo = max(0, idx[0] - pad)
    # No tail pad: those frames are the signer dropping their hands, and
    # ending on them leaves the avatar frozen with stumps for arms — which
    # reads as the hands having failed rather than the sign having finished.
    hi = idx[-1] + 1
    return frames[lo:hi]


def _match_hands(hand_result, pose_img) -> dict:
    """Assign each detected hand to a body wrist by image-space proximity."""
    out = {"lh": None, "rh": None}
    if hand_result.multi_hand_landmarks is None:
        return out
    if hand_result.multi_hand_world_landmarks is None:
        return out

    targets = {}
    if pose_img is not None:
        lw, rw = pose_img[L_WRIST], pose_img[R_WRIST]
        targets = {"lh": np.array(lw[:2]), "rh": np.array(rw[:2])}
    if not targets:
        return out

    claims = []
    for img_lm, world_lm in zip(
        hand_result.multi_hand_landmarks, hand_result.multi_hand_world_landmarks
    ):
        wrist_xy = np.array([img_lm.landmark[0].x, img_lm.landmark[0].y])
        world = np.array(
            [[lm.x, -lm.y, -lm.z] for lm in world_lm.landmark], dtype=np.float32
        )
        world -= world[0]                        # wrist to origin
        for side, target in targets.items():
            claims.append((float(np.linalg.norm(wrist_xy - target)), side, world))

    # Greedy nearest-first: each hand takes one wrist, each wrist takes one hand.
    claims.sort(key=lambda c: c[0])
    used_world = []
    for dist, side, world in claims:
        if out[side] is not None or dist > 0.28:
            continue
        if any(world is w for w in used_world):
            continue
        out[side] = world
        used_world.append(world)
    return out


def _roi_hand(rgb, pose_img, side: str) -> Optional[np.ndarray]:
    """Look again at one hand, in close-up, after the full-frame pass lost it.

    The crop is squared off and padded rather than stretched to fit, because
    the hand model reads a distorted hand as a differently shaped one.
    """
    if pose_img is None:
        return None
    wrist_i, pinky_i, index_i = _HAND_LOOK[side]
    wrist = pose_img[wrist_i]
    if wrist[2] < 0.3:                       # body model has lost the arm too
        return None

    h, w = rgb.shape[:2]
    wx, wy = wrist[0] * w, wrist[1] * h
    kx = (pose_img[pinky_i][0] + pose_img[index_i][0]) / 2 * w
    ky = (pose_img[pinky_i][1] + pose_img[index_i][1]) / 2 * h
    # Span the wrist-to-knuckles distance with room for fingers, with a floor
    # for the frames where the hand points straight at the camera and that
    # distance collapses to nothing.
    r = max(float(np.hypot(kx - wx, ky - wy)) * 2.6, w * 0.09)
    cx, cy = (wx + kx) / 2, (wy + ky) / 2

    x0, y0 = int(round(cx - r)), int(round(cy - r))
    x1, y1 = int(round(cx + r)), int(round(cy + r))
    px0, py0 = max(0, -x0), max(0, -y0)
    px1, py1 = max(0, x1 - w), max(0, y1 - h)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None

    crop = rgb[y0:y1, x0:x1]
    if px0 or py0 or px1 or py1:
        crop = cv2.copyMakeBorder(crop, py0, py1, px0, px1, cv2.BORDER_REPLICATE)
    crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)

    res = _get_roi_hands().process(np.ascontiguousarray(crop))
    if not res.multi_hand_world_landmarks:
        return None
    lm = res.multi_hand_world_landmarks[0]
    world = np.array([[p.x, -p.y, -p.z] for p in lm.landmark], dtype=np.float32)
    return world - world[0]


def _palm_frame(hand: np.ndarray) -> Optional[np.ndarray]:
    """An orthonormal frame for a hand: where it points, and which way it faces."""
    fwd = hand[H_MIDDLE_MCP] - hand[H_WRIST]
    nor = np.cross(hand[H_INDEX_MCP] - hand[H_WRIST],
                   hand[H_PINKY_MCP] - hand[H_WRIST])
    nf, nn = np.linalg.norm(fwd), np.linalg.norm(nor)
    if nf < 1e-6 or nn < 1e-9:
        return None
    z = fwd / nf
    x = np.cross(nor / nn, z)
    nx = np.linalg.norm(x)
    if nx < 1e-6:
        return None
    x /= nx
    return np.stack([x, np.cross(z, x), z], axis=1)     # columns


def _smooth_hands(
    seq: List[Optional[np.ndarray]], window: int
) -> List[Optional[np.ndarray]]:
    """Average a hand without averaging away its handshape.

    Smoothing raw landmark positions works on a still hand and quietly ruins a
    turning one. The same curled finger pointing two ways across a window
    averages to something straighter than either, so the shape flattens
    precisely during the movement that carries the meaning — and a flat hand
    is a handshape of its own, so the damage reads as a different sign rather
    than as blur.

    So split each hand into where it is pointing and what shape it is in.
    The shape lives in the palm's own frame, where a rotating hand is
    stationary and averaging is safe; the frame is averaged separately and
    pushed back to the nearest true rotation.
    """
    if window < 3:
        return seq
    half = window // 2
    n = len(seq)
    frames = [None if h is None else _palm_frame(h) for h in seq]
    shapes: List[Optional[np.ndarray]] = []
    for h, R in zip(seq, frames):
        shapes.append(None if h is None or R is None else (h - h[H_WRIST]) @ R)

    out: List[Optional[np.ndarray]] = []
    for i, cur in enumerate(seq):
        if cur is None:
            out.append(None)
            continue
        lo, hi = max(0, i - half), min(n, i + half + 1)
        if frames[i] is None:                       # degenerate palm: fall back
            stack = [seq[j] for j in range(lo, hi) if seq[j] is not None]
            out.append(np.mean(stack, axis=0) if stack else cur)
            continue
        sh = [shapes[j] for j in range(lo, hi) if shapes[j] is not None]
        fr = [frames[j] for j in range(lo, hi) if frames[j] is not None]
        wrist = np.mean([seq[j][H_WRIST] for j in range(lo, hi)
                         if seq[j] is not None], axis=0)
        R = np.mean(fr, axis=0)
        u, _, vt = np.linalg.svd(R)                 # nearest rotation
        R = u @ vt
        if np.linalg.det(R) < 0:
            u[:, -1] *= -1
            R = u @ vt
        out.append((np.mean(sh, axis=0) @ R.T + wrist).astype(np.float32))
    return out


def extract_pose(
    video_path: Path,
    word: str,
    out_path: Optional[Path] = None,
    target_fps: int = POSE_TARGET_FPS,
) -> Optional[Path]:
    """Run Pose + Hands over a clip and write normalised skeleton frames."""
    out_path = out_path or (POSE_DIR / f"{word.replace(' ', '_')}.json")
    if out_path.exists() and out_path.stat().st_size > 200:
        return out_path

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / target_fps)))
    max_frames = int(POSE_MAX_SECONDS * src_fps)

    pose_model = _get_pose()
    hands = _get_hands()

    times: List[float] = []
    poses: List[np.ndarray] = []
    raw = {"lh": [], "rh": []}
    frame_no = 0
    body_scale = 0.3

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame_no > max_frames:
                break
            if frame_no % step:
                frame_no += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = pose_model.process(rgb)

            world = result.pose_world_landmarks
            if world is None:
                frame_no += 1
                continue

            pose = np.array(
                [[lm.x, -lm.y, -lm.z] for lm in world.landmark], dtype=np.float32
            )
            hips = (pose[23] + pose[24]) / 2.0
            pose -= hips

            shoulder_span = float(np.linalg.norm(pose[11] - pose[12]))
            if shoulder_span > 1e-3:
                body_scale = shoulder_span

            pose_img = None
            if result.pose_landmarks is not None:
                pose_img = [
                    (lm.x, lm.y, lm.visibility) for lm in result.pose_landmarks.landmark
                ]

            matched = _match_hands(hands.process(rgb), pose_img)
            for side in ("lh", "rh"):
                if matched[side] is None:
                    matched[side] = _roi_hand(rgb, pose_img, side)

            times.append(frame_no / src_fps)
            poses.append(pose)
            # Anchor each metric hand onto its body wrist.
            for side, wrist_idx in (("lh", L_WRIST), ("rh", R_WRIST)):
                hand = matched[side]
                raw[side].append(None if hand is None else hand + pose[wrist_idx])
            frame_no += 1
    finally:
        cap.release()

    if len(poses) < 3:
        return None

    poses = _smooth(poses, POSE_SMOOTH_WINDOW)
    for side in ("lh", "rh"):
        raw[side] = _fill_gaps(raw[side], POSE_HOLD_FRAMES)
        raw[side] = _smooth_hands(raw[side], POSE_SMOOTH_WINDOW)

    frames = []
    has_hand = []
    for i, pose in enumerate(poses):
        lh, rh = raw["lh"][i], raw["rh"][i]
        frames.append(
            {
                "t": _round(times[i]),
                "pose": np.asarray(pose)[POSE_KEEP].round(POSE_DECIMALS).tolist(),
                "lh": None if lh is None else np.asarray(lh).round(POSE_DECIMALS).tolist(),
                "rh": None if rh is None else np.asarray(rh).round(POSE_DECIMALS).tolist(),
            }
        )
        has_hand.append(lh is not None or rh is not None)

    frames = _trim_idle(frames, has_hand)
    if len(frames) < 3:
        return None

    t0 = frames[0]["t"]
    for f in frames:
        f["t"] = _round(f["t"] - t0)

    payload = {
        "word": word,
        "fps": target_fps,
        "duration": frames[-1]["t"],
        "scale": round(body_scale, 4),
        "frames": frames,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return out_path


def pose_path_for(word: str) -> Path:
    return POSE_DIR / f"{word.replace(' ', '_')}.json"


def has_pose(word: str) -> bool:
    p = pose_path_for(word)
    return p.exists() and p.stat().st_size > 200
