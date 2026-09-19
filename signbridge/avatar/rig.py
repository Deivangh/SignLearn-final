"""Drive a rigged glTF character from the extracted skeleton frames.

The procedural figure in :mod:`renderer` is correct by construction — its
geometry *is* the measurement. A rigged character is not: it has its own bone
lengths and rest pose, and it takes *rotations*, not positions. So the job
here is retargeting, and the whole risk of it lives in the hands.

Body bones use per-bone swing. For a bone B whose child sits at a fixed local
offset ``u``:

* in the rest pose the child points along ``dr = Qr * u`` (world),
* we want it to point along ``dt``, taken from two landmarks,
* so ``Qn = delta(dr -> dt) * Qr`` puts it there, and the bone's local
  rotation is ``inverse(Qparent_new) * Qn``.

Bones are walked parents-first so ``Qparent_new`` is always known.

Hands do not use that, and the reason is the whole point of this module.
Swing is a *two-point* fit: it aims a bone but says nothing about roll. Aim
the hand bone from wrist to middle knuckle and the palm's facing is left to
whatever the minimal-arc rotation happens to produce — measured against this
rig and this corpus, that put the palm a mean of 107° out, and 164° out at the
ninetieth percentile. A hand turned back-to-front does not read as a hand with
the wrong roll; it reads as a flat slab, because the curl of the fingers is
hidden behind the knuckles. So the hand bone is fitted as a *frame*: forward
from wrist to middle knuckle, and a palm normal from the knuckle span. Two
independent axes pin all three degrees of freedom, and the residual palm error
drops to a mean of 0.7°.

The fingers then get anatomy imposed on them. A real PIP or DIP joint is a
hinge — it flexes toward the palm and does nothing else. MediaPipe's landmarks
disagree, carrying a mean 15° of sideways deviation at PIP and up to 89°,
which is what makes a retargeted hand look boneless. Each finger therefore
gets one hinge axis, taken from its proximal segment and the palm normal, and
the two outer joints are re-seated into that plane *at their measured angle* —
the curl the tracker saw is kept to within a degree, the sideways splay it
invented is discarded.

Finally, nothing here ever snaps to the rig's rest pose. That pose has exactly
0° of flexion in every finger: it is a flat plank of a hand, and it is a
legitimate handshape, so dropping into it mid-sign reads as a sign rather than
as missing data. Frames with no detected hand ease toward a slack, slightly
curled hand instead, and every bone is slewed with a time constant rather than
set outright, which also absorbs the tracker's frame-to-frame jitter.
"""
from __future__ import annotations

import json
from typing import List

from ..config import STATIC_URL
from ..ui.theme import COMPONENT_CSS

# MediaPipe pose landmark indices (into the 25 we keep).
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP = 23, 24

# Body bones: one entry per driven bone, (bone, source, from, to).
# "pose" reads the body landmarks; virtual points are computed in JS.
# Order is parents-first so a parent's new world rotation is always known.
#
# Hands are not in this list — they are solved as frames, not as aimed bones
# (see the module docstring), and ship as their own table below.
_FINGERS = [
    # name,   MediaPipe landmark chain (joint, then the three phalanx ends)
    ("Thumb", [1, 2, 3, 4]),
    ("Index", [5, 6, 7, 8]),
    ("Middle", [9, 10, 11, 12]),
    ("Ring", [13, 14, 15, 16]),
    ("Pinky", [17, 18, 19, 20]),
]

# A slack hand, in degrees of flexion per joint, used wherever the tracker
# found no hand. Not the rig's rest pose: that one has straight fingers, which
# is a real handshape and so cannot mean "no data".
_SLACK = {
    "Thumb": (6.0, 10.0, 8.0),
    "Index": (14.0, 28.0, 14.0),
    "Middle": (16.0, 32.0, 16.0),
    "Ring": (18.0, 34.0, 17.0),
    "Pinky": (20.0, 34.0, 18.0),
}

# Joint limits in degrees, (min flexion, max flexion). Negative is
# hyperextension, which fingers do a little of and hinges do almost none.
# The knuckle's ceiling is 90 because that is where its measure runs out: it
# is limited through the sine of the angle, which stops separating 100° from
# 80°. A knuckle does not reach 90° anyway, so the ceiling never binds and the
# floor — the one that matters — stays exact.
_LIMITS = {"mcp": (-25.0, 90.0), "pip": (-8.0, 120.0), "dip": (-8.0, 100.0)}


def _bone_specs() -> List[dict]:
    """Body only. Arms stop at the wrist; the hand takes over from there."""
    specs: List[dict] = [
        {"bone": "Spine1", "src": "pose", "a": "hipMid", "b": "shMid"},
        {"bone": "Neck", "src": "pose", "a": "shMid", "b": "headMid"},
    ]
    for side, sh, el, wr in (
        ("Left", L_SH, L_EL, L_WR),
        ("Right", R_SH, R_EL, R_WR),
    ):
        specs += [
            {"bone": f"{side}Arm", "src": "pose", "a": sh, "b": el},
            {"bone": f"{side}ForeArm", "src": "pose", "a": el, "b": wr},
        ]
    return specs


def _hand_specs() -> List[dict]:
    """One entry per hand: the bones to drive and the landmarks to read.

    ``frame`` names the three landmarks that pin the palm — wrist, index
    knuckle, pinky knuckle — and the rig bones that stand at the same three
    places, so the fit is rig-measured rather than assumed.
    """
    hands = []
    for side, src in (("Left", "lh"), ("Right", "rh")):
        fingers = []
        for name, lm in _FINGERS:
            fingers.append({
                "name": name,
                "hinge": name != "Thumb",       # the thumb is not a hinge
                "lm": lm,
                "bones": [f"{side}Hand{name}{k}" for k in (1, 2, 3)],
            })
        hands.append({
            "side": side,
            "src": src,
            "bone": f"{side}Hand",
            "frame": {
                "fwd": [0, 9], "span": [5, 17],
                "fwdBone": f"{side}HandMiddle1",
                "spanBones": [f"{side}HandIndex1", f"{side}HandPinky1"],
            },
            "fingers": fingers,
        })
    return hands


_JS = r"""
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const D = __PAYLOAD__;
const SPECS = __SPECS__;
const HANDS = __HANDS__;
const SLACK = __SLACK__;
const LIMITS = __LIMITS__;
const MODEL = __MODEL__;
const UID = '__UID__';

const wrap   = document.getElementById('screen_' + UID);
const ribbon = document.getElementById('ribbon_' + UID);
const meta   = document.getElementById('meta_' + UID);
const playBtn= document.getElementById('play_' + UID);
const speedSel = document.getElementById('speed_' + UID);
const zoomBtn = document.getElementById('zoom_' + UID);
const viewBtn = document.getElementById('view_' + UID);
const status = document.getElementById('status_' + UID);
const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

let W = wrap.clientWidth || 720;
let H = Math.round(Math.min(520, Math.max(360, W * 0.62)));

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, W / H, 0.01, 200);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(W, H);
renderer.outputColorSpace = THREE.SRGBColorSpace;
wrap.appendChild(renderer.domElement);

scene.add(new THREE.HemisphereLight(0xDCEAF2, 0x1A2730, 1.6));
const keyL = new THREE.DirectionalLight(0xFFF4E2, 2.1); keyL.position.set(1, 2, 3);
scene.add(keyL);
const fillL = new THREE.DirectionalLight(0xBFD8E8, 0.9); fillL.position.set(-2, 1, 1.5);
scene.add(fillL);
const rimL = new THREE.DirectionalLight(0xE0479E, 0.5); rimL.position.set(-1, 1, -2);
scene.add(rimL);

/* ---------------- rig ---------------- */
const bones = {};
let rig = null, ready = false;
const rest = new Map();          // body bone -> { q: rest world quat, u: local child dir }
const hands = [];                // one solved record per hand
const smoothed = new Map();      // bone uuid -> the local quat actually applied

const DEG = Math.PI / 180;
const TAU_BODY = 0.030;          // slew time constants, seconds
const TAU_HAND = 0.050;
const TAU_SLACK = 0.150;
const HOLD_S = 0.14;             // ride out a blink before easing to slack

const _v = new THREE.Vector3(), _v2 = new THREE.Vector3();
const _q = new THREE.Quaternion(), _qp = new THREE.Quaternion();
const _qn = new THREE.Quaternion(), _dr = new THREE.Vector3();
const _n = new THREE.Vector3(), _n2 = new THREE.Vector3();
const _a1 = new THREE.Vector3(), _a2 = new THREE.Vector3();
const _fwd = new THREE.Vector3(), _nor = new THREE.Vector3();
const _ax = new THREE.Vector3(), _pdir = new THREE.Vector3(), _dt = new THREE.Vector3();
const _praw = new THREE.Vector3(), _pfix = new THREE.Vector3();
const _nh = new THREE.Vector3();
const _bx = new THREE.Vector3(), _by = new THREE.Vector3(), _bz = new THREE.Vector3();
const _m = new THREE.Matrix4();
const _qf = new THREE.Quaternion(), _qt = new THREE.Quaternion();

function boneKey(name) { return name.replace(/^.*mixamorig[:_]?/i, ''); }

/* An orthonormal frame from a forward axis and a normal, as a quaternion.
   Gram-Schmidt keeps the forward axis exact and lets the normal set only the
   roll, which is the one thing aiming a bone cannot give us. */
function frameQuat(fwd, nor, out) {
  _bz.copy(fwd);
  if (_bz.lengthSq() < 1e-12) return null;
  _bz.normalize();
  _bx.crossVectors(nor, _bz);
  if (_bx.lengthSq() < 1e-12) return null;   // normal parallel to forward
  _bx.normalize();
  _by.crossVectors(_bz, _bx);
  _m.makeBasis(_bx, _by, _bz);
  return out.setFromRotationMatrix(_m);
}

/* The minimal rotation taking `from` onto `to`, with the antiparallel case
   handled rather than left to produce a NaN. */
function swingTo(from, to, out) {
  const d = from.dot(to);
  if (d < -0.999999) {
    _v2.set(1, 0, 0).cross(from);
    if (_v2.lengthSq() < 1e-8) _v2.set(0, 1, 0).cross(from);
    _v2.normalize();
    return out.set(_v2.x, _v2.y, _v2.z, 0);
  }
  _v2.crossVectors(from, to);
  return out.set(_v2.x, _v2.y, _v2.z, 1 + d).normalize();
}

function lmSub(h, i, j, out) {
  return out.set(h[j][0] - h[i][0], h[j][1] - h[i][1], h[j][2] - h[i][2]);
}

/* The palm's normal, averaged over two independent triangles across the
   knuckles. One cross product alone goes unstable exactly when the hand turns
   edge-on to the camera, which is where a good half of the handshapes live. */
function palmNormal(h, out) {
  lmSub(h, 0, 5, _a1); lmSub(h, 0, 17, _a2);
  _n.crossVectors(_a1, _a2);
  lmSub(h, 0, 9, _a1); lmSub(h, 5, 17, _a2);
  _n2.crossVectors(_a1, _a2);
  if (_n.lengthSq() > 1e-16) _n.normalize(); else _n.set(0, 0, 0);
  if (_n2.lengthSq() > 1e-16) _n2.normalize(); else _n2.set(0, 0, 0);
  out.copy(_n).add(_n2);
  return out.lengthSq() > 1e-12 ? out.normalize() : null;
}

new GLTFLoader().load(MODEL, (gltf) => {
  rig = gltf.scene;
  rig.traverse(o => {
    if (o.isBone) bones[boneKey(o.name)] = o;
    if (o.isMesh) { o.frustumCulled = false; }
  });
  rig.updateMatrixWorld(true);

  // Capture the rest pose once: each driven bone's world rotation and the
  // direction to its child in the bone's own space.
  for (const s of SPECS) {
    const b = bones[s.bone];
    if (!b) continue;
    const child = b.children.find(c => c.isBone);
    if (!child) continue;
    const u = child.position.clone();
    if (u.lengthSq() < 1e-12) continue;
    u.normalize();
    const q = new THREE.Quaternion();
    b.getWorldQuaternion(q);
    rest.set(s.bone, { q, u, local: b.quaternion.clone() });
  }
  for (const H of HANDS) captureHandRest(H);

  scene.add(rig);
  collectMaterials();
  frameCamera();
  ready = true;
  status.textContent = '';
  goTo(0, true);
}, (p) => {
  if (p.total) status.textContent =
    'Loading signer… ' + Math.round(p.loaded / p.total * 100) + '%';
}, (e) => {
  status.textContent = 'Could not load the avatar model.';
  console.error(e);
});

/* Measure the rig's own hand the way the tracker measures a real one, so the
   two frames can be compared directly. Also builds a slack pose, because the
   rest pose is a flat hand and a flat hand is a handshape in its own right. */
function captureHandRest(H) {
  const hand = bones[H.bone];
  const fb = bones[H.frame.fwdBone];
  const ib = bones[H.frame.spanBones[0]], pb = bones[H.frame.spanBones[1]];
  if (!hand || !fb || !ib || !pb) return;

  const hp = new THREE.Vector3(), fp = new THREE.Vector3();
  const ip = new THREE.Vector3(), pp = new THREE.Vector3();
  hand.getWorldPosition(hp); fb.getWorldPosition(fp);
  ib.getWorldPosition(ip); pb.getWorldPosition(pp);

  const fwd = fp.clone().sub(hp);
  const nor = ip.clone().sub(hp).cross(pp.clone().sub(hp));
  if (fwd.lengthSq() < 1e-12 || nor.lengthSq() < 1e-12) return;
  fwd.normalize(); nor.normalize();

  const qFrame = new THREE.Quaternion();
  if (!frameQuat(fwd, nor, qFrame)) return;
  const qWorld = new THREE.Quaternion();
  hand.getWorldQuaternion(qWorld);

  // Flexion curls toward the palm. Which way that is flips between a left and
  // a right hand, in the landmarks and in the rig alike, because the two are
  // mirror images, so one sign serves both frames.
  const sign = H.side === 'Left' ? -1 : 1;
  const rec = {
    src: H.src, bone: hand, local: hand.quaternion.clone(),
    qWorld, qFrameInv: qFrame.clone().invert(), sign,
    fingers: [], miss: 1e9,
  };

  for (const F of H.fingers) {
    const chain = F.bones.map(n => bones[n]);
    if (chain.some(b => !b)) continue;
    const f = { spec: F, bones: chain, rest: [], slack: [] };
    let ok = true;
    for (const b of chain) {
      const child = b.children.find(c => c.isBone);
      if (!child || child.position.lengthSq() < 1e-12) { ok = false; break; }
      const u = child.position.clone().normalize();
      const q = new THREE.Quaternion();
      b.getWorldQuaternion(q);
      f.rest.push({ q, u, local: b.quaternion.clone(),
                    dir: u.clone().applyQuaternion(q) });
    }
    if (!ok) continue;

    const axis = new THREE.Vector3().crossVectors(f.rest[0].dir, nor);
    if (axis.lengthSq() < 1e-12) continue;
    axis.normalize();

    const slack = SLACK[F.name] || [0, 0, 0];
    let cum = 0;
    let qParent = qWorld.clone();
    for (let k = 0; k < 3; k++) {
      cum += slack[k] * DEG * sign;
      const qw = new THREE.Quaternion()
        .setFromAxisAngle(axis, cum).multiply(f.rest[k].q);
      f.slack.push(qParent.clone().invert().multiply(qw));
      qParent = qw;
    }
    rec.fingers.push(f);
  }
  hands.push(rec);
}

/* ---------------- landmark helpers ---------------- */
function mid(p, i, j, out) {
  return out.set((p[i][0]+p[j][0])/2, (p[i][1]+p[j][1])/2, (p[i][2]+p[j][2])/2);
}
function point(f, key, out) {
  const p = f.pose;
  if (key === 'hipMid')  return mid(p, 23, 24, out);
  if (key === 'shMid')   return mid(p, 11, 12, out);
  if (key === 'headMid') return mid(p, 7, 8, out);
  return out.set(p[key][0], p[key][1], p[key][2]);
}

/* Stage a bone's target: written straight in, so the children solved after it
   see a correct parent, and remembered, so the smoothing pass can slew to it
   instead of jumping. */
const staged = [];
function stage(b, qWorldTarget) {
  b.parent.getWorldQuaternion(_qp);
  const local = _qp.invert().multiply(qWorldTarget);
  b.quaternion.copy(local);
  staged.push({ b, local: local.clone() });
}
function stageLocal(b, local) {
  b.quaternion.copy(local);
  staged.push({ b, local: local.clone() });
}

/* ---------------- hand solve ---------------- */
function solveHand(rec, h, dt) {
  if (!h) {
    rec.miss += dt;
    // A blink in the tracker is not a change of handshape: hold what we have
    // for a beat, and only then ease out of it.
    if (rec.miss < HOLD_S) return TAU_HAND;
    stageLocal(rec.bone, rec.local);
    for (const f of rec.fingers)
      for (let k = 0; k < 3; k++) stageLocal(f.bones[k], f.slack[k]);
    return TAU_SLACK;
  }
  rec.miss = 0;

  // The hand as a frame, not as an aimed bone: forward from wrist to middle
  // knuckle, roll from the knuckle span. This is what keeps the palm facing
  // where the signer put it.
  lmSub(h, 0, 9, _fwd);
  const nor = palmNormal(h, _nor);
  if (nor && frameQuat(_fwd, nor, _qf)) {
    _qt.copy(_qf).multiply(rec.qFrameInv).multiply(rec.qWorld);
    stage(rec.bone, _qt);
  }
  // A degenerate palm means the hand is exactly edge-on for this one frame.
  // Leaving the wrist where it was rides through it; re-aiming it would spin
  // the roll to an arbitrary value and read as a flick.

  if (nor) _nh.copy(nor).multiplyScalar(rec.sign);   // normal, palm side

  for (const f of rec.fingers) {
    const lm = f.spec.lm;

    // One hinge axis for the whole finger, from its knuckle segment and the
    // palm. PIP and DIP both turn about it and nothing else, so the finger
    // comes out planar — which is what a finger is.
    lmSub(h, lm[0], lm[1], _praw);
    let hinge = f.spec.hinge && nor && _praw.lengthSq() > 1e-12;
    if (hinge) {
      _praw.normalize();
      _ax.crossVectors(_praw, nor);
      if (_ax.lengthSq() < 1e-12) hinge = false; else _ax.normalize();
    }
    _pfix.copy(_praw);            // the chain as actually built, joint by joint

    for (let k = 0; k < 3; k++) {
      const r = f.rest[k];
      lmSub(h, lm[k], lm[k + 1], _dt);
      if (_dt.lengthSq() < 1e-12) { stageLocal(f.bones[k], r.local); continue; }
      _dt.normalize();

      if (hinge && k === 0) {
        // The knuckle is not a hinge — fingers spread, and that spread is
        // meaning — so only how far back it bends is limited. Splitting the
        // direction into its palmward component and the rest lets the limit
        // apply to one without touching the other. 2.4% of knuckles in this
        // corpus arrive bent further back than a hand can go.
        const lo = Math.sin(LIMITS.mcp[0] * DEG);
        const hiL = Math.min(1, Math.sin(LIMITS.mcp[1] * DEG));
        const c = _dt.dot(_nh);
        const want = Math.max(lo, Math.min(hiL, c));
        if (want !== c) {
          _v2.copy(_dt).addScaledVector(_nh, -c);      // the in-plane part
          if (_v2.lengthSq() > 1e-12) {
            _v2.normalize().multiplyScalar(Math.sqrt(Math.max(0, 1 - want * want)));
            _dt.copy(_v2).addScaledVector(_nh, want);
          }
        }
      }

      if (hinge && k > 0) {
        // Keep the angle the tracker measured between the two raw segments —
        // that curl is the whole content of a handshape — and drop the
        // sideways component, which no finger can do and every tracker
        // invents. Measured here at a mean 15° and up to 89°.
        lmSub(h, lm[k - 1], lm[k], _pdir);
        if (_pdir.lengthSq() > 1e-12) {
          _pdir.normalize();
          let ang = Math.acos(Math.max(-1, Math.min(1, _pdir.dot(_dt))));
          if (_v2.crossVectors(_pdir, _dt).dot(_ax) < 0) ang = -ang;
          ang *= rec.sign;                          // positive is palmward
          const lim = k === 1 ? LIMITS.pip : LIMITS.dip;
          ang = Math.max(lim[0] * DEG, Math.min(lim[1] * DEG, ang));
          // Hang it off the joint we actually built, not off the raw one, so
          // the error does not accumulate down the finger.
          _dt.copy(_pfix).applyQuaternion(
            _q.setFromAxisAngle(_ax, ang * rec.sign)).normalize();
        }
      }
      _pfix.copy(_dt);

      _dr.copy(r.u).applyQuaternion(r.q);
      _qn.copy(swingTo(_dr, _dt, _q)).multiply(r.q);
      stage(f.bones[k], _qn);
    }
  }
  return TAU_HAND;
}

function applyFrame(f, dt) {
  if (!ready || !f || !f.pose) return;
  staged.length = 0;

  for (const s of SPECS) {
    const b = bones[s.bone];
    const r = rest.get(s.bone);
    if (!b || !r) continue;
    const a = point(f, s.a, _v);
    const c = point(f, s.b, _v2);
    _dr.subVectors(c, a);
    if (_dr.lengthSq() < 1e-10) { stageLocal(b, r.local); continue; }
    _dr.normalize();
    _v.copy(r.u).applyQuaternion(r.q);          // rest child direction, world
    _qn.copy(swingTo(_v, _dr, _q)).multiply(r.q);
    stage(b, _qn);
  }

  const bodyCount = staged.length;
  const taus = [];
  for (const rec of hands) {
    const from = staged.length;
    const tau = solveHand(rec, f[rec.src], dt);
    for (let i = from; i < staged.length; i++) taus[i] = tau;
  }

  // Slew, rather than set. Two jobs: it absorbs the tracker's frame-to-frame
  // jitter, measured on this corpus at a mean 8.5° on a fingertip with peaks
  // past 140°, and it turns the handover to the slack pose into a relaxation
  // instead of a snap.
  for (let i = 0; i < staged.length; i++) {
    const st = staged[i];
    const tau = i < bodyCount ? TAU_BODY : (taus[i] || TAU_HAND);
    let prev = smoothed.get(st.b.uuid);
    if (!prev) { prev = st.local.clone(); smoothed.set(st.b.uuid, prev); }
    if (!(dt > 0)) prev.copy(st.local);
    else prev.slerp(st.local, 1 - Math.exp(-dt / tau));
    st.b.quaternion.copy(prev);
  }
  rig.updateMatrixWorld(true);
}

/* ---------------- camera ---------------- */
let az = 0, el = 0.02, drag = null, zoomed = true;
let focusY = 1.5, reach = 1.0;

function frameCamera() {
  const hips = bones['Hips'], top = bones['HeadTop_End'] || bones['Head'];
  if (!hips || !top) return;
  const a = new THREE.Vector3(), b = new THREE.Vector3();
  hips.getWorldPosition(a); top.getWorldPosition(b);
  reach = Math.max(b.y - a.y, 0.2);
  place();
}
function place() {
  // "Close in" frames the signing space: waist to just above the head.
  focusY = zoomed ? (2.084 + reach * 0.62) : (2.084 + reach * 0.35);
  const r = zoomed ? reach * 2.05 : reach * 3.1;
  camera.position.set(
    Math.sin(az) * Math.cos(el) * r,
    focusY + Math.sin(el) * r,
    Math.cos(az) * Math.cos(el) * r
  );
  camera.lookAt(0, focusY, 0);
}

/* Close-ups point a tight camera at each hand.

   The body cannot be hidden by render layers here — the whole character is a
   single skinned mesh, so the torso would fill the frame right behind the
   hand. A clipping plane sitting just behind the hand solves it instead: the
   trunk is cut away and the handshape reads against empty space.

   Aim at the middle knuckle rather than the wrist bone, or half the frame is
   forearm. */
renderer.localClippingEnabled = true;
const insetCam = new THREE.PerspectiveCamera(24, 1, 0.01, 200);
const clipPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
const rigMaterials = [];
const _hp = new THREE.Vector3(), _wp = new THREE.Vector3();

function collectMaterials() {
  rig.traverse(o => {
    if (!o.isMesh) return;
    for (const m of (Array.isArray(o.material) ? o.material : [o.material])) {
      if (m && !rigMaterials.includes(m)) rigMaterials.push(m);
    }
  });
}
function setClipping(on) {
  for (const m of rigMaterials) {
    m.clippingPlanes = on ? [clipPlane] : null;
    m.clipShadows = on;
  }
}

function renderInset(side, x, y, size) {
  const palm = bones[side + 'HandMiddle1'] || bones[side + 'Hand'];
  const wrist = bones[side + 'Hand'];
  if (!palm || !wrist) return;
  palm.getWorldPosition(_hp);
  wrist.getWorldPosition(_wp);

  // Clip behind whichever of wrist/palm sits furthest back, with enough
  // margin that fingers angled away from the camera are not sliced off.
  clipPlane.constant = -(Math.min(_hp.z, _wp.z) - reach * 0.22);
  setClipping(true);

  // Far enough that a spread hand fits the frame with room to spare.
  insetCam.position.set(_hp.x, _hp.y, _hp.z + reach * 0.78);
  insetCam.lookAt(_hp);
  insetCam.aspect = 1; insetCam.updateProjectionMatrix();
  renderer.setViewport(x, y, size, size);
  renderer.setScissor(x, y, size, size);
  renderer.setScissorTest(true);
  renderer.clearDepth();
  renderer.render(scene, insetCam);
  renderer.setScissorTest(false);
  setClipping(false);
}

renderer.domElement.style.cursor = 'grab';
renderer.domElement.addEventListener('pointerdown', e => {
  drag = { x: e.clientX, y: e.clientY, az, el };
  renderer.domElement.setPointerCapture(e.pointerId);
  renderer.domElement.style.cursor = 'grabbing';
});
renderer.domElement.addEventListener('pointermove', e => {
  if (!drag) return;
  az = drag.az - (e.clientX - drag.x) * 0.006;
  el = Math.max(-0.4, Math.min(0.55, drag.el + (e.clientY - drag.y) * 0.004));
  place();
});
['pointerup','pointercancel','pointerleave'].forEach(k =>
  renderer.domElement.addEventListener(k, () => {
    drag = null; renderer.domElement.style.cursor = 'grab';
  }));

/* ---------------- playback ---------------- */
let clips = D.clips.slice();
let idx = 0, clock = 0, playing = false, speed = 1.0, current = null;
const cache = new Map();

async function loadClip(i) {
  const c = clips[i];
  if (!c || !c.pose) return null;
  if (cache.has(c.pose)) return cache.get(c.pose);
  try {
    const r = await fetch(c.pose, { cache: 'force-cache' });
    if (!r.ok) throw new Error(r.status);
    const j = await r.json();
    cache.set(c.pose, j);
    return j;
  } catch (e) { cache.set(c.pose, null); return null; }
}

function paintRibbon() {
  [...ribbon.children].forEach((el, i) => {
    el.classList.toggle('done', i < idx);
    el.classList.toggle('active', i === idx);
    if (i !== idx) el.style.removeProperty('--p');
    if (i === idx) el.scrollIntoView({ block:'nearest', inline:'center',
      behavior: REDUCED ? 'auto' : 'smooth' });
  });
  meta.textContent = `${Math.min(idx + 1, clips.length)} / ${clips.length}`;
}

async function goTo(i, autoplay = true) {
  idx = Math.max(0, Math.min(i, clips.length - 1));
  clock = 0;
  paintRibbon();
  current = await loadClip(idx);
  playing = autoplay && !!current;
  playBtn.textContent = playing ? 'Pause' : 'Play';
  if (idx + 1 < clips.length) loadClip(idx + 1);
}

function lerpPts(a, b, t) {
  if (!a || !b) return a || b;
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = [a[i][0] + (b[i][0]-a[i][0])*t,
              a[i][1] + (b[i][1]-a[i][1])*t,
              a[i][2] + (b[i][2]-a[i][2])*t];
  }
  return out;
}

function step(dt) {
  if (!playing || !current) return;
  clock += dt * speed;
  const F = current.frames;
  const dur = current.duration || F[F.length-1].t;
  if (clock >= dur) {
    if (idx < clips.length - 1) { goTo(idx + 1, true); }
    else {
      playing = false; playBtn.textContent = 'Replay';
      const last = ribbon.children[idx];
      if (last) last.style.setProperty('--p', '100%');
    }
    return;
  }
  let hi = 1;
  while (hi < F.length - 1 && F[hi].t < clock) hi++;
  const lo = hi - 1;
  const span = Math.max(F[hi].t - F[lo].t, 1e-4);
  const t = Math.max(0, Math.min(1, (clock - F[lo].t) / span));
  applyFrame({
    pose: lerpPts(F[lo].pose, F[hi].pose, t),
    lh: lerpPts(F[lo].lh, F[hi].lh, t),
    rh: lerpPts(F[lo].rh, F[hi].rh, t),
  }, dt * speed);
  const el2 = ribbon.children[idx];
  if (el2) el2.style.setProperty('--p', (clock / dur * 100).toFixed(1) + '%');
}

/* ---------------- controls ---------------- */
playBtn.addEventListener('click', () => {
  if (!playing && idx === clips.length - 1 && current && clock >= (current.duration || 0)) {
    goTo(0, true); return;
  }
  playing = !playing;
  playBtn.textContent = playing ? 'Pause' : 'Play';
});
document.getElementById('prev_' + UID).addEventListener('click', () => goTo(idx - 1, true));
document.getElementById('next_' + UID).addEventListener('click', () => goTo(idx + 1, true));
document.getElementById('again_' + UID).addEventListener('click', () => goTo(idx, true));
speedSel.addEventListener('change', () => { speed = parseFloat(speedSel.value); });

let insetOn = true;
zoomBtn.addEventListener('click', () => {
  insetOn = !insetOn;
  zoomBtn.classList.toggle('on', insetOn);
  zoomBtn.setAttribute('aria-pressed', String(insetOn));
});
viewBtn.addEventListener('click', () => {
  zoomed = !zoomed;
  viewBtn.textContent = zoomed ? 'Show full body' : 'Close in';
  place();
});
[...ribbon.children].forEach((el, i) => el.addEventListener('click', () => goTo(i, true)));
window.addEventListener('keydown', e => {
  if (e.key === ' ') { e.preventDefault(); playBtn.click(); }
  if (e.key === 'ArrowRight') goTo(idx + 1, true);
  if (e.key === 'ArrowLeft') goTo(idx - 1, true);
});

new ResizeObserver(() => {
  W = wrap.clientWidth || W;
  H = Math.round(Math.min(520, Math.max(360, W * 0.62)));
  camera.aspect = W / H; camera.updateProjectionMatrix();
  renderer.setSize(W, H);
}).observe(wrap);

let last = performance.now();
(function loop(now) {
  requestAnimationFrame(loop);
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  step(dt);
  renderer.setViewport(0, 0, W, H);
  renderer.setScissorTest(false);
  renderer.render(scene, camera);
  if (insetOn && ready) {
    const s = Math.round(Math.min(168, Math.max(104, W * 0.21)));
    renderInset('Right', 12, 12, s);
    renderInset('Left', W - s - 12, 12, s);
  }
  window.__sbReady = ready;
})(last);
"""

_EXTRA_CSS = """
.sb-stage .sb-screen { position: relative; }
.sb-btn.tog { opacity: .55; }
.sb-btn.tog.on { opacity: 1; }
.sb-status { position: absolute; inset: 0; display: grid; place-items: center;
             color: rgba(233,228,216,.72); font-size: .9rem; pointer-events: none; }
"""


def _chip(gloss: str, tag: str) -> str:
    tag_html = f'<span class="tag">{tag}</span>' if tag else ""
    cls = " spelled" if tag == "spelled" else ""
    return f'<button class="sb-gl{cls}" type="button">{gloss.upper()}{tag_html}</button>'


def build_rig_html(clips: List[dict], uid: str = "0",
                   model_url: str = f"{STATIC_URL}/avatar.glb") -> str:
    """``clips`` = [{"gloss": str, "pose": url|None, "tag": str}, ...]"""
    playable = [c for c in clips if c.get("pose")]
    if not playable:
        return (
            f"<style>{COMPONENT_CSS}</style>"
            '<div class="sb-stage"><div class="sb-empty">'
            "No motion data for the words on this page yet.<br>"
            "Download clips and extract their skeletons, then reload:<br>"
            "<code>python scripts/warm_cache.py --common 200 --crops --poses</code><br>"
            "Until then, switch to <b>Real signer</b> above."
            "</div></div>"
        )

    chips = "".join(_chip(c["gloss"], c.get("tag", "")) for c in playable)
    js = (
        _JS.replace("__PAYLOAD__", json.dumps({"clips": playable}))
        .replace("__SPECS__", json.dumps(_bone_specs()))
        .replace("__HANDS__", json.dumps(_hand_specs()))
        .replace("__SLACK__", json.dumps(_SLACK))
        .replace("__LIMITS__", json.dumps(_LIMITS))
        .replace("__MODEL__", json.dumps(model_url))
        .replace("__UID__", uid)
    )

    return f"""<style>{COMPONENT_CSS}{_EXTRA_CSS}</style>
<div class="sb-stage" id="stage_{uid}">
  <div class="sb-screen" id="screen_{uid}">
    <div class="sb-status" id="status_{uid}">Loading signer&hellip;</div>
  </div>
  <div class="sb-ribbon" id="ribbon_{uid}" role="list" aria-label="Sign sequence">{chips}</div>
  <div class="sb-bar">
    <button class="sb-btn" id="prev_{uid}" title="Previous sign" aria-label="Previous sign">&#8592;</button>
    <button class="sb-btn primary" id="play_{uid}">Play</button>
    <button class="sb-btn" id="next_{uid}" title="Next sign" aria-label="Next sign">&#8594;</button>
    <button class="sb-btn" id="again_{uid}">Repeat sign</button>
    <select class="sb-speed" id="speed_{uid}" aria-label="Playback speed">
      <option value="0.35">0.35&times;</option>
      <option value="0.5">0.5&times;</option>
      <option value="0.75">0.75&times;</option>
      <option value="1" selected>1&times;</option>
      <option value="1.5">1.5&times;</option>
    </select>
    <button class="sb-btn tog on" id="zoom_{uid}" aria-pressed="true"
            title="Show a close-up of each hand">Hand close-ups</button>
    <button class="sb-btn" id="view_{uid}">Show full body</button>
    <span class="sb-meta" id="meta_{uid}"></span>
  </div>
</div>
<script type="importmap">
{{"imports":{{
  "three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
}}}}
</script>
<script type="module">{js}</script>"""
