"""Build the 3D avatar stage.

The avatar is driven directly by the skeleton frames from ``pose_extract``.
It deliberately does *not* depend on a rigged GLB with Mixamo bone names —
that approach breaks the moment someone swaps the model, it forces every user
to source a rig before the app does anything, and retargeting twenty finger
joints onto a humanoid rig goes subtly wrong in exactly the place a reader
cannot afford it. Instead the figure is built procedurally from the landmarks
themselves: solid, skin-toned, anatomically proportioned, and correct by
construction because the geometry *is* the measurement.

What makes a sign readable is the hand, so the hand gets the detail budget:
a real palm slab, five fingers of tapered phalanges, knuckles that read as
knuckles. Everything else is proportioned to stay out of its way.

Two close-up viewports render the same scene from a camera locked to each
hand. A signer's handshape is small in a full-body frame — this is the
difference between "something happened" and "that was an F handshape".

If you later want a photoreal character, the swap point is ``buildFigure()``
in the JS below: keep the frame data, retarget it onto your rig's bones.
"""
from __future__ import annotations

import json
from typing import List

from ..ui.theme import COMPONENT_CSS

# --- pose landmark topology (indices into the 25 kept landmarks) --------
# [from, to, radius_at_from, radius_at_to]
LIMB_BONES = [
    [11, 13, 0.047, 0.038],   # left upper arm
    [13, 15, 0.038, 0.024],   # left forearm
    [12, 14, 0.047, 0.038],   # right upper arm
    [14, 16, 0.038, 0.024],   # right forearm
]
# Shoulders, elbows, wrists, hips — the joints that need a visible pivot.
LIMB_JOINTS = [
    [11, 0.050], [12, 0.050],
    [13, 0.038], [14, 0.038],
    # The wrist tapers into the hand rather than ending in a knob. At the old
    # 30mm it was a ball wider than the palm it fed, and it buried the first
    # third of the palm slab — the arm read as ending in a fist.
    [15, 0.024], [16, 0.024],
    [23, 0.046], [24, 0.046],
]
# Torso outline, walked as a loop: shoulders across, then hips back.
TORSO_LOOP = [11, 12, 24, 23]

# --- hand topology -----------------------------------------------------
# Digit radii are set against the measured landmark geometry, not against a
# photo of a hand. In the extracted library the phalanges run 20-32mm long and
# neighbouring fingers sit a median 21mm apart (12mm at the tenth percentile,
# i.e. whenever they close). Anatomical 20mm-thick fingers on that skeleton are
# as wide as the gaps between them, so they interpenetrate into a single mass
# and every segment comes out shorter than it is thick — a bead, not a bone.
# Slimming the digits to ~13mm buys back a visible channel of air between them,
# which is the whole difference between reading a handshape and guessing it.
HAND_BONES = [
    [0, 1, 0.0090, 0.0086], [1, 2, 0.0086, 0.0080],
    [2, 3, 0.0080, 0.0070], [3, 4, 0.0070, 0.0052],     # thumb, the thickest digit
    [5, 6, 0.0070, 0.0063], [6, 7, 0.0063, 0.0055], [7, 8, 0.0055, 0.0042],
    [9, 10, 0.0072, 0.0065], [10, 11, 0.0065, 0.0056], [11, 12, 0.0056, 0.0043],
    [13, 14, 0.0067, 0.0060], [14, 15, 0.0060, 0.0052], [15, 16, 0.0052, 0.0041],
    [17, 18, 0.0059, 0.0053], [18, 19, 0.0053, 0.0046], [19, 20, 0.0046, 0.0037],
]
# Knuckle and joint beads, [landmark, radius]. Fingertips get a rounded cap.
# Each bead is a hair wider than the bone it caps — enough to read as a knuckle,
# not so much that consecutive beads merge across a 25mm phalanx.
HAND_JOINTS = [
    [1, 0.0090], [2, 0.0084], [3, 0.0073], [4, 0.0055],
    [5, 0.0074], [6, 0.0066], [7, 0.0057], [8, 0.0045],
    [9, 0.0076], [10, 0.0068], [11, 0.0058], [12, 0.0046],
    [13, 0.0071], [14, 0.0063], [15, 0.0054], [16, 0.0044],
    [17, 0.0063], [18, 0.0056], [19, 0.0048], [20, 0.0040],
]
# The palm slab is extruded from this outline.
PALM_LOOP = [0, 1, 5, 9, 13, 17]

# MediaPipe sizes each hand from its own image crop, frame by frame, so the
# hand both pulses within a clip (median 14% swing, 42% on the worst word) and
# lands at wildly different sizes between words (0.35x to 0.61x of shoulder
# width across the library). Pinning it to one proportion of the signer's
# shoulders removes both. 0.52 is the library's own median, which is also where
# anatomy puts it — so this steadies the hand at normal size rather than
# enlarging it.
HAND_TO_SHOULDER = 0.52

_JS = r"""
const D = __PAYLOAD__;
const T = __TOPO__;
const UID = '__UID__';

const root = document.getElementById('stage_' + UID);
const canvasWrap = document.getElementById('screen_' + UID);
const ribbon = document.getElementById('ribbon_' + UID);
const meta = document.getElementById('meta_' + UID);
const playBtn = document.getElementById('play_' + UID);
const speedSel = document.getElementById('speed_' + UID);
const zoomBtn = document.getElementById('zoom_' + UID);
const viewBtn = document.getElementById('view_' + UID);

const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const C = {
  skin:   0xE2AC81,
  skinLo: 0xC98C60,
  shirt:  0x2E4A5A,
  hair:   0x241A17,
  eye:    0x1A1512,
  trailA: new THREE.Color(0xF0A202),
  trailB: new THREE.Color(0xE0479E),
};

/* ---------------- scene ---------------- */
const scene = new THREE.Scene();

let W = canvasWrap.clientWidth || 720;
let H = Math.round(Math.min(520, Math.max(360, W * 0.62)));

const camera = new THREE.PerspectiveCamera(34, W / H, 0.05, 100);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(W, H);
renderer.shadowMap.enabled = false;
canvasWrap.appendChild(renderer.domElement);

/* Layer 0 is the full scene, 1 is the left hand alone, 2 the right — the
   close-up cameras switch to those. Lights have to live on every layer they
   are meant to illuminate, or an isolated hand renders pitch black. */
const ALL_LAYERS = [0, 1, 2];
function lightAll(l) { ALL_LAYERS.forEach(n => l.layers.enable(n)); scene.add(l); }

lightAll(new THREE.HemisphereLight(0xDCEAF2, 0x1A2730, 1.05));
const keyL = new THREE.DirectionalLight(0xFFF4E2, 1.25);
keyL.position.set(1.1, 1.9, 2.4);
lightAll(keyL);
const fillL = new THREE.DirectionalLight(0xBFD8E8, 0.55);
fillL.position.set(-1.9, 0.6, 1.2);
lightAll(fillL);
const rimL = new THREE.DirectionalLight(0xE0479E, 0.42);
rimL.position.set(-1.2, 1.0, -2.0);
lightAll(rimL);

(function ground() {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(128,128,4,128,128,124);
  g.addColorStop(0, 'rgba(0,0,0,0.46)');
  g.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = g; ctx.fillRect(0,0,256,256);
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(1.7, 1.7),
    new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false })
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = -0.86;
  scene.add(mesh);
})();

/* ---------------- materials ---------------- */
/* Darken each surface as it turns away from the eye, so every solid gets its
   own drawn silhouette.

   This is the fix for the flat hand. Splayed fingers sit at wide angles and
   shade differently from one another, so they read fine. Flatten the hand and
   the four finger axes come within about six degrees of parallel: four
   cylinders, one skin colour, one light direction, no shadow map — identical
   shading, no value break anywhere across them. The geometry has a clear 8mm
   of air in every gap and you still cannot see it, because nothing in the
   image marks where one finger stops and the next starts. A seam at each
   silhouette puts that boundary back.

   Doing it by view angle rather than by shadows costs no geometry, no second
   pass and no shadow map — the fragment already knows which way it faces.
   `normal` and `vViewPosition` are both view-space here, so their dot product
   is 1 head-on and 0 at the silhouette. */
function edgeShade(mat, floorV, width) {
  mat.onBeforeCompile = (sh) => {
    sh.fragmentShader = sh.fragmentShader.replace(
      '#include <dithering_fragment>',
      `#include <dithering_fragment>
	float _facing = abs(dot(normalize(normal), normalize(vViewPosition)));
	gl_FragColor.rgb *= mix(${floorV.toFixed(3)}, 1.0, smoothstep(0.0, ${width.toFixed(3)}, _facing));`
    );
  };
  // Without this, two MeshStandardMaterials with matching parameters share a
  // compiled program and whichever compiled first wins - the injection would
  // silently apply to both materials or to neither.
  mat.customProgramCacheKey = () => 'edge:' + floorV + ':' + width;
  return mat;
}

const matSkin  = new THREE.MeshStandardMaterial({ color: C.skin,   roughness: 0.62, metalness: 0.02 });
const matSkinLo= new THREE.MeshStandardMaterial({ color: C.skinLo, roughness: 0.66, metalness: 0.02 });
const matShirt = new THREE.MeshStandardMaterial({ color: C.shirt,  roughness: 0.82, metalness: 0.02 });

/* Hands get their own materials so the seam can be pushed hard there without
   drawing a heavy outline around the head. The body keeps a light touch of the
   same treatment, which is what stops a forearm swung across the chest from
   melting into the shirt behind it. */
const matHand   = edgeShade(matSkin.clone(),   0.30, 0.62);
const matHandLo = edgeShade(matSkinLo.clone(), 0.30, 0.62);
edgeShade(matSkin,  0.70, 0.34);
edgeShade(matShirt, 0.76, 0.30);
const matHair  = new THREE.MeshStandardMaterial({ color: C.hair,   roughness: 0.78, metalness: 0.03 });
const matEye   = new THREE.MeshStandardMaterial({ color: C.eye,    roughness: 0.30, metalness: 0.05 });

/* ---------------- figure ---------------- */
const figure = new THREE.Group();
figure.position.y = -0.14;
scene.add(figure);

const UP = new THREE.Vector3(0, 1, 0);

/* A tapered segment. Each gets its own geometry so the taper is baked in and
   only the length has to change per frame. */
function makeSegment(rA, rB, mat, seg) {
  const g = new THREE.CylinderGeometry(rB, rA, 1, seg || 12, 1, false);
  const m = new THREE.Mesh(g, mat);
  figure.add(m);
  return m;
}
function makeBead(r, mat, seg) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(r, seg || 14, (seg || 14) - 2), mat);
  figure.add(m);
  return m;
}

const vA = new THREE.Vector3(), vB = new THREE.Vector3(), dir = new THREE.Vector3();
function placeSegment(mesh, a, b) {
  vA.set(a[0], a[1], a[2]); vB.set(b[0], b[1], b[2]);
  dir.subVectors(vB, vA);
  const len = dir.length();
  if (len < 1e-5) { mesh.visible = false; return; }
  mesh.visible = true;
  mesh.position.copy(vA).addScaledVector(dir, 0.5);
  mesh.scale.set(1, len, 1);
  mesh.quaternion.setFromUnitVectors(UP, dir.normalize());
}
function setPos(mesh, p) { mesh.position.set(p[0], p[1], p[2]); mesh.visible = true; }

/* An extruded slab from a closed outline — used for the torso and the palm.
   Rebuilt per frame from live landmarks, so it bends with the body instead of
   being a rigid box strapped to it. */
function makeSlab(n, thickness, mat) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 2 * 3), 3));
  const idx = [];
  for (let i = 1; i < n - 1; i++) idx.push(0, i, i + 1);           // front fan
  for (let i = 1; i < n - 1; i++) idx.push(n, n + i + 1, n + i);   // back fan
  for (let i = 0; i < n; i++) {                                    // rim
    const j = (i + 1) % n;
    idx.push(i, n + i, n + j, i, n + j, j);
  }
  geo.setIndex(idx);
  const m = new THREE.Mesh(geo, mat);
  m.userData = { n, thickness };
  figure.add(m);
  return m;
}

const _o = new THREE.Vector3(), _p = new THREE.Vector3(), _q = new THREE.Vector3();
const _n = new THREE.Vector3(), _e1 = new THREE.Vector3(), _e2 = new THREE.Vector3();
function updateSlab(mesh, pts, expand) {
  const { n, thickness } = mesh.userData;
  const arr = mesh.geometry.attributes.position.array;
  _o.set(0,0,0);
  for (const p of pts) _o.add(_p.set(p[0], p[1], p[2]));
  _o.multiplyScalar(1 / pts.length);
  // Sample the normal from three points spread around the loop. Adjacent
  // vertices on a resampled spline are nearly collinear, and a normal built
  // from those flips at random as the body turns.
  const a = pts[0], b = pts[Math.floor(pts.length / 3)], c = pts[Math.floor(pts.length * 2 / 3)];
  _e1.set(b[0]-a[0], b[1]-a[1], b[2]-a[2]);
  _e2.set(c[0]-a[0], c[1]-a[1], c[2]-a[2]);
  _n.crossVectors(_e1, _e2);
  if (_n.lengthSq() < 1e-12) { mesh.visible = false; return; }
  _n.normalize().multiplyScalar(thickness * 0.5);
  mesh.visible = true;
  for (let i = 0; i < n; i++) {
    _p.set(pts[i][0], pts[i][1], pts[i][2]);
    if (expand) _p.addScaledVector(_q.subVectors(_p, _o).normalize(), expand);
    arr[i*3]   = _p.x + _n.x; arr[i*3+1]   = _p.y + _n.y; arr[i*3+2]   = _p.z + _n.z;
    const k = (n + i) * 3;
    arr[k] = _p.x - _n.x; arr[k+1] = _p.y - _n.y; arr[k+2] = _p.z - _n.z;
  }
  mesh.geometry.attributes.position.needsUpdate = true;
  mesh.geometry.computeVertexNormals();
  mesh.geometry.computeBoundingSphere();
}

/* ---- build the body ---- */
/* Four landmarks make a trapezoid, and a trapezoid reads as a paper cutout.
   A six-point outline — shoulders, waist, hips — splined at low tension gives
   a torso that tapers like a body without ballooning into an egg. */
const TORSO_PTS = 24;
const _tv = Array.from({length: 6}, () => new THREE.Vector3());
function lerp3(a, b, t) {
  return [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t];
}
function torsoOutline(P) {
  // Pull the shoulder corners in a little. Run the shell all the way out to
  // the joint and its rim saws through the upper arm as the arm swings.
  const shL = lerp3(P[11], P[12], 0.10), shR = lerp3(P[12], P[11], 0.10);
  const hipL = P[23], hipR = P[24];
  const ring = [
    lerp3(shL, hipL, -0.04),        // just above the shoulder line
    lerp3(shR, hipR, -0.04),
    lerp3(shR, hipR, 0.52),         // waist, right
    hipR,
    hipL,
    lerp3(shL, hipL, 0.52),         // waist, left
  ];
  let cx = 0, cy = 0, cz = 0;
  for (const p of ring) { cx += p[0]; cy += p[1]; cz += p[2]; }
  cx /= 6; cy /= 6; cz /= 6;
  ring.forEach((p, k) => {
    // A fixed nudge outward, so the shell covers the joint beads without
    // scaling with how far the landmark happens to sit from the centre.
    const dx = p[0]-cx, dy = p[1]-cy, dz = p[2]-cz;
    const len = Math.hypot(dx, dy, dz) || 1;
    _tv[k].set(p[0] + dx/len*0.018, p[1] + dy/len*0.018, p[2] + dz/len*0.018);
  });
  const curve = new THREE.CatmullRomCurve3(_tv, true, 'catmullrom', 0.18);
  return curve.getPoints(TORSO_PTS - 1).slice(0, TORSO_PTS).map(v => [v.x, v.y, v.z]);
}

const parts = {
  torso: makeSlab(TORSO_PTS, 0.17, matShirt),
  limbs: T.limbBones.map(b => makeSegment(b[2], b[3], matSkin)),
  joints: T.limbJoints.map(j => makeBead(j[1], matSkin)),
  neck:  makeSegment(0.048, 0.042, matSkin),
  head:  makeBead(0.108, matSkin, 22),
  hair:  makeBead(0.116, matHair, 22),
  eyes:  [makeBead(0.0135, matEye, 10), makeBead(0.0135, matEye, 10)],
  hands: {},
};
parts.head.scale.set(0.94, 1.12, 1.0);
parts.hair.scale.set(0.96, 1.06, 0.98);

/* Sleeves: the upper arms read as clothed, which stops the figure looking
   like a mannequin and gives the skin of the forearm something to contrast
   against. */
parts.sleeves = [makeSegment(0.062, 0.050, matShirt), makeSegment(0.062, 0.050, matShirt)];

for (const side of ['lh', 'rh']) {
  const h = {
    /* A 30mm slab on a ~55mm-wide palm is a brick, and it swallowed the
       finger bases whole. Thinner, and the knuckles sit proud of it. */
    palm:   makeSlab(T.palmLoop.length, 0.018, matHand),
    bones:  T.handBones.map(b => makeSegment(b[2], b[3], matHand, 12)),
    joints: T.handJoints.map(j => makeBead(j[1], matHandLo, 12)),
    wrist:  makeBead(0.016, matHand, 14),
  };
  const layer = side === 'lh' ? 1 : 2;
  [h.palm, h.wrist, ...h.bones, ...h.joints].forEach(m => m.layers.enable(layer));
  parts.hands[side] = h;
}

function setVisible(on) { figure.visible = on; }

/* ---------------- fingertip trail ----------------
   The hands are the words, so the index finger leaves a mark. It makes the
   *path* of a fast sign legible, which is the hardest thing to read off a
   flat video. */
/* A continuous line, not a spray of points. Sampled sparsely the path breaks
   into confetti that reads as noise over the torso; drawn as one stroke it
   reads as a gesture. */
const TRAIL_N = REDUCED ? 0 : 26;
function makeTrail() {
  if (!TRAIL_N) return null;
  const pos = new Float32Array(TRAIL_N * 3), col = new Float32Array(TRAIL_N * 3);
  const c = new THREE.Color();
  for (let i = 0; i < TRAIL_N; i++) {
    const t = i / (TRAIL_N - 1);
    c.copy(C.trailA).lerp(C.trailB, t).multiplyScalar(1 - t * 0.9);
    col[i*3] = c.r; col[i*3+1] = c.g; col[i*3+2] = c.b;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const line = new THREE.Line(geo, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.75,
    depthWrite: false, blending: THREE.AdditiveBlending
  }));
  line.frustumCulled = false;
  line.visible = false;
  figure.add(line);
  return { line, buf: [] };
}
const trails = { lh: makeTrail(), rh: makeTrail() };
// Off by default: a lingering stroke across the chest competes with the
// handshape it is meant to explain. It earns its place on a fast sign you
// are trying to unpick, which is what the toggle is for.
let trailOn = false;

function pushTrail(tr, p) {
  if (!tr) return;
  tr.buf.unshift([p[0], p[1], p[2]]);
  if (tr.buf.length > TRAIL_N) tr.buf.pop();
  // Until the buffer has a real path in it, collapse every vertex onto the
  // newest point so no stray segment shoots off to the origin.
  const arr = tr.line.geometry.attributes.position.array;
  for (let i = 0; i < TRAIL_N; i++) {
    const s = tr.buf[Math.min(i, tr.buf.length - 1)];
    arr[i*3] = s[0]; arr[i*3+1] = s[1]; arr[i*3+2] = s[2];
  }
  tr.line.geometry.attributes.position.needsUpdate = true;
  tr.line.geometry.computeBoundingSphere();
}
function clearTrails() {
  for (const k of ['lh','rh']) if (trails[k]) {
    trails[k].buf = [];
    trails[k].line.visible = false;
  }
}

/* ---------------- pose application ---------------- */
function lerpPts(a, b, t) {
  if (!a || !b) return a || b;
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = [
      a[i][0] + (b[i][0] - a[i][0]) * t,
      a[i][1] + (b[i][1] - a[i][1]) * t,
      a[i][2] + (b[i][2] - a[i][2]) * t,
    ];
  }
  return out;
}

const handCentre = { lh: null, rh: null };
const handNormal = { lh: null, rh: null };

/* ---------------- hand scale ----------------
   Measured along the bone chain, so the number is a true scale and not a
   reading of how curled the fingers happen to be — a fist and a flat hand
   give the same chain length. That lets us rescale about the wrist to a fixed
   proportion of the signer's shoulders without touching the handshape itself. */
const CHAIN = [[0,9],[9,10],[10,11],[11,12]];
function normaliseHand(pts, shoulder) {
  let chain = 0;
  for (const [a, b] of CHAIN) {
    chain += Math.hypot(pts[b][0]-pts[a][0], pts[b][1]-pts[a][1], pts[b][2]-pts[a][2]);
  }
  if (chain < 1e-4 || shoulder < 1e-4) return pts;
  // Clamped: a badly tracked frame should not get to inflate the hand across
  // the whole frame, and forcing it all the way would be inventing data.
  const k = Math.max(0.65, Math.min(1.75, (T.handToShoulder * shoulder) / chain));
  const w = pts[0];
  const out = new Array(pts.length);
  for (let i = 0; i < pts.length; i++) {
    out[i] = [
      w[0] + (pts[i][0] - w[0]) * k,
      w[1] + (pts[i][1] - w[1]) * k,
      w[2] + (pts[i][2] - w[2]) * k,
    ];
  }
  return out;
}

function applyFrame(f) {
  if (!f || !f.pose) return;
  const P = f.pose;

  updateSlab(parts.torso, torsoOutline(P), 0);
  T.limbBones.forEach((b, i) => placeSegment(parts.limbs[i], P[b[0]], P[b[1]]));
  T.limbJoints.forEach((j, i) => setPos(parts.joints[i], P[j[0]]));

  // Sleeve covers the top ~55% of each upper arm.
  [[11,13,0],[12,14,1]].forEach(([s, e, k]) => {
    const mid = [
      P[s][0] + (P[e][0]-P[s][0]) * 0.55,
      P[s][1] + (P[e][1]-P[s][1]) * 0.55,
      P[s][2] + (P[e][2]-P[s][2]) * 0.55,
    ];
    placeSegment(parts.sleeves[k], P[s], mid);
  });

  const midSh = [(P[11][0]+P[12][0])/2, (P[11][1]+P[12][1])/2, (P[11][2]+P[12][2])/2];
  const ears = P[7] && P[8];
  const headC = ears
    ? [(P[7][0]+P[8][0])/2, (P[7][1]+P[8][1])/2 + 0.012, (P[7][2]+P[8][2])/2]
    : [midSh[0], midSh[1] + 0.25, midSh[2]];
  placeSegment(parts.neck, midSh, [headC[0], headC[1] - 0.07, headC[2]]);
  setPos(parts.head, headC);

  // Face direction: nose relative to the ear midpoint tells us where the
  // signer is looking, which matters — ISL uses gaze and head tilt.
  if (ears && P[0]) {
    const fwd = new THREE.Vector3(P[0][0]-headC[0], P[0][1]-headC[1], P[0][2]-headC[2]).normalize();
    const right = new THREE.Vector3(P[8][0]-P[7][0], P[8][1]-P[7][1], P[8][2]-P[7][2]).normalize();
    const up = new THREE.Vector3().crossVectors(right, fwd).normalize();
    // Ride the eyes on the head's surface. Offsetting each axis independently
    // buries one of them inside the skull as soon as the head turns.
    const _eyeDir = new THREE.Vector3();
    parts.eyes.forEach((eye, i) => {
      const s = i === 0 ? -1 : 1;
      _eyeDir.copy(fwd).addScaledVector(right, s * 0.42).addScaledVector(up, 0.26).normalize();
      eye.position.set(headC[0], headC[1], headC[2]).addScaledVector(_eyeDir, 0.104);
      eye.visible = true;
    });
    parts.hair.position.set(headC[0], headC[1], headC[2]).addScaledVector(fwd, -0.022).addScaledVector(up, 0.022);
    parts.hair.visible = true;
  } else {
    parts.eyes.forEach(e => e.visible = false);
    parts.hair.visible = false;
  }

  const shoulder = Math.hypot(P[11][0]-P[12][0], P[11][1]-P[12][1], P[11][2]-P[12][2]);

  for (const side of ['lh', 'rh']) {
    const raw = f[side];
    const pts = raw && normaliseHand(raw, shoulder);
    const g = parts.hands[side];
    if (!pts) {
      g.palm.visible = false; g.wrist.visible = false;
      g.bones.forEach(b => b.visible = false);
      g.joints.forEach(d => d.visible = false);
      handCentre[side] = null;
      handNormal[side] = null;
      continue;
    }
    updateSlab(g.palm, T.palmLoop.map(i => pts[i]), 0.0025);
    T.handBones.forEach((b, i) => placeSegment(g.bones[i], pts[b[0]], pts[b[1]]));
    T.handJoints.forEach((j, i) => setPos(g.joints[i], pts[j[0]]));
    setPos(g.wrist, pts[0]);
    if (trailOn) pushTrail(trails[side], pts[8]);
    // Centre the close-up on the hand's own centroid. Anchoring it to the
    // wrist and nudging by a fixed 0.14m put the palm 12.4 degrees off axis in
    // a 13-degree half-frame — the hand rode the very top edge of the inset
    // and any handshape reaching upward had its fingertips cropped away.
    let cx = 0, cy = 0, cz = 0;
    for (const q of pts) { cx += q[0]; cy += q[1]; cz += q[2]; }
    handCentre[side] = [cx / pts.length, cy / pts.length, cz / pts.length];

    // Palm normal, from the wrist out to the index and pinky knuckles. The
    // close-up steers by it. Chirality does not matter: which face of the palm
    // we end up on is decided later, against the camera we already have.
    const u = [pts[5][0]-pts[0][0], pts[5][1]-pts[0][1], pts[5][2]-pts[0][2]];
    const v = [pts[17][0]-pts[0][0], pts[17][1]-pts[0][1], pts[17][2]-pts[0][2]];
    const nx = u[1]*v[2] - u[2]*v[1],
          ny = u[2]*v[0] - u[0]*v[2],
          nz = u[0]*v[1] - u[1]*v[0];
    const nl = Math.hypot(nx, ny, nz);
    handNormal[side] = nl < 1e-6 ? null : [nx/nl, ny/nl, nz/nl];
  }
  for (const k of ['lh','rh']) if (trails[k]) {
    trails[k].line.visible = trailOn && !!f[k] && trails[k].buf.length > 3;
  }
}

/* ---------------- playback ---------------- */
let clips = D.clips.slice();
let idx = 0, clock = 0, playing = false, speed = 1.0;
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
  } catch (e) {
    cache.set(c.pose, null);
    return null;
  }
}

function paintRibbon() {
  [...ribbon.children].forEach((el, i) => {
    el.classList.toggle('done', i < idx);
    el.classList.toggle('active', i === idx);
    if (i !== idx) el.style.removeProperty('--p');
    if (i === idx) el.scrollIntoView({ block:'nearest', inline:'center', behavior: REDUCED ? 'auto' : 'smooth' });
  });
  meta.textContent = `${Math.min(idx + 1, clips.length)} / ${clips.length}`;
}

let current = null;

async function goTo(i, autoplay = true) {
  idx = Math.max(0, Math.min(i, clips.length - 1));
  clock = 0;
  clearTrails();
  paintRibbon();
  const data = await loadClip(idx);
  current = data;
  if (!data) setVisible(false);
  playing = autoplay;
  playBtn.textContent = playing ? 'Pause' : 'Play';
  // Prefetch the next sign so playback never stalls between words.
  if (idx + 1 < clips.length) loadClip(idx + 1);
}

function step(dt) {
  if (!playing || !current) return;
  clock += dt * speed;
  const F = current.frames;
  const dur = current.duration || F[F.length - 1].t;

  if (clock >= dur) {
    if (idx < clips.length - 1) { goTo(idx + 1, true); }
    else {
      playing = false; playBtn.textContent = 'Replay';
      const last = ribbon.children[idx];
      if (last) last.style.setProperty('--p','100%');
    }
    return;
  }

  let hi = 1;
  while (hi < F.length - 1 && F[hi].t < clock) hi++;
  const lo = hi - 1;
  const span = Math.max(F[hi].t - F[lo].t, 1e-4);
  const t = Math.max(0, Math.min(1, (clock - F[lo].t) / span));

  setVisible(true);
  applyFrame({
    pose: lerpPts(F[lo].pose, F[hi].pose, t),
    lh: lerpPts(F[lo].lh, F[hi].lh, t),
    rh: lerpPts(F[lo].rh, F[hi].rh, t),
  });

  const el = ribbon.children[idx];
  if (el) el.style.setProperty('--p', (clock / dur * 100).toFixed(1) + '%');
}

/* ---------------- cameras ---------------- */
/* Default framing is the signing space — chest to just above the head. A
   full-body shot wastes half the frame on legs that never say anything. */
let az = 0, el = 0.05, drag = null, zoomed = true;
/* The skeleton spans y=0 at the hips to about y=0.75 at the crown, shifted
   down by figure.position.y. "Close in" frames the signing space — waist to
   above the head, where every sign happens. */
function place() {
  const r = zoomed ? 1.58 : 2.10;
  const target = zoomed ? 0.24 : 0.14;
  camera.position.set(
    Math.sin(az) * Math.cos(el) * r,
    target + Math.sin(el) * r,
    Math.cos(az) * Math.cos(el) * r
  );
  camera.lookAt(0, target, 0);
}
place();

/* The close-up shows the hand and nothing else. Isolating it on its own
   render layer is what makes that possible — otherwise the torso sits right
   behind the hand and fills the little frame with a blue wall. */
const insetCam = new THREE.PerspectiveCamera(26, 1, 0.01, 20);
/* 0.46m at 26 degrees shows a 0.21m tall window. A hand spans about 0.17m, so
   it fills roughly four fifths of the inset with margin for fingers at full
   stretch — close enough to read a handshape, loose enough never to clip one. */
const INSET_DIST = 0.46;
/* Where each close-up watches from. It steers toward the palm rather than
   sitting on the orbit, because the orbit is usually looking at the hand from
   the side: across the library the palm sits a median 64 degrees off the
   camera, and is edge-on past 60 degrees in 57% of all frames. Edge-on, a flat
   hand is an 18mm sliver with its whole handshape hidden behind its own
   thickness — no amount of shading rescues that, only a different angle. */
const insetDir = { lh: new THREE.Vector3(0, 0, 1), rh: new THREE.Vector3(0, 0, 1) };
const _orbit = new THREE.Vector3(), _want = new THREE.Vector3(), _pn = new THREE.Vector3();

function renderInset(side, x, y, size) {
  const c = handCentre[side];
  if (!c) return false;
  insetCam.layers.set(side === 'lh' ? 1 : 2);

  _orbit.set(Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el));
  const n = handNormal[side];
  const d = insetDir[side];
  if (n) {
    _pn.set(n[0], n[1], n[2]);
    // Stay on the face of the palm we are already nearer to. Following the raw
    // normal swings the camera a full half-turn the instant the signer turns
    // their hand over, which is unwatchable and loses the sign in the process.
    if (_pn.dot(d) < 0) _pn.negate();
    // Three quarters of the way round, not all of it: dead-on flattens the
    // handshape into a stencil with no depth to read, and leaving some orbit
    // in means the close-up still answers the angle the reader asked for.
    _want.copy(_orbit).multiplyScalar(0.35).addScaledVector(_pn, 0.65);
    if (_want.lengthSq() < 1e-6) _want.copy(_orbit);
    _want.normalize();
  } else {
    _want.copy(_orbit);
  }
  // Drift toward it over about a fifth of a second. Snapping to the palm every
  // frame would strobe the close-up on exactly the fast signs it exists for.
  d.lerp(_want, 0.09).normalize();
  // Never quite overhead: at the pole lookAt has no stable up vector and the
  // inset rolls on its own axis.
  if (Math.abs(d.y) > 0.86) { d.y = Math.sign(d.y) * 0.86; d.normalize(); }

  insetCam.position.set(
    c[0] + d.x * INSET_DIST, c[1] + d.y * INSET_DIST, c[2] + d.z * INSET_DIST
  );
  insetCam.lookAt(c[0], c[1], c[2]);
  insetCam.aspect = 1; insetCam.updateProjectionMatrix();
  renderer.setViewport(x, y, size, size);
  renderer.setScissor(x, y, size, size);
  renderer.setScissorTest(true);
  // Clear the inset to an opaque panel. Left transparent it composites the
  // hand straight onto whatever the main camera drew behind it, and a hand
  // over a torso over another hand is the opposite of a close-up.
  renderer.setClearColor(0x0D1820, 0.94);
  renderer.clear(true, true, false);
  renderer.render(scene, insetCam);
  renderer.setClearColor(0x000000, 0);
  renderer.setScissorTest(false);
  return true;
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
  el = Math.max(-0.40, Math.min(0.58, drag.el + (e.clientY - drag.y) * 0.004));
  place();
});
['pointerup','pointercancel','pointerleave'].forEach(k =>
  renderer.domElement.addEventListener(k, () => {
    drag = null; renderer.domElement.style.cursor = 'grab';
  })
);

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
  zoomBtn.setAttribute('aria-pressed', String(insetOn));
  zoomBtn.classList.toggle('on', insetOn);
});
const trailBtn = document.getElementById('trail_' + UID);
trailBtn.addEventListener('click', () => {
  trailOn = !trailOn;
  trailBtn.setAttribute('aria-pressed', String(trailOn));
  trailBtn.classList.toggle('on', trailOn);
  if (!trailOn) clearTrails();
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
  W = canvasWrap.clientWidth || W;
  H = Math.round(Math.min(520, Math.max(360, W * 0.62)));
  camera.aspect = W / H; camera.updateProjectionMatrix();
  renderer.setSize(W, H);
}).observe(canvasWrap);

/* ---------------- loop ---------------- */
let last = performance.now();
(function loop(now) {
  requestAnimationFrame(loop);
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  step(dt);

  renderer.setViewport(0, 0, W, H);
  renderer.setScissorTest(false);
  renderer.render(scene, camera);

  if (insetOn) {
    const s = Math.round(Math.min(208, Math.max(132, W * 0.26)));
    const pad = 12;
    renderInset('rh', pad, pad, s);
    renderInset('lh', W - s - pad, pad, s);
  }
})(last);

setVisible(false);
goTo(0, true);
"""

_EXTRA_CSS = """
.sb-stage .sb-screen { position: relative; }
.sb-btn.tog { opacity: .55; }
.sb-btn.tog.on { opacity: 1; }
.sb-hint { font-size: .78rem; opacity: .62; padding: 0 .2rem; }
"""


def _chip(gloss: str, tag: str) -> str:
    tag_html = f'<span class="tag">{tag}</span>' if tag else ""
    cls = " spelled" if tag == "spelled" else ""
    return (
        f'<button class="sb-gl{cls}" type="button">'
        f"{gloss.upper()}{tag_html}</button>"
    )


def build_avatar_html(clips: List[dict], uid: str = "0") -> str:
    """``clips`` = [{"gloss": str, "pose": url|None, "tag": str}, ...]"""
    playable = [c for c in clips if c.get("pose")]
    if not playable:
        return (
            f"<style>{COMPONENT_CSS}</style>"
            '<div class="sb-stage"><div class="sb-empty">'
            "No motion data for the words on this page yet.<br>"
            "Download clips and extract their skeletons, then reload:<br>"
            "<code>python scripts/warm_cache.py --common 200 --poses</code><br>"
            "Until then, switch to <b>Dictionary video</b> above."
            "</div></div>"
        )

    chips = "".join(_chip(c["gloss"], c.get("tag", "")) for c in playable)
    topo = {
        "limbBones": LIMB_BONES,
        "limbJoints": LIMB_JOINTS,
        "torsoLoop": TORSO_LOOP,
        "handBones": HAND_BONES,
        "handJoints": HAND_JOINTS,
        "palmLoop": PALM_LOOP,
        "handToShoulder": HAND_TO_SHOULDER,
    }

    js = (
        _JS.replace("__PAYLOAD__", json.dumps({"clips": playable}))
        .replace("__TOPO__", json.dumps(topo))
        .replace("__UID__", uid)
    )

    return f"""<style>{COMPONENT_CSS}{_EXTRA_CSS}</style>
<div class="sb-stage" id="stage_{uid}">
  <div class="sb-screen" id="screen_{uid}"></div>
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
    <button class="sb-btn tog" id="trail_{uid}" aria-pressed="false"
            title="Draw the path each index finger travels">Motion trail</button>
    <button class="sb-btn" id="view_{uid}">Show full body</button>
    <span class="sb-meta" id="meta_{uid}"></span>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.152.2/build/three.min.js"></script>
<script>{js}</script>"""
