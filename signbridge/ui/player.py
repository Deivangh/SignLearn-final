"""Sequential dictionary-clip player.

Clips stream from ``static/videos`` rather than being base64-inlined. A page
of forty signs is tens of megabytes; inlining that into the HTML makes the
Streamlit page crawl and re-sends every byte on each rerun.

The gloss ribbon is shared with the avatar stage so the two modes feel like
one instrument with two views, not two different screens.
"""
from __future__ import annotations

import json
from typing import List

from .theme import COMPONENT_CSS

_JS = r"""
const D = __PAYLOAD__;
const v = document.getElementById('vid___UID__');
const ribbon = document.getElementById('ribbon___UID__');
const meta = document.getElementById('meta___UID__');
const playBtn = document.getElementById('play___UID__');
const speedSel = document.getElementById('speed___UID__');
const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

let idx = 0, wanted = true, zoomed = true;

/* Zoom the real signer to fill the screen.

   Every crop box was widened at extraction time until its *pixel* aspect
   matched the stage's, so sizing the video to 1/w by 1/h of the stage and
   sliding it into place lands the box exactly on the frame with no letterbox
   and no distortion. Clips with no box fall back to the whole frame. */
function frame() {
  const c = zoomed ? D.clips[idx].crop : null;
  if (!c) {
    v.style.width = '100%'; v.style.height = '100%';
    v.style.left = '0'; v.style.top = '0';
    v.style.objectFit = 'contain';
    return;
  }
  v.style.width  = (100 / c.w) + '%';
  v.style.height = (100 / c.h) + '%';
  v.style.left   = (-100 * c.x / c.w) + '%';
  v.style.top    = (-100 * c.y / c.h) + '%';
  v.style.objectFit = 'fill';
}

function paint() {
  [...ribbon.children].forEach((el, i) => {
    el.classList.toggle('done', i < idx);
    el.classList.toggle('active', i === idx);
    if (i !== idx) el.style.removeProperty('--p');
    if (i === idx) el.scrollIntoView({ block:'nearest', inline:'center', behavior: REDUCED ? 'auto' : 'smooth' });
  });
  meta.textContent = `${Math.min(idx + 1, D.clips.length)} / ${D.clips.length}`;
}

function load(i, autoplay) {
  idx = Math.max(0, Math.min(i, D.clips.length - 1));
  v.src = D.clips[idx].src;
  v.playbackRate = parseFloat(speedSel.value);
  v.load();
  frame();
  paint();
  if (autoplay) v.play().catch(() => {
    playBtn.textContent = 'Play';   // browser blocked autoplay; wait for a tap
    wanted = false;
  });
}

v.addEventListener('timeupdate', () => {
  const el = ribbon.children[idx];
  if (el && v.duration) el.style.setProperty('--p', (v.currentTime / v.duration * 100).toFixed(1) + '%');
});
v.addEventListener('ended', () => {
  if (idx < D.clips.length - 1) load(idx + 1, true);
  else { playBtn.textContent = 'Replay'; wanted = false;
         const el = ribbon.children[idx]; if (el) el.style.setProperty('--p','100%'); }
});
v.addEventListener('error', () => {
  // A clip that will not load should not stall the whole page.
  const el = ribbon.children[idx];
  if (el) { el.classList.add('done'); el.title = 'Clip unavailable'; }
  if (idx < D.clips.length - 1) load(idx + 1, wanted);
});
v.addEventListener('play', () => { playBtn.textContent = 'Pause'; wanted = true; });
v.addEventListener('pause', () => { if (!v.ended) playBtn.textContent = 'Play'; });

playBtn.addEventListener('click', () => {
  if (v.ended && idx === D.clips.length - 1) { load(0, true); return; }
  if (v.paused) v.play(); else v.pause();
});
document.getElementById('prev___UID__').addEventListener('click', () => load(idx - 1, true));
document.getElementById('next___UID__').addEventListener('click', () => load(idx + 1, true));
document.getElementById('again___UID__').addEventListener('click', () => { v.currentTime = 0; v.play(); });
speedSel.addEventListener('change', () => { v.playbackRate = parseFloat(speedSel.value); });
[...ribbon.children].forEach((el, i) => el.addEventListener('click', () => load(i, true)));

const zoomBtn = document.getElementById('zoom___UID__');
zoomBtn.addEventListener('click', () => {
  zoomed = !zoomed;
  zoomBtn.classList.toggle('on', zoomed);
  zoomBtn.setAttribute('aria-pressed', String(zoomed));
  zoomBtn.textContent = zoomed ? 'Zoomed to signer' : 'Whole frame';
  frame();
});

window.addEventListener('keydown', e => {
  if (e.key === ' ') { e.preventDefault(); playBtn.click(); }
  if (e.key === 'ArrowRight') load(idx + 1, true);
  if (e.key === 'ArrowLeft') load(idx - 1, true);
});

load(0, true);
"""


_EXTRA_CSS = """
/* The stage is the crop window; the video is oversized inside it and slid
   into position, so it has to be clipped and absolutely placed. */
/* Capped so the 4:3 stage plus the ribbon and controls still fit inside the
   fixed component height Streamlit gives us on a wide screen. */
.sb-frame { position: relative; overflow: hidden; aspect-ratio: 4/3;
            width: 100%; max-width: 693px; margin: 0 auto;
            background: #08141B; }
.sb-frame video { position: absolute; display: block; }
.sb-btn.tog { opacity: .55; }
.sb-btn.tog.on { opacity: 1; }
"""


def _chip(gloss: str, tag: str, was: str = "") -> str:
    tag_html = f'<span class="tag">{tag}</span>' if tag else ""
    cls = " spelled" if tag == "spelled" else ""
    # A stand-in sign names the word it is covering for, so the reader is
    # never left thinking the dictionary had their word when it did not.
    title = f' title="Standing in for &quot;{was}&quot;"' if was else ""
    return (f'<button class="sb-gl{cls}" type="button"{title}>'
            f"{gloss.upper()}{tag_html}</button>")


def build_video_html(clips: List[dict], uid: str = "0") -> str:
    """``clips`` = [{"gloss": str, "src": url, "tag": str}, ...]"""
    if not clips:
        return (
            f"<style>{COMPONENT_CSS}</style>"
            '<div class="sb-stage"><div class="sb-empty">'
            "Nothing on this page matched the dictionary yet.<br>"
            "Try a clearer photo, or widen <b>How much to sign</b> in the sidebar."
            "</div></div>"
        )

    chips = "".join(
        _chip(c["gloss"], c.get("tag", ""), c.get("was", "")) for c in clips
    )
    js = _JS.replace("__PAYLOAD__", json.dumps({"clips": clips})).replace("__UID__", uid)

    return f"""<style>{COMPONENT_CSS}{_EXTRA_CSS}</style>
<div class="sb-stage">
  <div class="sb-screen sb-frame">
    <video id="vid_{uid}" playsinline preload="auto" muted></video>
  </div>
  <div class="sb-ribbon" id="ribbon_{uid}" role="list" aria-label="Sign sequence">{chips}</div>
  <div class="sb-bar">
    <button class="sb-btn" id="prev_{uid}" title="Previous sign" aria-label="Previous sign">&#8592;</button>
    <button class="sb-btn primary" id="play_{uid}">Play</button>
    <button class="sb-btn" id="next_{uid}" title="Next sign" aria-label="Next sign">&#8594;</button>
    <button class="sb-btn" id="again_{uid}">Repeat sign</button>
    <select class="sb-speed" id="speed_{uid}" aria-label="Playback speed">
      <option value="0.5">0.5&times;</option>
      <option value="0.75">0.75&times;</option>
      <option value="1" selected>1&times;</option>
      <option value="1.5">1.5&times;</option>
    </select>
    <button class="sb-btn tog on" id="zoom_{uid}" aria-pressed="true"
            title="Fill the screen with the signer instead of the whole shot"
            >Zoomed to signer</button>
    <span class="sb-meta" id="meta_{uid}"></span>
  </div>
</div>
<script>{js}</script>"""
