"""Design tokens.

Direction: the source is paper, the output is signal. Printed text sits on a
warm paper card; everything the app *produces* — glosses, motion, the avatar —
lives on deep petrol ink and is lit in marigold and magenta, the two inks
Indian letterpress reaches for when it wants attention.

Hands are drawn in marigold and the body in cool grey, because in a sign
language the hands are the words and the body is the grammar. The colour
split is the linguistics, not decoration.

Glosses are set in caps monospace with wide tracking. That is the actual
convention in sign linguistics notation, not a stylistic flourish.
"""
from __future__ import annotations

TOKENS = {
    "ink": "#0E1D26",
    "ink_2": "#16303E",
    "ink_3": "#1E4152",
    "paper": "#E9E4D8",
    "paper_dim": "#C9C2B2",
    "signal": "#F0A202",      # marigold — the hands, the active gloss
    "pulse": "#E0479E",       # magenta — motion trail, tail of the gradient
    "jade": "#3DBFA0",        # resolved / found
    "rust": "#D65437",        # unresolved / fingerspelled
    "mist": "#9FB8C4",        # body bones, secondary text
}

FONT_LINK = (
    "https://fonts.googleapis.com/css2?"
    "family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,700;12..96,800&"
    "family=Instrument+Sans:wght@400;500;600&"
    "family=JetBrains+Mono:wght@500;700&"
    "family=Noto+Serif+Devanagari:wght@400;600&display=swap"
)


def _vars() -> str:
    return "\n".join(f"  --{k.replace('_', '-')}: {v};" for k, v in TOKENS.items())


BASE_CSS = f"""
@import url('{FONT_LINK}');

:root {{
{_vars()}
  --display: 'Bricolage Grotesque', system-ui, sans-serif;
  --body: 'Instrument Sans', system-ui, sans-serif;
  --mono: 'JetBrains Mono', ui-monospace, monospace;
  --indic: 'Noto Serif Devanagari', serif;
}}
"""

# Injected into the Streamlit page itself.
APP_CSS = BASE_CSS + """
.stApp {
  background:
    radial-gradient(1100px 620px at 12% -8%, #1B3E4F 0%, transparent 62%),
    radial-gradient(760px 520px at 92% 4%, #2A2140 0%, transparent 58%),
    var(--ink);
}

html, body, [class*="css"] { font-family: var(--body); }

h1, h2, h3 { font-family: var(--display); letter-spacing: -0.02em; }

/* ---- masthead ---- */
.sb-mast { padding: 6px 0 22px; border-bottom: 1px solid rgba(159,184,196,.16); margin-bottom: 26px; }
.sb-mast h1 {
  font-size: clamp(2.1rem, 5.2vw, 3.4rem);
  font-weight: 800; line-height: .95; margin: 0;
  color: var(--paper);
}
.sb-mast h1 em {
  font-style: normal;
  background: linear-gradient(96deg, var(--signal), var(--pulse));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.sb-mast p {
  margin: 12px 0 0; max-width: 56ch; color: var(--mist);
  font-size: 1.02rem; line-height: 1.5;
}
.sb-eyebrow {
  font-family: var(--mono); font-size: .68rem; font-weight: 700;
  letter-spacing: .22em; text-transform: uppercase; color: var(--signal);
  margin-bottom: 14px; display: block;
}

/* ---- the page as paper ---- */
.sb-paper {
  background: var(--paper); color: #22303A;
  border-radius: 3px; padding: 26px 30px;
  box-shadow: 0 22px 48px rgba(0,0,0,.44);
  font-size: .95rem; line-height: 1.72;
  max-height: 340px; overflow-y: auto;
}
.sb-paper .indic { font-family: var(--indic); font-size: 1.05rem; }
.sb-paper mark { background: rgba(240,162,2,.34); padding: 0 2px; border-radius: 2px; }

/* ---- stats ---- */
.sb-stats { display: flex; gap: 30px; flex-wrap: wrap; margin: 18px 0 4px; }
.sb-stat .n {
  font-family: var(--display); font-size: 1.85rem; font-weight: 700;
  color: var(--paper); line-height: 1;
}
.sb-stat .l {
  font-family: var(--mono); font-size: .62rem; letter-spacing: .17em;
  text-transform: uppercase; color: var(--mist); margin-top: 6px;
}

/* ---- gloss chips ---- */
.sb-gloss-row { display: flex; flex-wrap: wrap; gap: 7px; margin: 10px 0 20px; }
.sb-chip {
  font-family: var(--mono); font-size: .72rem; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase;
  padding: 5px 11px; border-radius: 2px;
  border: 1px solid transparent;
}
.sb-chip.ok   { background: rgba(61,191,160,.13); color: var(--jade);   border-color: rgba(61,191,160,.34); }
.sb-chip.near { background: rgba(240,162,2,.13);  color: var(--signal); border-color: rgba(240,162,2,.34); }
.sb-chip.spell{ background: rgba(214,84,55,.12);  color: var(--rust);   border-color: rgba(214,84,55,.34); }
.sb-chip.miss { background: rgba(159,184,196,.08); color: var(--mist);  border-color: rgba(159,184,196,.22);
                text-decoration: line-through; }

.sb-sentence { border-left: 2px solid rgba(159,184,196,.2); padding: 2px 0 2px 16px; margin-bottom: 18px; }
.sb-sentence.key { border-left-color: var(--signal); }
.sb-sentence .src { color: var(--mist); font-size: .88rem; margin-bottom: 8px; line-height: 1.5; }
.sb-sentence .no {
  font-family: var(--mono); font-size: .6rem; letter-spacing: .16em;
  color: rgba(159,184,196,.62); display: block; margin-bottom: 5px;
}

/* ---- streamlit widget polish ---- */
.stButton > button {
  font-family: var(--body); font-weight: 600; border-radius: 3px;
  border: 1px solid rgba(240,162,2,.42); background: rgba(240,162,2,.09);
  color: var(--signal); transition: background .15s ease, transform .1s ease;
}
.stButton > button:hover { background: rgba(240,162,2,.2); color: var(--signal); }
.stButton > button:active { transform: translateY(1px); }

section[data-testid="stSidebar"] { background: rgba(9,20,27,.82); border-right: 1px solid rgba(159,184,196,.13); }
section[data-testid="stSidebar"] h2 { font-size: 1rem; }

[data-testid="stFileUploaderDropzone"] {
  border: 1px dashed rgba(240,162,2,.4); background: rgba(240,162,2,.04); border-radius: 4px;
}

.sb-note {
  font-size: .84rem; color: var(--mist); line-height: 1.55;
  border-left: 2px solid rgba(240,162,2,.4); padding-left: 13px; margin: 14px 0;
}

@media (prefers-reduced-motion: reduce) {
  * { animation-duration: .01ms !important; transition-duration: .01ms !important; }
}
"""

# Shared by both embedded components (video player and avatar stage).
COMPONENT_CSS = BASE_CSS + """
* { box-sizing: border-box; }
body { margin: 0; background: transparent; font-family: var(--body); color: var(--paper); }

.sb-stage {
  background: linear-gradient(168deg, var(--ink-2) 0%, var(--ink) 100%);
  border: 1px solid rgba(159,184,196,.15);
  border-radius: 5px; overflow: hidden;
}

.sb-screen { position: relative; width: 100%; background: #08141B; }
.sb-screen video, .sb-screen canvas { display: block; width: 100%; height: 100%; }

/* ---- signature element: the gloss ribbon ----
   A scrubbable strip of the whole sentence in linguistic gloss notation.
   The marigold-to-magenta fill under the active gloss advances in time with
   the clip, so the ribbon is simultaneously the caption, the progress bar,
   and the transport. */
.sb-ribbon {
  display: flex; gap: 0; overflow-x: auto; overflow-y: hidden;
  background: rgba(8,20,27,.62);
  border-top: 1px solid rgba(159,184,196,.14);
  scrollbar-width: thin; scrollbar-color: rgba(240,162,2,.4) transparent;
}
.sb-ribbon::-webkit-scrollbar { height: 3px; }
.sb-ribbon::-webkit-scrollbar-thumb { background: rgba(240,162,2,.4); }

.sb-gl {
  position: relative; flex: 0 0 auto; cursor: pointer;
  padding: 15px 17px 17px; background: none; border: 0;
  font-family: var(--mono); font-size: .74rem; font-weight: 700;
  letter-spacing: .13em; text-transform: uppercase;
  color: rgba(159,184,196,.5); white-space: nowrap;
  transition: color .2s ease;
}
.sb-gl:hover { color: var(--paper); }
.sb-gl.done { color: rgba(233,228,216,.72); }
.sb-gl.active { color: var(--paper); }
.sb-gl::after {
  content: ''; position: absolute; left: 0; bottom: 0; height: 2px; width: 0%;
  background: linear-gradient(90deg, var(--signal), var(--pulse));
}
.sb-gl.done::after { width: 100%; opacity: .38; }
.sb-gl.active::after { width: var(--p, 0%); }
.sb-gl:focus-visible { outline: 2px solid var(--signal); outline-offset: -3px; }

.sb-gl .tag {
  display: block; font-size: .54rem; letter-spacing: .16em; font-weight: 500;
  margin-top: 5px; color: rgba(159,184,196,.5);
}
.sb-gl.spelled .tag { color: var(--rust); }

/* ---- transport ---- */
.sb-bar {
  display: flex; align-items: center; gap: 9px;
  padding: 11px 14px; border-top: 1px solid rgba(159,184,196,.1);
}
.sb-btn {
  display: inline-flex; align-items: center; justify-content: center;
  height: 32px; min-width: 32px; padding: 0 11px;
  border-radius: 3px; border: 1px solid rgba(159,184,196,.22);
  background: rgba(159,184,196,.06); color: var(--paper);
  font-family: var(--body); font-size: .8rem; font-weight: 600; cursor: pointer;
  transition: background .15s ease, border-color .15s ease;
}
.sb-btn:hover { background: rgba(240,162,2,.16); border-color: rgba(240,162,2,.5); }
.sb-btn.primary { background: var(--signal); color: #0E1D26; border-color: var(--signal); }
.sb-btn.primary:hover { background: #FFB524; }
.sb-btn:focus-visible { outline: 2px solid var(--pulse); outline-offset: 2px; }

.sb-meta {
  margin-left: auto; font-family: var(--mono); font-size: .66rem;
  letter-spacing: .12em; color: var(--mist); text-transform: uppercase;
}
.sb-speed {
  background: rgba(159,184,196,.06); color: var(--paper); font-family: var(--mono);
  font-size: .7rem; border: 1px solid rgba(159,184,196,.22); border-radius: 3px;
  padding: 6px 7px;
}

.sb-empty {
  padding: 46px 24px; text-align: center; color: var(--mist);
  font-size: .9rem; line-height: 1.6;
}
"""
