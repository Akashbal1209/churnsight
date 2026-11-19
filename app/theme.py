"""Aurora glass theme for the Streamlit dashboard: animated colour background + glass cards.

Pure CSS (no JavaScript), so it works on small hosts and inside Streamlit's sandbox.
"""

from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

PURPLE, TEAL, PINK, AMBER, RED, GREEN = "#AFA9EC", "#5DCAA5", "#ED93B1", "#FAC775", "#F09595", "#97C459"
TEXT, MUTED = "#F4F2FF", "#B9B4D9"

CSS = """
<style>
[data-testid="stApp"] { background: #15102B; }
[data-testid="stApp"]::before {
  content: ""; position: fixed; inset: -25%; z-index: 0; pointer-events: none;
  background:
    radial-gradient(circle at 20% 25%, rgba(127,119,221,0.55), transparent 38%),
    radial-gradient(circle at 80% 20%, rgba(29,158,117,0.40), transparent 36%),
    radial-gradient(circle at 60% 85%, rgba(212,83,126,0.38), transparent 40%),
    radial-gradient(circle at 10% 90%, rgba(55,138,221,0.30), transparent 35%);
  filter: blur(30px);
  animation: aurora 26s ease-in-out infinite alternate;
}
@keyframes aurora {
  0%   { transform: translate(0, 0) rotate(0deg) scale(1); }
  33%  { transform: translate(6%, -4%) rotate(8deg) scale(1.08); }
  66%  { transform: translate(-5%, 5%) rotate(-6deg) scale(1.04); }
  100% { transform: translate(3%, 3%) rotate(4deg) scale(1.1); }
}
@media (prefers-reduced-motion: reduce) { [data-testid="stApp"]::before { animation: none; } }
[data-testid="stHeader"], [data-testid="stMain"], [data-testid="stAppViewContainer"] { background: transparent; }
[data-testid="stMainBlockContainer"] { position: relative; z-index: 1; max-width: 1200px; padding-top: 2rem; }

.glass {
  background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18);
  border-radius: 16px; padding: 16px 18px; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  height: 100%; min-height: 122px; box-sizing: border-box;
}
.glass .lbl { color: #B9B4D9; font-size: 13px; }
.glass .val { color: #FFFFFF; font-size: 30px; font-weight: 600; line-height: 1.2; margin: 2px 0; }
.glass .note { color: #CFCBEA; font-size: 13px; }
.hero h1 { color: #FFFFFF; font-size: 34px; font-weight: 700; line-height: 1.2; margin: 0 0 6px; }
.hero p { color: #CFCBEA; font-size: 16px; margin: 0; }
.brand { display:flex; align-items:center; gap:10px; color:#FFFFFF; font-size:18px; font-weight:600; margin-bottom: 18px; }
.brand span.dot { width:28px; height:28px; border-radius:8px; background: linear-gradient(135deg,#7F77DD,#D4537E); display:inline-block; }
.brand small { color:#B9B4D9; font-weight:400; font-size:13px; }
.insight { border-left: 3px solid #FAC775; border-radius: 0; padding: 4px 0 4px 12px; color:#F4F2FF; font-size: 15px; margin: 6px 0 14px; }
.action { display:flex; gap:12px; align-items:flex-start; }
.action .num { flex:none; width:28px; height:28px; border-radius:50%; background:#FAC775; color:#412402; font-weight:700; display:flex; align-items:center; justify-content:center; }
.action b { color:#FFFFFF; } .action div { color:#CFCBEA; font-size: 14px; }
.quote { background: rgba(255,255,255,0.06); border-radius: 10px; padding: 8px 12px; color:#E9E6FF; font-style: italic; margin: 6px 0; }

[data-baseweb="tab-list"] { gap: 6px; background: rgba(255,255,255,0.06); padding: 6px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.12); }
[data-baseweb="tab"] { border-radius: 10px !important; padding: 6px 14px !important; color: #CFCBEA !important; }
[data-baseweb="tab"][aria-selected="true"] { background: rgba(255,255,255,0.16) !important; color: #FFFFFF !important; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none; }
[data-testid="stExpander"] details { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.14); border-radius: 14px; }
[data-testid="stMetric"] { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18); border-radius: 16px; padding: 12px 16px; }
[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(255,255,255,0.06); border-radius: 16px; }
</style>
"""


def apply() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def brand() -> None:
    st.markdown('<div class="brand"><span class="dot"></span>ChurnSight '
                '<small>· what customers say, what they do, who to help first</small></div>', unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
                unsafe_allow_html=True)
    st.write("")


def card(label: str, value: str, note: str = "", color: str = "#FFFFFF") -> None:
    st.markdown(
        f'<div class="glass"><div class="lbl">{html.escape(label)}</div>'
        f'<div class="val" style="color:{color}">{html.escape(value)}</div>'
        f'<div class="note">{html.escape(note)}</div></div>', unsafe_allow_html=True)


def insight(text: str) -> None:
    st.markdown(f'<div class="insight">{html.escape(text)}</div>', unsafe_allow_html=True)


def action(num: int, title: str, detail: str) -> None:
    st.markdown(f'<div class="glass action"><span class="num">{num}</span><span><b>{html.escape(title)}</b>'
                f'<div>{html.escape(detail)}</div></span></div>', unsafe_allow_html=True)


def quote(text: str) -> None:
    st.markdown(f'<div class="quote">“{html.escape(text)}”</div>', unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 340) -> go.Figure:
    """Transparent, light-text Plotly styling that sits on the aurora background."""
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=13), margin=dict(l=10, r=60, t=30, b=10),
        colorway=[PURPLE, TEAL, PINK, AMBER, "#85B7EB", GREEN],
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.12, x=0),
        hoverlabel=dict(bgcolor="#2A2250", font_color=TEXT),
    )
    fig.update_traces(cliponaxis=False, selector=dict(type="bar"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.10)", zerolinecolor="rgba(255,255,255,0.2)", color=MUTED)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.10)", zerolinecolor="rgba(255,255,255,0.2)", color=MUTED)
    return fig
