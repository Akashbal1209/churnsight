"""Bento "Ocean" theme for the Streamlit dashboard: light blue-grey page, white rounded tiles,
pastel KPI tiles, one blue accent and a gentle "rise" animation (tiles fade up one by one).

Pure CSS (no JavaScript), so it works on small hosts and inside Streamlit's sandbox.
Animations replay when a tab is opened and switch off for users who prefer reduced motion.
"""

from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

# ---- palette -------------------------------------------------------------------------------
ACCENT, ACCENT_DARK, ACCENT_SOFT = "#1D4ED8", "#1E40AF", "#BFD3F8"
GOOD, BAD, WARN, ALERT = "#047857", "#E11D48", "#B45309", "#F59E0B"
TEXT, BODY, MUTED, FAINT = "#0F1222", "#333849", "#5E6478", "#8E93A4"
PAGE, TILE, TRACK, LINE = "#EAF0F8", "#FFFFFF", "#F0F1F4", "#E3E5EA"
# pastel fills for KPI tiles (cycle through them in a row)
BLUE_T, MINT_T, LILAC_T, CREAM_T = "#DBEAFE", "#D9F5EF", "#EDE9FE", "#FEF3C7"
TINTS = [BLUE_T, MINT_T, LILAC_T, CREAM_T]

# Older names, kept so existing code keeps working.
PURPLE, TEAL, PINK, AMBER, RED, GREEN = ACCENT, GOOD, "#7C3AED", WARN, BAD, GOOD

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

@keyframes cs-rise { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: none; } }

html, body, [data-testid="stApp"] { font-family: Inter, system-ui, sans-serif; }
[data-testid="stApp"] { background: #EAF0F8; color: #0F1222; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 1200px; padding-top: 1.6rem; }
h1, h2, h3, [data-testid="stHeadingWithActionElements"] { font-family: 'Space Grotesk', Inter, sans-serif !important;
  letter-spacing: -0.02em; color: #0F1222; }
h3 { font-size: 1.15rem !important; font-weight: 600 !important; }

/* ---- rise: tiles and columns fade up one by one (replays when a tab opens) ---- */
.cs-tile, .cs-hero, .cs-action, .cs-insight, [class*="st-key-tile_"], [data-testid="stExpander"],
[data-testid="stHorizontalBlock"] > * { animation: cs-rise .55s cubic-bezier(.2,.7,.2,1) both; }
[data-testid="stHorizontalBlock"] > :nth-child(2) { animation-delay: .07s; }
[data-testid="stHorizontalBlock"] > :nth-child(3) { animation-delay: .14s; }
[data-testid="stHorizontalBlock"] > :nth-child(4) { animation-delay: .21s; }
[data-testid="stHorizontalBlock"] > :nth-child(5) { animation-delay: .28s; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }

/* ---- brand + headline ---- */
.cs-brand { display: flex; align-items: center; gap: 10px; margin-bottom: 18px; }
.cs-brand .logo { width: 34px; height: 34px; border-radius: 12px; background: #0F1222;
  display: flex; align-items: center; justify-content: center; }
.cs-brand b { font-family: 'Space Grotesk', sans-serif; font-size: 19px; letter-spacing: -0.02em; color: #0F1222; }
.cs-brand small { color: #5E6478; font-size: 13px; }
.cs-hero .eyebrow { display: inline-block; font-size: 12px; font-weight: 600; color: #1E40AF; background: #DBEAFE;
  padding: 4px 10px; border-radius: 999px; margin-bottom: 12px; }
.cs-hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: 36px; line-height: 1.1; font-weight: 600;
  letter-spacing: -0.03em; color: #0F1222; margin: 0 0 8px; padding: 0; }
.cs-hero p { color: #4A5063; font-size: 15px; margin: 0; max-width: 720px; }

/* ---- tiles ---- */
.cs-tile { background: #FFFFFF; border-radius: 24px; padding: 18px 20px; height: 100%; min-height: 122px;
  box-sizing: border-box; }
.cs-tile .lbl { color: #5E6478; font-size: 13px; }
.cs-tile .val { font-family: 'Space Grotesk', sans-serif; font-size: 32px; font-weight: 600; letter-spacing: -0.03em;
  line-height: 1.15; margin: 4px 0; color: #0F1222; }
.cs-tile .note { color: #5E6478; font-size: 13px; }
.cs-action { background: #FFFFFF; border-radius: 24px; padding: 18px; display: flex; gap: 12px; height: 100%;
  box-sizing: border-box; }
.cs-action .num { flex: none; width: 28px; height: 28px; border-radius: 50%; background: #DBEAFE; color: #1E40AF;
  font-weight: 700; font-size: 13px; display: flex; align-items: center; justify-content: center; }
.cs-action b { color: #0F1222; font-size: 14px; } .cs-action div { color: #5E6478; font-size: 13px; margin-top: 4px; }
.cs-insight { background: #FEF3C7; color: #78350F; border-radius: 16px; padding: 14px 16px; font-size: 14px; margin: 6px 0 14px; }
.cs-step { display: flex; flex-wrap: wrap; gap: 4px 16px; padding: 10px 0; font-size: 14px; }
.cs-step b { flex: 0 0 240px; color: #0F1222; font-weight: 600; } .cs-step span { flex: 1 1 280px; color: #4A5063; }
.cs-quote { background: #F5F6F8; border-radius: 12px; padding: 10px 14px; color: #0F1222; font-style: italic; margin: 6px 0; }

/* ---- Streamlit widgets ---- */
/* tabs as a white pill bar; active tab is a dark pill (new and older Streamlit tab markup) */
[data-testid="stTabs"] [role="tablist"], [data-baseweb="tab-list"] { gap: 2px; background: #FFFFFF; padding: 5px;
  border-radius: 999px; width: fit-content; max-width: 100%; flex-wrap: wrap; border: 0; margin-bottom: 10px; }
[data-testid="stTab"], [data-baseweb="tab"] { border-radius: 999px !important; padding: 6px 14px !important;
  color: #4A5063 !important; height: auto !important; border: 0 !important; background: transparent; }
[data-testid="stTab"] p, [data-baseweb="tab"] p { font-size: 13px !important; font-weight: 500; }
[data-testid="stTab"][aria-selected="true"], [data-baseweb="tab"][aria-selected="true"] {
  background: #0F1222 !important; color: #FFFFFF !important; }
[data-testid="stTab"][aria-selected="true"] p, [data-baseweb="tab"][aria-selected="true"] p { color: #FFFFFF !important; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"], [data-testid="stTab"] .react-aria-SelectionIndicator { display: none; }
/* white tile around charts and lists: `with T.tile():` */
[class*="st-key-tile_"] { background: #FFFFFF; border-radius: 24px; padding: 18px 20px; }
[data-testid="stExpander"] details { background: #FFFFFF; border: 0; border-radius: 16px; }
[data-testid="stMetric"] { background: #FFFFFF; border-radius: 24px; padding: 12px 16px; }
</style>
"""

_LOGO = ('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" '
         'stroke-linecap="round" aria-hidden="true"><rect x="3" y="3" width="7" height="7" rx="2"/>'
         '<rect x="14" y="3" width="7" height="7" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/>'
         '<rect x="14" y="14" width="7" height="7" rx="2"/></svg>')


def _md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


_tiles = 0


def apply() -> None:
    global _tiles
    _tiles = 0  # stable tile keys on every rerun, so tiles don't remount on each click
    _md(CSS)


def tile():
    """A white rounded tile for charts and lists:  `with T.tile(): st.plotly_chart(...)`."""
    global _tiles
    _tiles += 1
    return st.container(key=f"tile_{_tiles}")


def brand() -> None:
    _md(f'<div class="cs-brand"><span class="logo">{_LOGO}</span><b>ChurnSight</b>'
        '<small>· what customers say, what they do, who to help first</small></div>')


def hero(title: str, subtitle: str, eyebrow: str = "") -> None:
    tag = f'<span class="eyebrow">{html.escape(eyebrow)}</span>' if eyebrow else ""
    _md(f'<div class="cs-hero">{tag}<h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>')
    st.write("")


def card(label: str, value: str, note: str = "", color: str = TEXT, tint: str = TILE) -> None:
    """A tile with a big number. `tint` gives KPI rows their pastel fill (see TINTS)."""
    _md(f'<div class="cs-tile" style="background:{tint}"><div class="lbl">{html.escape(label)}</div>'
        f'<div class="val" style="color:{color}">{html.escape(value)}</div>'
        f'<div class="note">{html.escape(note)}</div></div>')


def insight(text: str) -> None:
    _md(f'<div class="cs-insight">{html.escape(text)}</div>')


def action(num: int, title: str, detail: str) -> None:
    _md(f'<div class="cs-action"><span class="num">{num}</span><span><b>{html.escape(title)}</b>'
        f'<div>{html.escape(detail)}</div></span></div>')


def step(title: str, detail: str) -> None:
    """One row of a list inside a tile: bold title on the left, plain-language detail on the right."""
    _md(f'<div class="cs-step"><b>{html.escape(title)}</b><span>{html.escape(detail)}</span></div>')


def quote(text: str) -> None:
    _md(f'<div class="cs-quote">“{html.escape(text)}”</div>')


def style(fig: go.Figure, height: int = 340) -> go.Figure:
    """Clean light Plotly styling that sits inside a white tile."""
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=BODY, size=13), margin=dict(l=10, r=60, t=30, b=10),
        colorway=[ACCENT, GOOD, "#7C3AED", ALERT, "#0EA5E9", BAD],
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.12, x=0),
        hoverlabel=dict(bgcolor=TILE, bordercolor=LINE, font_color=TEXT),
    )
    fig.update_traces(cliponaxis=False, selector=dict(type="bar"))
    fig.update_xaxes(gridcolor=TRACK, zerolinecolor=LINE, linecolor=LINE, color=MUTED)
    fig.update_yaxes(gridcolor=TRACK, zerolinecolor=LINE, linecolor=LINE, color=MUTED)
    return fig
