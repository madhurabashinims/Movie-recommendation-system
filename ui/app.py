# ui/app.py  — CineMatch  |  Netflix-dark streaming UI
import sys, os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.loaders        import load_all
from backend.recommender    import recommend_movies_multimodal, get_weight_explanation, get_sentiment_weights
from backend.predictors     import predict_user_like_movie
from backend.explainability import generate_llm_explanation
from backend.chatbot        import chatbot_recommendation
from backend.ultra_filter   import ultra_filter_recommend, generate_ultra_explanation, GENRE_ALIASES

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS — Netflix / Prime dark streaming aesthetic
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=DM+Serif+Display&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:         #0f172a;
    --bg2:        #131c30;
    --bg3:        #1e293b;
    --bg4:        #263147;
    --accent:     #38bdf8;
    --accent2:    #0ea5e9;
    --accent3:    #7dd3fc;
    --green:      #4ade80;
    --teal:       #2dd4bf;
    --yellow:     #facc15;
    --red:        #f87171;
    --txt:        #f1f5f9;
    --txt2:       #94a3b8;
    --txt3:       #64748b;
    --border:     rgba(255,255,255,0.08);
    --border2:    rgba(56,189,248,0.25);
    --card-bg:    #1e293b;
    --card-hover: #263147;
    --shadow:     0 8px 32px rgba(0,0,0,0.5);
    --shadow-accent: 0 0 24px rgba(56,189,248,0.15);
    --radius:     14px;
    --radius-sm:  8px;
    --font:       'Plus Jakarta Sans', sans-serif;
    --font-serif: 'DM Serif Display', serif;
    font-size:    16px;
}

html, body { font-size: 16px !important; }

/* Streamlit chrome overrides */
.stApp { background: var(--bg) !important; color: var(--txt) !important; font-family: var(--font) !important; font-size: 16px !important; }
.stApp > header { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
/* ── Full-width layout: zero all container padding ── */
.block-container,
.main .block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stAppViewBlockContainer"] {
    padding: 0 !important;
    max-width: 100% !important;
    width: 100% !important;
}
/* Ensure stApp itself doesn't clip content */
.stApp {
    overflow-x: hidden !important;
}
/* Streamlit v1.3x main content wrapper */
section[data-testid="stMain"] {
    padding: 0 !important;
    width: 100% !important;
}
[data-testid="stVerticalBlock"] {
    width: 100% !important;
    min-width: 0 !important;
}
/* Prevent horizontal scroll from hero breaking out */
.cm-hero {
    max-width: 100vw !important;
    box-sizing: border-box !important;
}
.stDeployButton { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }
footer { display: none !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }

/* Make all Streamlit text inherit our dark theme */
p, span, div, label, h1, h2, h3 { color: var(--txt) !important; font-family: var(--font) !important; }

/* Base readable font sizes */
p { font-size: 15px !important; line-height: 1.7 !important; }
label { font-size: 14px !important; }
.stTextInput input, .stTextArea textarea { font-size: 15px !important; }
.stSelectbox div[data-testid="stMarkdownContainer"] p { font-size: 15px !important; }

/* ── Inputs & selects ── */
.stTextInput input, .stTextArea textarea, .stSelectbox select {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--txt) !important;
    font-family: var(--font) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: var(--txt3) !important;
}

/* ── Selectbox ── */
div[data-testid="stSelectbox"] > div > div {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--txt) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #0f172a !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font) !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 14px rgba(56,189,248,0.25) !important;
}
.stButton > button:hover {
    background: var(--accent3) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(56,189,248,0.35) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #131c30 !important;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
    padding: 0 2rem !important;
    gap: 0 !important;
    width: 100% !important;
    box-sizing: border-box !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--txt2) !important;
    font-family: var(--font) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.03em !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.85rem 1.4rem !important;
    transition: all 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--txt) !important; }
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] {
    background: #0f172a !important;
    padding: 0 !important;
    width: 100% !important;
    box-sizing: border-box !important;
}
[data-testid="stTabsTabPanel"] {
    padding: 0 !important;
    background: #0f172a !important;
    width: 100% !important;
    box-sizing: border-box !important;
    overflow-x: hidden !important;
}
[data-testid="stTabsTabPanel"] > div {
    padding: 0 !important;
    width: 100% !important;
}
/* Fix the inner stVerticalBlock inside tabs */
[data-testid="stTabsTabPanel"] [data-testid="stVerticalBlockBorderWrapper"] {
    padding: 0 !important;
    border: none !important;
}

/* Content inside tab that is NOT a hero gets its own padding */
.tab-content {
    padding: 1.8rem 2.5rem 2.5rem;
    box-sizing: border-box;
    width: 100%;
    overflow-x: hidden;
    max-width: 1600px;        /* cap on ultra-wide monitors */
    margin: 0 auto;           /* center on ultra-wide */
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--txt2) !important;
    font-family: var(--font) !important;
}
.streamlit-expanderContent {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
}
.stSlider div[data-testid="stThumbValue"] { color: var(--accent) !important; }

/* ── Multiselect ── */
.stMultiSelect [data-baseweb="select"] > div {
    background: var(--bg3) !important;
    border-color: var(--border) !important;
}
.stMultiSelect span[data-baseweb="tag"] {
    background: rgba(56,189,248,0.15) !important;
    color: var(--accent) !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}
[data-testid="metric-container"] label { color: var(--txt2) !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: var(--font) !important;
    font-weight: 800 !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stChatMessage"] * { color: var(--txt) !important; }
[data-testid="stChatMessage"][data-message-author-role="user"] {
    background: rgba(56,189,248,0.08) !important;
    border-color: var(--border2) !important;
}
[data-testid="stChatMessage"][data-message-author-role="user"] * {
    color: var(--txt) !important;
}
[data-testid="stChatMessage"][data-message-author-role="assistant"] * {
    color: var(--txt) !important;
}
.stChatFloatingInputContainer,
[data-testid="stChatFloatingInputContainer"] {
    background: var(--bg2) !important;
    border-top: 1px solid var(--border) !important;
}
[data-testid="stChatInputTextArea"] {
    background: var(--bg3) !important;
    color: var(--txt) !important;
    border-color: var(--border) !important;
}
[data-testid="stChatInputSubmitButton"] button { background: var(--accent) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg3) !important;
    border: 1px dashed var(--border2) !important;
    border-radius: var(--radius) !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Fix Streamlit expander — hide Material Icon bleed-through ── */
/* The "_arr" / "arrow_right" text is a Material Symbols icon char.
   It bleeds through when the icon font loads before CSS.
   We suppress it by zeroing font-size on the icon span only. */
[data-testid="stExpander"] summary span:first-child {
    font-size: 0 !important;
    line-height: 0 !important;
    width: 0 !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    font-size: 14px !important;
    color: #94a3b8 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    padding: 10px 14px !important;
}
[data-testid="stExpander"] summary:hover { color: #f1f5f9 !important; }
/* Also target the Streamlit v1.32+ expander structure */
button[data-testid="stBaseButton-header"] {
    font-size: 14px !important;
    color: #94a3b8 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
}
button[data-testid="stBaseButton-header"] p {
    font-size: 14px !important;
    color: #94a3b8 !important;
}
/* Nuclear: hide any span that contains only a single icon character */
[data-testid="stExpander"] .streamlit-expanderHeader span[data-testid="stExpanderToggleIcon"],
[data-testid="stExpander"] summary > span:not([class]) {
    display: none !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   CUSTOM COMPONENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

/* ── Navbar ── */
.cm-nav {
    position: sticky; top: 0; z-index: 999;
    background: rgba(15,23,42,0.92);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(255,255,255,0.08);
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 2.5rem; height: 64px;
    width: 100%;
    box-sizing: border-box;
    left: 0; right: 0;
}
.cm-logo {
    display: flex; align-items: center; gap: 10px;
    font-family: var(--font-serif) !important;
    font-size: 1.5rem; color: var(--accent) !important;
    letter-spacing: -0.02em; white-space: nowrap;
}
.cm-logo-dot { color: var(--txt) !important; }
.cm-logo svg { width: 28px; height: 28px; fill: var(--accent); }
.cm-nav-pills { display: flex; gap: 6px; }
.cm-pill {
    background: rgba(56,189,248,0.10);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 20px; padding: 3px 12px;
    font-size: 0.70rem; font-weight: 700;
    color: var(--accent) !important;
    letter-spacing: 0.07em; text-transform: uppercase;
}

/* ── Hero section ── */
.cm-hero {
    background: linear-gradient(160deg, #080f1f 0%, #0d1a35 35%, #0a172e 65%, #080f1f 100%);
    padding: 4rem 2.5rem 3rem;
    border-bottom: 1px solid rgba(56,189,248,0.14);
    text-align: center; position: relative; overflow: hidden;
    width: 100%;
    /* Netflix-style bottom fade */
    box-shadow:
        0 1px 0 rgba(56,189,248,0.10),
        inset 0 -60px 80px rgba(15,23,42,0.5);
}
.cm-hero::before {
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 50% -20%, rgba(56,189,248,0.13) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 15% 60%, rgba(14,165,233,0.06) 0%, transparent 50%),
        radial-gradient(ellipse 50% 40% at 85% 60%, rgba(125,211,252,0.05) 0%, transparent 50%);
    pointer-events: none;
}
/* glowing top-line accent */
.cm-hero::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg,
        transparent 0%,
        rgba(56,189,248,0.0) 10%,
        rgba(56,189,248,0.6) 50%,
        rgba(56,189,248,0.0) 90%,
        transparent 100%);
}
.cm-hero-eyebrow {
    font-size: 11px; font-weight: 800;
    text-transform: uppercase; letter-spacing: 0.22em;
    color: #38bdf8 !important;
    margin-bottom: 1rem;
    display: inline-flex; align-items: center; gap: 10px;
}
.cm-hero-eyebrow::before, .cm-hero-eyebrow::after {
    content: '';
    display: inline-block; width: 32px; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.7));
    vertical-align: middle;
}
.cm-hero-eyebrow::after {
    background: linear-gradient(90deg, rgba(56,189,248,0.7), transparent);
}
.cm-hero-title {
    font-family: var(--font-serif) !important;
    font-size: clamp(2.2rem, 5vw, 3.4rem);
    color: #f8fafc !important;
    line-height: 1.06; margin-bottom: 0.8rem;
    text-shadow: 0 4px 40px rgba(0,0,0,0.5);
    letter-spacing: -0.01em;
}
.cm-hero-title span {
    color: #38bdf8 !important;
    text-shadow: 0 0 60px rgba(56,189,248,0.35);
}
.cm-hero-sub {
    font-size: 16px; color: #94a3b8 !important;
    margin: 0 auto 1.6rem; font-weight: 400;
    max-width: 600px; line-height: 1.75;
}
.cm-hero-pills {
    display: flex; flex-wrap: wrap; gap: 8px;
    justify-content: center; margin-top: 0.8rem;
}
.cm-hero-pill {
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.22);
    border-radius: 20px; padding: 6px 16px;
    font-size: 12px; font-weight: 700;
    color: #7dd3fc !important;
    letter-spacing: 0.04em;
    white-space: nowrap;
    transition: all 0.2s;
    text-transform: uppercase;
}
.cm-hero-pill:hover {
    background: rgba(56,189,248,0.16);
    border-color: rgba(56,189,248,0.45);
    color: #e0f2fe !important;
}
.cm-hero-input-wrap {
    max-width: 680px; margin: 0 auto;
    position: relative;
}
.cm-hero-input-wrap .stTextInput input {
    font-size: 1rem !important;
    padding: 0.9rem 1.2rem !important;
    height: auto !important;
    background: rgba(30,41,59,0.85) !important;
    border: 1px solid rgba(56,189,248,0.3) !important;
    border-radius: 12px !important;
}
.cm-hero-hint {
    font-size: 13px; color: var(--txt3) !important;
    margin-top: 0.6rem;
}
.cm-hero-hint em { color: #7dd3fc !important; font-style: normal; }

/* ── Page section header ── */
.cm-section-title {
    font-family: var(--font) !important;
    font-size: 22px; font-weight: 800;
    color: var(--txt) !important;
    letter-spacing: -0.03em; margin-bottom: 6px;
}
.cm-section-sub {
    font-size: 14px; color: var(--txt2) !important;
    margin-bottom: 1.6rem; line-height: 1.6;
}

/* ── Weight bar ── */
.wbar {
    display: flex; align-items: center; gap: 10px;
    background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 8px;
    padding: 10px 16px; margin-bottom: 1.2rem;
    font-size: 14px; color: #94a3b8 !important;
}
.wbar b { color: var(--accent) !important; font-weight: 700; }
.wbar-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent); flex-shrink: 0;
    box-shadow: 0 0 8px var(--accent);
}

/* ── Query understanding panel ── */
.qup {
    background: rgba(56,189,248,0.05);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 10px;
    padding: 0.9rem 1.2rem; margin-bottom: 1.4rem;
    overflow: hidden;
}
.qup-label {
    font-size: 11px; font-weight: 800;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: #38bdf8 !important; margin-bottom: 8px;
}
.qup-tags { display: flex; flex-wrap: wrap; gap: 6px; }
.qup-tag {
    background: rgba(56,189,248,0.10);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 20px; padding: 3px 12px;
    font-size: 13px; font-weight: 600;
    color: #7dd3fc !important;
}
.qup-tag.neg { background: rgba(248,113,113,0.12); border-color: rgba(248,113,113,0.3); color: var(--red) !important; }
.qup-tag.era { background: rgba(250,204,21,0.12); border-color: rgba(250,204,21,0.3); color: var(--yellow) !important; }
.qup-tag.excl { background: rgba(251,146,60,0.12); border-color: rgba(251,146,60,0.3); color: #fb923c !important; }

/* ── Movie GRID ── */
.movie-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
    gap: 1.2rem;
    margin-top: 0.5rem;
    box-sizing: border-box;
    width: 100%;
}
@media (max-width: 900px) {
    .movie-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 540px) {
    .movie-grid { grid-template-columns: 1fr; }
}

/* ── Movie Card (grid) ── */
.mcard-grid {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    transition: transform 0.25s, box-shadow 0.25s, border-color 0.25s;
    cursor: pointer; position: relative;
    animation: fadeUp 0.4s ease both;
}
.mcard-grid:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 20px 48px rgba(0,0,0,0.6), 0 0 0 1px rgba(56,189,248,0.3);
    border-color: rgba(56,189,248,0.35);
    z-index: 10;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.mcard-grid-poster {
    position: relative; width: 100%;
    padding-top: 150%; /* 2:3 ratio */
    background: var(--bg3); overflow: hidden;
}
.mcard-grid-poster img {
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    object-fit: cover;
    transition: transform 0.35s, filter 0.35s;
}
.mcard-grid:hover .mcard-grid-poster img {
    transform: scale(1.05);
    filter: brightness(0.6);
}
.mcard-grid-poster .no-poster {
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 2.5rem; color: var(--txt3);
}

/* Score badge on poster */
.score-badge {
    position: absolute; top: 8px; right: 8px;
    font-size: 12px; font-weight: 800;
    padding: 3px 9px; border-radius: 20px;
    letter-spacing: 0.03em; z-index: 2;
}
.score-green  { background: rgba(74,222,128,0.9);  color: #052e16; }
.score-teal   { background: rgba(45,212,191,0.9);  color: #042f2e; }
.score-yellow { background: rgba(250,204,21,0.9);  color: #422006; }
.score-gray   { background: rgba(100,116,139,0.85); color: #f1f5f9; }

/* Hover overlay */
.mcard-overlay {
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 8px;
    opacity: 0; transition: opacity 0.25s;
}
.mcard-grid:hover .mcard-overlay { opacity: 1; }
.overlay-btn {
    background: rgba(56,189,248,0.92);
    color: #0f172a !important;
    border: none; border-radius: 6px;
    padding: 6px 16px; font-family: var(--font);
    font-size: 0.75rem; font-weight: 700; cursor: pointer;
    width: 140px; text-align: center;
    transition: background 0.15s;
}
.overlay-btn:hover { background: white; }
.overlay-btn.secondary {
    background: rgba(255,255,255,0.15);
    color: white !important;
    border: 1px solid rgba(255,255,255,0.3);
}

/* Card body */
.mcard-grid-body { padding: 0.8rem; }
.mcard-grid-title {
    font-weight: 700; font-size: 15px;
    color: var(--txt) !important; margin-bottom: 5px;
    white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; line-height: 1.3;
}
.mcard-grid-genres { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px; }
.gchip-dark {
    background: rgba(56,189,248,0.10);
    border: 1px solid rgba(56,189,248,0.20);
    color: var(--accent3) !important;
    font-size: 0.60rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 1px 7px; border-radius: 20px;
}
.mcard-grid-expl {
    font-size: 13px; color: var(--txt2) !important;
    line-height: 1.6;
}

/* ── Movie card ROW (chatbot + ultra) ── */
.mcard-row {
    display: flex; background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius); overflow: hidden;
    margin-bottom: 8px;
    transition: border-color 0.2s, box-shadow 0.2s;
    animation: fadeUp 0.3s ease both;
}
.mcard-row:hover {
    border-color: rgba(56,189,248,0.3);
    box-shadow: var(--shadow-accent);
}
.mcard-row-poster {
    width: 70px; flex-shrink: 0;
    background: var(--bg3); overflow: hidden;
}
.mcard-row-poster img { width: 100%; height: 100%; object-fit: cover; display: block; }
.mcard-row-body { flex: 1; padding: 0.7rem 1rem; border-left: 1px solid var(--border); }
.mcard-row-title { font-weight: 700; font-size: 15px; color: var(--txt) !important; margin-bottom: 5px; }
.mcard-row-genres { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 5px; }
.mcard-row-expl { font-size: 13px; color: var(--txt2) !important; line-height: 1.65; }
.mcard-row-score {
    width: 72px; flex-shrink: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 0.8rem; border-left: 1px solid var(--border);
    background: var(--bg4); gap: 5px;
}
.score-pct {
    font-weight: 800; font-size: 17px;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.score-bar-wrap {
    width: 100%; background: rgba(255,255,255,0.07);
    border-radius: 99px; height: 4px; overflow: hidden;
}
.score-bar { height: 100%; border-radius: 99px; }
.score-lbl {
    font-size: 11px; color: #64748b !important;
    text-transform: uppercase; letter-spacing: 0.06em; font-weight: 700;
}

/* ── Skeleton loading ── */
.skeleton {
    background: linear-gradient(90deg, var(--bg3) 25%, var(--bg4) 50%, var(--bg3) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.4s infinite;
    border-radius: var(--radius-sm);
}
@keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
.skeleton-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius); overflow: hidden;
}
.skeleton-poster { width: 100%; padding-top: 150%; }
.skeleton-body   { padding: 0.8rem; }
.skeleton-line   { height: 10px; margin-bottom: 8px; }

/* ── Empty state ── */
.empty-state {
    background: var(--bg3); border: 1px dashed rgba(56,189,248,0.2);
    border-radius: var(--radius); padding: 3rem 2rem; text-align: center;
}
.empty-state-icon { font-size: 3rem; margin-bottom: 1rem; }
.empty-state-title {
    font-size: 1.1rem; font-weight: 700;
    color: var(--txt) !important; margin-bottom: 0.5rem;
}
.empty-state-sub { font-size: 0.85rem; color: var(--txt2) !important; }

/* ── Verdict badges ── */
.verdict {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 8px 20px; border-radius: 20px;
    font-size: 0.9rem; font-weight: 800;
    letter-spacing: 0.05em; margin-bottom: 8px;
}
.v-like    { background: rgba(74,222,128,0.15); border: 1px solid var(--green); color: var(--green) !important; }
.v-maybe   { background: rgba(250,204,21,0.12); border: 1px solid var(--yellow); color: var(--yellow) !important; }
.v-unlikely{ background: rgba(248,113,113,0.12); border: 1px solid var(--red); color: var(--red) !important; }

/* ── About grid ── */
.ag { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }
.ac {
    display: flex; gap: 14px; align-items: flex-start;
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.1rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.ac:hover { border-color: rgba(56,189,248,0.3); box-shadow: var(--shadow-accent); }
.ac-icon {
    width: 36px; height: 36px; flex-shrink: 0;
    background: rgba(56,189,248,0.12);
    border-radius: 8px; display: flex;
    align-items: center; justify-content: center;
}
.ac-icon svg { width: 18px; height: 18px; fill: var(--accent); }
.ac-label { font-size: 15px; font-weight: 700; color: var(--txt) !important; }

/* ── Filter panel ── */
.filter-panel {
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.2rem 1.4rem;
    margin-bottom: 1.4rem;
}
.filter-title {
    font-size: 11px; font-weight: 800;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: #94a3b8 !important; margin-bottom: 1rem;
}

/* ── Radar chart dark ── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Stale result count ── */
.result-meta {
    font-size: 14px; color: #64748b !important;
    margin-bottom: 1.2rem;
}
.result-meta b { color: #38bdf8 !important; }

/* ── Mood pill for chatbot ── */
.mood-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(56,189,248,0.10);
    border: 1px solid rgba(56,189,248,0.22);
    border-radius: 20px; padding: 4px 14px;
    font-size: 12px; font-weight: 700;
    color: #38bdf8 !important;
    letter-spacing: 0.06em; text-transform: uppercase;
    margin: 6px 0 10px;
}

/* ── Score pills for radar ── */
.score-pills { display: flex; gap: 5px; flex-wrap: wrap; justify-content: center; margin-top: 4px; }
.spill {
    background: var(--bg4); border: 1px solid var(--border);
    border-radius: 20px; padding: 1px 8px;
    font-size: 0.64rem; color: var(--txt2) !important; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════
if "chat_history"   not in st.session_state: st.session_state.chat_history   = []
if "ultra_results"  not in st.session_state: st.session_state.ultra_results  = None
if "ultra_parsed"   not in st.session_state: st.session_state.ultra_parsed   = None
if "ultra_query"    not in st.session_state: st.session_state.ultra_query    = ""

# ══════════════════════════════════════════════════════════════════════════
#  DATA LOAD
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_context():
    return load_all()

with st.spinner("Loading CineMatch…"):
    context = load_context()

# ══════════════════════════════════════════════════════════════════════════
#  NAVBAR
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="cm-nav">
  <div class="cm-logo">
    <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
      <path d="M18 4l2 4h-3l-2-4h-2l2 4h-3l-2-4H8l2 4H7L5 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V4h-4z"/>
    </svg>
    Cine<span class="cm-logo-dot">Match</span>
  </div>
  <div class="cm-nav-pills">
    <span class="cm-pill">Multimodal AI</span>
    <span class="cm-pill">Explainable</span>
    <span class="cm-pill">7 Novel Techniques</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════
TMDB_IMG = "https://image.tmdb.org/t/p/w300"
TMDB_IMG_LG = "https://image.tmdb.org/t/p/w500"

def score_class(s: float) -> str:
    if s >= 0.90: return "score-green"
    if s >= 0.75: return "score-teal"
    if s >= 0.60: return "score-yellow"
    return "score-gray"

def score_color(s: float) -> str:
    if s >= 0.90: return "#4ade80"   # green
    if s >= 0.75: return "#2dd4bf"   # teal
    if s >= 0.60: return "#facc15"   # yellow
    return "#64748b"                  # gray

def pct(s: float) -> str:
    return f"{min(int(round(s * 100)), 100)}%"

def genre_chips_dark(genres_str: str) -> str:
    """Genre chips with full inline styles — safe in all Streamlit render contexts."""
    return "".join(
        f'<span style="background:rgba(56,189,248,0.10);border:1px solid rgba(56,189,248,0.22);'
        f'color:#7dd3fc;font-size:11px;font-weight:700;letter-spacing:0.06em;'
        f'text-transform:uppercase;padding:1px 7px;border-radius:20px;'
        f'margin-right:3px;margin-bottom:2px;display:inline-block;">{g.strip()}</span>'
        for g in str(genres_str).split("|") if g.strip()
    )

def poster_url(row, size="w300") -> str | None:
    base = f"https://image.tmdb.org/t/p/{size}"
    if "poster_path" in row and pd.notna(row.get("poster_path")):
        return f"{base}{row['poster_path']}"
    return None

def make_radar(ncf: float, text: float, vis: float) -> go.Figure:
    cats = ["NCF", "Text", "Visual", "NCF"]
    vals = [ncf, text, vis, ncf]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor="rgba(56,189,248,0.08)",
        line=dict(color="#38bdf8", width=2),
        marker=dict(size=5, color="#38bdf8"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,1],
                tickfont=dict(size=7, color="#64748b"),
                gridcolor="rgba(255,255,255,0.06)"),
            angularaxis=dict(
                tickfont=dict(size=9, color="#94a3b8"),
                gridcolor="rgba(255,255,255,0.06)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10,b=10,l=20,r=20), height=150, showlegend=False,
    )
    return fig

def movie_card_grid(row, delay_idx: int = 0) -> str:
    """Render a Netflix-style poster card for the grid layout."""
    p = poster_url(row)
    poster_html = (
        f'<img src="{p}" loading="lazy" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;" />'
        if p else
        '<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:0.7rem;color:#475569;font-weight:600;letter-spacing:0.05em;">NO POSTER</div>'
    )
    score = float(row.get("score", row.get("final_score", 0.5)))
    sc    = score_class(score)
    chips = genre_chips_dark(row.get("genres",""))
    expl  = str(row.get("reason", row.get("explanation", "")))[:160]
    title = str(row.get("title", ""))
    delay = f"animation-delay:{delay_idx * 0.05}s"

    return f"""
<div class="mcard-grid" style="{delay}">
  <div class="mcard-grid-poster">
    {poster_html}
    <span class="score-badge {sc}">{pct(score)} Match</span>
    <div class="mcard-overlay">
      <div style="background:rgba(56,189,248,0.92);color:#0f172a;border:none;
        border-radius:8px;padding:7px 18px;font-family:'Plus Jakarta Sans',sans-serif;
        font-size:0.78rem;font-weight:700;text-align:center;">▶ View Details</div>
    </div>
  </div>
  <div class="mcard-grid-body">
    <div class="mcard-grid-title" title="{title}">{title}</div>
    <div class="mcard-grid-genres">{chips}</div>
    <div class="mcard-grid-expl">{expl}</div>
  </div>
</div>"""

def movie_card_row(row, show_score: bool = True) -> str:
    """Render a horizontal card — fully inline styles, no CSS class deps."""
    p = poster_url(row, "w200")
    poster_html = (
        f'<img src="{p}" loading="lazy" style="width:100%;height:100%;object-fit:cover;display:block;" />'
        if p else
        '<div style="width:100%;min-height:90px;display:flex;align-items:center;'
        'justify-content:center;font-size:11px;color:#475569;font-weight:600;letter-spacing:0.04em;">NO POSTER</div>'
    )
    score   = float(row.get("score", row.get("final_score", 0.5)))
    sc      = score_color(score)
    bar_pct = min(int(score * 100), 100)
    score_pct_str = pct(score)
    title   = str(row.get("title", ""))
    expl    = str(row.get("reason", ""))

    # Build genre chips inline
    genres_str = str(row.get("genres", ""))
    chips = "".join(
        f'<span style="background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.25);'
        f'color:#7dd3fc;font-size:11px;font-weight:700;letter-spacing:0.06em;'
        f'text-transform:uppercase;padding:1px 7px;border-radius:20px;'
        f'margin-right:4px;margin-bottom:2px;display:inline-block;">{g.strip()}</span>'
        for g in genres_str.split("|") if g.strip()
    )

    score_col = ""
    if show_score:
        score_col = (
            f'<div style="width:70px;flex-shrink:0;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:0.8rem;border-left:1px solid rgba(255,255,255,0.08);background:#1a2540;gap:5px;">' +
            f'<div style="font-weight:800;font-size:16px;color:{sc};font-family:Plus Jakarta Sans,sans-serif;">{score_pct_str}</div>' +
            f'<div style="width:100%;background:rgba(255,255,255,0.08);border-radius:99px;height:4px;overflow:hidden;">' +
            f'<div style="width:{bar_pct}%;background:{sc};height:100%;border-radius:99px;"></div></div>' +
            f'<div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;font-weight:700;">Match</div></div>'
        )

    return (
        f'<div style="display:flex;background:#1e293b;border:1px solid rgba(255,255,255,0.08);border-radius:14px;overflow:hidden;margin-bottom:10px;box-shadow:0 4px 20px rgba(0,0,0,0.3);">' +
        f'<div style="width:70px;flex-shrink:0;background:#131c30;overflow:hidden;">{poster_html}</div>' +
        f'<div style="flex:1;padding:0.75rem 1rem;border-left:1px solid rgba(255,255,255,0.06);">' +
        f'<div style="font-weight:700;font-size:15px;color:#f1f5f9;margin-bottom:5px;font-family:Plus Jakarta Sans,sans-serif;">{title}</div>' +
        f'<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px;">{chips}</div>' +
        f'<div style="font-size:13px;color:#94a3b8;line-height:1.65;">{expl}</div>' +
        f'</div>{score_col}</div>'
    )

def render_query_understanding(parsed: dict):
    """Render the query understanding panel with coloured tags."""
    tags = []
    if parsed.get("anchor_movie"):
        tags.append(f'<span class="qup-tag">Anchor: {parsed["anchor_movie"]}</span>')
    if parsed.get("anchor_delta"):
        for d in parsed["anchor_delta"]:
            tags.append(f'<span class="qup-tag">↗ {d}</span>')
    if parsed.get("negations"):
        for n in parsed["negations"]:
            tags.append(f'<span class="qup-tag neg">Not: {n}</span>')
    if parsed.get("era_range"):
        yr = parsed["era_range"]
        tags.append(f'<span class="qup-tag era">Era: {yr[0]}–{yr[1]}</span>')
    if parsed.get("exclude_genres"):
        for eg in parsed["exclude_genres"]:
            tags.append(f'<span class="qup-tag excl">Excl: {eg}</span>')
    if parsed.get("mood_text"):
        tags.append(f'<span class="qup-tag">Mood: {parsed["mood_text"]}</span>')
    if parsed.get("theme_text"):
        tags.append(f'<span class="qup-tag">Theme: {parsed["theme_text"]}</span>')
    if not tags:
        return
    st.markdown(
        f'<div class="qup"><div class="qup-label">Detected Preferences</div>'
        f'<div class="qup-tags">{"".join(tags)}</div></div>',
        unsafe_allow_html=True,
    )

def render_empty_state():
    st.markdown("""
    <div class="empty-state">
      <div class="empty-state-icon" style="font-size:2rem;color:#38bdf8;margin-bottom:1rem;">?</div>
      <div class="empty-state-title">No perfect match found</div>
      <div class="empty-state-sub">Try relaxing filters or changing your description.<br>
      For example, remove era constraints or excluded genres.</div>
    </div>""", unsafe_allow_html=True)

def skeleton_grid(n: int = 5):
    cols = "".join(["""
    <div class="skeleton-card">
      <div class="skeleton skeleton-poster"></div>
      <div class="skeleton-body">
        <div class="skeleton skeleton-line" style="width:80%"></div>
        <div class="skeleton skeleton-line" style="width:55%"></div>
        <div class="skeleton skeleton-line" style="width:70%"></div>
      </div>
    </div>"""] * n)
    st.markdown(f'<div class="movie-grid">{cols}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "  For You  ",
    "  Movie Assistant  ",
    "  Will I Like This?  ",
    "  Ultra Specific  ",
    "  About  ",
])

# ──────────────────────────────────────────────────────────────────────────
# TAB 1 — FOR YOU
# ──────────────────────────────────────────────────────────────────────────
with tabs[0]:

    # Hero section
    st.markdown("""
    <div class="cm-hero">
      <div class="cm-hero-eyebrow">Personalised for you</div>
      <div class="cm-hero-title">Your Personal <span>Cinema</span></div>
      <div class="cm-hero-sub" style="margin:0 auto 1.2rem;">
        Hybrid NCF + SBERT + ResNet recommendations, fused with real-time mood detection
      </div>
      <div class="cm-hero-pills">
        <span class="cm-hero-pill">Neural Collaborative Filtering</span>
        <span class="cm-hero-pill">Sentence-BERT Embeddings</span>
        <span class="cm-hero-pill">Emotion-Adaptive Weights</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

    col_sel, col_mood, col_btn = st.columns([2, 3, 1], vertical_alignment="bottom")
    with col_sel:
        user_id = st.selectbox("User", sorted(context["ratings"].userId.unique()), key="rec_user")
    with col_mood:
        chat_msg = st.text_input(
            "Current mood",
            placeholder="e.g. I want something adventurous and uplifting…",
            key="mood",
        )
    with col_btn:
        run_recs = st.button("Get Picks", use_container_width=True)

    if run_recs:
        alpha, beta, gamma = get_sentiment_weights(chat_msg)
        mood_label = get_weight_explanation(chat_msg, alpha, beta, gamma)
        st.markdown(
            f'<div class="wbar"><div class="wbar-dot"></div>'
            f'{mood_label} &nbsp;·&nbsp; <b>NCF</b> {alpha:.2f} &nbsp;'
            f'<b>Text</b> {beta:.2f} &nbsp;<b>Visual</b> {gamma:.2f}</div>',
            unsafe_allow_html=True,
        )
        skeleton_grid(5)
        with st.spinner(""):
            recs, viz_data = recommend_movies_multimodal(
                user_id=user_id, context=context, chat_msg=chat_msg, top_n=10,
            )
        # rebuild after spinner
        st.rerun() if recs is None else None

        recent_titles = (
            context["movies"][
                context["movies"].movieId.isin(
                    context["ratings"][context["ratings"].userId == user_id].movieId
                )
            ]["title"].tolist()[:3]
        )
        viz_lookup = {}
        if viz_data is not None and not viz_data.empty:
            for _, vr in viz_data.iterrows():
                viz_lookup[vr["title"]] = vr.to_dict()

        st.markdown(
            f'<div class="result-meta"><b>{len(recs)}</b> personalised picks for User {user_id}</div>',
            unsafe_allow_html=True,
        )

        # Render grid
        cards_html = ""
        for i, (_, row) in enumerate(recs.iterrows()):
            explanation = generate_llm_explanation({
                "recommended_movie": row["title"],
                "genres":            row["genres"],
                "recent_movies":     recent_titles,
            })
            score = float(viz_lookup.get(row["title"], {}).get("text", 0.5))
            p     = poster_url(row)
            poster_html = (
                f'<img src="{p}" loading="lazy" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;" />' if p
                else '<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:0.7rem;color:#475569;font-weight:600;letter-spacing:0.05em;">NO POSTER</div>'
            )
            sc    = score_class(score)
            chips = genre_chips_dark(row.get("genres",""))
            title = str(row.get("title",""))
            delay = f"animation-delay:{i*0.05}s"
            expl  = str(explanation)

            sc_colors = {"score-green":"#4ade80","score-teal":"#2dd4bf","score-yellow":"#facc15","score-gray":"#64748b"}
            sc_col = sc_colors.get(sc, "#64748b")
            cards_html += f"""
<div style="background:#1e293b;border:1px solid rgba(255,255,255,0.08);
  border-radius:14px;overflow:hidden;
  transition:transform 0.25s,box-shadow 0.25s;
  {delay.replace("animation-delay","animation-delay")};
  animation:fadeUp 0.4s ease both;">
  <div style="position:relative;width:100%;padding-top:150%;background:#131c30;overflow:hidden;">
    {poster_html}
    <span style="position:absolute;top:8px;right:8px;font-size:12px;font-weight:800;
      padding:3px 9px;border-radius:20px;letter-spacing:0.04em;z-index:2;
      background:rgba(15,23,42,0.85);color:{sc_col};
      border:1px solid {sc_col};">{pct(score)} Match</span>
  </div>
  <div style="padding:0.8rem;">
    <div style="font-weight:700;font-size:15px;color:#f1f5f9;margin-bottom:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:'Plus Jakarta Sans',sans-serif;" title="{title}">{title}</div>
    <div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px;">{chips}</div>
    <div style="font-size:0.73rem;color:#94a3b8;line-height:1.55;">{expl}</div>
  </div>
</div>"""

        st.markdown(f'<div class="movie-grid">{cards_html}</div>', unsafe_allow_html=True)

        # Radar breakdown section
        if viz_lookup:
            st.markdown("<div style='margin-top:2.5rem'></div>", unsafe_allow_html=True)
            st.markdown(
                '<div class="cm-section-title">Score Breakdown</div>'
                '<div class="cm-section-sub">NCF × Text × Visual weights per title</div>',
                unsafe_allow_html=True,
            )
            rcols = st.columns(min(5, len(recs)))
            for i, (_, row) in enumerate(recs.head(5).iterrows()):
                vr = viz_lookup.get(row["title"])
                if vr and i < len(rcols):
                    with rcols[i]:
                        st.plotly_chart(
                            make_radar(
                                float(vr.get("ncf",0)),
                                float(vr.get("text",0)),
                                float(vr.get("poster",0)),
                            ),
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )
                        st.markdown(
                            f"<div style='text-align:center;font-size:0.70rem;"
                            f"color:#94a3b8;font-weight:600;overflow:hidden;"
                            f"text-overflow:ellipsis;white-space:nowrap;'>"
                            f"{row['title'][:22]}…</div>",
                            unsafe_allow_html=True,
                        )

# ──────────────────────────────────────────────────────────────────────────
    st.markdown('</div>', unsafe_allow_html=True)  # close tab-content

# TAB 2 — MOVIE ASSISTANT  (custom chat UI — avoids st.chat_message sanitiser)
# ──────────────────────────────────────────────────────────────────────────

def render_chat_bubble_user(text: str) -> str:
    return f"""
<div style="display:flex;justify-content:flex-end;margin-bottom:12px;gap:10px;align-items:flex-start;">
  <div style="max-width:70%;background:linear-gradient(135deg,#0ea5e9,#38bdf8);
    color:#0f172a;padding:12px 16px;border-radius:18px 18px 4px 18px;
    font-size:15px;font-weight:500;line-height:1.7;
    box-shadow:0 4px 16px rgba(14,165,233,0.3);">
    {text}
  </div>
  <div style="width:32px;height:32px;border-radius:50%;background:#38bdf8;
    display:flex;align-items:center;justify-content:center;
    flex-shrink:0;font-size:0.7rem;font-weight:700;color:#0f172a;margin-top:2px;">YOU</div>
</div>"""

def render_chat_bubble_bot(reply: str, mood_display: str, genres: list,
                            recs, context_movies=None) -> str:
    genre_str = " · ".join(genres[:3]) if genres else ""
    pill_text = f"{mood_display}" + (f"  ·  {genre_str}" if genre_str else "")

    cards_html = ""
    if recs is not None and not recs.empty:
        header = """<div style="font-size:11px;color:#64748b;letter-spacing:0.12em;
            text-transform:uppercase;margin:12px 0 8px;
            border-top:1px solid rgba(255,255,255,0.07);padding-top:12px;">
            Recommended for you</div>"""
        cards = ""
        for _, row in recs.iterrows():
            p      = poster_url(row, "w200")
            title  = str(row.get("title",""))
            genres_str = str(row.get("genres",""))
            reason = str(row.get("reason",""))
            score  = float(row.get("score", row.get("final_score", 0.5)))
            sc     = score_color(score)
            bar_w  = min(int(score*100), 100)
            pct_s  = pct(score)

            chip_html = "".join(
                f'<span style="background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.25);' +
                f'color:#7dd3fc;font-size:11px;font-weight:700;letter-spacing:0.06em;' +
                f'text-transform:uppercase;padding:1px 6px;border-radius:20px;' +
                f'margin-right:3px;display:inline-block;">{g.strip()}</span>'
                for g in genres_str.split("|") if g.strip()
            )
            img_html = (
                f'<img src="{p}" style="width:100%;height:100%;object-fit:cover;display:block;">' if p
                else '<div style="width:100%;height:100%;min-height:90px;display:flex;align-items:center;justify-content:center;font-size:0.6rem;color:#475569;font-weight:600;letter-spacing:0.05em;">NO POSTER</div>'
            )

            cards += f"""
<div style="display:flex;background:#263147;border:1px solid rgba(255,255,255,0.09);
  border-radius:12px;overflow:hidden;margin-bottom:8px;">
  <div style="width:65px;min-height:90px;flex-shrink:0;background:#1a2540;overflow:hidden;">{img_html}</div>
  <div style="flex:1;padding:10px 12px;border-left:1px solid rgba(255,255,255,0.06);">
    <div style="font-weight:700;font-size:15px;color:#f1f5f9;margin-bottom:4px;">{title}</div>
    <div style="margin-bottom:5px;">{chip_html}</div>
    <div style="font-size:13px;color:#94a3b8;line-height:1.65;">{reason}</div>
  </div>
  <div style="width:66px;flex-shrink:0;display:flex;flex-direction:column;
    align-items:center;justify-content:center;padding:10px;
    border-left:1px solid rgba(255,255,255,0.06);background:#1e2d45;gap:4px;">
    <div style="font-weight:800;font-size:16px;color:{sc};">{pct_s}</div>
    <div style="width:38px;background:rgba(255,255,255,0.08);border-radius:99px;height:4px;overflow:hidden;">
      <div style="width:{bar_w}%;background:{sc};height:100%;border-radius:99px;"></div>
    </div>
    <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;font-weight:700;">Match</div>
  </div>
</div>"""
        cards_html = header + cards

    return f"""
<div style="display:flex;justify-content:flex-start;margin-bottom:16px;gap:10px;align-items:flex-start;">
  <div style="width:32px;height:32px;border-radius:50%;background:#1e293b;
    border:2px solid #38bdf8;display:flex;align-items:center;
    justify-content:center;flex-shrink:0;font-size:0.65rem;font-weight:700;color:#38bdf8;margin-top:2px;">AI</div>
  <div style="max-width:78%;background:#1e293b;border:1px solid rgba(255,255,255,0.08);
    padding:14px 16px;border-radius:4px 18px 18px 18px;
    box-shadow:0 4px 20px rgba(0,0,0,0.3);">
    <div style="font-size:15px;color:#e2e8f0;line-height:1.75;margin-bottom:10px;">{reply}</div>
    <div style="display:inline-flex;align-items:center;gap:5px;
      background:rgba(56,189,248,0.10);border:1px solid rgba(56,189,248,0.25);
      border-radius:20px;padding:4px 12px;font-size:12px;font-weight:700;
      color:#38bdf8;letter-spacing:0.06em;text-transform:uppercase;">{pill_text}</div>
    {cards_html}
  </div>
</div>"""


with tabs[1]:

    # ── Hero banner ───────────────────────────────────────────────
    st.markdown("""
    <div class="cm-hero" style="padding:2.6rem 2.5rem 2rem;">
      <div class="cm-hero-eyebrow">Emotion-Aware Recommendations</div>
      <div class="cm-hero-title">Movie <span>Assistant</span></div>
      <div class="cm-hero-sub" style="margin:0 auto 1.2rem;">
        Tell me how you feel — I'll find exactly the right film for your mood
      </div>
      <div class="cm-hero-pills">
        <span class="cm-hero-pill">7-Class Emotion Detection</span>
        <span class="cm-hero-pill">Intent Parsing</span>
        <span class="cm-hero-pill">Diversity Filtering</span>
        <span class="cm-hero-pill">Groq LLM Replies</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    h1, h2 = st.columns([4, 1], vertical_alignment="bottom")
    with h1:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    with h2:
        if st.button("Clear chat", use_container_width=True, key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    with st.expander("Personalisation Settings", expanded=False):
        chat_user_id = st.selectbox(
            "Personalise for user",
            sorted(context["ratings"].userId.unique()),
            key="chat_user",
        )
    if "chat_user" not in st.session_state:
        st.session_state["chat_user"] = sorted(context["ratings"].userId.unique())[0]

    # ── Chat history container ─────────────────────────────────
    st.markdown("""
    <div style="min-height:60px;margin-top:12px;margin-bottom:4px;
      padding:20px 24px;background:#0d1829;border:1px solid rgba(255,255,255,0.06);
      border-radius:14px;max-height:72vh;overflow-y:auto;" id="chat-container">
    """, unsafe_allow_html=True)

    chat_html = ""
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            chat_html += render_chat_bubble_user(entry["content"])
        elif entry["role"] == "bot":
            data        = entry.get("data", {})
            recs        = data.get("recommendations")
            chat_html  += render_chat_bubble_bot(
                reply        = entry["content"],
                mood_display = data.get("mood_display", ""),
                genres       = data.get("genres", []),
                recs         = recs,
            )

    if chat_html:
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='text-align:center;padding:2rem;color:#64748b;"
            "font-size:0.88rem;color:#64748b;'>Start a conversation — tell me how you feel</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Chat input ─────────────────────────────────────────────
    user_message = st.chat_input("e.g. I had a rough day, something funny and light please…")

    if user_message and user_message.strip():
        uid = st.session_state.get(
            "chat_user", sorted(context["ratings"].userId.unique())[0]
        )
        st.session_state.chat_history.append({"role": "user", "content": user_message.strip()})
        with st.spinner("Finding the right movies…"):
            result = chatbot_recommendation(
                user_id=uid, message=user_message.strip(), context=context,
            )
        st.session_state.chat_history.append({
            "role":    "bot",
            "content": result["reply"],
            "data":    result,
        })
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # close tab-content

# ──────────────────────────────────────────────────────────────────────────
# TAB 3 — WILL I LIKE THIS?
# ──────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("""
    <div class="cm-hero" style="padding:2.6rem 2.5rem 2rem;">
      <div class="cm-hero-eyebrow">Preference Prediction Engine</div>
      <div class="cm-hero-title">Will I Like <span>This?</span></div>
      <div class="cm-hero-sub" style="margin:0 auto 1.2rem;">
        Enter any movie — get an AI-powered compatibility score against your taste profile
      </div>
      <div class="cm-hero-pills">
        <span class="cm-hero-pill">NCF Score</span>
        <span class="cm-hero-pill">Narrative Match</span>
        <span class="cm-hero-pill">Visual Poster Analysis</span>
        <span class="cm-hero-pill">LLM Explanation</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    user_id2 = st.selectbox("User", sorted(context["ratings"].userId.unique()), key="predict_user")

    ca, cb_col = st.columns(2)
    with ca:
        title  = st.text_input("Movie Title", placeholder="e.g. Interstellar")
    with cb_col:
        genres = st.text_input("Genres", placeholder="e.g. Sci-Fi|Drama")
    plot = st.text_area("Plot Summary", placeholder="Brief synopsis…", height=100)

    st.markdown(
        "<p style='font-size:13px;font-weight:700;letter-spacing:0.08em;"
        "text-transform:uppercase;color:#94a3b8;margin-bottom:6px;margin-top:8px;'>"
        "Poster (optional)</p>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload poster image", type=["jpg","png"], label_visibility="collapsed")
    if uploaded:
        pc, _ = st.columns([1, 4])
        with pc:
            st.image(uploaded, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_pred = st.button("Analyse Compatibility", key="run_pred")

    if run_pred:
        with st.spinner("Analysing…"):
            result = predict_user_like_movie(
                user_id=user_id2,
                movie_text=f"{title} {genres} {plot}",
                context=context,
                uploaded_poster=uploaded,
            )
        verdict    = result["verdict"]
        confidence = result["confidence"]
        badge_map  = {
            "like":     ("v-like",     "Good Match",     "High compatibility"),
            "maybe":    ("v-maybe",    "◎ Possible Match", "Moderate compatibility"),
            "unlikely": ("v-unlikely", "Low Match",      "May not suit your taste"),
        }
        cls, label, sublabel = badge_map.get(verdict, badge_map["unlikely"])
        st.markdown(
            f'<div class="verdict {cls}">{label}</div>'
            f"<p style='font-size:14px;color:#94a3b8;margin-top:-4px;margin-bottom:1rem;'>"
            f"{sublabel} · {confidence} confidence</p>",
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Compatibility", f"{result['final_score']:.0%}")
        m2.metric("Narrative Match", f"{result['text_score']:.0%}")
        m3.metric("Visual Match", f"{result['poster_score']:.0%}" if result["used_poster"] else "—")

        exp = generate_llm_explanation({
            "recommended_movie": title, "genres": genres,
            "recent_movies": [], "verdict": verdict,
        })
        st.markdown(
            f'<div class="mcard-row" style="margin-top:1.2rem;">'
            f'<div class="mcard-row-body" style="border-left:none;">'
            f'<div class="mcard-row-expl">{exp}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)  # close tab-content

# ──────────────────────────────────────────────────────────────────────────
# TAB 4 — ULTRA SPECIFIC
# ──────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("""
    <div class="cm-hero" style="padding:2.6rem 2.5rem 2rem;">
      <div class="cm-hero-eyebrow">7 Novel Techniques</div>
      <div class="cm-hero-title">Ultra-Specific <span>Search</span></div>
      <div class="cm-hero-sub">
        Natural language filtering with negation, anchors, era ranges and mood-theme decoupling
      </div>
      <div class="cm-hero-pills">
        <span class="cm-hero-pill">Negation-Aware Vectors</span>
        <span class="cm-hero-pill">Anchor + Delta Shift</span>
        <span class="cm-hero-pill">Mood-Theme Decoupling</span>
        <span class="cm-hero-pill">Era Filtering</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Hero query input ──────────────────────────────────────────
    ultra_query = st.text_area(
        "Describe what you want to watch",
        placeholder=(
            'e.g. "like Inception but simpler and funnier, NOT like Twilight, '
            'dark but hopeful ending, 90s vibes, no horror"'
        ),
        height=100,
        label_visibility="visible",
        key="ultra_query_input",
    )
    st.markdown(
        '<div class="cm-hero-hint">Examples: &nbsp;'
        '<em>animation with animals cheerful adventure kids</em> &nbsp;·&nbsp; '
        '<em>dark sci-fi mind-bending not horror 2010s</em> &nbsp;·&nbsp; '
        '<em>like The Dark Knight but funnier</em></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Filter panel ──────────────────────────────────────────────
    with st.expander("Optional Filters", expanded=False):
        st.markdown('<div class="filter-title">Filters</div>', unsafe_allow_html=True)
        uf1, uf2, uf3 = st.columns(3)
        with uf1:
            exclude_genres_ui = st.multiselect(
                "Exclude genres",
                options=sorted(GENRE_ALIASES.keys()),
                key="ultra_excl_genres",
            )
            sort_by = st.selectbox(
                "Sort by",
                ["Best match", "Most popular", "Newest"],
                key="ultra_sort",
            )
        with uf2:
            mood_weight_ui = st.slider(
                "Mood weight", 0.0, 1.0, 0.35, 0.05,
                help="Higher = emotional tone matters more than content theme",
                key="ultra_mood_w",
            )
            theme_weight_ui = round(1.0 - mood_weight_ui, 2)
            st.markdown(
                f"<div style='font-size:13px;color:#94a3b8;margin-top:-6px;'>"
                f"Theme weight: <b style='color:#38bdf8;'>{theme_weight_ui}</b></div>",
                unsafe_allow_html=True,
            )
        with uf3:
            top_n_ui = st.slider("Results", 5, 20, 10, key="ultra_top_n")
            min_year, max_year = st.slider(
                "Year range", 1970, 2025, (1990, 2025), key="ultra_year_range"
            )

    run_ultra = st.button("Find Movies", key="ultra_run", use_container_width=False)

    if run_ultra and ultra_query.strip():
        skeleton_grid(5)
        with st.spinner("Parsing query and searching…"):
            results_df, parsed = ultra_filter_recommend(
                query                = ultra_query.strip(),
                context              = context,
                exclude_genres_extra = exclude_genres_ui,
                mood_theme_weights   = (mood_weight_ui, theme_weight_ui),
                top_n                = top_n_ui,
            )
        st.session_state.ultra_results = results_df
        st.session_state.ultra_parsed  = parsed
        st.session_state.ultra_query   = ultra_query.strip()

    # ── Query understanding panel ────────────────────────────────
    if st.session_state.ultra_parsed:
        render_query_understanding(st.session_state.ultra_parsed)

    # ── Results ──────────────────────────────────────────────────
    results_df = st.session_state.ultra_results
    if results_df is not None:
        if results_df.empty:
            render_empty_state()
        else:
            st.markdown(
                f'<div style="font-size:14px;color:#64748b;margin-bottom:1.2rem;">' 
                f'<b style="color:#38bdf8;">{len(results_df)}</b> results for: '
                f'<em style="color:#94a3b8;">{st.session_state.ultra_query}</em></div>',
                unsafe_allow_html=True,
            )
            # Batch generate ALL explanations first (one spinner, no loop re-renders)
            if "ultra_expls" not in st.session_state or                st.session_state.get("ultra_expls_query") != st.session_state.ultra_query:
                with st.spinner("Generating explanations…"):
                    expls = []
                    for _, row in results_df.iterrows():
                        e = generate_ultra_explanation(
                            movie_title = row["title"],
                            genres      = row.get("genres",""),
                            parsed      = st.session_state.ultra_parsed or {},
                        )
                        expls.append(e)
                st.session_state.ultra_expls = expls
                st.session_state.ultra_expls_query = st.session_state.ultra_query
            else:
                expls = st.session_state.ultra_expls

            # Render all cards at once
            all_cards_html = ""
            for i, (_, row) in enumerate(results_df.iterrows()):
                row_copy = row.copy()
                row_copy["reason"] = expls[i] if i < len(expls) else ""
                all_cards_html += movie_card_row(row_copy, show_score=True)
            st.markdown(all_cards_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close tab-content

# ──────────────────────────────────────────────────────────────────────────
# TAB 5 — ABOUT
# ──────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("""
    <div class="cm-hero" style="padding:2.6rem 2.5rem 2rem;text-align:left;">
      <div class="cm-hero-eyebrow" style="text-align:left;">Research Grade · Mini Project</div>
      <div class="cm-hero-title" style="text-align:left;">About <span>CineMatch</span></div>
      <div class="cm-hero-sub" style="text-align:left;max-width:640px;">
        A multimodal recommender combining Neural Collaborative Filtering, Sentence-BERT, 
        and ResNet-50 visual features — with 7 novel techniques verified against 
        RecSys literature (2020–2025).
      </div>
      <div class="cm-hero-pills" style="justify-content:flex-start;">
        <span class="cm-hero-pill">MovieLens 25M</span>
        <span class="cm-hero-pill">TMDB Metadata</span>
        <span class="cm-hero-pill">Groq LLM</span>
        <span class="cm-hero-pill">7 Novel Methods</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='cm-section-title'>Core Techniques</div>"
        "<div class='cm-section-sub'>Novel contributions verified against RecSys literature</div>",
        unsafe_allow_html=True,
    )

    ICONS = {
        "ncf":   '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14.5v-9l6 4.5-6 4.5z"/>',
        "time":  '<path d="M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zm4.24 14L11 13V7h1.5v5.25l4.5 2.67-1.26 2.08z"/>',
        "post":  '<path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3 3.5-4.5 4.5 6H5l3.5-4.5z"/>',
        "mood":  '<path d="M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm3.5-9c.83 0 1.5-.67 1.5-1.5S16.33 8 15.5 8 14 8.67 14 9.5s.67 1.5 1.5 1.5zm-7 0c.83 0 1.5-.67 1.5-1.5S9.33 8 8.5 8 7 8.67 7 9.5 7.67 11 8.5 11zm3.5 6.5c2.33 0 4.31-1.46 5.11-3.5H6.89c.8 2.04 2.78 3.5 5.11 3.5z"/>',
        "neg":   '<path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>',
        "anchor":'<path d="M17 15l1.55 1.55c-.96 1.69-3.33 3.04-5.55 3.37V11h3V9h-3V7.46C14.15 7.04 15 5.65 15 4c0-1.65-1.35-3-3-3S9 2.35 9 4c0 1.65.85 3.04 2 3.46V9H8v2h3v8.92c-2.22-.33-4.59-1.68-5.55-3.37L7 15l-4-3v3c0 3.88 4.92 7 9 7s9-3.12 9-7v-3l-4 3z"/>',
        "chat":  '<path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>',
    }
    features = [
        ("ncf",    "Neural Collaborative Filtering",      "Learns user-item interaction patterns from 25M ratings"),
        ("time",   "Time-Decay User Profiles",            "exp(−0.3×years) weighting — recent taste matters more"),
        ("post",   "Genre-Conditioned Poster Scoring",    "Visual similarity × Jaccard(genre overlap) boost"),
        ("mood",   "Sentiment-Adaptive Fusion Weights",   "α/β/γ shift dynamically with detected user emotion"),
        ("neg",    "Negation-Aware Vector Subtraction",   "query_vec −= 0.45 × mean(neg_vecs) — novel in RecSys"),
        ("anchor", "Anchor + Delta Query Shift",          "Start from a reference film, shift toward modifiers"),
        ("chat",   "7-Class Emotion → Genre Pipeline",    "DistilRoBERTa + genre inference + diversity filter"),
    ]

    st.markdown('<div class="ag">', unsafe_allow_html=True)
    for key, label, desc in features:
        svg = f'<svg viewBox="0 0 24 24">{ICONS[key]}</svg>'
        st.markdown(f"""
        <div class="ac">
          <div class="ac-icon">{svg}</div>
          <div>
            <div class="ac-label">{label}</div>
            <div style="font-size:13px;color:#94a3b8;margin-top:4px;line-height:1.6;">{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='display:flex;gap:8px;flex-wrap:wrap;'>"
        + "".join(
            f"<span class='cm-pill'>{t}</span>"
            for t in ["Research Grade","Explainable AI","MovieLens 25M",
                      "TMDB Metadata","Groq LLM","Emotion-Aware","Label Scarce"]
        )
        + "</div>",
        unsafe_allow_html=True,
    )