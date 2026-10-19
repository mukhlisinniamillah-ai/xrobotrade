# app.py
# ============================================================
# FIX: Pastikan folder app.py ada di sys.path
# (workaround Python 3.14 + Streamlit yang kadang tidak
#  menambahkan working dir ke sys.path)
# ============================================================
import os
import sys

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import base64
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core import (
    BROKER_LIST,
    analyze_broker_summary,
    get_historical_data,
    get_technical_data,
    find_support_resistance,
    analyze_trend,
    analyze_volume,
    analyze_rsi,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Xrobotrade Analyst",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "bg":     "#f6f1e9",
    "panel":  "#ffffff",
    "border": "#e5dccb",
    "text":   "#1a1c1f",
    "muted":  "#8a7d6b",
    "amber":  "#c07a33",
    "cyan":   "#4a4f55",
    "green":  "#4a7c59",
    "red":    "#b0453a",
    "gray":   "#8a7d6b",
}

# ============================================================
# CREATOR PROFILE — isi sesuai Anda
# ============================================================
CREATOR_NAME      = "Mukhlisin Niamillah"
CREATOR_ROLE      = "Trader &amp; Investor · Stockmarket Analyst"
CREATOR_TAGLINE   = "“Cuman seseorang yang sedang menghadapi berbagai rasa takutnya.”"
CREATOR_BIO       = (
    "Trader &amp; investor aktif di pasar modal. INFJ. Mendalami psikologi, "
    "teknologi, dan finansial market. Menjalani hidup dengan prinsip Cacing-Cacing, Naga-Naga."
)
CREATOR_GITHUB    = "https://github.com/mukhlisinniamillah-ai"
CREATOR_INSTAGRAM = "https://www.instagram.com/niam.munehisa17/"
CREATOR_EMAIL     = "Mukhlisinniamillah@gmail.com"

# Foto: taruh file profile.jpg / profile.png di folder yang sama dengan app.py.
# Atau isi CREATOR_PHOTO_URL dengan URL gambar remote.
CREATOR_PHOTO_URL  = ""
CREATOR_PHOTO_PATH = ""


def _get_creator_photo_html():
    if CREATOR_PHOTO_URL:
        return f'<img src="{CREATOR_PHOTO_URL}" class="creator-photo" alt="{CREATOR_NAME}" />'

    for path in (CREATOR_PHOTO_PATH, "profile.png", "profile.jpeg", "creator.jpg"):
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                ext  = path.rsplit(".", 1)[-1].lower()
                mime = "image/png" if ext == "png" else "image/jpeg"
                return f'<img src="data:{mime};base64,{b64}" class="creator-photo" alt="{CREATOR_NAME}" />'
            except Exception:
                continue

    return '<div class="creator-photo-fallback">MN</div>'


def creator_card_html():
    photo = _get_creator_photo_html()
    return f"""
    <div class="creator-card">
        <div class="creator-photo-wrap">{photo}</div>
        <div class="creator-name">{CREATOR_NAME}</div>
        <div class="creator-role">{CREATOR_ROLE}</div>
        <div class="creator-tagline">{CREATOR_TAGLINE}</div>
        <div class="creator-divider"></div>
        <div class="creator-bio">{CREATOR_BIO}</div>
        <div class="creator-links">
            <a href="{CREATOR_GITHUB}"    target="_blank" rel="noopener">GitHub</a>
            <a href="{CREATOR_INSTAGRAM}" target="_blank" rel="noopener">Instagram</a>
            <a href="mailto:{CREATOR_EMAIL}">Email</a>
        </div>
    </div>
    """


# ============================================================
# GLOBAL CSS — tema editorial cream + perbaikan readability
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #f6f1e9;
    --sidebar: #efe8db;
    --surface: #ffffff;
    --border: #e5dccb;
    --border-strong: #d7c4ad;
    --text: #1a1c1f;
    --text-soft: #4a4f55;
    --muted: #8a7d6b;
    --bronze: #c07a33;
    --bronze-deep: #a86526;
    --green: #4a7c59;
    --red: #b0453a;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text);
    -webkit-font-smoothing: antialiased;
}

.stApp { background-color: var(--bg); color: var(--text); }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background-color: var(--sidebar);
    border-right: 1px solid var(--border-strong);
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span { color: var(--text) !important; }
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.04em;
    font-size: 0.72rem !important;
}

/* ---------- Headings ---------- */
h1, h2, h3, h4 {
    font-family: 'Fraunces', Georgia, serif !important;
    font-weight: 500 !important;
    letter-spacing: -0.015em;
    color: var(--text) !important;
}
h4 { font-weight: 500 !important; font-size: 1.35rem !important; }

/* ---------- Buttons ---------- */
.stButton > button {
    background-color: var(--text);
    color: var(--bg);
    border: 1px solid var(--text);
    border-radius: 2px;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-size: 0.75rem;
    padding: 0.7rem 1.6rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background-color: var(--bronze);
    border-color: var(--bronze);
    color: #ffffff;
    box-shadow: 0 6px 16px rgba(192,122,51,0.22);
}
.stButton > button:active,
.stButton > button:focus {
    background-color: var(--bronze-deep) !important;
    border-color: var(--bronze-deep) !important;
    color: #ffffff !important;
    box-shadow: none !important;
}

/* ---------- Tabs ---------- */
button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    padding-top: 0.9rem;
    padding-bottom: 0.9rem;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--text) !important;
    font-weight: 600;
}
div[data-baseweb="tab-highlight"] { background-color: var(--bronze) !important; }
div[data-baseweb="tab-border"]     { background-color: var(--border) !important; }

/* ---------- Metrics ---------- */
div[data-testid="stMetric"] {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.9rem 1rem;
    box-shadow: 0 1px 2px rgba(26,28,31,0.035);
}
div[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
div[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem !important;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-feature-settings: "tnum";
}

/* ---------- Expander ---------- */
div[data-testid="stExpander"] {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
}
div[data-testid="stExpander"] summary {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text);
}

/* ---------- Inputs ---------- */
.stTextInput input, .stNumberInput input, div[data-baseweb="select"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}

/* ============================================================
   CRITICAL READABILITY FIX
   Memaksa semua teks native Streamlit berwarna gelap,
   terlepas dari setting dark/light mode OS pengguna.
   ============================================================ */
.stApp, [data-testid="stAppViewContainer"] {
    --text-color: #1a1c1f !important;
    --background-color: #f6f1e9 !important;
    --secondary-background-color: #ffffff !important;
    --primary-color: #c07a33 !important;
}
[data-testid="stMarkdownContainer"] p { color: #1a1c1f !important; }

/* Text/Number/TextArea inputs */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
.stDateInput input, .stTimeInput input {
    color: #1a1c1f !important;
    background-color: #ffffff !important;
    border: 1px solid #e5dccb !important;
    caret-color: #c07a33 !important;
}
.stTextInput input::placeholder, .stNumberInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #8a7d6b !important;
    opacity: 1 !important;
}
.stNumberInput button {
    background-color: #ffffff !important;
    color: #1a1c1f !important;
    border: 1px solid #e5dccb !important;
}
.stNumberInput button:hover {
    background-color: #efe8db !important;
    color: #a86526 !important;
}
.stNumberInput button svg { fill: #1a1c1f !important; }

/* Selectbox */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #e5dccb !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] div,
div[data-baseweb="select"] input {
    color: #1a1c1f !important;
    background-color: transparent !important;
}
div[data-baseweb="select"] svg { fill: #1a1c1f !important; }

/* Dropdown portal */
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li,
div[data-baseweb="menu"],
ul[data-testid="stSelectboxVirtualDropdown"],
ul[role="listbox"],
li[role="option"] {
    background-color: #ffffff !important;
    color: #1a1c1f !important;
}
div[data-baseweb="popover"] li:hover,
ul[role="listbox"] li:hover,
li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background-color: #efe8db !important;
    color: #a86526 !important;
}

/* Expander isi */
[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #e5dccb !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * { color: #1a1c1f !important; }
[data-testid="stExpander"] p,
[data-testid="stExpander"] label { color: #1a1c1f !important; }

/* Spinner */
[data-testid="stSpinner"],
[data-testid="stSpinner"] p,
[data-testid="stSpinner"] span,
[data-testid="stSpinner"] div,
.stSpinner p { color: #1a1c1f !important; }

/* Alerts */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div { color: #1a1c1f !important; }

/* Caption */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p { color: #8a7d6b !important; }

/* Sidebar label & caption */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span { color: #1a1c1f !important; }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small { color: #8a7d6b !important; }

/* Checkbox / Radio */
.stCheckbox label, .stCheckbox label span,
.stRadio label,    .stRadio label span { color: #1a1c1f !important; }

/* Tooltip */
div[data-baseweb="tooltip"], div[data-baseweb="tooltip"] * { color: #f6f1e9 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #f6f1e9; }
::-webkit-scrollbar-thumb { background: #d7c4ad; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #c07a33; }

/* ============================================================
   CUSTOM COMPONENTS
   ============================================================ */
.term-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 1px 3px rgba(26,28,31,0.04);
}
.term-header .brand {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 500;
    font-size: 1.55rem;
    color: var(--text);
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.term-header .brand span {
    color: var(--bronze);
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.62rem;
    display: block;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.term-header .clock {
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    font-size: 0.7rem;
    text-align: right;
    line-height: 1.8;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.card {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(26,28,31,0.03);
}
.card-accent-amber { border-left: 3px solid var(--bronze); }
.card-accent-cyan  { border-left: 3px solid var(--text-soft); }
.card-accent-green { border-left: 3px solid var(--green); }
.card-accent-red   { border-left: 3px solid var(--red); }
.card-accent-gray  { border-left: 3px solid var(--muted); }

.card-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.66rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--muted);
    margin-bottom: 0.55rem;
}
.card-value {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.18rem;
    font-weight: 500;
    color: var(--text);
    line-height: 1.45;
}
.card-note {
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    color: var(--text-soft);
    margin-top: 0.55rem;
    line-height: 1.6;
}

.badge {
    display: inline-block;
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.3rem 0.75rem;
    border-radius: 2px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge-green { background: rgba(74,124,89,0.09);   color: var(--green);  border: 1px solid rgba(74,124,89,0.28); }
.badge-red   { background: rgba(176,69,58,0.09);   color: var(--red);    border: 1px solid rgba(176,69,58,0.28); }
.badge-amber { background: rgba(192,122,51,0.09);  color: var(--bronze); border: 1px solid rgba(192,122,51,0.30); }
.badge-gray  { background: rgba(138,125,107,0.09); color: var(--muted);  border: 1px solid rgba(138,125,107,0.28); }

table.sr-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.83rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
}
table.sr-table td {
    padding: 0.55rem 0.8rem;
    border-bottom: 1px solid var(--border);
    color: var(--text);
}
table.sr-table tr:last-child td { border-bottom: none; }
table.sr-table td:first-child {
    color: var(--text-soft);
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 0.04em;
}
table.sr-table td:last-child {
    text-align: right;
    font-weight: 600;
    font-feature-settings: "tnum";
}

/* ---------- Creator card (sidebar) ---------- */
.creator-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.2rem 1rem 1rem;
    margin-bottom: 1rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(26,28,31,0.04);
}
.creator-photo-wrap {
    width: 74px;
    height: 74px;
    margin: 0 auto 0.7rem;
    border-radius: 50%;
    padding: 2px;
    background: linear-gradient(135deg, var(--bronze), var(--bronze-deep));
    box-shadow: 0 3px 10px rgba(192,122,51,0.22);
}
.creator-photo, .creator-photo-fallback {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    object-fit: cover;
    display: block;
    background: var(--surface);
    border: 2px solid var(--surface);
    color: var(--bronze);
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600;
    font-size: 1.3rem;
    text-align: center;
}
.creator-photo-fallback {
    display: flex;
    align-items: center;
    justify-content: center;
    line-height: 1;
}
.creator-name {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 0.98rem;
    font-weight: 500;
    color: var(--text);
    letter-spacing: -0.01em;
    line-height: 1.3;
    margin-bottom: 0.12rem;
}
.creator- {
    font-family: 'Inter', sans-serif;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.55rem;
}
.creator-role {
    font-family: 'Inter', sans-serif;
    font-size: 0.62rem;
    font-weight: 600;
    color: var(--bronze);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    line-height: 1.5;
    margin-bottom: 0.55rem;
}
.creator-tagline {
    font-family: 'Fraunces', Georgia, serif;
    font-style: italic;
    font-size: 0.74rem;
    color: var(--text-soft);
    line-height: 1.55;
    margin-bottom: 0.7rem;
}
.creator-divider { height: 1px; background: var(--border); margin: 0.7rem 0; }
.creator-bio {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: var(--text-soft);
    line-height: 1.6;
    text-align: left;
    margin-bottom: 0.7rem;
}
.creator-links {
    display: flex;
    justify-content: center;
    gap: 0.7rem;
    flex-wrap: wrap;
}
.creator-links a {
    font-family: 'Inter', sans-serif;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    text-decoration: none;
    padding: 0.2rem 0.3rem;
    border-bottom: 1px solid transparent;
    transition: all 0.2s ease;
}
.creator-links a:hover {
    color: var(--bronze);
    border-bottom-color: var(--bronze);
    opacity: 1;
}

.footer-note {
    text-align: center;
    color: var(--muted);
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 2.6rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border);
    line-height: 1.9;
}
.footer-note a { color: var(--bronze); text-decoration: none; }
.footer-note a:hover { text-decoration: underline; }
hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
def badge(text, kind):
    return f'<span class="badge badge-{kind}">{text}</span>'


def card(label, value_html, note_html="", accent=""):
    accent_cls = f"card-accent-{accent}" if accent else ""
    note = f'<div class="card-note">{note_html}</div>' if note_html else ""
    return f"""
    <div class="card {accent_cls}">
        <div class="card-label">{label}</div>
        <div class="card-value">{value_html}</div>
        {note}
    </div>
    """


# ============================================================
# HEADER
# ============================================================
now_str = datetime.now().strftime("%d %b %Y \u2022 %H:%M")
st.markdown(f"""
<div class="term-header">
    <div class="brand">XROBOTRADE ANALYST<span>Bandarmology &amp; Technical Intelligence</span></div>
    <div class="clock">Session: {now_str}<br>Source: Yahoo Finance</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    # --- Profil pembuat ---
    st.markdown(creator_card_html(), unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(
        '<div class="card-label" style="margin-top:0;">DATA BROKER SUMMARY</div>',
        unsafe_allow_html=True,
    )
    PERIODS = ["1 MINGGU", "1 BULAN", "3 BULAN", "6 BULAN", "1 TAHUN"]

    def broker_inputs(period_key, role_key, count=3):
        rows = []
        for i in range(1, count + 1):
            c1, c2, c3 = st.columns([1.4, 1, 1])
            broker = c1.selectbox(
                f"{role_key} broker {i}", BROKER_LIST,
                key=f"{role_key}_{period_key}_{i}_broker",
                label_visibility="collapsed",
            )
            lot = c2.number_input(
                f"{role_key} lot {i}", min_value=0.0, step=1000.0,
                key=f"{role_key}_{period_key}_{i}_lot",
                label_visibility="collapsed",
            )
            price = c3.number_input(
                f"{role_key} harga {i}", min_value=0.0, step=10.0,
                key=f"{role_key}_{period_key}_{i}_price",
                label_visibility="collapsed",
            )
            rows.append({"broker": broker, "lot": lot, "price": price})
        return rows

    broker_data = []
    for p in PERIODS:
        with st.expander(p, expanded=(p == "1 TAHUN")):
            st.caption("TOP BUYER — Broker / Lot / Harga")
            buyers = broker_inputs(p, "buy", 3)
            st.caption("TOP SELLER — Broker / Lot / Harga")
            sellers = broker_inputs(p, "sell", 3)
        broker_data.append({
            "period": p,
            "buyers":  [(b["broker"], b["lot"], b["price"]) for b in buyers
                        if b["broker"] and b["lot"] > 0 and b["price"] > 0],
            "sellers": [(b["broker"], b["lot"], b["price"]) for b in sellers
                        if b["broker"] and b["lot"] > 0 and b["price"] > 0],
        })

    st.markdown("---")
    symbol  = st.text_input("KODE SAHAM", value="BBCA.JK")
    period  = st.selectbox("PERIODE DATA TEKNIKAL", ["1mo", "3mo", "6mo", "1y"], index=2)
    analyze_btn = st.button("JALANKAN ANALISIS", type="primary", use_container_width=True)


# ============================================================
# MAIN LOGIC
# ============================================================
if analyze_btn:
    with st.spinner("Sedang memproses data..."):
        broker_result = analyze_broker_summary(broker_data)
        hist, err = get_historical_data(symbol, period=period)
        if err:
            st.error(err)
            st.stop()

        tech_result, err2 = get_technical_data(hist)
        if err2:
            st.error(err2)
            st.stop()

        current_price = tech_result["current_price"]
        ma5           = tech_result["ma5"].iloc[-1]
        ma10          = tech_result["ma10"].iloc[-1]
        ma20          = tech_result["ma20"].iloc[-1]
        ma50          = tech_result["ma50"].iloc[-1]
        volume_today  = tech_result["volume_today"]
        avg_volume_20 = tech_result["avg_volume_20"]
        rsi           = tech_result["rsi"].iloc[-1]
        sr_data       = find_support_resistance(hist, order=5)

        trend        = analyze_trend(current_price, ma5, ma10, ma20, ma50)
        vol_analysis = analyze_volume(volume_today, avg_volume_20)
        rsi_analysis = analyze_rsi(rsi)

    # ---- Sinyal ----
    signal = "LEMAH"
    signal_desc = "Tidak ada konfirmasi sinyal yang kuat"

    if "AKUMULASI" in broker_result["status"] and "UPTREND" in trend:
        signal, signal_desc = "KUAT", "Bandar akumulasi + teknikal bullish"
    elif "DISTRIBUSI" in broker_result["status"] and "DOWNTREND" in trend:
        signal, signal_desc = "KUAT", "Bandar distribusi + teknikal bearish"
    elif "AKUMULASI" in broker_result["status"] and ("SIDEWAYS" in trend or "UPTREND LEMAH" in trend):
        signal, signal_desc = "SEDANG", "Akumulasi terdeteksi, tren belum konfirmasi penuh"
    elif "DISTRIBUSI" in broker_result["status"] and ("SIDEWAYS" in trend or "DOWNTREND LEMAH" in trend):
        signal, signal_desc = "SEDANG", "Distribusi terdeteksi, tren belum konfirmasi penuh"
    elif "LOW RISK" in broker_result["status"]:
        signal, signal_desc = "LEMAH", "Bukan pergerakan bandar utama"

    if signal == "KUAT" and "AKUMULASI" in broker_result["status"]:
        rec_text = "BUY — dengan analisis lanjutan. Sesuaikan risk/reward &amp; money management pribadi."
        rec_type = "green"
    elif signal == "KUAT" and "DISTRIBUSI" in broker_result["status"]:
        rec_text = "HINDARI / SELL WITH TRAILING STOP — potensi koreksi atau breakdown."
        rec_type = "red"
    elif signal == "SEDANG" and "AKUMULASI" in broker_result["status"]:
        rec_text = "PANTAU — beli hanya jika ada konfirmasi tren naik."
        rec_type = "amber"
    elif signal == "LEMAH":
        rec_text = "WAIT AND SEE — belum ada sinyal kuat."
        rec_type = "gray"
    else:
        rec_text = "Analisis lebih lanjut diperlukan."
        rec_type = "gray"

    status_kind = (
        "green" if "AKUMULASI" in broker_result["status"]
        else "red" if "DISTRIBUSI" in broker_result["status"]
        else "gray"
    )

    tab1, tab2, tab3 = st.tabs(["OVERVIEW", "BROKER SUMMARY", "TECHNICAL ANALYSIS"])

    # ==================== TAB 1: OVERVIEW ====================
    with tab1:
        st.markdown(f"#### {symbol.upper()}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("HARGA TERAKHIR", f"Rp {current_price:,.0f}")
        m2.metric("RSI (14)", f"{rsi:.1f}")
        m3.metric("VOLUME HARI INI", f"{volume_today:,}")
        vol_ratio = (volume_today / avg_volume_20 * 100) if avg_volume_20 else 0
        m4.metric("VS RATA-RATA 20D", f"{vol_ratio:.0f}%")

        colA, colB = st.columns(2)
        with colA:
            st.markdown(
                card("STATUS BANDAR",
                     badge(broker_result["status"], status_kind),
                     accent=status_kind),
                unsafe_allow_html=True,
            )
        with colB:
            sig_html = (
                f'{badge(signal, rec_type)} &nbsp; '
                f'<span style="color:var(--muted); font-size:0.85rem; '
                f'font-family:\'Inter\',sans-serif;">{signal_desc}</span>'
            )
            st.markdown(
                card("SINYAL & REKOMENDASI", sig_html,
                     note_html=rec_text, accent=rec_type),
                unsafe_allow_html=True,
            )

        st.markdown(
            card("TREN TEKNIKAL",
                 f'<span style="font-family:\'Inter\',sans-serif; font-size:0.95rem;">{trend}</span>',
                 accent="cyan"),
            unsafe_allow_html=True,
        )

    # ==================== TAB 2: BROKER SUMMARY ====================
    with tab2:
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown(
                card("STATUS", badge(broker_result["status"], status_kind),
                     accent=status_kind),
                unsafe_allow_html=True,
            )
            if broker_result["bandar_text"] != "-":
                st.markdown(
                    card("PELAKU UTAMA",
                         f'<span style="font-family:\'Inter\',sans-serif; font-size:0.92rem;">{broker_result["bandar_text"]}</span>'),
                    unsafe_allow_html=True,
                )
            st.markdown(
                card("RISIKO",
                     f'<span style="font-family:\'Inter\',sans-serif; font-size:0.92rem;">{broker_result["risk"]}</span>'),
                unsafe_allow_html=True,
            )
        with bc2:
            if broker_result["korban"] != "-":
                st.markdown(
                    card("KORBAN POTENSIAL",
                         f'<span style="font-family:\'Inter\',sans-serif; font-size:0.92rem;">{broker_result["korban"]}</span>'),
                    unsafe_allow_html=True,
                )
            if broker_result["avg_buy"] > 0:
                st.markdown(
                    card("RATA-RATA HARGA BELI (1 TAHUN)",
                         f"Rp {broker_result['avg_buy']:,.2f}",
                         accent="green"),
                    unsafe_allow_html=True,
                )
            if broker_result["avg_sell"] > 0:
                st.markdown(
                    card("RATA-RATA HARGA JUAL (1 TAHUN)",
                         f"Rp {broker_result['avg_sell']:,.2f}",
                         accent="red"),
                    unsafe_allow_html=True,
                )

    # ==================== TAB 3: TECHNICAL ====================
    with tab3:
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("MA5",  f"{ma5:,.0f}")
        t2.metric("MA10", f"{ma10:,.0f}")
        t3.metric("MA20", f"{ma20:,.0f}")
        t4.metric("MA50", f"{ma50:,.0f}")

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
            row_heights=[0.55, 0.2, 0.25],
            subplot_titles=("HARGA & MOVING AVERAGE", "VOLUME", "RSI (14)"),
        )
        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=hist["Close"],
            name="Price",
            increasing_line_color=COLORS["green"],
            decreasing_line_color=COLORS["red"],
            increasing_fillcolor=COLORS["green"],
            decreasing_fillcolor=COLORS["red"],
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=tech_result["ma5"],
                                 name="MA5",  line=dict(color=COLORS["amber"], width=1.2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=tech_result["ma10"],
                                 name="MA10", line=dict(color=COLORS["cyan"],  width=1.2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=tech_result["ma20"],
                                 name="MA20", line=dict(color="#a89078",        width=1.2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=tech_result["ma50"],
                                 name="MA50", line=dict(color="#1a1c1f",        width=1.2)),
                      row=1, col=1)

        vol_colors = [COLORS["green"] if c >= o else COLORS["red"]
                      for c, o in zip(hist["Close"], hist["Open"])]
        fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], name="Volume",
                             marker_color=vol_colors, opacity=0.85),
                      row=2, col=1)

        fig.add_trace(go.Scatter(x=hist.index, y=tech_result["rsi"],
                                 name="RSI",
                                 line=dict(color=COLORS["amber"], width=1.4)),
                      row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color=COLORS["red"],
                      opacity=0.55, row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=COLORS["green"],
                      opacity=0.55, row=3, col=1)

        if sr_data["resistance"]:
            for r in sr_data["resistance"]:
                fig.add_hline(y=r, line_dash="dash",
                              line_color=COLORS["red"], opacity=0.30,
                              row=1, col=1)
        if sr_data["support"]:
            for s in sr_data["support"]:
                fig.add_hline(y=s, line_dash="dash",
                              line_color=COLORS["green"], opacity=0.30,
                              row=1, col=1)

        fig.update_layout(
            height=760,
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            paper_bgcolor=COLORS["bg"],
            plot_bgcolor=COLORS["panel"],
            font=dict(family="Inter, sans-serif", color=COLORS["text"], size=11),
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor=COLORS["border"], borderwidth=1,
                font=dict(size=10, color=COLORS["muted"]),
            ),
            margin=dict(l=10, r=10, t=45, b=10),
        )
        fig.update_xaxes(gridcolor=COLORS["border"], showline=False, zeroline=False)
        fig.update_yaxes(gridcolor=COLORS["border"], showline=False, zeroline=False)
        for ann in fig.layout.annotations:
            ann.font = dict(family="Inter, sans-serif", size=10, color=COLORS["muted"])
        st.plotly_chart(fig, use_container_width=True)

        colv, colr = st.columns(2)
        with colv:
            st.markdown(
                card("ANALISIS VOLUME",
                     f'<span style="font-family:\'Inter\',sans-serif; font-size:0.88rem;">{vol_analysis}</span>'),
                unsafe_allow_html=True,
            )
        with colr:
            st.markdown(
                card("ANALISIS RSI",
                     f'<span style="font-family:\'Inter\',sans-serif; font-size:0.88rem;">{rsi_analysis}</span>'),
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="card-label" style="margin-top:0.6rem;">SUPPORT &amp; RESISTANCE</div>',
            unsafe_allow_html=True,
        )
        sr_c1, sr_c2 = st.columns(2)
        with sr_c1:
            if sr_data["resistance"]:
                rows_html = "".join(
                    f"<tr><td>Resistance {i+1}</td>"
                    f"<td style='color:var(--red);'>{r:,.2f}</td></tr>"
                    for i, r in enumerate(sr_data["resistance"])
                )
            else:
                rows_html = "<tr><td colspan='2' style='color:var(--muted);'>Tidak terdeteksi</td></tr>"
            st.markdown(f"<table class='sr-table'>{rows_html}</table>",
                        unsafe_allow_html=True)
        with sr_c2:
            if sr_data["support"]:
                rows_html = "".join(
                    f"<tr><td>Support {i+1}</td>"
                    f"<td style='color:var(--green);'>{s:,.2f}</td></tr>"
                    for i, s in enumerate(sr_data["support"])
                )
            else:
                rows_html = "<tr><td colspan='2' style='color:var(--muted);'>Tidak terdeteksi</td></tr>"
            st.markdown(f"<table class='sr-table'>{rows_html}</table>",
                        unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="card" style="text-align:center; padding:3.5rem 1.5rem;">
        <div class="card-label" style="font-size:0.72rem;">STATUS</div>
        <div class="card-value" style="font-size:1rem; color:var(--muted);
             font-family:'Inter',sans-serif; font-weight:400;">
            Masukkan data broker summary &amp; kode saham di sidebar,
            lalu jalankan analisis.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="footer-note">'
    'Xrobotrade Analyst &middot; Data: Yahoo Finance &middot; '
    'Dibuat oleh <a href="https://github.com/mukhlisinniamillah-ai" target="_blank">'
    'Mukhlisin Niamillah</a> &middot; Untuk tujuan edukasi, bukan saran investasi resmi'
    '</div>',
    unsafe_allow_html=True,
)