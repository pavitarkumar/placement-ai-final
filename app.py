"""
app.py
------
AI-Powered Student Placement Prediction System
Mobile-responsive · Dark glassmorphism UI · Authentication · Plotly charts

Pages (protected after login):
  1. Dashboard        5. Model Evaluation
  2. Data Explorer    6. PCA & Clustering
  3. Visualizations   7. Predict
  4. Train & Tune
"""
from firebase.auth import (
    login_user,
    create_user,
)
from firebase.firebase_config import auth
import os, sys, json, hashlib, warnings
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocessing import (
    load_data, generate_sample_data, get_missing_summary,
    handle_missing_values, prepare_features_target, encode_and_scale,
    validate_columns, NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL, ALL_FEATURE_COLS,
)
from train_model import (
    train_with_gridsearch, evaluate_model, get_feature_importances, MODELS_DIR,
)

warnings.filterwarnings("ignore")

# ── Chart constants ────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_PLACED    = "#3b82f6"
COLOR_NOTPLACED = "#ef4444"
COLOR_ACCENT    = "#8b5cf6"
COLOR_GREEN     = "#10b981"
COLOR_YELLOW    = "#f59e0b"
CHART_BG = PAPER_BG = "rgba(0,0,0,0)"
MODEL_COLORS = [COLOR_PLACED, COLOR_ACCENT, COLOR_GREEN, COLOR_YELLOW]
MODEL_FILLS  = ["rgba(59,130,246,0.15)", "rgba(139,92,246,0.15)",
                "rgba(16,185,129,0.15)", "rgba(245,158,11,0.15)"]

# ── Auth constants ─────────────────────────────────────────────────────────────
USERS_FILE = os.path.join(BASE_DIR, "models", "users.json")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Placement Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS — dark theme + auth UI + mobile responsive
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Variables ── */
:root {
  --bg:          #070b14;
  --bg2:         #0d1526;
  --glass:       rgba(255,255,255,0.04);
  --glass-b:     rgba(255,255,255,0.08);
  --glass-h:     rgba(255,255,255,0.07);
  --blue:   #3b82f6; --purple: #8b5cf6;
  --green:  #10b981; --red:    #ef4444; --yellow: #f59e0b;
  --t1: #f1f5f9; --t2: #94a3b8; --t3: #475569;
}

/* ── Base ── */
html, body, [data-testid="stApp"] {
  background: var(--bg) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--t1) !important;
  overflow-x: hidden !important;
}

/* ── Remove horizontal scroll ── */
* { box-sizing: border-box !important; }

/* ════════════════════════════════════════════════════
   HEADER / NAVBAR — fix the black bar overlap
   Streamlit's header is position:fixed at the top.
   We must (a) style it to match our theme and
   (b) give block-container enough padding-top so
   content is never hidden behind it.
   ════════════════════════════════════════════════════ */

/* 1. Style the sticky header bar */
[data-testid="stHeader"] {
  background: rgba(7,11,20,.92) !important;
  backdrop-filter: blur(20px) !important;
  -webkit-backdrop-filter: blur(20px) !important;
  border-bottom: 1px solid rgba(255,255,255,.06) !important;
  box-shadow: 0 1px 24px rgba(0,0,0,.4) !important;
  z-index: 1000 !important;
}

/* 2. Remove the rainbow decoration strip at the very top */
[data-testid="stDecoration"] { display: none !important; }

/* 3. Hide toolbar action buttons (deploy, share, ⋮ menu)
      while keeping the sidebar hamburger ≡ intact       */
[data-testid="stToolbarActions"] { visibility: hidden !important; width: 0 !important; }
button[data-testid="baseButton-header"]  { display: none !important; }

/* 4. Sidebar collapsed control stays accessible on mobile */
[data-testid="stSidebarCollapsedControl"] { z-index: 1001 !important; }

/* ── Block container — proper top clearance for the header ── */
/* Streamlit header is ~3.75rem (60px) tall.  We add a bit of
   breathing room on top so the first widget is never clipped.  */
.main .block-container {
  max-width: 100% !important;
  overflow-x: hidden !important;
  /* top: header height (≈3.75rem) + extra breathing (1rem) */
  padding: 4.75rem 2rem 3rem !important;
}

/* Sidebar top clearance */
[data-testid="stSidebar"] > div:first-child {
  padding-top: 4rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a1020 0%, #070b14 100%) !important;
  border-right: 1px solid var(--glass-b) !important;
  min-width: 240px !important;
}
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif !important; }

/* ── Glass card ── */
.glass-card {
  background: var(--glass); border: 1px solid var(--glass-b);
  border-radius: 16px; padding: 24px;
  backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  transition: all .3s; margin-bottom: 16px;
}
.glass-card:hover { background: var(--glass-h); border-color: rgba(59,130,246,.2); }

/* ── KPI card ── */
.kpi-card {
  background: var(--glass); border: 1px solid var(--glass-b);
  border-radius: 16px; padding: 20px 16px; text-align: center;
  position: relative; overflow: hidden; transition: all .3s;
  height: 100%;
}
.kpi-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0;
  height: 3px; border-radius: 16px 16px 0 0;
}
.kpi-card.blue::before   { background: linear-gradient(90deg,#3b82f6,#60a5fa); }
.kpi-card.purple::before { background: linear-gradient(90deg,#8b5cf6,#a78bfa); }
.kpi-card.green::before  { background: linear-gradient(90deg,#10b981,#34d399); }
.kpi-card.yellow::before { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
.kpi-card.red::before    { background: linear-gradient(90deg,#ef4444,#f87171); }
.kpi-icon  { font-size: 1.6rem; margin-bottom: 8px; }
.kpi-value { font-size: 1.8rem; font-weight: 800; color: var(--t1); line-height: 1; }
.kpi-label { font-size: .68rem; color: var(--t3); text-transform: uppercase;
             letter-spacing: .08em; margin-top: 6px; font-weight: 600; }
.kpi-delta { font-size: .75rem; margin-top: 6px; font-weight: 600; }
.kpi-delta.up      { color: var(--green); }
.kpi-delta.neutral { color: var(--t3); }

/* ── KPI row wrapper ── */
.kpi-row {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;
  margin-bottom: 24px;
}

/* ── Section header ── */
.section-header {
  display: flex; align-items: center; gap: 10px;
  margin: 24px 0 16px; padding-bottom: 12px; border-bottom: 1px solid var(--glass-b);
}
.section-header h2 { font-size: 1rem; font-weight: 700; color: var(--t1); margin: 0; }
.section-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--blue);
               box-shadow: 0 0 8px var(--blue); flex-shrink: 0; }

/* ── Hero banner ── */
.hero-banner {
  background: linear-gradient(135deg,rgba(59,130,246,.12),rgba(139,92,246,.08) 50%,rgba(7,11,20,0));
  border: 1px solid rgba(59,130,246,.2); border-radius: 20px;
  padding: 40px 48px; margin-bottom: 28px; position: relative; overflow: hidden;
}
.hero-banner::after {
  content: ''; position: absolute; top: -50%; right: -10%;
  width: 400px; height: 400px;
  background: radial-gradient(circle,rgba(59,130,246,.08),transparent 70%); pointer-events: none;
}
.hero-title {
  font-size: 2rem; font-weight: 800; line-height: 1.2; margin: 0 0 10px;
  background: linear-gradient(135deg,#f1f5f9,#93c5fd);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub { font-size: .95rem; color: var(--t2); margin: 0 0 20px; }
.badge {
  display: inline-block; background: rgba(59,130,246,.12);
  border: 1px solid rgba(59,130,246,.25); color: #93c5fd;
  border-radius: 20px; padding: 3px 10px; font-size: .68rem;
  font-weight: 600; margin: 3px; letter-spacing: .02em;
}

/* ── Metric override ── */
[data-testid="metric-container"] {
  background: var(--glass); border: 1px solid var(--glass-b);
  border-radius: 14px; padding: 14px 16px;
}
[data-testid="metric-container"] label {
  color: var(--t3) !important; font-size: .68rem !important;
  text-transform: uppercase; letter-spacing: .06em;
}
[data-testid="stMetricValue"] { color: var(--blue) !important;
  font-size: 1.5rem !important; font-weight: 800 !important; }

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg,#2563eb,#1d4ed8) !important; color: white !important;
  border: none !important; border-radius: 10px !important; padding: 10px 20px !important;
  font-weight: 600 !important; font-size: .88rem !important;
  transition: all .2s !important; box-shadow: 0 4px 14px rgba(37,99,235,.35) !important;
  width: 100% !important;
}
.stButton > button:hover { transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(37,99,235,.5) !important; }
.stDownloadButton > button {
  background: linear-gradient(135deg,#059669,#047857) !important; color: white !important;
  border: none !important; border-radius: 10px !important; font-weight: 600 !important;
  box-shadow: 0 4px 14px rgba(5,150,105,.3) !important; width: 100% !important;
}
.stProgress > div > div { background: linear-gradient(90deg,#3b82f6,#8b5cf6) !important;
  border-radius: 4px !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--glass); border-radius: 12px; padding: 4px; gap: 4px;
  border: 1px solid var(--glass-b); flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px;
  color: var(--t2); font-weight: 500; font-size: .82rem; }
.stTabs [aria-selected="true"] { background: var(--blue) !important; color: white !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
  background: var(--glass) !important; border: 1px solid var(--glass-b) !important;
  border-radius: 10px !important; color: var(--t1) !important; font-weight: 600 !important;
}

/* ── Alerts ── */
.stAlert { border-radius: 12px !important; }
.stSuccess { background: rgba(16,185,129,.1)  !important; border-color: rgba(16,185,129,.3) !important; }
.stWarning { background: rgba(245,158,11,.1)  !important; border-color: rgba(245,158,11,.3) !important; }
.stError   { background: rgba(239,68,68,.1)   !important; border-color: rgba(239,68,68,.3)  !important; }
.stInfo    { background: rgba(59,130,246,.1)   !important; border-color: rgba(59,130,246,.3) !important; }

/* ── Sidebar identity card ── */
.sidebar-logo { text-align: center; padding: 16px 10px 20px;
  border-bottom: 1px solid var(--glass-b); margin-bottom: 14px; }
.sidebar-logo .logo-icon  { font-size: 2.2rem; display: block; margin-bottom: 6px; }
.sidebar-logo .logo-title { font-size: .9rem; font-weight: 700; color: var(--t1); }
.sidebar-logo .logo-sub   { font-size: .65rem; color: var(--t3); margin-top: 2px;
  letter-spacing: .08em; text-transform: uppercase; }
.status-card {
  background: var(--glass); border: 1px solid var(--glass-b);
  border-radius: 12px; padding: 10px 12px; font-size: .75rem; margin-top: 8px;
}
.status-card .st { font-weight: 700; margin-bottom: 4px; }
.status-card .sr { color: var(--t3); }
.status-card .sv { color: var(--t1); font-weight: 500; }

/* ── Prediction result ── */
.result-placed {
  background: linear-gradient(135deg,rgba(16,185,129,.15),rgba(5,150,105,.08));
  border: 1px solid rgba(16,185,129,.35); border-radius: 18px;
  padding: 28px 20px; text-align: center; margin: 16px 0;
}
.result-not-placed {
  background: linear-gradient(135deg,rgba(239,68,68,.15),rgba(185,28,28,.08));
  border: 1px solid rgba(239,68,68,.35); border-radius: 18px;
  padding: 28px 20px; text-align: center; margin: 16px 0;
}
.result-icon  { font-size: 2.6rem; display: block; margin-bottom: 8px; }
.result-label { font-size: 1.6rem; font-weight: 800; margin-bottom: 4px; }
.result-conf  { font-size: .9rem; color: var(--t2); }
.best-model-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: linear-gradient(135deg,rgba(245,158,11,.2),rgba(234,179,8,.1));
  border: 1px solid rgba(245,158,11,.4); border-radius: 20px;
  padding: 5px 14px; font-size: .8rem; font-weight: 700; color: #fbbf24; margin: 10px 0;
}

/* ── Animated gradient line ── */
@keyframes shimmer { 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }
.gradient-line {
  height: 2px;
  background: linear-gradient(90deg,#3b82f6,#8b5cf6,#10b981,#f59e0b,#3b82f6);
  background-size: 400% 400%; animation: shimmer 5s ease infinite;
  border-radius: 2px; margin: 0 0 24px;
}

/* ── Footer ── */
.footer { text-align: center; color: var(--t3); font-size: .68rem;
  padding: 24px 0 8px; border-top: 1px solid var(--glass-b);
  margin-top: 48px; letter-spacing: .02em; }

/* ════════════════════════════════════════════════════
   AUTH PAGE STYLES
   ════════════════════════════════════════════════════ */
.auth-wrapper {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; min-height: 80vh; padding: 24px 16px;
}
.auth-logo-ring {
  width: 80px; height: 80px; border-radius: 50%; margin: 0 auto 20px;
  background: linear-gradient(135deg,#3b82f6,#8b5cf6);
  display: flex; align-items: center; justify-content: center;
  font-size: 2.2rem; box-shadow: 0 0 40px rgba(59,130,246,.35);
  animation: glow 3s ease-in-out infinite;
}
@keyframes glow {
  0%,100% { box-shadow: 0 0 30px rgba(59,130,246,.35); }
  50%      { box-shadow: 0 0 60px rgba(139,92,246,.5); }
}
.auth-card {
  width: 100%; max-width: 420px;
  background: rgba(13,21,38,.85);
  border: 1px solid rgba(59,130,246,.25);
  border-radius: 24px; padding: 40px 36px;
  backdrop-filter: blur(30px); -webkit-backdrop-filter: blur(30px);
  box-shadow: 0 25px 80px rgba(0,0,0,.5), 0 0 0 1px rgba(255,255,255,.04);
  position: relative; overflow: hidden;
}
.auth-card::before {
  content: '';
  position: absolute; top: -60%; left: -30%;
  width: 250px; height: 250px; border-radius: 50%;
  background: radial-gradient(circle,rgba(59,130,246,.08),transparent 70%);
  pointer-events: none;
}
.auth-title { font-size: 1.6rem; font-weight: 800; color: var(--t1);
  margin: 0 0 6px; text-align: center; }
.auth-sub   { font-size: .85rem; color: var(--t2);
  text-align: center; margin: 0 0 28px; }
.auth-divider {
  display: flex; align-items: center; gap: 12px; margin: 20px 0;
  color: var(--t3); font-size: .75rem;
}
.auth-divider::before, .auth-divider::after {
  content: ''; flex: 1; height: 1px; background: var(--glass-b);
}
.auth-switch {
  text-align: center; font-size: .82rem; color: var(--t3); margin-top: 20px;
}
.auth-switch a { color: var(--blue); font-weight: 600; cursor: pointer; }
.auth-features {
  display: flex; justify-content: center; gap: 16px; margin-top: 24px; flex-wrap: wrap;
}
.auth-feat {
  display: flex; align-items: center; gap: 5px;
  font-size: .72rem; color: var(--t3);
}
.auth-user-badge {
  display: flex; align-items: center; gap: 8px;
  background: var(--glass); border: 1px solid var(--glass-b);
  border-radius: 20px; padding: 6px 14px; font-size: .8rem;
  color: var(--t2); margin-bottom: 6px;
}
.auth-user-badge .avatar {
  width: 26px; height: 26px; border-radius: 50%;
  background: linear-gradient(135deg,#3b82f6,#8b5cf6);
  display: flex; align-items: center; justify-content: center;
  font-size: .7rem; font-weight: 700; color: white; flex-shrink: 0;
}

/* Input overrides for auth */
.auth-card .stTextInput > div > div {
  background: rgba(255,255,255,.04) !important;
  border: 1px solid rgba(255,255,255,.1) !important;
  border-radius: 10px !important; color: var(--t1) !important;
}
.auth-card .stTextInput > div > div:focus-within {
  border-color: rgba(59,130,246,.5) !important;
  box-shadow: 0 0 0 3px rgba(59,130,246,.1) !important;
}
.auth-card .stTextInput label { color: var(--t2) !important; font-size: .82rem !important; }
.auth-card .stForm { background: transparent !important; border: none !important; padding: 0 !important; }
.auth-card [data-testid="stFormSubmitButton"] > button {
  background: linear-gradient(135deg,#2563eb,#7c3aed) !important;
  border: none !important; border-radius: 12px !important;
  padding: 12px !important; font-size: .95rem !important; font-weight: 700 !important;
  letter-spacing: .02em !important; width: 100% !important;
  box-shadow: 0 8px 24px rgba(37,99,235,.4) !important;
  transition: all .2s !important;
}
.auth-card [data-testid="stFormSubmitButton"] > button:hover {
  transform: translateY(-2px) !important; box-shadow: 0 12px 32px rgba(37,99,235,.55) !important;
}

/* ════════════════════════════════════════════════════
   MOBILE RESPONSIVE
   Every breakpoint must preserve padding-top ≥ header
   height so content never slides behind the navbar.
   Header height reference:
     desktop  ≈ 60px  (3.75rem)
     tablet   ≈ 52px  (3.25rem)
     mobile   ≈ 48px  (3.00rem)
   ════════════════════════════════════════════════════ */

/* ── Tablet (≤1024px) ── */
@media (max-width: 1024px) {
  .main .block-container {
    /* keep header clearance: 3.25rem + 1rem breathing = 4.25rem */
    padding: 4.25rem 1.25rem 2rem !important;
  }
  [data-testid="stSidebar"] > div:first-child { padding-top: 3.5rem !important; }
  .hero-banner { padding: 28px 28px !important; }
  .hero-title  { font-size: 1.6rem !important; }
  .kpi-row     { grid-template-columns: repeat(3, 1fr) !important; }
}

/* ── Mobile (≤768px) — full stack ── */
@media (max-width: 768px) {
  /* Header clearance: 3.00rem + 0.75rem breathing = 3.75rem */
  .main .block-container {
    padding: 3.75rem .875rem 2rem !important;
  }
  [data-testid="stSidebar"] > div:first-child { padding-top: 3rem !important; }

  /* Hero section — compact */
  .hero-banner {
    padding: 18px 16px !important;
    border-radius: 14px !important;
    margin-bottom: 16px !important;
  }
  .hero-title  { font-size: 1.25rem !important; line-height: 1.25 !important; }
  .hero-sub    { font-size: .80rem !important; margin-bottom: 10px !important; }
  .badge       { font-size: .60rem !important; padding: 2px 7px !important; margin: 2px !important; }
  .gradient-line { margin: 0 0 16px !important; }

  /* Stack ALL st.columns */
  div[data-testid="stHorizontalBlock"] {
    flex-direction: column !important;
    gap: 0 !important;
  }
  div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
  div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div {
    width: 100% !important;
    min-width: 100% !important;
    flex: 1 1 100% !important;
    padding-left: 0  !important;
    padding-right: 0 !important;
  }

  /* KPI grid — 2 columns on mobile */
  .kpi-row   { grid-template-columns: repeat(2, 1fr) !important; gap: 8px !important;
               margin-bottom: 16px !important; }
  .kpi-card  { padding: 14px 10px !important; border-radius: 12px !important; }
  .kpi-value { font-size: 1.4rem !important; }
  .kpi-icon  { font-size: 1.2rem !important; margin-bottom: 4px !important; }
  .kpi-label { font-size: .58rem !important; }

  /* Glass cards */
  .glass-card { padding: 16px 12px !important; border-radius: 12px !important;
                margin-bottom: 10px !important; }

  /* Section header */
  .section-header { margin: 14px 0 10px !important; padding-bottom: 8px !important; }
  .section-header h2 { font-size: .88rem !important; }

  /* Tabs — horizontal scroll, no wrap */
  .stTabs [data-baseweb="tab-list"] {
    flex-wrap: nowrap !important; overflow-x: auto !important;
    -webkit-overflow-scrolling: touch !important;
    scrollbar-width: none !important;
    gap: 2px !important; padding: 3px !important;
  }
  .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none !important; }
  .stTabs [data-baseweb="tab"] { font-size: .76rem !important; padding: 6px 10px !important;
    white-space: nowrap !important; }

  /* Metrics */
  [data-testid="metric-container"] { padding: 10px 12px !important; border-radius: 12px !important; }
  [data-testid="stMetricValue"]    { font-size: 1.15rem !important; }

  /* Forms — full-width, touch-friendly */
  .stTextInput > div > div  { border-radius: 10px !important; font-size: .9rem !important; }
  .stSelectbox > div > div  { border-radius: 10px !important; }
  [data-testid="stFormSubmitButton"] > button { font-size: .88rem !important; padding: 11px !important; }
  .stSlider { width: 100% !important; }

  /* Buttons — touch-friendly tap targets */
  .stButton > button { padding: 11px 16px !important; font-size: .88rem !important;
    border-radius: 10px !important; min-height: 44px !important; }
  .stDownloadButton > button { min-height: 44px !important; }

  /* Result cards */
  .result-placed, .result-not-placed { padding: 20px 14px !important; border-radius: 14px !important; }
  .result-label { font-size: 1.25rem !important; }
  .result-icon  { font-size: 2rem !important; }
  .best-model-badge { font-size: .75rem !important; padding: 4px 10px !important; }

  /* Tables — allow horizontal scroll inside, not the whole page */
  [data-testid="stDataFrame"] { overflow-x: auto !important; -webkit-overflow-scrolling: touch !important; }
  [data-testid="stDataFrame"] iframe { max-width: 100% !important; }

  /* Sidebar */
  [data-testid="stSidebar"] { min-width: 0 !important; }
  .status-card { font-size: .72rem !important; }
  .auth-user-badge { font-size: .75rem !important; }

  /* Footer */
  .footer { font-size: .62rem !important; margin-top: 24px !important; padding-top: 16px !important; }
}

/* ── Small phone (≤480px) ── */
@media (max-width: 480px) {
  /* Header clearance stays; just tighten side padding */
  .main .block-container {
    padding: 3.75rem .625rem 1.5rem !important;
  }

  /* Hero even more compact */
  .hero-banner  { padding: 14px 12px !important; border-radius: 12px !important; }
  .hero-title   { font-size: 1.1rem !important; }
  .hero-sub     { font-size: .75rem !important; }

  /* KPI grid stays 2-col but tighter */
  .kpi-row      { gap: 6px !important; }
  .kpi-card     { padding: 12px 8px !important; }
  .kpi-value    { font-size: 1.25rem !important; }
  .kpi-label    { font-size: .56rem !important; }
  .kpi-icon     { font-size: 1.1rem !important; }

  /* Text sizing */
  .section-header h2 { font-size: .82rem !important; }
  p, li, .stMarkdown   { font-size: .88rem !important; }

  /* Auth card edge-to-edge */
  .auth-card { padding: 20px 14px !important; border-radius: 14px !important; }

  /* Bigger tap targets */
  .stButton > button, .stDownloadButton > button {
    min-height: 48px !important;
    font-size: .88rem !important;
  }
  [data-testid="stFormSubmitButton"] > button { min-height: 48px !important; }
}

/* ── Very small (≤360px, narrow Android) ── */
@media (max-width: 360px) {
  .main .block-container { padding: 3.75rem .5rem 1.25rem !important; }
  .hero-title   { font-size: 1rem !important; }
  .kpi-value    { font-size: 1.1rem !important; }
  .section-header h2 { font-size: .78rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH UTILITIES
# ══════════════════════════════════════════════════════════════════════════════




# ══════════════════════════════════════════════════════════════════════════════
#  AUTH STATE
# ══════════════════════════════════════════════════════════════════════════════
for _k, _v in [("logged_in", False), ("username", ""), ("auth_tab", "login")]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH PAGE
# ══════════════════════════════════════════════════════════════════════════════

def _show_auth() -> None:
    """Render login / signup page. Uses st-native elements so widgets render correctly."""

    # ── Auth-specific CSS ────────────────────────────────────────────────────
    st.markdown("""
    <style>
      /* Hide sidebar entirely on auth page */
      [data-testid="stSidebar"],
      [data-testid="stSidebarCollapsedControl"] { display: none !important; }

      /* Deep-space background with soft radial glows */
      [data-testid="stApp"] {
        background:
          radial-gradient(ellipse at 25% 20%, rgba(59,130,246,.1) 0%, transparent 55%),
          radial-gradient(ellipse at 75% 80%, rgba(139,92,246,.08) 0%, transparent 55%),
          #070b14 !important;
      }

      /* Turn the block-container itself into the glass card */
      .main .block-container {
        max-width: 460px !important;
        padding: 2.5rem 2rem !important;
        margin: 4vh auto 0 !important;
        background: rgba(13,21,38,.92) !important;
        border: 1px solid rgba(59,130,246,.22) !important;
        border-radius: 24px !important;
        box-shadow: 0 30px 90px rgba(0,0,0,.55), 0 0 60px rgba(59,130,246,.07) !important;
        backdrop-filter: blur(32px) !important;
        -webkit-backdrop-filter: blur(32px) !important;
      }

      /* Input fields on auth page */
      .stTextInput > div > div {
        background: rgba(255,255,255,.04) !important;
        border: 1px solid rgba(255,255,255,.1) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
      }
      .stTextInput > div > div:focus-within {
        border-color: rgba(59,130,246,.55) !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,.12) !important;
      }
      .stTextInput label { color: #94a3b8 !important; font-size: .82rem !important; }

      /* Submit button gradient */
      [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg,#2563eb,#7c3aed) !important;
        border: none !important; border-radius: 12px !important;
        font-size: .95rem !important; font-weight: 700 !important;
        padding: 11px !important; letter-spacing: .02em !important;
        box-shadow: 0 8px 24px rgba(37,99,235,.4) !important;
        transition: all .2s !important; width: 100% !important;
        color: white !important;
      }
      [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(37,99,235,.55) !important;
      }

      /* Auth tabs */
      .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,.04) !important;
        border-radius: 12px !important; padding: 4px !important;
        border: 1px solid rgba(255,255,255,.07) !important;
        margin-bottom: 20px !important;
      }
      .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important; font-weight: 600 !important;
        font-size: .85rem !important; color: #64748b !important;
      }
      .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg,#2563eb,#7c3aed) !important;
        color: white !important;
      }

      /* Mobile: full-width card */
      @media (max-width: 520px) {
        .main .block-container {
          margin: 0 !important; border-radius: 0 !important;
          min-height: 100vh !important; padding: 1.5rem 1rem !important;
          border-left: none !important; border-right: none !important;
        }
      }
    </style>
    """, unsafe_allow_html=True)

    # ── Logo & branding ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:10px 0 24px">
      <div style="display:inline-flex;align-items:center;justify-content:center;
           width:76px;height:76px;border-radius:50%;
           background:linear-gradient(135deg,#3b82f6,#8b5cf6);
           font-size:2.2rem;
           box-shadow:0 0 0 8px rgba(59,130,246,.12),0 0 40px rgba(59,130,246,.35);
           animation:glow 3s ease-in-out infinite;margin-bottom:16px">
        🎓
      </div>
      <div style="font-size:1.55rem;font-weight:800;color:#f1f5f9;
           background:linear-gradient(135deg,#f1f5f9,#93c5fd);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           background-clip:text">
        Placement AI
      </div>
      <div style="font-size:.78rem;color:#475569;margin-top:5px;letter-spacing:.05em;
           text-transform:uppercase">
        AI-Powered Campus Placement Predictor
      </div>
    </div>
    """, unsafe_allow_html=True)

       # ── Tabs: Sign In | Register ─────────────────────────────────────────────
    t_login, t_signup = st.tabs(["🔐  Sign In", "✨  Create Account"])

    # ── LOGIN TAB ────────────────────────────────────────────────────────────
    with t_login:

        with st.form("login_form", clear_on_submit=False):

            username = st.text_input(
                "Username",
                placeholder="Enter your username"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password"
            )

            submit = st.form_submit_button(
                "Sign In →",
                use_container_width=True
            )

        # =========================================================
        # LOGIN LOGIC
        # =========================================================

        if submit:

            if not username or not password:

                st.error(
                    "Please fill in all fields."
                )

            else:

                success, user = login_user(
                    username,
                    password,
                )

                if success:

                    st.session_state.logged_in = True

                    st.session_state.username = (
                        username.strip().lower()
                    )

                    st.success(
                        f"✅ Welcome back, {user['name']}!"
                    )

                    st.rerun()

                else:

                    st.error(
                        "❌ Invalid username or password."
                    )

        # =========================================================
        # FORGOT PASSWORD
        # =========================================================

        st.markdown(
            """
            <div style='margin-top:14px'></div>
            """,
            unsafe_allow_html=True
        )

        forgot_email = st.text_input(
            "Reset Password Email",
            placeholder="Enter your registered email",
            key="forgot_email"
        )

        if st.button(
            "📩 Send Password Reset Email",
            use_container_width=True,
            key="forgot_password"
        ):

            if not forgot_email:

                st.warning(
                    "Please enter your email."
                )

            else:

                try:

                    auth.send_password_reset_email(
                        forgot_email
                    )

                    st.success(
                        "✅ Password reset email sent successfully!"
                    )

                except Exception:

                    st.error(
                        "❌ Failed to send reset email."
                    )

    # ── SIGNUP TAB ───────────────────────────────────────────────────────────
    with t_signup:

        with st.form("signup_form", clear_on_submit=True):

            full_name = st.text_input(
                "Full Name",
                placeholder="Your full name"
            )

            email = st.text_input(
                "Email",
                placeholder="you@example.com"
            )

            username = st.text_input(
                "Username",
                placeholder="Choose a username"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="At least 6 characters"
            )

            confirm = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Repeat password"
            )

            submit = st.form_submit_button(
                "Create Account →",
                use_container_width=True
            )

        if submit:

            if not all([
                full_name,
                email,
                username,
                password,
                confirm
            ]):

                st.error(
                    "Please fill in all fields."
                )

            elif len(password) < 6:

                st.error(
                    "Password must be at least 6 characters."
                )

            elif password != confirm:

                st.error(
                    "Passwords do not match."
                )

            else:

                success, message = create_user(
                    full_name,
                    email,
                    username,
                    password,
                )

                if success:

                    st.success(message)

                else:

                    st.error(message)
    # ── Feature badges ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;justify-content:center;gap:20px;margin-top:24px;flex-wrap:wrap">
      <div style="display:flex;align-items:center;gap:5px;font-size:.72rem;color:#334155">🔒 Secure</div>
      <div style="display:flex;align-items:center;gap:5px;font-size:.72rem;color:#334155">🤖 4 ML Models</div>
      <div style="display:flex;align-items:center;gap:5px;font-size:.72rem;color:#334155">📊 Charts</div>
      <div style="display:flex;align-items:center;gap:5px;font-size:.72rem;color:#334155">📱 Mobile</div>
    </div>
    """, unsafe_allow_html=True)


# ── Auth gate ──────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    _show_auth()
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE (main app)
# ══════════════════════════════════════════════════════════════════════════════
_STATE_DEFAULTS = {
    "df_raw": None, "df_clean": None,
    "X_train": None, "X_test":  None,
    "y_train": None, "y_test":  None,
    "encoders": None, "scaler":  None,
    "feature_names": None, "target_le": None,
    "train_results": None, "eval_results": None,
}
for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Auto-load pre-trained bundle ──────────────────────────────────────────────
_SAVED_PATH = os.path.join(BASE_DIR, "models", "session_artifacts.joblib")
if st.session_state.train_results is None and os.path.exists(_SAVED_PATH):
    try:
        _bundle = joblib.load(_SAVED_PATH)
        for _k, _v in _bundle.items():
            if _k in _STATE_DEFAULTS:
                st.session_state[_k] = _v
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def plotly_layout(fig, title: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=title, font=dict(size=13, color="#f1f5f9", family="Inter"), x=0, pad=dict(l=4)),
        plot_bgcolor=CHART_BG, paper_bgcolor=PAPER_BG,
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 8, b=10),
        font=dict(family="Inter", color="#94a3b8", size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)",
                    borderwidth=1, font=dict(size=10)),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    )
    return fig


def section(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="section-header"><span class="section-dot"></span>'
        f'<h2>{icon}&nbsp; {title}</h2></div>',
        unsafe_allow_html=True,
    )


def kpi(icon: str, value, label: str, color: str = "blue") -> str:
    return (
        f'<div class="kpi-card {color}">'
        f'<div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'</div>'
    )


def best_model_name() -> str | None:
    er = st.session_state.eval_results
    return max(er, key=lambda n: er[n]["accuracy"]) if er else None


@st.cache_data(show_spinner=False)
def _compute_pca(X: np.ndarray, n: int = 2):
    p = PCA(n_components=n, random_state=42)
    return p.fit_transform(X), p.explained_variance_ratio_


@st.cache_data(show_spinner=False)
def _compute_pca_full(X: np.ndarray):
    p = PCA(random_state=42); p.fit(X)
    return p.explained_variance_ratio_


@st.cache_data(show_spinner=False)
def _kmeans_inertias(X: np.ndarray, k_max: int = 10):
    return [KMeans(n_clusters=ki, random_state=42, n_init=10).fit(X).inertia_ for ki in range(2, k_max+1)]


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # User identity
    
    _disp_name = st.session_state.username

    _initials = "".join(
    w[0].upper()
    for w in _disp_name.split()[:2]
    )

    st.markdown(
        f'<div class="sidebar-logo">'
        f'<span class="logo-icon">🎓</span>'
        f'<div class="logo-title">Placement AI</div>'
        f'<div class="logo-sub">Dashboard v3.0</div>'
        f'</div>'
        f'<div class="auth-user-badge" style="margin-bottom:14px">'
        f'<div class="avatar">{_initials}</div>'
        f'<div><div style="color:#f1f5f9;font-weight:600;font-size:.8rem">{_disp_name}</div>'
        f'<div style="font-size:.66rem;color:#475569">@{st.session_state.username}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "nav",
        ["🏠  Dashboard", "📂  Data Explorer", "📊  Visualizations",
         "🤖  Train & Tune", "📈  Model Evaluation",
         "🔬  PCA & Clustering", "🎯  Predict"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Dataset status
    _df_sb = st.session_state.df_raw
    if _df_sb is not None:
        _placed_n    = int((_df_sb[TARGET_COL] == "Placed").sum()) if TARGET_COL in _df_sb.columns else "—"
        _notplaced_n = int((_df_sb[TARGET_COL] == "Not Placed").sum()) if TARGET_COL in _df_sb.columns else "—"
        st.markdown(
            f'<div class="status-card">'
            f'<div class="st" style="color:#3b82f6">✅ Dataset Loaded</div>'
            f'<div class="sr">Rows: <span class="sv">{_df_sb.shape[0]:,}</span></div>'
            f'<div class="sr">Placed: <span class="sv">{_placed_n}</span></div>'
            f'<div class="sr">Not Placed: <span class="sv">{_notplaced_n}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-card">'
            '<div class="st" style="color:#ef4444">⚠ No Dataset</div>'
            '<div class="sr" style="color:#64748b">Upload CSV to begin</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    _bm = best_model_name()
    if _bm:
        _bm_acc = st.session_state.eval_results[_bm]["accuracy"] * 100
        st.markdown(
            f'<div class="status-card" style="margin-top:6px">'
            f'<div class="st" style="color:#10b981">✅ Models Trained</div>'
            f'<div class="sr">Best: <span class="sv">{_bm}</span></div>'
            f'<div class="sr">Accuracy: <span class="sv">{_bm_acc:.1f}%</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Logout button
    if st.button("🚪 Sign Out", use_container_width=True, key="logout_btn"):
        st.session_state.logged_in = False
        st.session_state.username  = ""
        st.rerun()

    st.markdown(
        '<div style="text-align:center;color:#1e293b;font-size:.6rem;margin-top:12px">'
        'Streamlit · Scikit-learn · Plotly'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":
    st.markdown(
        '<div class="hero-banner">'
        '<div class="hero-title">🎓 AI Student Placement<br>Prediction Dashboard</div>'
        '<div class="hero-sub">End-to-end ML pipeline · Hyperparameter Tuning · Interactive Insights</div>'
        '<div>'
        '<span class="badge">Logistic Regression</span>'
        '<span class="badge">KNN</span>'
        '<span class="badge">Random Forest</span>'
        '<span class="badge">Gradient Boosting</span>'
        '<span class="badge">GridSearchCV</span>'
        '<span class="badge">Cross-Validation</span>'
        '<span class="badge">PCA</span>'
        '<span class="badge">KMeans</span>'
        '</div></div>'
        '<div class="gradient-line"></div>',
        unsafe_allow_html=True,
    )

    _df  = st.session_state.df_raw
    _er  = st.session_state.eval_results
    _tr  = st.session_state.train_results
    _bm  = best_model_name()

    total_s    = _df.shape[0] if _df is not None else 0
    placed_c   = int((_df[TARGET_COL] == "Placed").sum()) if (_df is not None and TARGET_COL in _df.columns) else 0
    place_rate = f"{placed_c / total_s * 100:.1f}%" if total_s else "—"
    best_acc   = f"{_er[_bm]['accuracy']*100:.1f}%" if _bm else "—"
    n_models   = len(_tr) if _tr else 0

    st.markdown(
        '<div class="kpi-row">'
        + kpi("👥", f"{total_s:,}", "Total Students", "blue")
        + kpi("✅", placed_c if total_s else "—", "Placed", "green")
        + kpi("📊", place_rate, "Placement Rate", "purple")
        + kpi("🏆", best_acc, "Best Accuracy", "yellow")
        + kpi("⚡", f"{n_models}/4", "Models Trained", "blue" if n_models == 4 else "red")
        + '</div>',
        unsafe_allow_html=True,
    )

    if _df is not None and TARGET_COL in _df.columns:
        col_a, col_b = st.columns(2)
        with col_a:
            section("📊", "Placement Distribution")
            vc  = _df[TARGET_COL].value_counts()
            fig = go.Figure(go.Pie(
                labels=vc.index.tolist(), values=vc.values.tolist(), hole=0.55,
                marker=dict(colors=[COLOR_PLACED, COLOR_NOTPLACED], line=dict(color="#070b14", width=3)),
                textinfo="label+percent", textfont=dict(size=11, family="Inter"),
            ))
            fig.add_annotation(text=f"{sum(vc.values.tolist())}<br>students", x=0.5, y=0.5,
                               showarrow=False, font=dict(size=16, color="#f1f5f9", family="Inter"))
            plotly_layout(fig, height=300)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"responsive": True}
            )

        with col_b:
            section("📈", "CGPA vs Placement")
            if "CGPA" in _df.columns:
                fig = px.box(
                    _df, x=TARGET_COL, y="CGPA", color=TARGET_COL,
                    color_discrete_map={"Placed": COLOR_PLACED, "Not Placed": COLOR_NOTPLACED},
                    points="outliers",
                )
                plotly_layout(fig, height=300)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    if _er and _bm:
        section("🤖", "Model Performance Overview")
        _names = list(_er.keys())
        _df_m  = pd.DataFrame({
            "Model":     _names,
            "Accuracy":  [_er[n]["accuracy"]  for n in _names],
            "F1 Score":  [_er[n]["f1"]        for n in _names],
            "Precision": [_er[n]["precision"] for n in _names],
            "Recall":    [_er[n]["recall"]    for n in _names],
        })
        fig = go.Figure()
        for i, metric in enumerate(["Accuracy", "F1 Score", "Precision", "Recall"]):
            fig.add_trace(go.Bar(
                name=metric, x=_names, y=_df_m[metric],
                marker_color=MODEL_COLORS[i], opacity=0.85,
                text=[f"{v:.3f}" for v in _df_m[metric]],
                textposition="outside", textfont=dict(size=9),
            ))
        plotly_layout(fig, "Model Comparison — All Metrics", height=320)
        fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0.05)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"responsive": True}
        )
        st.markdown(
            f'<div class="best-model-badge">🏆 Best: {_bm} &nbsp;|&nbsp; {_er[_bm]["accuracy"]*100:.2f}% accuracy</div>',
            unsafe_allow_html=True,
        )

    section("🚀", "How to Use")
    _steps = [
        ("01","📂","Upload Data","Upload CSV or generate 500-row sample."),
        ("02","📊","Explore","Study distributions & correlations."),
        ("03","🤖","Train","Auto-tune 4 ML models with GridSearchCV."),
        ("04","📈","Evaluate","Compare Acc, F1, ROC-AUC, confusion matrices."),
        ("05","🎯","Predict","Instant placement prediction for any student."),
    ]
    _scols = st.columns(5)
    for _c, (_num, _ico, _t, _d) in zip(_scols, _steps):
        _c.markdown(
            f'<div class="glass-card" style="text-align:center;padding:18px 12px;min-height:160px">'
            f'<div style="font-size:.6rem;color:#3b82f6;font-weight:800;letter-spacing:.1em;margin-bottom:4px">{_num}</div>'
            f'<div style="font-size:1.6rem;margin-bottom:6px">{_ico}</div>'
            f'<div style="font-size:.82rem;font-weight:700;color:#f1f5f9;margin-bottom:4px">{_t}</div>'
            f'<div style="font-size:.7rem;color:#64748b;line-height:1.4">{_d}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂  Data Explorer":
    section("📂", "Upload & Explore Dataset")

    uploaded = st.file_uploader("Upload student placement CSV", type=["csv"])

    if uploaded is None:
        st.info("No file uploaded — generate a synthetic 500-row sample dataset.")
        if st.button("⚡ Generate 500-Row Sample Dataset"):
            with st.spinner("Generating…"):
                df_gen = generate_sample_data(500)
            st.session_state.df_raw   = df_gen
            st.session_state.df_clean = None
            st.success("✅ Sample dataset generated!")
            st.rerun()
    else:
        try:
            df_up = load_data(uploaded)
            missing_cols = validate_columns(df_up, require_target=False)
            if missing_cols:
                st.warning(f"⚠ Missing expected columns: `{', '.join(missing_cols)}`")
            st.session_state.df_raw   = df_up
            st.session_state.df_clean = None
            st.success(f"✅ Loaded: **{df_up.shape[0]:,} rows × {df_up.shape[1]} columns**")
        except Exception as exc:
            st.error(f"Failed to read CSV: {exc}"); st.stop()

    _df = st.session_state.df_raw
    if _df is None:
        st.stop()

    if st.session_state.df_clean is None:
        st.session_state.df_clean = handle_missing_values(_df)

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Preview", "📊 Statistics", "🩺 Missing Values", "📥 Download"])

    with tab1:
        n_rows = st.slider("Rows to show", 5, min(200, len(_df)), 15)
        st.dataframe(_df.head(n_rows), use_container_width=True, hide_index=True)

    with tab2:
        _c1, _c2 = st.columns([3, 2])
        with _c1:
            st.markdown("**Numerical Summary**")
            st.dataframe(_df.describe().T.round(3), use_container_width=True)
        with _c2:
            st.markdown("**Categorical Columns**")
            for _cn in [c for c in CATEGORICAL_COLS + [TARGET_COL] if c in _df.columns]:
                _vc = _df[_cn].value_counts().reset_index()
                _vc.columns = [_cn, "Count"]
                _vc["Pct"] = (_vc["Count"] / len(_df) * 100).round(1).astype(str) + "%"
                st.dataframe(_vc, use_container_width=True, hide_index=True)

    with tab3:
        _ms = get_missing_summary(_df)
        if _ms.empty:
            st.success("✅ No missing values found.")
        else:
            st.warning(f"Found missing values in **{len(_ms)}** column(s).")
            st.dataframe(_ms, use_container_width=True)
            _strat = st.selectbox("Fill strategy for numerical columns", ["median", "mean"])
            if st.button("Fix Missing Values"):
                st.session_state.df_clean = handle_missing_values(_df, strategy=_strat)
                st.success("✅ Missing values handled.")

    with tab4:
        _dl = st.session_state.df_clean if st.session_state.df_clean is not None else _df
        st.download_button("⬇ Download Cleaned CSV", _dl.to_csv(index=False).encode(),
                           "cleaned_dataset.csv", "text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Visualizations":
    _df = st.session_state.df_raw
    if _df is None:
        st.warning("⚠ Upload a dataset first."); st.stop()

    section("📊", "Interactive Data Visualizations")
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔥 Correlation", "🎻 Violin Plots", "🗂 Categorical"])

    with tab1:
        _num_cols = [c for c in NUMERICAL_COLS if c in _df.columns]
        _nr = -(-len(_num_cols) // 4)
        fig = make_subplots(rows=_nr, cols=4, subplot_titles=_num_cols,
                            horizontal_spacing=0.06, vertical_spacing=0.12)
        for i, _cn in enumerate(_num_cols):
            r, c = divmod(i, 4)
            if TARGET_COL in _df.columns:
                for status, color in {"Placed": COLOR_PLACED, "Not Placed": COLOR_NOTPLACED}.items():
                    grp = _df[_df[TARGET_COL] == status][_cn].dropna()
                    fig.add_trace(go.Histogram(x=grp, name=status, marker_color=color,
                                               opacity=0.65, showlegend=(i == 0), nbinsx=20),
                                  row=r+1, col=c+1)
            else:
                fig.add_trace(go.Histogram(x=_df[_cn].dropna(), marker_color=COLOR_PLACED,
                                           opacity=0.8, nbinsx=20), row=r+1, col=c+1)
        plotly_layout(fig, "Feature Distributions", height=620)
        fig.update_layout(barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        _num_df = _df[[c for c in NUMERICAL_COLS if c in _df.columns]].dropna()
        _corr   = _num_df.corr().round(3)
        _mask   = np.triu(np.ones_like(_corr, dtype=bool), k=1)
        fig = go.Figure(go.Heatmap(
            z=_corr.where(~_mask).values,
            x=_corr.columns.tolist(), y=_corr.index.tolist(),
            colorscale="RdBu_r", zmid=0,
            text=_corr.where(~_mask).round(2).values, texttemplate="%{text}",
            textfont=dict(size=9), hoverongaps=False,
            colorbar=dict(thickness=10, len=0.75),
        ))
        plotly_layout(fig, "Feature Correlation Heatmap", height=480)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if TARGET_COL in _df.columns:
            _sel = st.selectbox("Select feature", [c for c in NUMERICAL_COLS if c in _df.columns])
            fig = px.violin(_df, y=_sel, x=TARGET_COL, color=TARGET_COL, box=True, points="outliers",
                            color_discrete_map={"Placed": COLOR_PLACED, "Not Placed": COLOR_NOTPLACED})
            plotly_layout(fig, f"{_sel} by Placement Status", height=400)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Target column not found.")

    with tab4:
        _cats = [c for c in CATEGORICAL_COLS if c in _df.columns]
        if _cats and TARGET_COL in _df.columns:
            _sel_c = st.selectbox("Select categorical feature", _cats)
            _ct    = pd.crosstab(_df[_sel_c], _df[TARGET_COL]).reset_index()
            _ct_l  = _ct.melt(id_vars=_sel_c, var_name="Status", value_name="Count")
            fig = px.bar(_ct_l, x=_sel_c, y="Count", color="Status", barmode="group",
                         color_discrete_map={"Placed": COLOR_PLACED, "Not Placed": COLOR_NOTPLACED},
                         text="Count")
            plotly_layout(fig, f"{_sel_c} vs Placement Status", height=400)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Categorical columns or target not available.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: TRAIN & TUNE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Train & Tune":
    _df = st.session_state.df_raw
    if _df is None:
        st.warning("⚠ Upload a dataset first."); st.stop()

    section("⚙", "Model Training Configuration")

    _c1, _c2 = st.columns(2)
    with _c1:
        test_size = st.slider("Test split (%)", 10, 40, 20) / 100
    with _c2:
        cv_folds  = st.slider("CV folds", 3, 10, 5)

    if st.button("🚀 Train All Models with GridSearchCV"):
        if "df_clean" in st.session_state and st.session_state.df_clean is not None:
            _source_df = st.session_state.df_clean
        else:
            _source_df = _df

        _clean = handle_missing_values(_source_df)
        _miss  = validate_columns(_clean, require_target=True)
        if _miss:
            st.error(f"Missing required columns: `{', '.join(_miss)}`"); st.stop()

        with st.spinner("Preparing features…"):
            try:
                X, y, encoders, scaler, feature_names, target_le = prepare_features_target(_clean)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y)
                for _k, _v in [("X_train", X_train), ("X_test", X_test),
                                ("y_train", y_train), ("y_test", y_test),
                                ("encoders", encoders), ("scaler", scaler),
                                ("feature_names", feature_names), ("target_le", target_le)]:
                    st.session_state[_k] = _v
            except Exception as exc:
                st.error(f"Preprocessing error: {exc}"); st.stop()

        _pb  = st.progress(0)
        _txt = st.empty()

        def _cb(name, step, total):
            _pb.progress(int((step-1)/total*100))
            _txt.info(f"Training **{name}** ({step}/{total})…")

        with st.spinner("Running GridSearchCV across 4 models (~60s)…"):
            try:
                results = train_with_gridsearch(X_train, y_train, cv_folds=cv_folds, progress_callback=_cb)
            except Exception as exc:
                st.error(f"Training error: {exc}"); st.stop()

        _pb.progress(100)
        _txt.success("✅ All models trained and saved!")

        eval_results = {n: evaluate_model(r["model"], X_test, y_test) for n, r in results.items()}
        st.session_state.train_results = results
        st.session_state.eval_results  = eval_results

        try:
            joblib.dump({"df_raw": _clean, "df_clean": _clean,
                         "X_train": X_train, "X_test": X_test,
                         "y_train": y_train, "y_test": y_test,
                         "encoders": encoders, "scaler": scaler,
                         "feature_names": feature_names, "target_le": target_le,
                         "train_results": results, "eval_results": eval_results}, _SAVED_PATH)
        except Exception:
            pass

        section("📋", "Training Summary")
        _rows = []
        for _nm, _res in results.items():
            _ev = eval_results[_nm]
            _rows.append({"Model": _nm,
                          "CV Acc (mean±std)": f"{_res['cv_mean']:.4f} ± {_res['cv_std']:.4f}",
                          "Test Acc": f"{_ev['accuracy']:.4f}", "Precision": f"{_ev['precision']:.4f}",
                          "Recall": f"{_ev['recall']:.4f}", "F1": f"{_ev['f1']:.4f}",
                          "ROC-AUC": f"{_ev['roc_auc']:.4f}" if _ev.get("roc_auc") else "—",
                          "Best Params": str(_res["best_params"])})
        st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
        _bm = best_model_name()
        st.markdown(
            f'<div class="best-model-badge">🏆 Best: {_bm} — {eval_results[_bm]["accuracy"]*100:.2f}% accuracy</div>',
            unsafe_allow_html=True)

    elif st.session_state.train_results is not None:
        st.success("✅ Models already trained. See **Model Evaluation** for results.")
        if st.button("🔄 Retrain Models"):
            st.session_state.train_results = None
            st.session_state.eval_results  = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Model Evaluation":
    if st.session_state.train_results is None:
        st.warning("⚠ Train models first."); st.stop()

    _results = st.session_state.train_results
    _er      = st.session_state.eval_results
    _names   = list(_er.keys())
    _bm      = best_model_name()

    section("🏆", "Performance Summary")
    st.markdown(
        f'<div class="glass-card" style="border-color:rgba(245,158,11,.3);background:rgba(245,158,11,.05)">'
        f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">'
        f'<span style="font-size:1.8rem">🏆</span>'
        f'<div><div style="font-size:1rem;font-weight:800;color:#fbbf24">Best Model: {_bm}</div>'
        f'<div style="color:#94a3b8;font-size:.8rem;margin-top:4px">'
        f'Accuracy <b style="color:#f1f5f9">{_er[_bm]["accuracy"]*100:.2f}%</b> &nbsp;|&nbsp;'
        f'F1 <b style="color:#f1f5f9">{_er[_bm]["f1"]*100:.2f}%</b> &nbsp;|&nbsp;'
        f'Precision <b style="color:#f1f5f9">{_er[_bm]["precision"]*100:.2f}%</b> &nbsp;|&nbsp;'
        f'Recall <b style="color:#f1f5f9">{_er[_bm]["recall"]*100:.2f}%</b>'
        + (f' &nbsp;|&nbsp; ROC-AUC <b style="color:#f1f5f9">{_er[_bm]["roc_auc"]:.4f}</b>' if _er[_bm].get("roc_auc") else "")
        + '</div></div></div></div>', unsafe_allow_html=True)

    for _nm in _names:
        _ev = _er[_nm]
        st.markdown(
            f'<div style="color:#60a5fa;font-weight:700;font-size:.9rem;margin:14px 0 8px">'
            f'{_nm}{"  🏆" if _nm == _bm else ""}</div>',
            unsafe_allow_html=True)
        _mc1, _mc2, _mc3, _mc4, _mc5 = st.columns(5)
        _mc1.metric("Accuracy",  f"{_ev['accuracy']:.4f}")
        _mc2.metric("Precision", f"{_ev['precision']:.4f}")
        _mc3.metric("Recall",    f"{_ev['recall']:.4f}")
        _mc4.metric("F1 Score",  f"{_ev['f1']:.4f}")
        _mc5.metric("ROC-AUC",   f"{_ev['roc_auc']:.4f}" if _ev.get("roc_auc") else "—")

    section("📡", "Radar Chart — Model Profiles")
    _radar_m = ["Accuracy", "Precision", "Recall", "F1 Score"]
    fig = go.Figure()
    for i, _nm in enumerate(_names):
        _ev  = _er[_nm]
        vals = [_ev["accuracy"], _ev["precision"], _ev["recall"], _ev["f1"]]
        fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=_radar_m+[_radar_m[0]],
                                      fill="toself", name=_nm,
                                      line_color=MODEL_COLORS[i%len(MODEL_COLORS)],
                                      fillcolor=MODEL_FILLS[i%len(MODEL_FILLS)], opacity=0.85))
    plotly_layout(fig, "Model Performance Radar", height=380)
    fig.update_layout(polar=dict(bgcolor="rgba(255,255,255,0.02)",
        radialaxis=dict(visible=True, range=[0,1], gridcolor="rgba(255,255,255,0.07)", tickfont=dict(size=8)),
        angularaxis=dict(gridcolor="rgba(255,255,255,0.07)")))
    st.plotly_chart(fig, use_container_width=True)

    section("🔢", "Confusion Matrices")
    _cm_cols = st.columns(min(len(_names), 2))  # max 2 per row on mobile
    for idx, (col, _nm) in enumerate(zip(_cm_cols * 10, _names)):
        _cm = _er[_nm]["confusion_matrix"]
        _fig = go.Figure(go.Heatmap(
            z=_cm[::-1], x=["Pred Not Placed","Pred Placed"],
            y=["Actual Placed","Actual Not Placed"],
            colorscale=[[0,"#0d1526"],[1,"#3b82f6"]],
            text=_cm[::-1], texttemplate="%{text}",
            textfont=dict(size=14, color="white"), showscale=False))
        plotly_layout(_fig, f"{_nm}{' 🏆' if _nm == _bm else ''}", height=260)
        col.plotly_chart(_fig, use_container_width=True)

    section("📉", "Cross-Validation Scores")
    fig = go.Figure()
    for i, _nm in enumerate(_names):
        _s = _results[_nm]["cv_scores"]
        fig.add_trace(go.Bar(name=_nm, x=[f"Fold {j+1}" for j in range(len(_s))], y=_s,
                             marker_color=MODEL_COLORS[i%len(MODEL_COLORS)], opacity=0.85,
                             text=[f"{s:.3f}" for s in _s], textposition="outside", textfont=dict(size=9)))
    plotly_layout(fig, "StratifiedKFold CV Accuracy", height=340)
    fig.update_layout(barmode="group", yaxis_range=[0, 1.12])
    st.plotly_chart(fig, use_container_width=True)

    _tree_m = [n for n in ["Gradient Boosting","Random Forest"] if n in _results]
    if _tree_m and st.session_state.feature_names:
        section("⭐", "Feature Importances")
        _fi_tabs = st.tabs(_tree_m)
        for _tab, _mname in zip(_fi_tabs, _tree_m):
            with _tab:
                _fi_df = get_feature_importances(_results[_mname]["model"], st.session_state.feature_names)
                if not _fi_df.empty:
                    fig = px.bar(_fi_df, x="Importance", y="Feature", orientation="h",
                                 color="Importance",
                                 color_continuous_scale=[[0,"#1e3a5f"],[.5,"#3b82f6"],[1,"#93c5fd"]],
                                 text=_fi_df["Importance"].round(4))
                    plotly_layout(fig, f"{_mname} Importances", height=380)
                    fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
                    fig.update_traces(textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)

    section("📋", "Classification Reports")
    for _nm in _names:
        with st.expander(f"Report — {_nm}" + (" 🏆" if _nm == _bm else "")):
            st.code(_er[_nm]["report"], language="")

    section("📊", "Full Comparison Table")
    _comp = []
    for _nm in _names:
        _ev  = _er[_nm]; _res = _results[_nm]
        _comp.append({"Model": _nm+(" ★" if _nm == _bm else ""),
                      "Accuracy":  round(_ev["accuracy"],  4),
                      "Precision": round(_ev["precision"], 4),
                      "Recall":    round(_ev["recall"],    4),
                      "F1 Score":  round(_ev["f1"],        4),
                      "ROC-AUC":   round(_ev["roc_auc"], 4) if _ev.get("roc_auc") else "—",
                      "CV Mean":   round(_res["cv_mean"],  4),
                      "CV Std":    round(_res["cv_std"],   4)})
    _df_comp = pd.DataFrame(_comp)
    st.dataframe(_df_comp, use_container_width=True, hide_index=True)
    st.download_button("⬇ Download Comparison CSV", _df_comp.to_csv(index=False).encode(),
                       "model_comparison.csv", "text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: PCA & CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬  PCA & Clustering":
    if st.session_state.X_train is None:
        st.warning("⚠ Train models first."); st.stop()

    X_all = np.vstack([st.session_state.X_train, st.session_state.X_test])
    y_all = np.concatenate([st.session_state.y_train, st.session_state.y_test])
    _tle  = st.session_state.target_le

    section("🔭", "Principal Component Analysis")
    X_2d, ev_2d = _compute_pca(X_all, 2)
    labels = _tle.inverse_transform(y_all)

    fig = px.scatter(pd.DataFrame({"PC1": X_2d[:,0], "PC2": X_2d[:,1], "Status": labels}),
                     x="PC1", y="PC2", color="Status", opacity=0.7,
                     color_discrete_map={"Placed": COLOR_PLACED, "Not Placed": COLOR_NOTPLACED},
                     hover_data={"PC1":":.3f","PC2":":.3f"})
    plotly_layout(fig, f"PCA 2D  (PC1: {ev_2d[0]*100:.1f}%,  PC2: {ev_2d[1]*100:.1f}%)", height=440)
    fig.update_traces(marker=dict(size=5, line=dict(width=0)))
    st.plotly_chart(fig, use_container_width=True)

    _pm1, _pm2 = st.columns(2)
    _pm1.metric("PC1 Variance", f"{ev_2d[0]*100:.2f}%")
    _pm2.metric("PC2 Variance", f"{ev_2d[1]*100:.2f}%")

    section("📊", "Scree Plot")
    ev_full = _compute_pca_full(X_all)
    cum_var = np.cumsum(ev_full)
    cr      = list(range(1, len(ev_full)+1))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=cr, y=ev_full, name="Individual", marker_color=COLOR_PLACED, opacity=0.75), secondary_y=False)
    fig.add_trace(go.Scatter(x=cr, y=cum_var, name="Cumulative",
                             line=dict(color=COLOR_YELLOW, width=2.5), mode="lines+markers",
                             marker=dict(size=5)), secondary_y=True)
    plotly_layout(fig, "Scree Plot", height=340)
    fig.update_yaxes(title_text="Individual Variance", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative", secondary_y=True, range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

    section("🔵", "KMeans Clustering")
    k = st.slider("Number of clusters (k)", 2, 8, 3)
    _km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_2d)
    _df_km = pd.DataFrame({"PC1": X_2d[:,0], "PC2": X_2d[:,1],
                            "Cluster": [f"Cluster {c+1}" for c in _km.labels_]})
    fig = px.scatter(_df_km, x="PC1", y="PC2", color="Cluster",
                     color_discrete_sequence=px.colors.qualitative.Bold, opacity=0.65)
    fig.add_trace(go.Scatter(x=_km.cluster_centers_[:,0], y=_km.cluster_centers_[:,1],
                             mode="markers", name="Centroids",
                             marker=dict(symbol="x", size=14, color="white",
                                         line=dict(width=2, color="white"))))
    plotly_layout(fig, f"KMeans (k={k}) on PCA Space", height=440)
    st.plotly_chart(fig, use_container_width=True)

    section("📐", "Elbow Method")
    _ine = _kmeans_inertias(X_2d)
    fig  = go.Figure(go.Scatter(x=list(range(2,11)), y=_ine, mode="lines+markers",
                                line=dict(color=COLOR_PLACED, width=2.5),
                                marker=dict(size=8, color=COLOR_YELLOW, line=dict(width=2, color=COLOR_PLACED)),
                                fill="tozeroy", fillcolor="rgba(59,130,246,0.07)"))
    plotly_layout(fig, "Elbow Method — Inertia vs k", height=320)
    fig.update_xaxes(title_text="Clusters (k)", dtick=1)
    fig.update_yaxes(title_text="Inertia (WCSS)")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯  Predict":
    if st.session_state.train_results is None:
        st.warning("⚠ Train models first."); st.stop()

    section("🎯", "Student Placement Predictor")
    _bm = best_model_name()
    st.markdown(
        f'<div style="color:#94a3b8;font-size:.82rem;margin-bottom:16px">'
        f'Auto-selected: <b style="color:#fbbf24">{_bm}</b> '
        f'({st.session_state.eval_results[_bm]["accuracy"]*100:.1f}% accuracy)</div>',
        unsafe_allow_html=True)

    tab_single, tab_batch = st.tabs(["👤 Single Prediction", "📋 Batch Prediction"])

    with tab_single:
        with st.form("predict_form"):
            _f1, _f2, _f3 = st.columns(3)
            with _f1:
                st.markdown("**👤 Personal Info**")
                age    = st.number_input("Age", 18, 35, 22)
                gender = st.selectbox("Gender", ["Male","Female"])
                degree = st.selectbox("Degree", ["B.Tech","BCA","MCA","MBA","B.Sc","Other"])
                branch = st.selectbox("Branch", ["CS","IT","ECE","Mechanical","Civil","Other"])
                cgpa   = st.number_input("CGPA (0–10)", 0.0, 10.0, 7.5, 0.1)
            with _f2:
                st.markdown("**💼 Experience**")
                internships    = st.number_input("Internships",   0, 10, 1)
                projects       = st.number_input("Projects",      0, 20, 2)
                certifications = st.number_input("Certifications",0, 20, 2)
                backlogs       = st.number_input("Backlogs",      0, 20, 0)
            with _f3:
                st.markdown("**🧠 Skills**")
                coding_skills = st.slider("Coding (1–10)",     1, 10, 6)
                communication = st.slider("Communication (1–10)", 1, 10, 6)
                aptitude      = st.slider("Aptitude (0–100)",  0, 100, 65)
                soft_skills   = st.slider("Soft Skills (1–10)",1, 10, 6)

            _mk = list(st.session_state.train_results.keys())
            model_choice = st.selectbox("Model", _mk, index=_mk.index(_bm))
            submitted = st.form_submit_button("🔮 Predict Placement", use_container_width=True)

        if submitted:
            _inp = pd.DataFrame([{
                "Age": age, "Gender": gender, "Degree": degree, "Branch": branch,
                "CGPA": cgpa, "Internships": internships, "Projects": projects,
                "Coding_Skills": coding_skills, "Communication": communication,
                "Aptitude_Test": aptitude, "Soft_Skills": soft_skills,
                "Certifications": certifications, "Backlogs": backlogs,
            }])
            try:
                X_in, _, _ = encode_and_scale(_inp, fit=False,
                                              encoders=st.session_state.encoders,
                                              scaler=st.session_state.scaler)
            except Exception as exc:
                st.error(f"Preprocessing error: {exc}"); st.stop()

            _model = st.session_state.train_results[model_choice]["model"]
            _pred  = _model.predict(X_in)[0]
            try:
               _proba = _model.predict_proba(X_in)[0]
            except Exception:
               _proba = None
            _tle   = st.session_state.target_le
            _res   = _tle.inverse_transform([_pred])[0]
            _placed = _res == "Placed"
            _conf  = float(_proba.max()) * 100 if _proba is not None else None
            _cs    = f"Confidence: {_conf:.1f}%" if _conf else ""

            if _placed:
                st.markdown(f'<div class="result-placed"><span class="result-icon">✅</span>'
                            f'<div class="result-label" style="color:#34d399">PLACED</div>'
                            f'<div class="result-conf">{_cs}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-not-placed"><span class="result-icon">❌</span>'
                            f'<div class="result-label" style="color:#f87171">NOT PLACED</div>'
                            f'<div class="result-conf">{_cs}</div></div>', unsafe_allow_html=True)

            if _proba is not None:
                _classes = _tle.classes_
                _pc1, _pc2 = st.columns(2)
                with _pc1:
                    fig = go.Figure(go.Bar(x=_classes.tolist(), y=(_proba*100).tolist(),
                                          marker=dict(color=[COLOR_GREEN if c=="Placed" else COLOR_NOTPLACED for c in _classes], line=dict(width=0)),
                                          text=[f"{p*100:.1f}%" for p in _proba], textposition="outside",
                                          textfont=dict(size=12, color="#f1f5f9")))
                    plotly_layout(fig, f"Probability — {model_choice}", height=300)
                    fig.update_yaxes(range=[0, 115], title_text="Probability (%)")
                    st.plotly_chart(fig, use_container_width=True)
                with _pc2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta", value=_conf,
                        number=dict(suffix="%", font=dict(size=32, color="#f1f5f9")),
                        delta=dict(reference=50, increasing_color=COLOR_GREEN, decreasing_color=COLOR_NOTPLACED),
                        gauge=dict(axis=dict(range=[0,100], tickwidth=1, tickcolor="#475569"),
                                   bar=dict(color=COLOR_GREEN if _placed else COLOR_NOTPLACED),
                                   bgcolor="rgba(255,255,255,0.03)", borderwidth=1,
                                   bordercolor="rgba(255,255,255,0.1)",
                                   steps=[dict(range=[0,50], color="rgba(239,68,68,0.1)"),
                                          dict(range=[50,100], color="rgba(16,185,129,0.1)")],
                                   threshold=dict(line=dict(color=COLOR_YELLOW, width=2), thickness=0.8, value=50)),
                        title=dict(text="Confidence Score", font=dict(size=12, color="#94a3b8"))))
                    plotly_layout(fig, "", height=300)
                    st.plotly_chart(fig, use_container_width=True)

            _pd2 = _inp.copy()
            _pd2["Prediction"] = _res; _pd2["Model_Used"] = model_choice
            if _proba is not None:
                for _cls, _p in zip(_tle.classes_, _proba):
                    _pd2[f"P({_cls})"] = round(float(_p), 4)
            st.download_button("⬇ Download Prediction Report",
                               _pd2.to_csv(index=False).encode(),
                               "placement_prediction.csv", "text/csv", use_container_width=True)

    with tab_batch:
        st.info("Upload a CSV with the same feature columns (no Placement_Status needed).")
        _bfile = st.file_uploader("Upload batch CSV", type=["csv"], key="batch_upload")
        if _bfile is not None:
            _bm_key = list(st.session_state.train_results.keys())
            _bm_sel = st.selectbox("Model for batch", _bm_key, index=_bm_key.index(_bm), key="bsel")
            if st.button("🔮 Run Batch Prediction"):
                try:
                    _dfb = pd.read_csv(_bfile)
                    with st.spinner(f"Predicting {len(_dfb)} students…"):
                        X_b, _, _ = encode_and_scale(_dfb, fit=False,
                                                     encoders=st.session_state.encoders,
                                                     scaler=st.session_state.scaler)
                    _mb   = st.session_state.train_results[_bm_sel]["model"]
                    _pb   = _mb.predict(X_b)
                    _dfb["Prediction"] = st.session_state.target_le.inverse_transform(_pb)
                    _dfb["Model_Used"] = _bm_sel
                    if hasattr(_mb, "predict_proba"):
                        _pbp = _mb.predict_proba(X_b)
                        for _i, _cls in enumerate(st.session_state.target_le.classes_):
                            _dfb[f"P({_cls})"] = _pbp[:,_i].round(4)
                    st.success(f"✅ Predicted {len(_dfb)} students.")
                    st.dataframe(_dfb.head(25), use_container_width=True, hide_index=True)
                    st.download_button("⬇ Download Batch CSV", _dfb.to_csv(index=False).encode(),
                                       "batch_predictions.csv", "text/csv", use_container_width=True)
                except Exception as exc:
                    st.error(f"Batch prediction error: {exc}")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    '🎓 AI-Powered Student Placement Dashboard &nbsp;·&nbsp;'
    'Streamlit · Scikit-learn · Plotly<br>'
    'LR &nbsp;·&nbsp; KNN &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp; Gradient Boosting &nbsp;·&nbsp; GridSearchCV &nbsp;·&nbsp; PCA &nbsp;·&nbsp; KMeans'
    '</div>',
    unsafe_allow_html=True,
)
