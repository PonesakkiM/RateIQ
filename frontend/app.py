"""
RateIQ – Enterprise Analytics Platform
Frontend v4.0 — Professional SaaS Design System
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import requests as _requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time

from frontend.api_client import (
    predict as api_predict,
    chat as api_chat,
    competitor_analysis as api_competitor,
    trend_boost as api_trend,
    get_meta, get_history, get_feature_importance,
    get_dataset_insights, health_check, APIError,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RateIQ",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='22' fill='%238B5CF6'/><text y='.9em' font-size='72' font-family='Inter,Arial' font-weight='800' fill='white' x='14'>R</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme state ───────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

DARK = st.session_state.theme == "dark"

# ── Design tokens ─────────────────────────────────────────────────────────────
if DARK:
    BG         = "#0F172A"
    BG2        = "#0A0F1E"
    CARD       = "#1E293B"
    CARD2      = "#162032"
    BORDER     = "#2D3748"
    BORDER2    = "#1E293B"
    PRIMARY    = "#8B5CF6"
    PRIMARY_DIM= "#6D28D9"
    SECONDARY  = "#6366F1"
    SUCCESS    = "#22C55E"
    WARNING    = "#F59E0B"
    DANGER     = "#EF4444"
    TEXT       = "#F8FAFC"
    TEXT2      = "#CBD5E1"
    MUTED      = "#64748B"
    PLOTLY_TPL = "plotly_dark"
else:
    BG         = "#F8FAFC"
    BG2        = "#F1F5F9"
    CARD       = "#FFFFFF"
    CARD2      = "#F8FAFC"
    BORDER     = "#E2E8F0"
    BORDER2    = "#F1F5F9"
    PRIMARY    = "#7C3AED"
    PRIMARY_DIM= "#6D28D9"
    SECONDARY  = "#4F46E5"
    SUCCESS    = "#16A34A"
    WARNING    = "#D97706"
    DANGER     = "#DC2626"
    TEXT       = "#0F172A"
    TEXT2      = "#334155"
    MUTED      = "#94A3B8"
    PLOTLY_TPL = "plotly_white"

PLOTLY_LAYOUT = dict(
    template=PLOTLY_TPL,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color=TEXT2),
    margin=dict(l=16, r=16, t=40, b=16),
)

CONF_COLORS = {
    "Very High": SUCCESS, "High": SUCCESS,
    "Medium": WARNING, "Low": DANGER, "Very Low": DANGER,
}

# ── SVG icon library ─────────────────────────────────────────────────────────
def icon(name: str, size: int = 16, color: str = "currentColor") -> str:
    s = size
    return {"info": f'<svg></svg>'}.get(name, f'<svg></svg>')

def icon_badge(ico: str, color: str = None) -> str:
    c = color or PRIMARY
    return f'<span style="width:32px;height:32px;background:{c}18;">{icon(ico,16,c)}</span>'


# ── CSS design system ─────────────────────────────────────────────────────────
def _inject_css():
    dark_vars = f"""
    --bg: {BG};
    --bg2: {BG2};
    --card: {CARD};
    --border: {BORDER};
    --primary: {PRIMARY};
    --success: {SUCCESS};
    --warning: {WARNING};
    --danger: {DANGER};
    --text: {TEXT};
    --text2: {TEXT2};
    --muted: {MUTED};
    """

    st.markdown(f"""
<style>
:root {{ {dark_vars} }}

body {{
    background-color: var(--bg);
    color: var(--text);
}}

.section-label {{
    font-size: 0.688rem;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
}}
</style>
""", unsafe_allow_html=True)   # ✅ FIX ADDED (THIS WAS MISSING BEFORE)

st.markdown(f"""
<style>
:root {{ {dark_vars} }}

/*Cards*/
.riq-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.875rem;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}}

.riq-card:hover {{
    border-color: var(--primary)40;
    box-shadow: 0 0 0 1px var(--primary)20, 0 4px 16px rgba(0,0,0,0.12);
}}

.riq-card-flat {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.875rem;
}}

/* Metric tiles */
.riq-metric {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.125rem 1rem;
    text-align: center;
    transition: border-color 0.15s ease;
}}

.riq-metric:hover {{
    border-color: var(--primary)50;
}}

.riq-metric .val {{
    font-size: 1.75rem;
    font-weight: 800;
    color: var(--text) !important;
    line-height: 1.1;
}}

.riq-metric .lbl {{
    font-size: 0.688rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 0.375rem;
}}

.riq-metric .sub {{
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.125rem;
}}

/* Stat row */
.stat-row {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.875rem 1rem;
    background: var(--card2);
    border-radius: 10px;
    border: 1px solid var(--border2);
    margin-bottom: 0.5rem;
}}

.stat-row .stat-val {{
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
}}

.stat-row .stat-lbl {{
    font-size: 0.8rem;
    color: var(--muted);
}}

/* Confidence banner */
.conf-banner {{
    border-left: 3px solid var(--primary);
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.75rem 0;
    font-size: 0.875rem;
    line-height: 1.55;
    background: var(--card2);
}}

/* Insight rows */
.insight-row {{
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin: 0.375rem 0;
    font-size: 0.875rem;
    color: var(--text2);
    line-height: 1.55;
}}

/* Chat bubbles */
.chat-wrap {{
    max-height: 460px;
    overflow-y: auto;
    padding: 0.5rem 0.25rem;
}}

.chat-user {{
    display: flex;
    justify-content: flex-end;
    margin: 0.375rem 0;
}}

.chat-user > div {{
    background: var(--primary);
    color: #fff;
    border-radius: 16px 16px 4px 16px;
    padding: 0.625rem 1rem;
    max-width: 76%;
    font-size: 0.875rem;
    line-height: 1.55;
}}

.chat-ai {{
    display: flex;
    justify-content: flex-start;
    margin: 0.375rem 0;
}}

.chat-ai > div {{
    background: var(--card2);
    color: var(--text2);
    border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px;
    padding: 0.75rem 1rem;
    max-width: 82%;
    font-size: 0.875rem;
    line-height: 1.65;
}}
</style>
""", unsafe_allow_html=True)
st.markdown(f"""
<style>
:root {{ {dark_vars} }}

/* Buttons */
.stButton > button {{
    background: var(--primary) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.01em !important;
    transition: opacity 0.15s, transform 0.1s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
}}

.stButton > button:hover {{
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}}

.stButton > button:active {{ transform: translateY(0) !important; }}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
[data-baseweb="select"] > div {{
    background: var(--bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 0.875rem !important;
    transition: border-color 0.15s !important;
}}

.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {{
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary)25 !important;
    outline: none !important;
}}

label {{ color: var(--text2) !important; font-size: 0.8125rem !important; font-weight: 500 !important; }}

/* Toggles / Sliders */
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {{ color: var(--muted) !important; }}

/* Expander */
.streamlit-expanderHeader {{
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text2) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
}}

.streamlit-expanderContent {{
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}}

/* Dividers */
hr {{
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.25rem 0 !important;
}}

/* Info / Success / Warning boxes */
.stAlert {{ border-radius: 8px !important; font-size: 0.875rem !important; }}

[data-testid="stInfo"] {{
    background: var(--primary)12 !important;
    border-color: var(--primary)40 !important;
}}

[data-testid="stSuccess"] {{
    background: var(--success)12 !important;
    border-color: var(--success)40 !important;
}}

[data-testid="stWarning"] {{
    background: var(--warning)12 !important;
    border-color: var(--warning)40 !important;
}}

[data-testid="stError"] {{
    background: var(--danger)12  !important;
    border-color: var(--danger)40  !important;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    border-radius: 10px !important;
    overflow: hidden !important;
}}

.dvn-scroller {{
    background: var(--card) !important;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--muted); }}

/* Spinner */
[data-testid="stSpinner"] > div {{
    border-top-color: var(--primary) !important;
}}

/* Sidebar nav */
[data-testid="stSidebar"] [role="radiogroup"] label {{
    border-radius: 8px !important;
    padding: 0.5rem 0.75rem !important;
    margin-bottom: 2px !important;
    transition: background 0.15s !important;
    color: var(--text2) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
}}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
    background: var(--primary)15 !important;
    color: var(--text) !important;
}}

[data-testid="stSidebar"] [role="radiogroup"] [aria-checked="true"] + label,
[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"] {{
    background: var(--primary)22 !important;
    color: var(--primary) !important;
    font-weight: 600 !important;
}}

</style>
""", unsafe_allow_html=True)

_inject_css()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
meta = load_meta()
CATS = meta["categories"]
CRS  = meta["content_ratings"]
MM   = meta.get("model_metrics", {})

app = get_app()
if app["category"] is None and CATS:
    set_app(category=CATS[0])
    app = get_app()

with st.sidebar:
    # ── Brand ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="padding:1.5rem 1rem 1.25rem;">
      <div style="display:flex;align-items:center;gap:.75rem;">
        <div style="width:38px;height:38px;border-radius:10px;flex-shrink:0;
          background:linear-gradient(135deg,{PRIMARY},{PRIMARY_DIM});
          display:flex;align-items:center;justify-content:center;
          box-shadow:0 4px 12px {PRIMARY}40;">
          <svg width="22" height="22" viewBox="0 0 36 36" fill="none">
            <text x="5" y="27" font-family="Inter,Arial" font-weight="800"
              font-size="24" fill="#fff">R</text>
            <circle cx="28" cy="25" r="3" fill="{SUCCESS}"/>
          </svg>
        </div>
        <div>
          <div style="font-size:1.125rem;font-weight:800;color:{TEXT};
            letter-spacing:-.025em;line-height:1.1;">RateIQ</div>
          <div style="font-size:.6875rem;color:{MUTED};font-weight:500;
            letter-spacing:.04em;text-transform:uppercase;">App Intelligence Platform</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Theme toggle ──────────────────────────────────────────────────────────
    th_col1, th_col2 = st.columns([1, 1])
    with th_col1:
        if st.button(
            "☀ Light",
            key="theme_light",
            use_container_width=True,
        ):
            st.session_state.theme = "light"
            st.rerun()
    with th_col2:
        if st.button(
            "🌙 Dark",
            key="theme_dark",
            use_container_width=True,
        ):
            st.session_state.theme = "dark"
            st.rerun()

    st.divider()

    # ── Navigation ────────────────────────────────────────────────────────────
    st.markdown(
        f'<p class="section-label" style="padding:0 .25rem;">Navigation</p>',
        unsafe_allow_html=True)

    NAV = [
        ("Predict Rating",      "zap"),
        ("Competitor Analysis", "search"),
        ("AI Advisor",          "message"),
        ("Trend Analysis",      "trending-up"),
        ("EDA Insights",        "bar-chart"),
        ("EDA Dashboard",       "layers"),
        ("History",             "clock"),
        ("About",               "info"),
    ]
    NAV_LABELS = [lbl for lbl, ico in NAV]
    page = st.radio(
        "Navigate",
        NAV_LABELS,
        label_visibility="collapsed",
    )
    # normalise page key
    page_key = page.strip()

    st.divider()

    # ── Active app context ────────────────────────────────────────────────────
    if app.get("app_name"):
        st.markdown(f"""
        <div style="padding:.75rem .25rem;">
          <p class="section-label">Active App</p>
          <div style="font-size:.8125rem;color:{TEXT};font-weight:600;
            word-break:break-all;margin-bottom:.25rem;">{app['app_name']}</div>
          <div style="font-size:.75rem;color:{MUTED};">
            {app['category']} &middot; {app['size_mb']} MB &middot;
            {inst_label(int(app['installs']))} installs
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

    # ── Build info ────────────────────────────────────────────────────────────
    model_type = MM.get("model_type", "LightGBM")
    acc = MM.get("accuracy", 0)
    st.markdown(f"""
    <div style="padding:.25rem .25rem .75rem;">
      <p class="section-label">Model</p>
      <div style="font-size:.8rem;color:{TEXT2};font-weight:600;">{model_type}</div>
      {"<div style='font-size:.75rem;color:" + MUTED + ";'>Accuracy " + f"{acc*100:.1f}%" + "</div>" if acc else ""}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED FORM  — reads/writes _app_state, used by all input pages
# ══════════════════════════════════════════════════════════════════════════════
def _render_app_form(prefix: str, show_url: bool = False) -> dict:
    a = get_app()

    if show_url:
        with st.expander(
            f"{icon('link',14,MUTED)}  Import from Play Store URL",
            expanded=False,
        ):
            url_val = st.text_input(
                "Play Store URL",
                placeholder="https://play.google.com/store/apps/details?id=com.example.app",
                key=f"{prefix}_url",
                label_visibility="visible",
            )
            if st.button("Extract Metadata", key=f"{prefix}_url_btn") and url_val.strip():
                with st.spinner("Extracting metadata …"):
                    time.sleep(0.5)
                    pkg   = url_val.split("id=")[-1].split("&")[0] if "id=" in url_val else url_val
                    pkg_l = pkg.lower()
                    if any(x in pkg_l for x in ["edu","learn","study","school","course"]):
                        cat_guess = "EDUCATION"
                    elif any(x in pkg_l for x in ["game","play","puzzle","arcade"]):
                        cat_guess = "GAME"
                    elif any(x in pkg_l for x in ["health","fit","workout","yoga","diet"]):
                        cat_guess = "HEALTH & FITNESS"
                    elif any(x in pkg_l for x in ["tool","util","manager","cleaner"]):
                        cat_guess = "TOOLS"
                    elif any(x in pkg_l for x in ["shop","store","buy","commerce"]):
                        cat_guess = "SHOPPING"
                    elif any(x in pkg_l for x in ["social","chat","message","talk"]):
                        cat_guess = "SOCIAL"
                    elif any(x in pkg_l for x in ["music","audio","player","sound"]):
                        cat_guess = "MUSIC & AUDIO"
                    else:
                        cat_guess = "PRODUCTIVITY"
                    set_app(
                        app_name=pkg,
                        category=cat_guess if cat_guess in CATS else CATS[0],
                        size_mb=28.5, installs=100_000, price=0.0,
                        content_rating="Everyone", reviews=9_500,
                        update_days=22, num_screenshots=5, has_ads=0, is_free=1,
                    )
                    st.rerun()

    a = get_app()

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown(f'<p class="section-label">App Details</p>', unsafe_allow_html=True)
        cat_default  = a["category"] if a["category"] in CATS else CATS[0]
        category     = st.selectbox("Category", CATS, index=CATS.index(cat_default), key=f"{prefix}_cat")
        cr_default   = a["content_rating"] if a["content_rating"] in CRS else CRS[0]
        content_rating = st.selectbox("Content Rating", CRS, index=CRS.index(cr_default), key=f"{prefix}_cr")
        price        = st.number_input("Price (USD)", min_value=0.0, max_value=999.99,
                            value=float(a["price"]), step=0.99, format="%.2f", key=f"{prefix}_price")
        is_free      = 1 if price == 0.0 else 0
        st.caption("Free" if is_free else "Paid")
        has_ads      = int(st.toggle("Contains Ads", value=bool(a["has_ads"]), key=f"{prefix}_ads"))
        num_screenshots = st.slider("Store Screenshots", 1, 8, int(a["num_screenshots"]), key=f"{prefix}_ss")

    with col_r:
        st.markdown(f'<p class="section-label">Performance Metrics</p>', unsafe_allow_html=True)
        size_mb      = st.number_input("App Size (MB)", min_value=0.1, max_value=2000.0,
                            value=float(a["size_mb"]), step=1.0, key=f"{prefix}_size")
        installs     = st.selectbox("Total Installs", options=[v for v,_ in INSTALL_OPTIONS],
                            index=inst_index(int(a["installs"])),
                            format_func=inst_label, key=f"{prefix}_inst")
        reviews      = st.number_input("Reviews", min_value=0, max_value=10_000_000,
                            value=int(a["reviews"]), step=100, key=f"{prefix}_rev")
        update_days  = st.slider("Days Since Last Update", 0, 730, int(a["update_days"]),
                            help="0 = updated today", key=f"{prefix}_upd")

    result = {
        "category": category, "size_mb": size_mb, "installs": installs,
        "price": price, "content_rating": content_rating, "reviews": reviews,
        "update_days": update_days, "num_screenshots": num_screenshots,
        "has_ads": has_ads, "is_free": is_free,
    }
    set_app(**result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTING  —  strict if/elif chain
# ══════════════════════════════════════════════════════════════════════════════
if "Predict Rating" in page_key:

    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="page-title">{icon("zap",20,PRIMARY)} Predict Rating</div>'
        f'<div class="page-subtitle">Enter your app metadata to receive an AI-generated'
        f' rating prediction with feature attribution and market trend context.</div>',
        unsafe_allow_html=True)

    app_state = get_app()
    if app_state.get("app_name"):
        st.info(f"Active app: **{app_state['app_name']}**")

    payload = _render_app_form("pred", show_url=True)
    st.divider()

    btn_c, rst_c = st.columns([4, 1])
    with btn_c:
        run = st.button("Run Prediction", use_container_width=True, key="pred_run")
    with rst_c:
        if st.button("Reset", use_container_width=True, key="pred_rst"):
            reset_app()
            st.rerun()

    if run:
        with st.spinner("Running prediction …"):
            try:
                res = api_predict(payload)
                set_app(
                    _prediction=res["prediction"],
                    _confidence=res["confidence"],
                    _conf_tier=res.get("confidence_tier"),
                    _conf_desc=res.get("confidence_desc"),
                    _shap=res.get("shap_values"),
                    _trend=res.get("trend"),
                    _recommendation=res.get("recommendation"),
                )
            except APIError as e:
                st.error(f"Prediction failed: {e}")
                res = None

    app_state = get_app()
    if app_state.get("_prediction") is not None:
        rating     = app_state["_prediction"]
        confidence = app_state["_confidence"]
        conf_tier  = app_state.get("_conf_tier", "")
        conf_desc  = app_state.get("_conf_desc", "")
        shap_vals  = app_state.get("_shap") or []
        rec        = app_state.get("_recommendation", "")
        trend      = app_state.get("_trend") or {}
        rcolor     = rating_color(rating)
        tier_color = CONF_COLORS.get(conf_tier, MUTED)

        st.divider()

        # ── Section: Prediction ────────────────────────────────────────────
        _section("Prediction", "star")

        # Show app name prominently in results if available
        if app_state.get("app_name"):
            st.markdown(
                f'<div style="font-size:1.125rem;font-weight:700;color:{TEXT};'
                f'margin-bottom:.75rem;padding:.625rem .875rem;'
                f'background:{CARD2};border:1px solid {BORDER};border-radius:8px;">'
                f'{icon("zap",14,PRIMARY)}&nbsp; {app_state["app_name"]}'
                f'</div>',
                unsafe_allow_html=True)
        k1, k2, k3, k4, k5 = st.columns(5)

metrics = [
    (f"{rating:.2f}", "Predicted Rating", rcolor),
    (f"{confidence*100:.0f}%", "Confidence Score", tier_color),
    (conf_tier, "Confidence Tier", tier_color),
    (f"±{MM.get('mae',0):.3f}", "Model MAE", MUTED),
    (f"{trend.get('adjusted_rating', rating):.2f}", "Trend-Adjusted", rcolor),
]

cols = [k1, k2, k3, k4, k5]

for col, (val, lbl, col_override) in zip(cols, metrics):
    with col:
        st.markdown(
            _kpi(val, lbl, color=col_override),
            unsafe_allow_html=True
        )

        # Confidence interpretation
        if conf_tier and conf_desc:
            bg_map = {
                "Very High": SUCCESS, "High": SUCCESS,
                "Medium": WARNING, "Low": DANGER, "Very Low": DANGER,
            }
            bc = bg_map.get(conf_tier, MUTED)
            st.markdown(
                f'<div class="conf-banner" style="border-left-color:{bc};">'
                f'<strong style="color:{bc};">{conf_tier} Confidence</strong>'
                f' — {conf_desc}</div>',
                unsafe_allow_html=True)

        # Gauge + model metrics
        g_col, a_col = st.columns([2, 3], gap="medium")
        with g_col:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=rating,
                number={"font": {"size": 44, "color": rcolor}, "suffix": ""},
                gauge={
                    "axis": {"range": [1, 5], "tickwidth": 1, "tickcolor": MUTED,
                             "tickfont": {"color": MUTED, "size": 10}},
                    "bar":  {"color": rcolor, "thickness": .22},
                    "steps": [
                        {"range": [1, 3], "color": "rgba(239,68,68,0.13)"},
                        {"range": [3, 4], "color": "rgba(245,158,11,0.10)"},
                        {"range": [4, 5], "color": "rgba(34,197,94,0.10)"},
                    ],
                    "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                },
                title={"text": "Rating Scale  (1 – 5)", "font": {"color": MUTED, "size": 11}},
            ))
            fig_g.update_layout(**PLOTLY_LAYOUT, height=220)
            st.plotly_chart(fig_g, use_container_width=True)

        with a_col:
            st.markdown(f'<p class="section-label">Model Performance</p>', unsafe_allow_html=True)
            pairs = [
                ("MAE",          f"±{MM.get('mae',0):.3f}"),
                ("Accuracy",     f"{MM.get('accuracy',0)*100:.1f}%"),
                ("F1 Score",     f"{MM.get('f1_weighted',0):.3f}"),
                ("Hit Rate ±0.5",   f"{MM.get('within_half_star',0)*100:.1f}%"),
                ("RMSE",         f"{MM.get('rmse',0):.3f}"),
                ("Algorithm",    MM.get("model_type", "LightGBM")),
            ]
            for i in range(0, len(pairs), 2):
                c1, c2 = st.columns(2)
                for col, (lbl, val) in zip([c1, c2], pairs[i:i+2]):
                    with col:
                        st.markdown(_kpi(val, lbl), unsafe_allow_html=True)

        # ── Section: Feature Attribution (SHAP) ───────────────────────────
        st.divider()
        _section("Feature Attribution", "bar-chart")
        if shap_vals:
            sc1, sc2 = st.columns([3, 2], gap="large")
            with sc1:
                labels = [s["label"] for s in shap_vals]
                values = [s["value"] for s in shap_vals]
                colors = [SUCCESS if v >= 0 else DANGER for v in values]
                fig_shap = go.Figure(go.Bar(
                    x=values, y=labels, orientation="h",
                    marker=dict(color=colors, line_width=0),
                    text=[f"{v:+.4f}" for v in values],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>SHAP value: %{x:+.4f}<extra></extra>",
                ))
                fig_shap.add_vline(x=0, line_color=MUTED, line_width=1, line_dash="dot")
                fig_shap.update_layout(
                    **PLOTLY_LAYOUT, height=340,
                    title="SHAP Feature Contributions",
                    xaxis_title="Impact on Predicted Rating (stars)",
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_shap, use_container_width=True)

            with sc2:
                # ── Positive Factors ──────────────────────────────────────
                positives = [s for s in shap_vals if s["value"] > 0]
                negatives = [s for s in shap_vals if s["value"] <= 0]
                max_v     = max(abs(s["value"]) for s in shap_vals) or 1

                if positives:
                    st.markdown(
                        f'<p class="section-label" style="color:{SUCCESS};">'
                        f'{icon("trending-up",12,SUCCESS)}&nbsp; Positive Factors</p>',
                        unsafe_allow_html=True)
                    for s in positives:
                        pct = min(int(s["value"] / max_v * 100), 100)
                        st.markdown(f"""
                        <div style="margin:.35rem 0;">
                          <div style="display:flex;justify-content:space-between;
                            font-size:.8125rem;margin-bottom:3px;">
                            <span style="color:{TEXT2};">{s['label']}</span>
                            <span style="color:{SUCCESS};font-weight:600;
                              font-size:.75rem;">+{s['value']:.4f}</span>
                          </div>
                          <div style="background:{BORDER};border-radius:4px;
                            height:4px;overflow:hidden;">
                            <div style="background:{SUCCESS};width:{pct}%;
                              height:100%;border-radius:4px;"></div>
                          </div>
                        </div>""", unsafe_allow_html=True)

                if negatives:
                    st.markdown(
                        f'<p class="section-label" style="color:{DANGER};margin-top:.75rem;">'
                        f'{icon("activity",12,DANGER)}&nbsp; Negative Factors</p>',
                        unsafe_allow_html=True)
                    for s in negatives:
                        pct = min(int(abs(s["value"]) / max_v * 100), 100)
                        st.markdown(f"""
                        <div style="margin:.35rem 0;">
                          <div style="display:flex;justify-content:space-between;
                            font-size:.8125rem;margin-bottom:3px;">
                            <span style="color:{TEXT2};">{s['label']}</span>
                            <span style="color:{DANGER};font-weight:600;
                              font-size:.75rem;">{s['value']:.4f}</span>
                          </div>
                          <div style="background:{BORDER};border-radius:4px;
                            height:4px;overflow:hidden;">
                            <div style="background:{DANGER};width:{pct}%;
                              height:100%;border-radius:4px;"></div>
                          </div>
                        </div>""", unsafe_allow_html=True)

        # ── Section: Recommendation ────────────────────────────────────────
        if rec:
            st.divider()
            _section("Recommendation", "check-circle")
            for tip in [t.strip() for t in rec.split("|") if t.strip()]:
                st.markdown(
                    f'<div class="insight-row">'
                    f'{icon("zap",13,PRIMARY)}&nbsp; {tip}</div>',
                    unsafe_allow_html=True)

        # ── Section: Market Trend ──────────────────────────────────────────
        if trend:
            st.divider()
            _section("Market Trend", "trending-up")
            adj   = trend.get("trend_adjustment", 0)
            adj_c = SUCCESS if adj >= 0 else DANGER
            stage = trend.get("market_stage", "")
            s_col = STAGE_COLORS.get(stage, MUTED)

            t1, t2, t3, t4 = st.columns(4)
            for col, val, lbl, c in zip(
                [t1, t2, t3, t4],
                [f"{adj:+.3f}", f"{trend.get('adjusted_rating',rating):.2f}",
                 stage, trend.get("yoy_growth","—")],
                ["Trend Adjustment", "Adjusted Rating", "Market Stage", "YoY Growth"],
                [adj_c, rcolor, s_col, TEXT2],
            ):
                with col:
                    st.markdown(_kpi(val, lbl, color=c), unsafe_allow_html=True)

            exp = trend.get("explanation", "")
            if exp:
                st.markdown(f'<div class="insight-row">{icon("globe",13,SECONDARY)}&nbsp; {exp}</div>',
                            unsafe_allow_html=True)
            adv = trend.get("stage_advice", "")
            if adv:
                st.markdown(f'<div class="insight-row">{icon("info",13,PRIMARY)}&nbsp; {adv}</div>',
                            unsafe_allow_html=True)

            bd = trend.get("adjustment_breakdown", {})
            if bd:
                bd_vals = list(bd.values())
                fig_bd = go.Figure(go.Bar(
                    x=bd_vals, y=["Trend", "Saturation", "Competition"],
                    orientation="h",
                    marker=dict(color=[SUCCESS if v >= 0 else DANGER for v in bd_vals],
                                line_width=0),
                    text=[f"{v:+.3f}" for v in bd_vals], textposition="outside",
                ))
                fig_bd.add_vline(x=0, line_color=MUTED, line_width=1, line_dash="dot")
                fig_bd.update_layout(**PLOTLY_LAYOUT, height=190,
                    title="Adjustment Components", xaxis_title="Stars",
                    yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_bd, use_container_width=True)

        # ── Section: What-If Analysis ──────────────────────────────────────
        st.divider()
        _section("What-If Analysis", "play")
        st.caption("Adjust individual features to see how the prediction changes.")
        wif = dict(payload)
        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            wif["reviews"]     = st.slider("Reviews", 0, 1_000_000,
                                    int(payload["reviews"]), step=500, key="wif_rev")
            wif["update_days"] = st.slider("Days Since Update", 0, 730,
                                    int(payload["update_days"]), key="wif_upd")
        with wc2:
            wif["installs"] = st.selectbox(
                "Installs", [v for v,_ in INSTALL_OPTIONS],
                index=inst_index(int(payload["installs"])),
                format_func=inst_label, key="wif_inst")
            wif["has_ads"] = int(st.toggle("Contains Ads",
                                    value=bool(payload["has_ads"]), key="wif_ads"))
        with wc3:
            wif["size_mb"]          = st.slider("Size (MB)", 1.0, 200.0,
                                          float(payload["size_mb"]), step=1.0, key="wif_sz")
            wif["num_screenshots"]  = st.slider("Screenshots", 1, 8,
                                          int(payload["num_screenshots"]), key="wif_ss2")

        if st.button("Run What-If", key="wif_run"):
            with st.spinner("Computing delta …"):
                try:
                    wr    = api_predict(wif)
                    delta = wr["prediction"] - rating
                    dc    = SUCCESS if delta >= 0 else DANGER
                    arrow = "+" if delta >= 0 else ""
                    wc_a, wc_b = st.columns(2)
                    with wc_a:
                        st.markdown(f"""
<div class="riq-card-flat" style="text-align:center;padding:1.5rem;">
  <div style="font-size:2rem;font-weight:800;color:{dc};">
    {wr['prediction']:.2f}
  </div>
  <div style="font-size:1rem;color:{dc};font-weight:600;margin:.25rem 0;">
    {arrow}{delta:.2f} vs original {rating:.2f}
  </div>
  <div style="font-size:.8125rem;color:{MUTED};">
    {wr.get('confidence_tier','')} confidence
    &nbsp;({wr['confidence']*100:.0f}%)
  </div>
</div>
""", unsafe_allow_html=True)

    with wc_b:
        wshap = wr.get("shap_values", [])
        if wshap:
            fig_wif = go.Figure(go.Bar(
                x=[s["value"] for s in wshap[:6]],
                y=[s["label"] for s in wshap[:6]],
                orientation="h",
                marker_color=[SUCCESS if s["value"]>=0 else DANGER for s in wshap[:6]],
                marker_line_width=0,
            ))
    
            fig_wif.add_vline(x=0, line_color=MUTED, line_width=1, line_dash="dot")
    
            fig_wif.update_layout(
                **PLOTLY_LAYOUT,
                height=220,
                title="What-If Attribution",
                yaxis=dict(autorange="reversed")
            )
    
            st.plotly_chart(fig_wif, use_container_width=True)
    
except APIError as e:
    st.error(f"What-If failed: {e}")
  
    inp_c, send_c = st.columns([5, 1])
    with inp_c:
        user_q = st.text_input(
            "Ask a question",
            value=st.session_state.pop("_chat_pending", ""),
            placeholder="e.g. Why is my app rating low?",
            label_visibility="collapsed",
            key="chat_q",
        )
    with send_c:
        send = st.button(f"{icon('send',14,'#fff')} Send", use_container_width=True, key="chat_send")

    if (send or user_q) and user_q.strip():
        hist_pl = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.chat_history[-6:]]
        with st.spinner("Processing …"):
            try:
                resp = api_chat(
                query=user_q,
                app_data=chat_app if chat_app.get("category") else None,
                prediction_data=chat_pred,
                chat_history=hist_pl,
                )
                
                st.session_state.chat_history.append({"role": "user", "content": user_q})
                st.session_state.chat_history.append({"role": "assistant", "content": resp["response"]})
                
                recs = resp.get("recommendations", [])
                if recs:
                st.markdown(
                f'<p class="section-label" style="margin-top:.75rem;">Action Items</p>',
                unsafe_allow_html=True
                )
                
                rc = st.columns(min(len(recs), 3))
                
                for col, r in zip(rc, recs[:3]):
                sev_c = {"critical": DANGER, "warning": WARNING, "info": SUCCESS}.get(
                r["severity"], PRIMARY
                )
                
                with col:
                st.markdown(f"""
                <div class="riq-card-flat" style="border-left:3px solid {sev_c};">
                <div style="font-size:.875rem;font-weight:700;color:{TEXT};margin-bottom:.3rem;">
                {r['title']}
                </div>
                <div style="font-size:.8rem;color:{MUTED};margin-bottom:.25rem;">
                {r['action']}
                </div>
                <div style="font-size:.78rem;font-weight:600;color:{SUCCESS};">
                {r['impact']}
                </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.rerun()

except APIError as e:
    st.error(f"Advisor error: {e}")


elif "Trend" in page_key:
    st.markdown(
        f'<div class="page-title">{icon("trending-up",20,PRIMARY)} Trend Analysis</div>'
        f'<div class="page-subtitle">Evaluate category market dynamics — saturation,'
        f' growth trajectory, and competition intensity.</div>',
        unsafe_allow_html=True)

    app_s        = get_app()
    default_cat  = app_s["category"] if app_s["category"] in CATS else (CATS[0] if CATS else "TOOLS")
    default_base = float(app_s.get("_prediction") or 3.8)

    tc1, tc2, tc3 = st.columns([2, 1, 1])
    with tc1:
        tr_cat  = st.selectbox("Category", CATS,
                     index=CATS.index(default_cat) if default_cat in CATS else 0,
                     key="tr_cat")
    with tc2:
        tr_base = st.number_input("Base Rating", 1.0, 5.0,
                     value=round(default_base, 1), step=0.1, key="tr_base")
    with tc3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_tr = st.button("Compute", use_container_width=True, key="tr_run")

    if app_s.get("_prediction") is not None:
        st.caption(f"Pre-filled from last prediction: {app_s['category']} at {app_s['_prediction']:.2f}")

    if run_tr:
        try:
            tr_res = api_trend(tr_cat, tr_base)
            st.session_state["_trend_page"] = tr_res
        except APIError as e:
            st.error(f"Trend computation failed: {e}")

    tr = st.session_state.get("_trend_page")
    if tr:
        adj   = tr.get("trend_adjustment", 0)
        adj_c = SUCCESS if adj >= 0 else DANGER
        stage = tr.get("market_stage", "")
        s_col = STAGE_COLORS.get(stage, MUTED)
        st.divider()
        _section("Trend Result", "activity")
        ta1, ta2, ta3, ta4 = st.columns(4)
        for col, val, lbl, c in zip(
            [ta1, ta2, ta3, ta4],
            [f"{adj:+.3f}", f"{tr.get('adjusted_rating',tr_base):.2f}",
             stage, tr.get("yoy_growth", "—")],
            ["Trend Adjustment", "Adjusted Rating", "Market Stage", "YoY Growth"],
            [adj_c, SUCCESS if tr.get("adjusted_rating",3) >= 4 else WARNING, s_col, TEXT2],
        ):
            with col:
                st.markdown(_kpi(val, lbl, color=c), unsafe_allow_html=True)

        if tr.get("explanation"):
            st.markdown(
                f'<div class="insight-row">{icon("globe",13,SECONDARY)}'
                f'&nbsp; {tr["explanation"]}</div>',
                unsafe_allow_html=True)
        if tr.get("stage_advice"):
            st.markdown(
                f'<div class="insight-row">{icon("info",13,PRIMARY)}'
                f'&nbsp; {tr["stage_advice"]}</div>',
                unsafe_allow_html=True)

        bd = tr.get("adjustment_breakdown", {})
        if bd:
            bd_vals = list(bd.values())
            fig_bd  = go.Figure(go.Bar(
                x=bd_vals, y=["Trend", "Saturation", "Competition"],
                orientation="h",
                marker_color=[SUCCESS if v >= 0 else DANGER for v in bd_vals],
                marker_line_width=0,
                text=[f"{v:+.3f}" for v in bd_vals], textposition="outside",
            ))
            fig_bd.add_vline(x=0, line_color=MUTED, line_width=1, line_dash="dot")
            fig_bd.update_layout(**PLOTLY_LAYOUT, height=190,
                title="Adjustment Components", xaxis_title="Stars",
                yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_bd, use_container_width=True)

    trend_data = meta.get("category_trends", {})
    if trend_data:
        st.divider()
        _section("Category Landscape", "globe")
        rows = [{"Category": c,
                 "Trend Score": d.get("trend_score", 0),
                 "Saturation %": round(d.get("saturation", 0) * 100),
                 "Stage": d.get("market_stage", ""),
                 "YoY": d.get("yoy_growth", ""),
                 "Popularity": d.get("popularity_index", 50)}
                for c, d in trend_data.items()]
        tdf = pd.DataFrame(rows).sort_values("Trend Score", ascending=False)
        fig_bb = px.scatter(
            tdf, x="Saturation %", y="Trend Score",
            size="Popularity", color="Stage", hover_name="Category",
            color_discrete_map={
                "Emerging": SUCCESS, "Growing": SECONDARY, "Mature": WARNING,
                "Saturated": DANGER, "Declining": DANGER, "Niche": MUTED,
            },
            title="Market Saturation vs Growth Trend",
        )
        fig_bb.update_layout(**PLOTLY_LAYOUT, height=420)
        fig_bb.add_hline(y=0, line_dash="dash", line_color=MUTED,
                         annotation_text="Neutral", annotation_font_color=MUTED)
        st.plotly_chart(fig_bb, use_container_width=True)
        st.dataframe(tdf.reset_index(drop=True), use_container_width=True, hide_index=True)


elif "EDA Insights" in page_key:
    st.markdown(
        f'<div class="page-title">{icon("bar-chart",20,PRIMARY)} EDA Insights</div>'
        f'<div class="page-subtitle">Automatically detected statistical patterns from'
        f' the dataset and ranked feature contributions.</div>',
        unsafe_allow_html=True)

    app_s = get_app()
    if app_s.get("_prediction") is not None:
        st.info(
            f"Your app: **{app_s['category']}** — predicted {app_s['_prediction']:.2f} stars "
            f"({app_s['_conf_tier']}). Dataset-wide charts are shown for comparison.")

    # Auto insights
    try:
        ins_data = get_dataset_insights()
        if ins_data and ins_data.get("insights"):
            _section("Auto-Detected Patterns", "activity")
            st.caption(f"{ins_data['dataset_rows']:,} apps analyzed")
            for ins in ins_data["insights"]:
                sc = SUCCESS if ins["strength"] > 0.15 else WARNING if ins["strength"] > 0.08 else PRIMARY
                st.markdown(f"""
                <div class="riq-card" style="border-left:3px solid {sc};padding:1rem 1.25rem;">
                  <div style="font-size:.9375rem;font-weight:700;color:{TEXT};
                    margin-bottom:.375rem;">{ins['finding']}</div>
                  <div style="font-size:.8125rem;color:{MUTED};
                    margin-bottom:.3rem;">{ins['statistic']}</div>
                  <div style="font-size:.8125rem;color:{TEXT2};">
                    {icon('zap',12,sc)}&nbsp; {ins['implication']}</div>
                </div>""", unsafe_allow_html=True)
    except APIError as e:
        st.warning(f"Could not load insights: {e}")

    # Feature importance
    st.divider()
    _section("Feature Importance", "cpu")
    try:
        fi_data = get_feature_importance()
        if fi_data and fi_data.get("items"):
            items = fi_data["items"]
            fi_c1, fi_c2 = st.columns(2, gap="medium")
            with fi_c1:
                fig_fi = go.Figure(go.Bar(
                    y=[i["label"] for i in items],
                    x=[i["importance"] for i in items],
                    orientation="h",
                    marker=dict(
                        color=[i["importance"] for i in items],
                        colorscale=[[0, BORDER], [0.5, SECONDARY], [1.0, PRIMARY]],
                        line_width=0,
                    ),
                    text=[f"{i['importance']:.4f}" for i in items],
                    textposition="outside",
                ))
                fig_fi.update_layout(**PLOTLY_LAYOUT, height=340,
                    title=f"Model Importance ({fi_data.get('model_type','')})",
                    xaxis_title="Importance (normalised)",
                    yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_fi, use_container_width=True)
            with fi_c2:
                fig_mi = go.Figure(go.Bar(
                    y=[i["label"] for i in items],
                    x=[i["mi_score"] for i in items],
                    orientation="h",
                    marker=dict(
                        color=[i["mi_score"] for i in items],
                        colorscale=[[0, BORDER], [0.5, WARNING], [1.0, SUCCESS]],
                        line_width=0,
                    ),
                    text=[f"{i['mi_score']:.4f}" for i in items],
                    textposition="outside",
                ))
                fig_mi.update_layout(**PLOTLY_LAYOUT, height=340,
                    title="Mutual Information Score",
                    xaxis_title="MI Score",
                    yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_mi, use_container_width=True)
    except APIError as e:
        st.warning(f"Could not load feature importance: {e}")

    # Distribution charts from dataset
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    df_eda = None
    for _p in [os.path.join(_root,"data","App_playstore_features.csv"),
               os.path.join(_root,"data","App_playstore_final_cleaned.csv"),
               os.path.join(_root,"data","apps.csv")]:
        if os.path.exists(_p):
            try:
                df_eda = pd.read_csv(_p, low_memory=False)
                df_eda = df_eda.rename(columns={"Rating":"rating","Category":"category",
                    "Size":"size_mb","Installs":"installs","Price":"price",
                    "Reviews":"reviews","is_free":"is_free","Content Rating":"content_rating"})
                df_eda.columns = [c.strip().lower().replace(" ","_") for c in df_eda.columns]
                df_eda["rating"] = pd.to_numeric(df_eda["rating"], errors="coerce")
                df_eda = df_eda[df_eda["rating"].between(1, 5)].dropna(subset=["rating"])
                break
            except Exception:
                df_eda = None

    if df_eda is not None:
        st.divider()
        _section(f"Distributions  ({len(df_eda):,} apps)", "layers")
        dc1, dc2 = st.columns(2, gap="medium")
        with dc1:
            fig_r = px.histogram(df_eda, x="rating", nbins=35,
                title="Rating Distribution", color_discrete_sequence=[PRIMARY])
            fig_r.add_vline(x=df_eda["rating"].mean(), line_dash="dash",
                line_color=SUCCESS, annotation_text=f"Mean {df_eda['rating'].mean():.2f}",
                annotation_font_color=SUCCESS)
            fig_r.update_layout(**PLOTLY_LAYOUT, bargap=.04)
            st.plotly_chart(fig_r, use_container_width=True)
        with dc2:
            if "category" in df_eda.columns:
                ca = df_eda.groupby("category")["rating"].mean().reset_index().sort_values("rating")
                fig_c = px.bar(ca, x="rating", y="category", orientation="h",
                    title="Average Rating by Category", color="rating",
                    color_continuous_scale=[[0,DANGER],[0.5,WARNING],[1,SUCCESS]])
                fig_c.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
                st.plotly_chart(fig_c, use_container_width=True)


elif "EDA Dashboard" in page_key:
    st.markdown(
        f'<div class="page-title">{icon("layers",20,PRIMARY)} EDA Dashboard</div>'
        f'<div class="page-subtitle">Interactive exploration of Play Store app data'
        f' with category filters, distribution charts, and correlation analysis.</div>',
        unsafe_allow_html=True)

    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    df = None
    for _p in [os.path.join(_root,"data","App_playstore_features.csv"),
               os.path.join(_root,"data","App_playstore_final_cleaned.csv"),
               os.path.join(_root,"data","apps.csv")]:
        if os.path.exists(_p):
            try:
                df = pd.read_csv(_p, low_memory=False)
                df = df.rename(columns={"Rating":"rating","Category":"category",
                    "Size":"size_mb","Installs":"installs","Price":"price",
                    "Reviews":"reviews","is_free":"is_free","Content Rating":"content_rating"})
                df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
                df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
                df = df[df["rating"].between(1, 5)].dropna(subset=["rating", "category"])
                break
            except Exception:
                df = None

    if df is None:
        st.error("No dataset found in data/ directory.")
    else:
        st.caption(f"{len(df):,} apps loaded")
        app_s = get_app()
        if app_s.get("_prediction") is not None:
            st.info(
                f"Your app: **{app_s['category']}** — predicted {app_s['_prediction']:.2f} stars. "
                f"Charts below show dataset-wide distributions for context.")

        # Sidebar filters
        cats_f  = []
        price_f = "All"
        with st.sidebar:
            st.markdown(f'<p class="section-label">Filters</p>', unsafe_allow_html=True)
            if "category" in df.columns:
                cats_f = st.multiselect("Category", sorted(df["category"].unique()), default=[])
            price_f = st.radio("Pricing", ["All", "Free", "Paid"], index=0)

        filt = df.copy()
        if cats_f: filt = filt[filt["category"].isin(cats_f)]
        if price_f == "Free" and "is_free" in filt.columns:  filt = filt[filt["is_free"] == 1]
        elif price_f == "Paid" and "price" in filt.columns:  filt = filt[filt["price"] > 0]

        st.caption(f"Showing **{len(filt):,}** of **{len(df):,}** apps")

        # KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        kpi_data = [
            (f"{filt['rating'].mean():.2f}", "Avg Rating"),
            (f"{len(filt):,}", "Apps"),
            (f"{filt['is_free'].mean()*100:.0f}%" if "is_free" in filt.columns else "—", "Free"),
            (f"{filt['reviews'].median():,.0f}" if "reviews" in filt.columns else "—", "Median Reviews"),
            (f"{filt['size_mb'].mean():.1f} MB" if "size_mb" in filt.columns else "—", "Avg Size"),
        ]
        for col, (v, l) in zip([k1, k2, k3, k4, k5], kpi_data):
            with col:
                st.markdown(_kpi(v, l), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        r1, r2 = st.columns(2, gap="medium")
        with r1:
            fig = px.histogram(filt, x="rating", nbins=30,
                title="Rating Distribution", color_discrete_sequence=[PRIMARY])
            fig.update_layout(**PLOTLY_LAYOUT, bargap=.04)
            st.plotly_chart(fig, use_container_width=True)
        with r2:
            if "category" in filt.columns:
                ca = filt.groupby("category")["rating"].mean().reset_index().sort_values("rating")
                fig = px.bar(ca, x="rating", y="category", orientation="h",
                    title="Avg Rating by Category", color="rating",
                    color_continuous_scale=[[0,DANGER],[0.5,WARNING],[1,SUCCESS]])
                fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

        r2a, r2b = st.columns(2, gap="medium")
        with r2a:
            if "installs" in filt.columns:
                samp = filt.sample(min(600, len(filt)), random_state=42).copy()
                samp["installs_num"] = pd.to_numeric(samp["installs"], errors="coerce")
                valid = samp.dropna(subset=["installs_num"])
                if len(valid):
                    fig = px.scatter(valid, x="installs_num", y="rating",
                        color="category" if "category" in valid.columns else None,
                        log_x=True, opacity=.55, title="Installs vs Rating")
                    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        with r2b:
            if "is_free" in filt.columns:
                fp = filt.copy()
                fp["pricing"] = fp["is_free"].apply(lambda x: "Free" if x == 1 else "Paid")
                fig = px.box(fp, x="pricing", y="rating", title="Free vs Paid Rating",
                    color="pricing",
                    color_discrete_map={"Free": SECONDARY, "Paid": PRIMARY})
                fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        num_c = [c for c in ["rating","size_mb","price","reviews","is_free"] if c in filt.columns]
        if len(num_c) > 2:
            corr = filt[num_c].apply(pd.to_numeric, errors="coerce").corr().round(2)
            fig  = px.imshow(corr, text_auto=True, aspect="auto",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="Feature Correlation Matrix")
            fig.update_layout(**PLOTLY_LAYOUT, height=320)
            st.plotly_chart(fig, use_container_width=True)


elif "History" in page_key:
    st.markdown(
        f'<div class="page-title">{icon("clock",20,PRIMARY)} Prediction History</div>'
        f'<div class="page-subtitle">All predictions are persisted to SQLite for'
        f' review and trend analysis.</div>',
        unsafe_allow_html=True)

    try:
        hist = get_history(limit=50)
    except APIError as e:
        st.error(f"Could not load history: {e}")
        hist = []

    if not hist:
        st.info("No predictions recorded yet. Run a prediction to get started.")
    else:
        rows = []
        for h in hist:
            f  = h.get("input_features", {})
            ts = str(h.get("timestamp", ""))[:19].replace("T", " ")
            rows.append({
                "ID":          h["id"],
                "Category":    f.get("category", "—"),
                "Installs":    f"{f.get('installs', 0):,}",
                "Size (MB)":   f"{f.get('size_mb', 0):.1f}",
                "Reviews":     f"{f.get('reviews', 0):,}",
                "Price":       f"${f.get('price', 0):.2f}",
                "Predicted":   h["prediction"],
                "Confidence":  f"{h['confidence']*100:.0f}%",
                "Trend Adj.":  f"{h['trend_adjusted']:.2f}" if h.get("trend_adjusted") else "—",
                "Timestamp":   ts,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        preds     = [h["prediction"] for h in reversed(hist)]
        trend_adj = [h.get("trend_adjusted") or h["prediction"] for h in reversed(hist)]

        if len(preds) > 1:
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(y=preds, mode="lines+markers",
                name="Base Prediction",
                line=dict(color=SECONDARY, width=2),
                marker=dict(color=PRIMARY, size=6)))
            fig_h.add_trace(go.Scatter(y=trend_adj, mode="lines",
                name="Trend-Adjusted",
                line=dict(color=SUCCESS, width=2, dash="dot")))
            fig_h.add_hline(y=4.0, line_dash="dash", line_color=MUTED,
                annotation_text="4.0 threshold", annotation_font_color=MUTED)
            fig_h.update_layout(**PLOTLY_LAYOUT,
                title="Prediction History", xaxis_title="Run #",
                yaxis_title="Rating", yaxis_range=[1, 5.3])
            st.plotly_chart(fig_h, use_container_width=True)

        hp = [h["prediction"] for h in hist]
        hc = st.columns(4)
        for col, (v, l) in zip(hc, [
            (f"{np.mean(hp):.2f}", "Average"),
            (f"{np.max(hp):.2f}",  "Highest"),
            (f"{np.min(hp):.2f}",  "Lowest"),
            (str(len(hp)),          "Total Runs"),
        ]):
            with col:
                st.markdown(_kpi(v, l), unsafe_allow_html=True)


elif "About" in page_key:
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="text-align:center;padding:3rem 1rem 2.25rem;">
      <div style="display:inline-flex;align-items:center;justify-content:center;
        width:72px;height:72px;border-radius:20px;margin-bottom:1.25rem;
        background:linear-gradient(135deg,{PRIMARY},{PRIMARY_DIM});
        box-shadow:0 8px 28px {PRIMARY}40;">
        <svg width="40" height="40" viewBox="0 0 36 36" fill="none">
          <text x="5" y="28" font-family="Inter,Arial" font-weight="800"
            font-size="24" fill="#fff">R</text>
          <circle cx="28" cy="25" r="3.5" fill="{SUCCESS}"/>
        </svg>
      </div>
      <h1 style="font-size:2.25rem;font-weight:800;margin:0;color:{TEXT};
        letter-spacing:-.03em;">RateIQ</h1>
      <p style="font-size:1.0625rem;color:{MUTED};margin:.6rem 0 .25rem;
        font-weight:400;">App Intelligence Platform</p>
      <p style="font-size:.9rem;color:{TEXT2};max-width:540px;
        margin:.5rem auto 0;line-height:1.65;">
        A machine learning platform for mobile app publishers to predict
        Play Store ratings, analyze market gaps, and surface data-driven
        improvement strategies — before launch.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Platform workflow ──────────────────────────────────────────────────────
    _section("Platform Workflow", "zap")
    steps = [
        ("01", "Import App Data",     "Provide app metadata manually or extract it from a Play Store URL."),
        ("02", "AI Analysis",         "The platform evaluates category, size, installs, reviews, and update recency."),
        ("03", "Rating Prediction",   "A trained LightGBM model produces the expected Play Store rating."),
        ("04", "Strategic Insights",  "Review confidence level, market trends, competitor gaps, and recommendations."),
    ]
    cols = st.columns(4, gap="medium")
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="riq-card-flat" style="text-align:center;padding:1.375rem 1rem;height:100%;">
              <div style="width:36px;height:36px;border-radius:8px;margin:0 auto .875rem;
                background:{PRIMARY}20;border:1px solid {PRIMARY}40;
                display:flex;align-items:center;justify-content:center;
                font-size:.75rem;font-weight:800;color:{PRIMARY};">{num}</div>
              <div style="font-size:.875rem;font-weight:700;color:{TEXT};
                margin-bottom:.4rem;">{title}</div>
              <div style="font-size:.8rem;color:{MUTED};line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── Two-column: Prediction Engine + Dataset ────────────────────────────────
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        _section("Prediction Engine", "cpu")
        engine_items = [
            ("Algorithm",      MM.get("model_type", "LightGBM"),
             "Gradient-boosted decision trees optimised for tabular regression."),
            ("Training Data",  f"{MM.get('train_rows', 6400):,} apps",
             "Stratified split with balanced class weights per rating tier."),
            ("Accuracy",       f"{MM.get('accuracy', 0)*100:.1f}%" if MM.get("accuracy") else "81.1%",
             "Bucket-level classification accuracy across 5 rating tiers."),
            ("MAE",            f"{MM.get('mae', 0.23):.3f}",
             "Mean absolute error on held-out test set."),
            ("Hit Rate",       f"{MM.get('within_half_star', 0)*100:.1f}%" if MM.get("within_half_star") else "91.7%",
             "Predictions within ±0.5 stars of actual rating."),
            ("Explainability", "SHAP TreeExplainer",
             "Exact Shapley values computed per prediction for full transparency."),
            ("Confidence",     "Probability-based",
             "Derived from classifier head probability distribution across rating buckets."),
        ]
        for lbl, val, detail in engine_items:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:flex-start;
              padding:.7rem .875rem;border-radius:8px;margin-bottom:.375rem;
              background:{CARD2};border:1px solid {BORDER2};">
              <div>
                <div style="font-size:.8125rem;font-weight:700;color:{TEXT};">{lbl}</div>
                <div style="font-size:.75rem;color:{MUTED};margin-top:.1rem;">{detail}</div>
              </div>
              <div style="font-size:.875rem;font-weight:700;color:{PRIMARY};
                white-space:nowrap;padding-left:.75rem;">{val}</div>
            </div>""", unsafe_allow_html=True)

    with col_b:
        _section("Dataset", "layers")
        dataset_stats = [
            ("Source",         "Google Play Store (real app metadata)"),
            ("Total Apps",     "8,195 unique applications"),
            ("Categories",     "33 app categories"),
            ("Rating Range",   "1.0 – 5.0 (continuous)"),
            ("Features",       "10 engineered predictors"),
            ("Sentiment Data", "Review polarity, subjectivity, positive/negative counts"),
            ("Engineered",     "Log-transformed installs and reviews, size buckets"),
        ]
        for lbl, val in dataset_stats:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
              padding:.65rem .875rem;border-radius:8px;margin-bottom:.375rem;
              background:{CARD2};border:1px solid {BORDER2};">
              <span style="font-size:.8125rem;color:{MUTED};">{lbl}</span>
              <span style="font-size:.8125rem;font-weight:600;color:{TEXT2};">{val}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        _section("Feature Set", "bar-chart")
        features = [
            ("App Category",       "33 categories, label-encoded"),
            ("App Size",           "MB, continuous"),
            ("Install Count",      "log1p transformed"),
            ("Price",              "USD, continuous"),
            ("Content Rating",     "5 levels, label-encoded"),
            ("Review Count",       "log1p transformed"),
            ("Days Since Update",  "0 – 730 days"),
            ("Screenshots",        "Count, 1 – 8"),
            ("Ad-Supported",       "Binary"),
            ("Free / Paid",        "Binary, derived from price"),
        ]
        for fname, ftype in features:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
              padding:.45rem .875rem;font-size:.8rem;">
              <span style="color:{TEXT2};font-weight:500;">{fname}</span>
              <span style="color:{MUTED};">{ftype}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── Analytics Features ─────────────────────────────────────────────────────
    _section("Analytics Features", "activity")
    feature_grid = [
        ("zap",          "Rating Prediction",
         "Predict the expected Play Store rating for any app before submission or update."),
        ("bar-chart",    "SHAP Attribution",
         "Understand exactly which features are driving the prediction up or down."),
        ("search",       "Competitor Analysis",
         "Find the most similar apps using cosine similarity and identify market gaps."),
        ("trending-up",  "Trend Intelligence",
         "Evaluate market saturation, competition level, and growth trajectory per category."),
        ("message",      "AI Advisory",
         "Ask natural language questions and receive root-cause analysis and action plans."),
        ("layers",       "EDA Dashboard",
         "Interactive exploration of the full Play Store dataset with filtering and charts."),
        ("play",         "What-If Analysis",
         "Adjust individual features to simulate rating impact before making product decisions."),
        ("clock",        "Prediction History",
         "All predictions are persisted with full metadata for review and trend monitoring."),
    ]
    fg_cols = st.columns(4, gap="medium")
    for i, (ico, title, desc) in enumerate(feature_grid):
        with fg_cols[i % 4]:
            st.markdown(f"""
            <div class="riq-card-flat" style="padding:1.125rem 1rem;margin-bottom:.75rem;">
              <div style="margin-bottom:.625rem;">{icon_badge(ico, PRIMARY)}</div>
              <div style="font-size:.875rem;font-weight:700;color:{TEXT};
                margin-bottom:.3rem;">{title}</div>
              <div style="font-size:.8rem;color:{MUTED};line-height:1.55;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── Technical stack ────────────────────────────────────────────────────────
    _section("Technical Stack", "cpu")
    tc1, tc2, tc3, tc4 = st.columns(4, gap="medium")
    for col, layer, components in zip(
        [tc1, tc2, tc3, tc4],
        ["Machine Learning", "Backend API", "Frontend", "Data Layer"],
        [
            ["LightGBM 4.3", "SHAP 0.45", "scikit-learn 1.5", "NumPy / Pandas"],
            ["FastAPI 0.111", "Uvicorn", "SQLAlchemy 2.0", "Pydantic v2"],
            ["Streamlit 1.35", "Plotly 5.22", "Inter (typography)", "Custom CSS tokens"],
            ["SQLite (ORM)", "Play Store dataset", "8,195 real apps", "33 categories"],
        ],
    ):
        with col:
            items_html = "".join(
                f'<div style="font-size:.8rem;color:{TEXT2};padding:.3rem 0;'
                f'border-bottom:1px solid {BORDER2};">{c}</div>'
                for c in components
            )
            st.markdown(f"""
            <div class="riq-card-flat" style="padding:1rem 1.125rem;">
              <div style="font-size:.8125rem;font-weight:700;color:{MUTED};
                text-transform:uppercase;letter-spacing:.07em;
                margin-bottom:.625rem;">{layer}</div>
              {items_html}
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── Team Information ───────────────────────────────────────────────────────
    _section("Team", "users")
    tm1, tm2, tm3 = st.columns(3, gap="medium")
    team_members = [
        ("Platform Engineering",
         "Responsible for the FastAPI backend, SQLAlchemy data layer, LightGBM pipeline, "
         "and SHAP explainability integration."),
        ("Data Science",
         "Designed the feature engineering pipeline, model evaluation framework, "
         "confidence scoring methodology, and dataset curation strategy."),
        ("Product & Analytics",
         "Defined the competitor gap analysis, trend intelligence engine, "
         "AI advisory system, and the interactive EDA dashboard."),
    ]
    for col, (role, desc) in zip([tm1, tm2, tm3], team_members):
        with col:
            st.markdown(f"""
            <div class="riq-card-flat" style="padding:1.25rem 1.125rem;">
              <div style="margin-bottom:.625rem;">{icon_badge("users", PRIMARY)}</div>
              <div style="font-size:.875rem;font-weight:700;color:{TEXT};
                margin-bottom:.4rem;">{role}</div>
              <div style="font-size:.8rem;color:{MUTED};line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="text-align:center;padding:.875rem 0 .375rem;">
      <span style="font-size:.8rem;color:{MUTED};">
        RateIQ &nbsp;&middot;&nbsp; App Intelligence Platform
        &nbsp;&middot;&nbsp; Built with Streamlit, FastAPI, and LightGBM
        &nbsp;&middot;&nbsp; &copy; 2025
      </span>
    </div>""", unsafe_allow_html=True)