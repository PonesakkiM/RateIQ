"""
RateIQ – CSS/styling helpers for Streamlit (Dark + Light themes)
"""

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg:        #0A0F1E;
    --bg2:       #0F172A;
    --card:      #1E293B;
    --card2:     #162032;
    --primary:   #6366F1;
    --primary2:  #818CF8;
    --accent:    #22C55E;
    --accent2:   #4ADE80;
    --text:      #E2E8F0;
    --muted:     #94A3B8;
    --border:    #334155;
    --danger:    #F87171;
    --warn:      #FBBF24;
    --radius:    12px;
    --shadow:    0 4px 24px rgba(0,0,0,.45);
    --glow:      0 0 20px rgba(99,102,241,.25);
}

* { font-family: 'Inter', 'Poppins', sans-serif !important; box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main { background-color: var(--bg) !important; color: var(--text) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background-color: #070D1A !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Header / toolbar hide ── */
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
#MainMenu { visibility: hidden; }

/* ── Cards ── */
.riq-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
    transition: box-shadow .2s;
}
.riq-card:hover { box-shadow: var(--glow); }

/* ── Metric tiles ── */
.riq-metric {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1rem;
    text-align: center;
    transition: transform .15s;
}
.riq-metric:hover { transform: translateY(-2px); box-shadow: var(--glow); }
.riq-metric .val { font-size: 2rem; font-weight: 800; color: var(--primary); line-height: 1.1; }
.riq-metric .lbl { font-size: 0.73rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin-top: .3rem; }

/* ── Chat UI ── */
.chat-bubble-user {
    background: linear-gradient(135deg, var(--primary), var(--primary2));
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: .75rem 1.1rem;
    margin: .4rem 0 .4rem auto;
    max-width: 75%;
    display: inline-block;
    font-size: .9rem;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(99,102,241,.35);
}
.chat-bubble-ai {
    background: var(--card2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 18px 18px 18px 4px;
    padding: .85rem 1.2rem;
    margin: .4rem auto .4rem 0;
    max-width: 82%;
    display: inline-block;
    font-size: .9rem;
    line-height: 1.65;
}
.chat-avatar { font-size: 1.4rem; margin-right: .5rem; vertical-align: middle; }
.chat-container { max-height: 480px; overflow-y: auto; padding: .5rem; scroll-behavior: smooth; }
.chat-row-user { display: flex; justify-content: flex-end; margin: .25rem 0; }
.chat-row-ai   { display: flex; justify-content: flex-start; margin: .25rem 0; }

/* ── Recommendation cards ── */
.rec-card {
    background: var(--card);
    border-left: 4px solid var(--primary);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin: .5rem 0;
}
.rec-card.critical { border-left-color: var(--danger); }
.rec-card.warning  { border-left-color: var(--warn); }
.rec-card.info     { border-left-color: var(--accent); }
.rec-title  { font-weight: 700; font-size: .9rem; color: var(--text); margin-bottom: .3rem; }
.rec-action { font-size: .85rem; color: var(--muted); margin-bottom: .2rem; }
.rec-impact { font-size: .8rem; font-weight: 600; color: var(--accent); }

/* ── Gap badges ── */
.gap-badge {
    display: inline-block;
    padding: .2rem .6rem;
    border-radius: 20px;
    font-size: .75rem;
    font-weight: 600;
}
.gap-badge.negative { background: rgba(248,113,113,.15); color: var(--danger); border: 1px solid rgba(248,113,113,.3); }
.gap-badge.positive { background: rgba(34,197,94,.12); color: var(--accent); border: 1px solid rgba(34,197,94,.3); }
.gap-badge.neutral  { background: rgba(148,163,184,.12); color: var(--muted); border: 1px solid rgba(148,163,184,.25); }

/* ── Insight pills ── */
.insight-pill {
    background: rgba(99,102,241,.10);
    border: 1px solid rgba(99,102,241,.25);
    border-radius: 8px;
    padding: .75rem 1rem;
    margin: .4rem 0;
    font-size: .88rem;
    color: var(--text);
    line-height: 1.5;
}

/* ── Trend stage badge ── */
.stage-badge {
    display: inline-block;
    padding: .3rem .9rem;
    border-radius: 20px;
    font-size: .8rem;
    font-weight: 700;
    letter-spacing: .04em;
    text-transform: uppercase;
}
.stage-Emerging  { background: rgba(34,197,94,.15);  color: #4ADE80; border: 1px solid rgba(34,197,94,.3); }
.stage-Growing   { background: rgba(99,102,241,.12); color: #818CF8; border: 1px solid rgba(99,102,241,.3); }
.stage-Mature    { background: rgba(251,191,36,.12); color: #FBBF24; border: 1px solid rgba(251,191,36,.3); }
.stage-Saturated { background: rgba(248,113,113,.12);color: #F87171; border: 1px solid rgba(248,113,113,.3); }
.stage-Declining { background: rgba(248,113,113,.12);color: #F87171; border: 1px solid rgba(248,113,113,.3); }
.stage-Niche     { background: rgba(148,163,184,.12);color: #94A3B8; border: 1px solid rgba(148,163,184,.3); }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--primary2)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: .55rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
    transition: opacity .2s, transform .1s !important;
}
.stButton > button:hover { opacity: .88 !important; transform: translateY(-1px) !important; }

/* ── Inputs ── */
.stTextInput input, .stNumberInput input,
.stTextArea textarea, [data-baseweb="select"] div {
    background: #0F172A !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.2) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] { font-weight: 600 !important; color: var(--muted) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--primary) !important; border-bottom: 2px solid var(--primary) !important; }

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Recommendation box ── */
.riq-rec {
    background: rgba(99,102,241,.08);
    border-left: 3px solid var(--primary);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    color: var(--text);
    font-size: .9rem;
    line-height: 1.6;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ── Markdown formatting ── */
.riq-markdown { font-size: .9rem; line-height: 1.7; color: var(--text); }
.riq-markdown strong { color: var(--primary2); }
.riq-markdown code { background: rgba(99,102,241,.15); padding: .1rem .4rem; border-radius: 4px; font-size: .85rem; }
</style>
"""

LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
:root {
    --bg: #F8FAFC; --bg2: #F1F5F9; --card: #FFFFFF; --card2: #F8FAFC;
    --primary: #4F46E5; --primary2: #7C3AED;
    --accent: #16A34A; --accent2: #15803D;
    --text: #0F172A; --muted: #64748B; --border: #E2E8F0;
    --danger: #DC2626; --warn: #D97706;
    --shadow: 0 4px 16px rgba(0,0,0,.08); --radius: 12px;
}
* { font-family: 'Inter', 'Poppins', sans-serif !important; }
html, body, [data-testid="stAppViewContainer"] { background-color: var(--bg) !important; color: var(--text) !important; }
[data-testid="stSidebar"] { background-color: #F1F5F9 !important; }
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu { visibility: hidden; }
.riq-card { background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.5rem; box-shadow: var(--shadow); margin-bottom: 1rem; }
.riq-metric { background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.25rem 1rem; text-align: center; }
.riq-metric .val { font-size: 2rem; font-weight: 800; color: var(--primary); }
.riq-metric .lbl { font-size: 0.73rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
.chat-bubble-user { background: linear-gradient(135deg, var(--primary), var(--primary2)); color: #fff; border-radius: 18px 18px 4px 18px; padding: .75rem 1.1rem; max-width: 75%; display: inline-block; font-size: .9rem; }
.chat-bubble-ai { background: #F1F5F9; color: var(--text); border: 1px solid var(--border); border-radius: 18px 18px 18px 4px; padding: .85rem 1.2rem; max-width: 82%; display: inline-block; font-size: .9rem; }
.chat-container { max-height: 480px; overflow-y: auto; padding: .5rem; }
.chat-row-user { display: flex; justify-content: flex-end; margin: .25rem 0; }
.chat-row-ai   { display: flex; justify-content: flex-start; margin: .25rem 0; }
.rec-card { background: var(--card); border-left: 4px solid var(--primary); border-radius: 0 10px 10px 0; padding: 1rem 1.2rem; margin: .5rem 0; }
.rec-card.critical { border-left-color: var(--danger); }
.rec-card.warning  { border-left-color: var(--warn); }
.rec-card.info     { border-left-color: var(--accent); }
.rec-title { font-weight: 700; font-size: .9rem; color: var(--text); }
.rec-action { font-size: .85rem; color: var(--muted); }
.rec-impact { font-size: .8rem; font-weight: 600; color: var(--accent); }
.gap-badge { display: inline-block; padding: .2rem .6rem; border-radius: 20px; font-size: .75rem; font-weight: 600; }
.gap-badge.negative { background: rgba(220,38,38,.1); color: var(--danger); }
.gap-badge.positive { background: rgba(22,163,74,.1); color: var(--accent); }
.insight-pill { background: rgba(79,70,229,.06); border: 1px solid rgba(79,70,229,.2); border-radius: 8px; padding: .75rem 1rem; margin: .4rem 0; font-size: .88rem; }
.stage-Emerging  { background: rgba(22,163,74,.1);  color: var(--accent);  border: 1px solid rgba(22,163,74,.2);  border-radius:20px; padding:.3rem .9rem; display:inline-block; font-size:.8rem; font-weight:700; }
.stage-Growing   { background: rgba(79,70,229,.08); color: var(--primary); border: 1px solid rgba(79,70,229,.2);  border-radius:20px; padding:.3rem .9rem; display:inline-block; font-size:.8rem; font-weight:700; }
.stage-Mature    { background: rgba(217,119,6,.08); color: var(--warn);    border: 1px solid rgba(217,119,6,.2);  border-radius:20px; padding:.3rem .9rem; display:inline-block; font-size:.8rem; font-weight:700; }
.stage-Saturated { background: rgba(220,38,38,.08); color: var(--danger);  border: 1px solid rgba(220,38,38,.2);  border-radius:20px; padding:.3rem .9rem; display:inline-block; font-size:.8rem; font-weight:700; }
.stage-Declining { background: rgba(220,38,38,.08); color: var(--danger);  border: 1px solid rgba(220,38,38,.2);  border-radius:20px; padding:.3rem .9rem; display:inline-block; font-size:.8rem; font-weight:700; }
.stage-Niche     { background: rgba(100,116,139,.08);color: var(--muted);  border: 1px solid rgba(100,116,139,.2);border-radius:20px; padding:.3rem .9rem; display:inline-block; font-size:.8rem; font-weight:700; }
.stButton > button { background: linear-gradient(135deg, var(--primary), var(--primary2)) !important; color: #fff !important; border: none !important; border-radius: 8px !important; padding: .55rem 1.8rem !important; font-weight: 600 !important; }
.riq-rec { background: rgba(79,70,229,.06); border-left: 3px solid var(--primary); border-radius: 0 10px 10px 0; padding: 1rem 1.2rem; font-size: .9rem; }
hr { border-color: var(--border) !important; }
.riq-markdown { font-size: .9rem; line-height: 1.7; }
</style>
"""


def inject_css(dark: bool = True) -> None:
    import streamlit as st
    st.markdown(DARK_CSS if dark else LIGHT_CSS, unsafe_allow_html=True)


def card(content_html: str) -> str:
    return f'<div class="riq-card">{content_html}</div>'


def metric_tile(value: str, label: str) -> str:
    return f'<div class="riq-metric"><div class="val">{value}</div><div class="lbl">{label}</div></div>'


def rec_card(title: str, action: str, impact: str, severity: str = "info") -> str:
    return (
        f'<div class="rec-card {severity}">'
        f'<div class="rec-title">{title}</div>'
        f'<div class="rec-action">{action}</div>'
        f'<div class="rec-impact">{impact}</div>'
        f'</div>'
    )


def gap_badge(label: str, kind: str = "neutral") -> str:
    return f'<span class="gap-badge {kind}">{label}</span>'


def insight_pill(text: str) -> str:
    return f'<div class="insight-pill">{text}</div>'


def stage_badge(stage: str) -> str:
    return f'<span class="stage-badge stage-{stage}">{stage}</span>'
