"""
RateIQ – Logo & Branding Helper
Loads logo_dark.png / logo_light.png from frontend/assets/
Falls back to SVG-based text logo if image files are absent.
"""
import os
import base64
import streamlit as st
from typing import Optional

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

LOGO_PATHS = {
    "dark":  os.path.join(_ASSETS_DIR, "logo_dark.png"),
    "light": os.path.join(_ASSETS_DIR, "logo_light.png"),
}


def _img_to_b64(path: str) -> Optional[str]:
    """Read image file and return base64 encoded data URI."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        mime = "image/png" if ext == "png" else f"image/{ext}"
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        return None


def logo_available(mode: str = "dark") -> bool:
    """Return True if logo image exists for the given mode."""
    return os.path.exists(LOGO_PATHS.get(mode, ""))


def get_logo_html(
    dark: bool = True,
    width: int = 56,
    show_text: bool = True,
    primary: str = "#6366F1",
    accent: str = "#22C55E",
    muted: str = "#94A3B8",
    text_col: str = "#E2E8F0",
) -> str:
    """
    Return the logo as an HTML string.
    - If the image file exists → renders <img> tag at the given width
    - Otherwise → renders inline SVG text logo as fallback
    """
    mode = "dark" if dark else "light"
    b64 = _img_to_b64(LOGO_PATHS[mode])

    if b64:
        img_html = (
            f'<img src="{b64}" width="{width}" height="{width}" '
            f'style="border-radius:14px;display:block;" alt="RateIQ logo">'
        )
        if show_text:
            return f"""
            <div style="display:flex;align-items:center;gap:.75rem;">
                {img_html}
                <div>
                    <div style="font-size:1.5rem;font-weight:900;
                        background:linear-gradient(135deg,{primary},{accent});
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        letter-spacing:-.02em;line-height:1.1;">RateIQ</div>
                    <div style="font-size:.65rem;color:{muted};letter-spacing:.1em;
                        text-transform:uppercase;margin-top:.1rem;">AI App Optimizer</div>
                </div>
            </div>"""
        return img_html

    # ── Fallback: inline SVG icon + text ─────────────────────────────────────
    bg      = "#1E1B4B" if dark else "#EEF2FF"
    icon_fg = "#FFFFFF"  if dark else primary
    return f"""
    <div style="display:flex;align-items:center;gap:.75rem;">
        <div style="width:{width}px;height:{width}px;border-radius:14px;
            background:{bg};display:flex;align-items:center;justify-content:center;
            flex-shrink:0;box-shadow:0 4px 16px rgba(99,102,241,.35);">
            <svg width="{int(width*0.6)}" height="{int(width*0.6)}"
                viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
                <!-- R letterform -->
                <text x="4" y="28" font-family="Inter,Arial,sans-serif"
                    font-weight="800" font-size="28" fill="{icon_fg}">R</text>
                <!-- dot accent (matches brand image) -->
                <circle cx="29" cy="26" r="3.5" fill="{accent}"/>
                <!-- small connector line -->
                <line x1="23" y1="22" x2="27" y2="25"
                    stroke="{accent}" stroke-width="1.8" stroke-linecap="round"/>
            </svg>
        </div>
        {"" if not show_text else f'''
        <div>
            <div style="font-size:1.5rem;font-weight:900;
                background:linear-gradient(135deg,{primary},{accent});
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                letter-spacing:-.02em;line-height:1.1;">RateIQ</div>
            <div style="font-size:.65rem;color:{muted};letter-spacing:.1em;
                text-transform:uppercase;margin-top:.1rem;">AI App Optimizer</div>
        </div>'''}
    </div>"""


def render_sidebar_logo(dark: bool = True, primary: str = "#6366F1",
                        accent: str = "#22C55E", muted: str = "#94A3B8",
                        text_col: str = "#E2E8F0") -> None:
    """Render the logo inside st.sidebar (call from within a `with st.sidebar:` block)."""
    html = get_logo_html(dark=dark, width=54, show_text=True,
                         primary=primary, accent=accent, muted=muted, text_col=text_col)
    st.markdown(
        f'<div style="padding:1.2rem .5rem 1.6rem;">{html}</div>',
        unsafe_allow_html=True,
    )


def render_page_logo(dark: bool = True, width: int = 40,
                     primary: str = "#6366F1", accent: str = "#22C55E",
                     muted: str = "#94A3B8", text_col: str = "#E2E8F0") -> None:
    """Render a compact logo at the top of a page (e.g. in a header column)."""
    html = get_logo_html(dark=dark, width=width, show_text=True,
                         primary=primary, accent=accent, muted=muted, text_col=text_col)
    st.markdown(html, unsafe_allow_html=True)
