"""Reusable Streamlit UI pieces.

The visual language (tones, chips, badges, bento tiles, job cards) is ported from Krish Naik's OpenMatch frontend
(AdityaU101/LinkedIn-Conversational-Assistant, branch Krish, frontend/components/ui.py), rewired to call this
project's pipeline directly instead of a FastAPI backend. Colours come from CSS variables in ui/theme.css, so the
pages work in light and dark mode.
"""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

# One palette, one meaning, on every page: good = matched / done, warn = partial / related, neutral = missing /
# not yet, bad = blocked or private, info = neutral information.
TONES = ("good", "warn", "neutral", "bad", "info")
STATUS_TONE = {"done": "good", "evaluated": "good", "pending": "neutral", "running": "info", "blocked": "bad",
               "supported": "good", "rejected": "bad", "approved": "good", "local": "good", "hosted": "info",
               "proprietary": "warn", "private": "bad", "public": "info"}
SOURCE_LABEL = {"kaggle_linkedin_postings": "LinkedIn (Kaggle 2023–24)", "kaggle_resume_dataset": "Kaggle resumes",
                "resume_corpus": "Resume corpus (private)", "upload": "Your upload"}


# Streamlit theme options per mode. Light is the default (.streamlit/config.toml); the toggle sets these options at
# runtime and reruns, which re-sends the theme to the browser. Options are process-wide, which suits a local app.
THEMES = {
    "light": {"base": "light", "primaryColor": "#0a66c2", "backgroundColor": "#ffffff",
              "secondaryBackgroundColor": "#f3f6fa", "textColor": "#0f172a", "borderColor": "#e2e8f0",
              "linkColor": "#0a66c2"},
    "dark": {"base": "dark", "primaryColor": "#4d9bff", "backgroundColor": "#0b1220",
             "secondaryBackgroundColor": "#141d2f", "textColor": "#e6eaf2", "borderColor": "#253149",
             "linkColor": "#6aaeff"},
}
# Colours for this app's own HTML (cards, badges, hero), per mode.
CSS_VARS = {
    "light": {"bg": "#ffffff", "surface": "#f3f6fa", "card": "#ffffff", "text": "#0f172a", "muted": "#64748b",
              "line": "#e2e8f0", "accent": "#0a66c2", "accent-soft": "#e8f1fb", "shadow": "0 1px 2px rgba(15,23,42,.06), 0 4px 16px rgba(15,23,42,.05)",
              "good-bg": "#e7f6ec", "good-fg": "#1f7a3f", "warn-bg": "#fdf3e1", "warn-fg": "#9a5b06",
              "neutral-bg": "#eef1f5", "neutral-fg": "#475569", "bad-bg": "#fdecea", "bad-fg": "#b42318",
              "info-bg": "#e8f1fb", "info-fg": "#0a5aad",
              "hero-from": "#0a66c2", "hero-to": "#0b4f99"},
    "dark": {"bg": "#0b1220", "surface": "#141d2f", "card": "#111a2b", "text": "#e6eaf2", "muted": "#94a3b8",
             "line": "#253149", "accent": "#4d9bff", "accent-soft": "rgba(77,155,255,.14)", "shadow": "0 1px 2px rgba(0,0,0,.4)",
             "good-bg": "rgba(52,199,89,.15)", "good-fg": "#5fd38a", "warn-bg": "rgba(245,166,35,.16)",
             "warn-fg": "#f5b452", "neutral-bg": "rgba(148,163,184,.14)", "neutral-fg": "#b6c2d3",
             "bad-bg": "rgba(240,82,82,.16)", "bad-fg": "#ff7b72", "info-bg": "rgba(77,155,255,.16)",
             "info-fg": "#7db8ff", "hero-from": "#1d4ed8", "hero-to": "#0a3d7a"},
}


def theme_mode() -> str:
    return st.session_state.setdefault("theme_mode", "light")


def _apply_theme(mode: str) -> None:
    for key, value in THEMES[mode].items():
        st._config.set_option(f"theme.{key}", value)


def sync_theme() -> None:
    """Make the server's theme match this session's mode (a new session starts in light mode). Streamlit sends the
    theme when a run starts, before widget callbacks, so a toggle needs one more run to reach the browser."""
    mode = theme_mode()
    if st.session_state.pop("_theme_changed", False) or \
            st._config.get_option("theme.backgroundColor") != THEMES[mode]["backgroundColor"]:
        _apply_theme(mode)
        st.rerun()


def _toggle_theme() -> None:
    st.session_state["theme_mode"] = "dark" if st.session_state.get("theme_dark") else "light"
    st.session_state["_theme_changed"] = True


def theme_toggle() -> None:
    st.toggle("Dark mode", value=theme_mode() == "dark", key="theme_dark", on_change=_toggle_theme)


def load_theme() -> None:
    css = ROOT / "ui" / "theme.css"
    palette = ";".join(f"--om-{k}:{v}" for k, v in CSS_VARS[theme_mode()].items())
    st.markdown(f"<style>:root{{{palette}}}\n{css.read_text() if css.exists() else ''}</style>",
                unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def badge_html(text, kind: str = "neutral", size: str = "0.82rem") -> str:
    kind = kind if kind in TONES else "neutral"
    return f'<span class="om-badge om-{kind}" style="font-size:{size}">{html.escape(str(text))}</span>'


def badge(text, kind: str = "neutral") -> None:
    st.markdown(badge_html(text, kind), unsafe_allow_html=True)


def status_badge_html(status: str | None) -> str:
    status = status or "pending"
    return badge_html(status.replace("_", " "), STATUS_TONE.get(status, "neutral"))


def score_tone(score: float | None) -> str:
    """Match score / coverage (0–1) -> the tone it is shown in."""
    if score is None:
        return "neutral"
    return "good" if score >= 0.6 else "warn" if score >= 0.3 else "neutral"


def chips_html(items: list, kind: str = "neutral", limit: int | None = None) -> str:
    shown = items if limit is None else items[:limit]
    out = "".join(badge_html(i, kind) for i in shown)
    if limit is not None and len(items) > limit:
        out += f'<span class="om-muted" style="font-size:0.8rem">+{len(items) - limit} more</span>'
    return out


def chips(items: list[str], kind: str = "neutral", empty: str = "—", limit: int | None = None) -> None:
    if not items:
        st.caption(empty)
        return
    st.markdown(chips_html(items, kind, limit), unsafe_allow_html=True)


def empty_state(text: str) -> None:
    st.markdown(f"<div class='om-empty'>{html.escape(text)}</div>", unsafe_allow_html=True)


def tile(key: str, height: int | None = None):
    """One bento tile: a bordered box (fixed height keeps tiles in a row aligned)."""
    return st.container(border=True, height=height, key=f"bento-{key}")


def stat(label: str, value, note: str = "") -> None:
    st.markdown(f"<div class='om-stat'><div class='om-stat-label'>{html.escape(label)}</div>"
                f"<div class='om-stat-value'>{html.escape(str(value))}</div>"
                f"<div class='om-stat-note'>{html.escape(note)}</div></div>", unsafe_allow_html=True)


def hero(label: str, value: str, note: str) -> None:
    st.markdown(f"<div class='om-hero'><div class='om-hero-label'>{html.escape(label)}</div>"
                f"<div class='om-hero-value'>{html.escape(value)}</div>"
                f"<div class='om-hero-note'>{html.escape(note)}</div></div>", unsafe_allow_html=True)


def row_html(rank, title: str, sub: str, right: str = "") -> str:
    return (f"<div class='om-row'><span class='om-rank'>{rank}</span><div class='om-row-main'><b>{html.escape(title)}"
            f"</b><br><small>{html.escape(sub)}</small></div><div>{right}</div></div>")


def nav_link(page: str, label: str) -> None:
    """st.page_link that is skipped when the page runs outside the app's navigation (single-page tests)."""
    try:
        st.page_link(page, label=label)
    except Exception:
        pass


def profile() -> dict | None:
    return st.session_state.get("profile")


def require_profile() -> dict | None:
    p = profile()
    if not p:
        st.info("Load a CV first: upload one or pick a sample on **My profile**.")
        nav_link("views/profile.py", "Go to My profile")
    return p


def privacy_note(p: dict | None) -> None:
    if not p:
        return
    if p.get("private", True):
        st.caption(":material/lock: Private CV: parsed, explained and checked on this computer only (local Ollama). "
                   "Nothing about it is sent to a hosted model.")
    else:
        st.caption("Public Kaggle CV (anonymized): still explained locally; job postings may be parsed on ASU Voyager.")


def provider_kind(cfg: dict, provider: str, model: str = "") -> str:
    if cfg["providers"][provider]["private_ok"]:
        return "local"
    proprietary = {(m["provider"], m["model"]) for m in cfg["benchmark_models"] if m.get("proprietary")}
    return "proprietary" if (provider, model) in proprietary or provider == "azure" else "hosted"
