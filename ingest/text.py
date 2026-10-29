"""Text cleaning and contact-detail masking (applied before any LLM sees a document)."""

import html
import re
import unicodedata

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
PHONE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)")
TAG = re.compile(r"<[^>]+>")
SPACES = re.compile(r"[ \t ]+")
BLANK_LINES = re.compile(r"\n\s*\n\s*\n+")


def clean_text(text: str) -> str:
    """Strip HTML, normalize unicode and whitespace; keep line breaks (section cues)."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(TAG.sub(" ", text))
    text = unicodedata.normalize("NFKC", text).replace("\r", "\n")
    text = SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return BLANK_LINES.sub("\n\n", text).strip()


def mask_contacts(text: str) -> tuple[str, int]:
    """Replace emails, URLs and phone numbers with placeholders. Returns (text, n_masked)."""
    n = 0
    for pattern, token in ((EMAIL, "[EMAIL]"), (URL, "[URL]"), (PHONE, "[PHONE]")):
        text, k = pattern.subn(token, text)
        n += k
    return text, n


def normalize_key(text: str) -> str:
    """Lowercase alphanumerics only — for exact-duplicate keys."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
