"""Silver (weak) span labels for training the CRF baseline: gazetteers + regex rules.

Used on TRAIN-split documents only. Gazetteers: O*NET tools, O*NET skill/knowledge element
names, O*NET job titles, and the posting's own title/company metadata. Labels are the 8
annotation labels (eval/annotation/ANNOTATION_GUIDE.md), so the CRF predicts what gold measures.
"""

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

ONET = Path("data/external/onet/db_31_0_csv")
US_STATES = ("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC "
             "ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC").split()
SECTION = re.compile(r"\b(summary|skills|technical skills|experience|work experience|professional experience|"
                     r"education|certifications?|projects|qualifications|requirements|responsibilities|"
                     r"what you.ll do|what we.re looking for|preferred qualifications|basic qualifications|"
                     r"benefits|about us|accomplishments|highlights)\b", re.IGNORECASE)
REQ_CUE = re.compile(r"\b(\d+\+?\s*(?:-\s*\d+\s*)?(?:years?|yrs)\.?\s+(?:of\s+)?(?:[\w-]+\s+){0,4}experience|"
                     r"degree|bachelor|master'?s|ph\.?d|diploma|certif\w*|licen[cs]e|required|must have|preferred|"
                     r"proficien\w*|experience (?:with|in)|knowledge of|familiar\w* with)\b", re.IGNORECASE)
DEGREE = re.compile(r"\b(?:Bachelor|Master|Associate|Doctor)(?:'?s)?(?: of| in)? (?:[A-Z][\w&]*\s?){1,5}|"
                    r"\b(?:B\.S\.|B\.A\.|M\.S\.|M\.A\.|MBA|Ph\.?D\.?|BSc|MSc|B\.Tech|M\.Tech)(?: (?:in|of) (?:[A-Z][\w&]*\s?){1,5})?|"
                    r"\b(?:BS|BA|MS|MA)(?: (?:in|of) (?:[A-Z][\w&]*\s?){1,5})")
CERT = re.compile(r"\b(?:[A-Z][\w+]*\s){0,4}(?:Certified|Certification|Certificate|License)\b(?:\s[A-Z][\w+]*){0,4}|"
                  r"\b(?:CPA|PMP|CISSP|CCNA|CDL|RN|LPN|CNA|CFA|SHRM-CP|PHR)\b")
DATE_RANGE = re.compile(r"(?:\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.? |\b\d{1,2}/)?"
                        r"\d{4}\s*(?:to|-|–)\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.? "
                        r"|\d{1,2}/)?(?:\d{4}|Present|Current)", re.IGNORECASE)
LOCATION = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+){0,2}, (?:" + "|".join(US_STATES) + r")\b|\b(?:Remote|Hybrid)\b")
PROJECT = re.compile(r"\bProject(?: Name)?:\s*[^.;\n]{3,80}")
# O*NET tool names that are also everyday words; too ambiguous for a gazetteer
COMMON_WORDS = {"access", "box", "chef", "go", "make", "word", "teams", "slack", "notes", "sage", "basic", "swift",
                "dash", "office", "project", "twitter", "facebook", "linkedin", "youtube", "instagram", "email",
                "calendar", "maps", "drive", "reader", "pages", "keynote", "numbers", "photos"}
KEEP_SHORT = {"R", "C", "C#", "C++", "Go", "SQL", "AWS", "GCP", "SAP", "SAS", "Git", "PHP", "CSS", "HTML", "Java"}
LABELS = ["Skill", "Requirement", "Experience", "Credential", "Project", "Company", "Occupation", "Location"]
# Long spans contain short ones (a Requirement contains Skills), so the CRF tags two flat layers.
LAYERS = {"outer": ["Requirement", "Experience", "Project"],
          "inner": ["Skill", "Credential", "Company", "Occupation", "Location"]}
LAYER_OF = {lab: layer for layer, labs in LAYERS.items() for lab in labs}


@lru_cache(maxsize=1)
def skill_gazetteer() -> dict[str, str]:
    """lower-cased name -> skill_type. O*NET tools are Tool/Technology; element names are Technical/Soft."""
    tools = pd.read_csv(ONET / "software_skills.csv")["Workplace Example"].dropna().unique()
    gaz = {}
    for t in tools:
        t = t.strip()
        if len(t) >= 3 or t in KEEP_SHORT:
            gaz[t.lower()] = "Tool/Technology"
    for f, kind in (("essential_skills", "Technical"), ("transferable_skills", "Soft"), ("knowledge", "Domain Knowledge")):
        for name in pd.read_csv(ONET / f"{f}.csv")["Element Name"].unique():
            if " " in name:   # one-word element names ("Design", "Monitoring") are too ambiguous in free text
                gaz.setdefault(name.lower(), kind)
    for w in COMMON_WORDS:
        gaz.pop(w, None)
    return gaz


@lru_cache(maxsize=1)
def title_gazetteer() -> set[str]:
    t = pd.read_csv(ONET / "job_titles.csv")
    titles = set(t["Job Title"].str.lower()) | set(pd.read_csv(ONET / "occupation_data.csv")["Title"].str.lower())
    return {x for x in titles if 2 <= len(x.split()) <= 5}


def _phrase_spans(text: str, phrases, case_sensitive=False) -> list[tuple[int, int]]:
    spans = []
    hay = text if case_sensitive else text.lower()
    for p in phrases:
        if not p:
            continue
        for m in re.finditer(r"(?<![\w+#])" + re.escape(p) + r"(?![\w+#])", hay):
            spans.append((m.start(), m.end()))
    return spans


def _gazetteer_spans(text: str, names: set[str], max_len: int = 5) -> list[tuple[int, int]]:
    """Longest-match n-gram lookup over word tokens (fast for large gazetteers)."""
    toks = [(m.start(), m.end()) for m in re.finditer(r"[\w+#.&/-]*[\w+#]", text)]
    low = text.lower()
    spans, i = [], 0
    while i < len(toks):
        hit = None
        for n in range(min(max_len, len(toks) - i), 0, -1):
            s, e = toks[i][0], toks[i + n - 1][1]
            if low[s:e] in names:
                hit = (s, e, n)
                break
        if hit:
            spans.append(hit[:2])
            i += hit[2]
        else:
            i += 1
    return spans


def silver_spans(text: str, doc_type: str, title: str = "", company: str = "") -> list[tuple[int, int, str]]:
    """Rule-based (start, end, label) spans; later rules never overwrite earlier ones."""
    gaz = skill_gazetteer()
    spans: list[tuple[int, int, str]] = []
    if doc_type == "JobPosting":
        for line in re.finditer(r"[^\n.;•]+", text):
            seg = line.group()
            if REQ_CUE.search(seg) and 4 <= len(seg.split()) <= 40:
                s = line.start() + len(seg) - len(seg.lstrip())
                spans.append((s, line.end(), "Requirement"))
        spans += [(s, e, "Occupation") for s, e in _phrase_spans(text, [title.lower()] if title else [])]
        spans += [(s, e, "Company") for s, e in _phrase_spans(text, [company] if company else [], case_sensitive=True)]
    else:
        for m in DATE_RANGE.finditer(text):
            # role header: the ~10 words before the date range
            s = max(0, text.rfind(" ", 0, max(0, m.start() - 70)))
            spans.append((s + 1 if s else 0, m.end(), "Experience"))
        spans += [(m.start(), m.end(), "Project") for m in PROJECT.finditer(text)]
        spans += [(s, e, "Occupation") for s, e in _gazetteer_spans(text, title_gazetteer())]
        spans += [(m.start(), m.start() + 12, "Company") for m in re.finditer(r"Company Name", text)]
    spans += [(m.start(), m.end(), "Credential") for m in DEGREE.finditer(text)]
    spans += [(m.start(), m.end(), "Credential") for m in CERT.finditer(text)]
    spans += [(m.start(), m.end(), "Location") for m in LOCATION.finditer(text)]
    spans += [(s, e, "Skill") for s, e in _gazetteer_spans(text, set(gaz))]
    return resolve(spans)


def resolve(spans):
    """Drop spans that overlap an earlier span of the same layer (first rule wins)."""
    out = []
    for s, e, lab in spans:
        if e <= s or any(not (e <= s2 or s >= e2) and LAYER_OF[lab] == LAYER_OF[l2] for s2, e2, l2 in out):
            continue
        out.append((s, e, lab))
    return sorted(out)
