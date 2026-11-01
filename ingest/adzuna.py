"""Live job postings from the Adzuna search API (proposal: live feeds).

  python -m ingest.adzuna "software engineer" --where Arizona     # one search = one API call per page

Each search is cached under data/live/adzuna/ (raw JSON, git-ignored) and every call is logged to calls.jsonl,
so a repeated search within MAX_AGE reuses the cache instead of spending a call. Adzuna returns a ~500-character
snippet of each description, not the full posting, so live jobs carry less text than the Kaggle postings.
Postings are public job ads (source "adzuna"); CV text is never sent to Adzuna, only the typed keywords.
"""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from ingest.text import clean_text

load_dotenv()
DIR = Path("data/live/adzuna")
CALLS = DIR / "calls.jsonl"
API = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
MAX_AGE = 6 * 3600                       # seconds a cached search is reused
SOURCE = "adzuna"

US_STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY",
    "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
    "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"}
SENIORITY = [(re.compile(r"\bintern(ship)?\b", re.I), "Internship"),
             (re.compile(r"\b(director|vp|vice president|head of)\b", re.I), "Director"),
             (re.compile(r"\b(chief|cto|ceo|cfo|president)\b", re.I), "Executive"),
             (re.compile(r"\b(senior|sr\.?|lead|principal|staff|manager)\b", re.I), "Mid-Senior level"),
             (re.compile(r"\b(junior|jr\.?|entry|graduate|new grad|associate)\b", re.I), "Entry level")]
REMOTE = re.compile(r"\bremote\b", re.I)
HYBRID = re.compile(r"\bhybrid\b", re.I)


class AdzunaError(RuntimeError):
    """Raised with a message that never contains the request URL (it carries the API key)."""


def _key(what: str, where: str, country: str, page: int, max_days: int | None = None) -> str:
    parts = [what.strip().lower(), where.strip().lower(), country, page] + ([max_days] if max_days else [])
    return hashlib.sha1(json.dumps(parts).encode()).hexdigest()[:12]


def calls_used() -> int:
    return sum(1 for line in open(CALLS) if '"cached": false' in line) if CALLS.exists() else 0


def _log(**row) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    with open(CALLS, "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **row}) + "\n")


def fetch_page(what: str, where: str = "", country: str = "us", page: int = 1, per_page: int = 50,
               max_days: int | None = None, force: bool = False) -> dict:
    """One search page (one API call), or the cached copy if it is younger than MAX_AGE.
    max_days: only postings created in the last N days (Adzuna's max_days_old)."""
    path = DIR / f"search_{_key(what, where, country, page, max_days)}.json"
    if path.exists() and not force and time.time() - path.stat().st_mtime < MAX_AGE:
        out = json.loads(path.read_text())
        _log(what=what, where=where, page=page, n=len(out["results"]), cached=True)
        return out
    app_id, app_key = os.getenv("ADZUNA_APP_ID"), os.getenv("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        raise AdzunaError("ADZUNA_APP_ID / ADZUNA_APP_KEY are not set in .env")
    params = {"app_id": app_id, "app_key": app_key, "results_per_page": per_page, "what": what,
              "content-type": "application/json"}
    if where.strip():
        params["where"] = where
    if max_days:
        params["max_days_old"] = max_days
    try:
        r = requests.get(API.format(country=country, page=page), params=params, timeout=30)
    except requests.RequestException as e:
        raise AdzunaError(f"Adzuna request failed ({type(e).__name__})") from None
    _log(what=what, where=where, page=page, status=r.status_code, cached=False,
         n=len(r.json().get("results", [])) if r.ok else 0)
    if not r.ok:
        body = r.text[:200].replace(app_key, "<key>").replace(app_id, "<id>")
        raise AdzunaError(f"Adzuna returned HTTP {r.status_code}: {body}")
    data = r.json()
    out = {"query": {"what": what, "where": where, "country": country, "page": page, "max_days": max_days},
           "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"), "count": data.get("count"),
           "results": data.get("results", [])}
    DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def to_frame(results: list[dict], fetched_at: str = "") -> pd.DataFrame:
    """Adzuna results -> the job corpus columns used by ranking (doc_id = adz_<adzuna id>)."""
    rows = []
    for x in results:
        title = clean_text(x.get("title", ""))
        text = clean_text(x.get("description", ""))
        area = (x.get("location") or {}).get("area") or []
        state = US_STATES.get(area[1], "Other") if len(area) > 1 else "US"
        mode = "Remote" if REMOTE.search(title + " " + text) else "Hybrid" if HYBRID.search(title + " " + text) \
            else "On-site"
        seniority = next((s for pat, s in SENIORITY if pat.search(title)), "Unknown")
        rows.append({"doc_id": f"adz_{x['id']}", "source": SOURCE, "title": title,
                     "company": (x.get("company") or {}).get("display_name", ""),
                     "location": (x.get("location") or {}).get("display_name", ""), "state": state,
                     "seniority": seniority, "work_mode": mode,
                     "industry": (x.get("category") or {}).get("label", ""), "text": text,
                     "url": x.get("redirect_url", ""), "created": x.get("created", ""),
                     "salary_min": x.get("salary_min"), "salary_max": x.get("salary_max"),
                     "fetched_at": fetched_at, "doc_type": "JobPosting"})
    return pd.DataFrame(rows)


def search(what: str, where: str = "", pages: int = 1, max_days: int | None = None,
           force: bool = False) -> pd.DataFrame:
    """Live postings for a keyword search (pages x 50 results); duplicates across pages are dropped."""
    frames = []
    for page in range(1, pages + 1):
        out = fetch_page(what, where, page=page, max_days=max_days, force=force)
        frames.append(to_frame(out["results"], out["fetched_at"]))
        if len(out["results"]) < 50:
            break
    df = pd.concat(frames, ignore_index=True) if frames else to_frame([])
    return df.drop_duplicates("doc_id").reset_index(drop=True)


def load_all() -> pd.DataFrame:
    """Every posting in the cache (newest fetch wins for a repeated id)."""
    frames = [to_frame(d["results"], d["fetched_at"])
              for d in (json.loads(p.read_text()) for p in sorted(DIR.glob("search_*.json"), key=os.path.getmtime))]
    if not frames:
        return to_frame([])
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates("doc_id", keep="last").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("--where", default="")
    ap.add_argument("--pages", type=int, default=1)
    a = ap.parse_args()
    df = search(a.what, a.where, a.pages)
    print(len(df), "postings;", calls_used(), "API calls used so far")
    print(df[["doc_id", "title", "company", "location", "state", "work_mode"]].head(10).to_string(index=False))
