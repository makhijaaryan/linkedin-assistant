"""Download the Lightcast Open Skills taxonomy (R3 skill normalization).

Needs LIGHTCAST_CLIENT_ID / LIGHTCAST_CLIENT_SECRET in .env (apply at
https://lightcast.io/taxonomies/skills-taxonomy/access).

Run from the repo root:  python scripts/fetch_lightcast.py
Writes data/external/lightcast/skills_<version>.json and .csv
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

TOKEN_URL = "https://auth.emsicloud.com/connect/token"
API = "https://api.lightcast.io/skills"
OUT_DIR = Path("data/external/lightcast")


def get_token():
    load_dotenv()
    client_id, secret = os.getenv("LIGHTCAST_CLIENT_ID"), os.getenv("LIGHTCAST_CLIENT_SECRET")
    if not (client_id and secret):
        sys.exit("Set LIGHTCAST_CLIENT_ID and LIGHTCAST_CLIENT_SECRET in .env first.")
    r = requests.post(TOKEN_URL, data={
        "client_id": client_id,
        "client_secret": secret,
        "grant_type": "client_credentials",
        "scope": os.getenv("LIGHTCAST_SCOPE", "lightcast_open_free"),
    }, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def main():
    headers = {"Authorization": f"Bearer {get_token()}"}

    meta = requests.get(f"{API}/versions/latest", headers=headers, timeout=30)
    meta.raise_for_status()
    meta = meta.json().get("data", meta.json())
    version = meta.get("version", "latest")
    print(f"Taxonomy version: {version}; skill count: {meta.get('skillCount', '?')}")

    # Ask for every available field (hierarchy + description are needed for RELATED_TO bridges).
    fields = meta.get("fields")
    params = {"fields": ",".join(fields)} if fields else {}
    r = requests.get(f"{API}/versions/{version}/skills", headers=headers, params=params, timeout=300)
    r.raise_for_status()
    skills = r.json()["data"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"skills_{version}.json").write_text(json.dumps(skills, indent=1))
    pd.json_normalize(skills).to_csv(OUT_DIR / f"skills_{version}.csv", index=False)
    print(f"Saved {len(skills)} skills to {OUT_DIR}/skills_{version}.{{json,csv}}")


if __name__ == "__main__":
    main()
