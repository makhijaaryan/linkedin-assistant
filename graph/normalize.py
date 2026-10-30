"""Skill and occupation normalization to O*NET 31.0 (R3; taxonomy per D14).

Classical: rule cleanup + alias table, then char-n-gram TF-IDF nearest alias.
Modern:    alias table, then sentence-embedding kNN; when the top two candidates are close,
           a local LLM picks (route normalize_tiebreak). Mentions with no confident match stay
           as free skills (taxonomy_id None), so nothing extracted is lost.
Which one the graph uses is config graph.normalizer; D4 keeps the modern one only if it wins.
"""

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ONET = Path("data/external/onet/db_31_0_csv")
CACHE = Path(".cache/models")
VENDORS = ("microsoft", "oracle", "apache", "amazon", "google", "adobe", "ibm", "sap", "autodesk", "atlassian",
           "salesforce.com", "salesforce", "intuit", "esri", "cisco", "vmware", "red hat", "the mathworks")
MANUAL_ALIASES = {"js": "javascript", "k8s": "kubernetes", "postgres": "postgresql", "golang": "go",
                  "nodejs": "node.js", "node": "node.js", "reactjs": "react", "react.js": "react", "vuejs": "vue.js",
                  "ms excel": "microsoft excel", "ms word": "microsoft word", "ms office": "microsoft office",
                  "ms powerpoint": "microsoft powerpoint", "powerpoint": "microsoft powerpoint", "gcp": "google cloud",
                  "google cloud platform": "google cloud", "azure": "microsoft azure software", "sklearn": "scikit-learn",
                  "ms sql": "microsoft sql server", "sql server": "microsoft sql server", "c sharp": "c#",
                  "power bi": "microsoft power bi", "excel": "microsoft excel", "word": "microsoft word",
                  "outlook": "microsoft outlook", "aws": "amazon web services aws software",
                  "sql": "structured query language sql", "tensorflow": "tensorflow", "ms project": "microsoft project"}
VERSION_TAIL = re.compile(r"\s+v?\d+(?:\.\d+)*[a-z]?$")
PARENS = re.compile(r"\s*\([^)]*\)")


def clean(name: str) -> str:
    s = PARENS.sub("", str(name).lower()).strip()
    s = re.sub(r"[\s_]+", " ", s)
    s = re.sub(r"\b(software|programming language|language|framework|skills?|proficiency|experience)$", "", s).strip()
    s = VERSION_TAIL.sub("", s).strip(" .,;:-")
    return MANUAL_ALIASES.get(s, s)


@dataclass
class Match:
    taxonomy_id: str | None
    canonical_name: str
    score: float
    method: str          # alias | tfidf | embedding | llm | none
    skill_type: str | None = None
    category: str | None = None
    hot_technology: bool = False


@lru_cache(maxsize=1)
def taxonomy() -> pd.DataFrame:
    """One row per taxonomy skill: taxonomy_id, canonical_name, skill_type, category, hot, occupations."""
    t = pd.read_csv(ONET / "software_skills.csv")
    g = t.groupby("Workplace Example")
    tools = pd.DataFrame({
        "canonical_name": g.size().index,
        "category": g["Element Name"].agg(lambda s: s.mode().iat[0]).values,
        "hot_technology": g["Hot Technology"].agg(lambda s: bool((s == "Y").any())).values,
        "occupations": g["O*NET-SOC Code"].agg(lambda s: sorted(set(s))).values,
    })
    tools["taxonomy_id"] = "onet:tool:" + tools["canonical_name"].map(lambda x: re.sub(r"[^a-z0-9]+", "-", x.lower()).strip("-"))
    tools["skill_type"] = "Tool/Technology"
    rows = [tools]
    for f, kind in (("essential_skills", "Technical"), ("transferable_skills", "Soft"), ("knowledge", "Domain Knowledge")):
        e = pd.read_csv(ONET / f"{f}.csv")
        e = e[(e["Scale ID"] == "IM") & (e["Data Value"] >= 3.5)]   # occupations where the element is important
        g = e.groupby(["Element ID", "Element Name"])["O*NET-SOC Code"].agg(lambda s: sorted(set(s))).reset_index()
        rows.append(pd.DataFrame({"taxonomy_id": "onet:" + g["Element ID"], "canonical_name": g["Element Name"],
                                  "category": f.replace("_", " "), "hot_technology": False,
                                  "occupations": g["O*NET-SOC Code"], "skill_type": kind}))
    return pd.concat(rows, ignore_index=True).drop_duplicates("taxonomy_id").reset_index(drop=True)


@lru_cache(maxsize=1)
def alias_table() -> dict[str, int]:
    """cleaned alias -> taxonomy row. Collisions go to the skill used by more occupations."""
    tax = taxonomy()
    spread = tax["occupations"].map(len).to_numpy()
    out: dict[str, int] = {}

    def put(alias, i):
        if alias and (alias not in out or spread[i] > spread[out[alias]]):
            out[alias] = i

    for i, name in enumerate(tax["canonical_name"]):
        low = name.lower()
        put(clean(name), i)
        put(low, i)
        acr = re.findall(r"\b([A-Z][A-Z0-9+#]{1,9})\b", name)
        if acr and len(acr[-1]) >= 2:                      # "Structured query language SQL" -> "sql"
            put(acr[-1].lower(), i)
            put(clean(name.replace(acr[-1], "")), i)
        for v in VENDORS:                                  # "Microsoft Excel" -> "excel"
            if low.startswith(v + " ") and len(low) > len(v) + 2:
                put(clean(name[len(v) + 1:]), i)
    for alias, target in MANUAL_ALIASES.items():
        if target in out:
            out[alias] = out[target]
    return out


class Normalizer:
    def __init__(self, method: str = "modern", router=None, accept_tfidf: float = 0.75,
                 accept_embed: float = 0.80, tie_margin: float = 0.03):
        self.method, self.router = method, router
        self.accept_tfidf, self.accept_embed, self.tie_margin = accept_tfidf, accept_embed, tie_margin
        self.tax = taxonomy()
        self.aliases = alias_table()
        self.alias_keys = list(self.aliases)
        self.alias_rows = np.array([self.aliases[k] for k in self.alias_keys])
        self._tfidf = self._emb = None

    def _match(self, i: int, score: float, method: str) -> Match:
        r = self.tax.iloc[i]
        return Match(r.taxonomy_id, r.canonical_name, round(float(score), 4), method, r.skill_type, r.category,
                     bool(r.hot_technology))

    def _tfidf_index(self):
        if self._tfidf is None:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
            self._tfidf = (vec, vec.fit_transform(self.alias_keys))
        return self._tfidf

    def _embed_index(self):
        if self._emb is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            path = CACHE / "alias_emb.npy"
            if path.exists() and len(np.load(path)) == len(self.alias_keys):
                emb = np.load(path)
            else:
                CACHE.mkdir(parents=True, exist_ok=True)
                emb = model.encode(self.alias_keys, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
                np.save(path, emb)
            self._emb = (model, emb)
        return self._emb

    def candidates(self, mentions: list[str], k: int = 5, method: str | None = None) -> list[list[tuple[int, float]]]:
        """Top-k (taxonomy row, score) per mention, de-duplicated by taxonomy row."""
        method = method or self.method
        keys = [clean(m) for m in mentions]
        if method == "classical":
            vec, X = self._tfidf_index()
            sims = (vec.transform(keys) @ X.T).toarray()
        else:
            model, emb = self._embed_index()
            with np.errstate(all="ignore"):   # spurious Accelerate matmul warnings on Apple silicon
                sims = model.encode(keys, batch_size=256, normalize_embeddings=True, show_progress_bar=False) @ emb.T
        out = []
        for row in sims:
            best = {}
            for j in np.argsort(-row)[: 4 * k]:
                t = int(self.alias_rows[j])
                best.setdefault(t, float(row[j]))
            out.append(sorted(best.items(), key=lambda x: -x[1])[:k])
        return out

    def normalize(self, mentions: list[str], contexts: list[str] | None = None, private: bool = True) -> list[Match]:
        contexts = contexts or [""] * len(mentions)
        results: list[Match | None] = [None] * len(mentions)
        todo = []
        for n, m in enumerate(mentions):
            key = clean(m)
            if key in self.aliases:
                results[n] = self._match(self.aliases[key], 1.0, "alias")
            else:
                todo.append(n)
        if todo:
            cands = self.candidates([mentions[n] for n in todo])
            accept = self.accept_tfidf if self.method == "classical" else self.accept_embed
            for n, c in zip(todo, cands):
                if not c or c[0][1] < accept:
                    results[n] = Match(None, clean(mentions[n]) or mentions[n], c[0][1] if c else 0.0, "none")
                elif (self.method == "modern" and self.router is not None and len(c) > 1
                      and c[0][1] - c[1][1] < self.tie_margin and c[1][1] >= accept):
                    results[n] = self.tiebreak(mentions[n], contexts[n], c[:3], private)
                else:
                    results[n] = self._match(c[0][0], c[0][1], "tfidf" if self.method == "classical" else "embedding")
        return results

    def tiebreak(self, mention: str, context: str, cands: list[tuple[int, float]], private: bool) -> Match:
        names = [self.tax.iloc[i].canonical_name for i, _ in cands]
        prompt = (f'Skill mention: "{mention}"\nContext: "{context[:300]}"\nWhich taxonomy entry is the same skill? '
                  f"Options: {json.dumps(names)}. Reply JSON {{\"choice\": <exact option or null>}}.")
        try:
            reply = self.router.chat("normalize_tiebreak", [{"role": "user", "content": prompt}], private=private)
            choice = json.loads(reply.text).get("choice")
        except Exception:
            choice = None
        for (i, s), name in zip(cands, names):
            if choice == name:
                return self._match(i, s, "llm")
        return self._match(cands[0][0], cands[0][1], "embedding")


@lru_cache(maxsize=1)
def occupation_titles() -> pd.DataFrame:
    occ = pd.read_csv(ONET / "occupation_data.csv")[["O*NET-SOC Code", "Title"]]
    alt = pd.read_csv(ONET / "job_titles.csv")[["O*NET-SOC Code", "Job Title"]].rename(columns={"Job Title": "alias"})
    occ_alias = occ.rename(columns={"Title": "alias"})
    t = pd.concat([occ_alias, alt]).dropna().drop_duplicates("alias")
    t = t.merge(occ, on="O*NET-SOC Code")
    return t.rename(columns={"O*NET-SOC Code": "soc", "Title": "title"}).reset_index(drop=True)


SENIORITY_WORDS = re.compile(r"\b(senior|sr|junior|jr|lead|principal|staff|chief|head|i{1,3}|iv|entry level|associate|"
                             r"intern|remote|hybrid|contract|part time|full time)\b\.?", re.IGNORECASE)


class OccupationNormalizer:
    """Job title -> O*NET-SOC occupation via char TF-IDF over O*NET titles and ~60K alternate titles."""

    def __init__(self, accept: float = 0.55):
        self.t = occupation_titles()
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.X = self.vec.fit_transform(self.t["alias"].str.lower())
        self.accept = accept

    def normalize(self, titles: list[str]) -> list[tuple[str | None, str | None, float]]:
        keys = [re.sub(r"\s+", " ", SENIORITY_WORDS.sub(" ", PARENS.sub("", str(x)))).strip().lower() for x in titles]
        sims = self.vec.transform(keys) @ self.X.T
        out = []
        for n in range(len(keys)):
            row = sims.getrow(n)
            if row.nnz == 0:
                out.append((None, None, 0.0))
                continue
            j = row.indices[row.data.argmax()]
            score = float(row.data.max())
            r = self.t.iloc[j]
            out.append((r.soc, r.title, score) if score >= self.accept else (None, None, score))
        return out
