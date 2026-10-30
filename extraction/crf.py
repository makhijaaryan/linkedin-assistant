"""Classical extraction baseline (R2): regex section cues + linear-chain CRF over spaCy tokens,
trained on silver labels (extraction/silver.py) from train-split documents only.

Two CRFs tag two flat BIO layers (outer: Requirement/Experience/Project; inner: Skill, Credential,
Company, Occupation, Location). Predicted spans are converted to the same ParsedCV/ParsedJob
schema as the LLM path, with the span itself as the evidence quote.

Train:   python -m extraction.crf --train   (writes .cache/models/crf_{outer,inner}.pkl)
"""

import argparse
import pickle
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import sklearn_crfsuite
import spacy

from extraction.silver import LAYER_OF, LAYERS, SECTION, silver_spans, skill_gazetteer
from ontology.schema import (CandidateItem, CredentialItem, Evidence, ExperienceItem, JobItem, LocationItem,
                             ParsedCV, ParsedJob, ProjectItem, RequirementItem, SkillItem)

MODELS = Path(".cache/models")
VERSION = "crf-1.0"
YEARS = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:-\s*\d+\s*)?(?:years?|yrs)", re.IGNORECASE)
PREFERRED = re.compile(r"\b(prefer\w*|nice to have|a plus|bonus|desired|ideally)\b", re.IGNORECASE)
EDU = {"bachelor": "Bachelor", "b.s": "Bachelor", "bs ": "Bachelor", "ba ": "Bachelor", "master": "Master",
       "m.s": "Master", "mba": "Master", "ph.d": "Doctorate", "phd": "Doctorate", "doctor": "Doctorate",
       "associate": "Associate", "high school": "High School", "diploma": "High School"}


@lru_cache(maxsize=1)
def nlp():
    return spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])


def token_features(doc) -> list[dict]:
    gaz = skill_gazetteer()
    section, feats = "none", []
    text = doc.text
    for i, t in enumerate(doc):
        if SECTION.fullmatch(t.text) or (i + 1 < len(doc) and SECTION.fullmatch(f"{t.text} {doc[i + 1].text}")):
            section = t.text.lower()
        w = t.text
        f = {"bias": 1.0, "w": w.lower(), "suf3": w[-3:].lower(), "pre2": w[:2].lower(), "shape": t.shape_,
             "pos": t.pos_, "upper": w.isupper(), "title": w.istitle(), "digit": w.isdigit(), "punct": t.is_punct,
             "gaz": w.lower() in gaz, "section": section,
             "line_start": t.idx == 0 or "\n" in text[max(0, t.idx - 2):t.idx]}
        for d in (-2, -1, 1, 2):
            j = i + d
            if 0 <= j < len(doc):
                f[f"{d}:w"] = doc[j].text.lower()
                f[f"{d}:pos"] = doc[j].pos_
                f[f"{d}:title"] = doc[j].text.istitle()
                f[f"{d}:gaz"] = doc[j].text.lower() in gaz
            else:
                f[f"{d}:edge"] = True
        feats.append(f)
    return feats


def bio_tags(doc, spans: list[tuple[int, int, str]], layer: str) -> list[str]:
    tags = ["O"] * len(doc)
    for s, e, lab in spans:
        if LAYER_OF[lab] != layer:
            continue
        idx = [t.i for t in doc if t.idx >= s and t.idx + len(t.text) <= e]
        for k, i in enumerate(idx):
            tags[i] = ("B-" if k == 0 else "I-") + lab
    return tags


def tags_to_spans(doc, tags, marginals) -> list[tuple[int, int, str, float]]:
    spans, cur = [], None
    for t, tag, m in zip(doc, tags, marginals):
        if tag.startswith("B-") or (tag.startswith("I-") and (cur is None or cur[2] != tag[2:])):
            if cur:
                spans.append(cur)
            cur = [t.idx, t.idx + len(t.text), tag[2:], [m[tag]]]
        elif tag.startswith("I-"):
            cur[1] = t.idx + len(t.text)
            cur[3].append(m[tag])
        else:
            if cur:
                spans.append(cur)
            cur = None
    if cur:
        spans.append(cur)
    return [(s, e, lab, sum(p) / len(p)) for s, e, lab, p in spans]


def train(docs: pd.DataFrame, max_tokens: int = 1500):
    MODELS.mkdir(parents=True, exist_ok=True)
    X, Y = [], {layer: [] for layer in LAYERS}
    for row, doc in zip(docs.itertuples(), nlp().pipe(docs["text"], batch_size=32)):
        doc = doc[:max_tokens].as_doc()
        spans = silver_spans(doc.text, row.doc_type, getattr(row, "title", "") or "", getattr(row, "company", "") or "")
        X.append(token_features(doc))
        for layer in LAYERS:
            Y[layer].append(bio_tags(doc, spans, layer))
    for layer in LAYERS:
        crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100,
                                   all_possible_transitions=True)
        crf.fit(X, Y[layer])
        pickle.dump(crf, open(MODELS / f"crf_{layer}.pkl", "wb"))
        print(layer, "trained on", len(X), "docs")


@lru_cache(maxsize=1)
def load_models():
    return {layer: pickle.load(open(MODELS / f"crf_{layer}.pkl", "rb")) for layer in LAYERS}


def _ev(text, s, e, conf):
    return Evidence(quote=text[s:e], confidence=round(conf, 3), char_start=s, char_end=e)


def _inside(spans, s, e, lab):
    return [x for x in spans if x[2] == lab and x[0] >= s and x[1] <= e]


def predict_spans(text: str) -> list[tuple[int, int, str, float]]:
    doc = nlp()(text)
    feats = token_features(doc)
    spans = []
    for layer, crf in load_models().items():
        tags = crf.predict_single(feats)
        spans += tags_to_spans(doc, tags, crf.predict_marginals_single(feats))
    return sorted(spans)


def extract_document(doc_id: str, doc_type: str, text: str, **_) -> tuple[ParsedCV | ParsedJob, dict]:
    spans = predict_spans(text)
    gaz = skill_gazetteer()
    common = {"doc_id": doc_id, "extractor": "classical/crf", "extractor_version": VERSION}
    by = lambda lab: [x for x in spans if x[2] == lab]
    loc = by("Location")
    location = None
    if loc:
        s, e, _, c = loc[0]
        q = text[s:e]
        city, _, state = q.partition(", ")
        location = LocationItem(city=city if state else None, state=state or None,
                                is_remote=q.lower() == "remote", evidence=_ev(text, s, e, c))
    skills = []
    seen = set()
    for s, e, _, c in by("Skill"):
        name = text[s:e]
        if name.lower() not in seen:
            seen.add(name.lower())
            skills.append(SkillItem(name=name, skill_type=gaz.get(name.lower(), "Technical"), evidence=_ev(text, s, e, c)))
    creds = []
    for s, e, _, c in by("Credential"):
        q = text[s:e]
        level = next((v for k, v in EDU.items() if k in q.lower() + " "), None)
        creds.append(CredentialItem(name=q, credential_type="Degree" if level else "Certification", level=level,
                                    evidence=_ev(text, s, e, c)))
    if doc_type == "JobPosting":
        occ, comp = by("Occupation"), by("Company")
        job = JobItem(title=text[occ[0][0]:occ[0][1]] if occ else "Unknown",
                      company=text[comp[0][0]:comp[0][1]] if comp else None, location=location,
                      work_mode="Remote" if location and location.is_remote else "Unknown")
        reqs = []
        for s, e, _, c in by("Requirement"):
            q = text[s:e]
            years = YEARS.search(q)
            level = next((v for k, v in EDU.items() if k in q.lower() + " "), None)
            rtype = "Education" if level else "Experience" if years else "Credential" if "certif" in q.lower() else "Skill"
            reqs.append(RequirementItem(text=q, requirement_type=rtype,
                                        importance="Preferred" if PREFERRED.search(q) else "Required",
                                        min_years=float(years.group(1)) if years else None, education_level=level,
                                        skills=[text[a:b] for a, b, _, _ in _inside(spans, s, e, "Skill")],
                                        evidence=_ev(text, s, e, c)))
        # skills outside any requirement still count as job skills: attach them as an "Other" requirement each
        in_req = {(a, b) for s, e, _, _ in by("Requirement") for a, b, _, _ in _inside(spans, s, e, "Skill")}
        for s, e, _, c in by("Skill"):
            if (s, e) not in in_req:
                reqs.append(RequirementItem(text=text[s:e], requirement_type="Skill", importance="Required",
                                            skills=[text[s:e]], evidence=_ev(text, s, e, c)))
        parsed = ParsedJob(**common, job=job, requirements=reqs)
    else:
        exps = []
        for s, e, _, c in by("Experience"):
            occ, comp = _inside(spans, s, e, "Occupation"), _inside(spans, s, e, "Company")
            exps.append(ExperienceItem(title=text[occ[0][0]:occ[0][1]] if occ else text[s:e][:60],
                                       company=text[comp[0][0]:comp[0][1]] if comp else None,
                                       evidence=_ev(text, s, e, c)))
        projects = [ProjectItem(name=text[s:e][:80], evidence=_ev(text, s, e, c)) for s, e, _, c in by("Project")]
        parsed = ParsedCV(**common, candidate=CandidateItem(location=location), skills=skills, experiences=exps,
                          credentials=creds, projects=projects)
    return parsed, {"calls": 0, "spans": len(spans)}


def training_docs(n_jobs: int = 2000, n_cvs: int = 1500, seed: int = 573) -> pd.DataFrame:
    jobs = pd.read_parquet("data/processed/jobs.parquet", columns=["doc_id", "title", "company", "text", "split",
                                                                  "is_canonical"])
    cvs = pd.read_parquet("data/processed/resumes.parquet", columns=["doc_id", "text", "split", "is_canonical"])
    jobs = jobs[(jobs["split"] == "train") & jobs["is_canonical"]].sample(n_jobs, random_state=seed)
    cvs = cvs[(cvs["split"] == "train") & cvs["is_canonical"]].sample(n_cvs, random_state=seed)
    return pd.concat([jobs.assign(doc_type="JobPosting"), cvs.assign(doc_type="CV", title="", company="")])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--jobs", type=int, default=2000)
    ap.add_argument("--cvs", type=int, default=1500)
    a = ap.parse_args()
    if a.train:
        train(training_docs(a.jobs, a.cvs))
