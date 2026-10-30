"""R3: extractions -> normalized, evidence-grounded Neo4j graph (ontology v1.0).

  python -m graph.build --extractions asu__gpt-oss-120b ollama__llama3.1-8b --set graph
  python -m graph.build --check          # evidence audit only
  python -m graph.build --snapshot v1    # dump to data/processed/graph_snapshots/v1.jsonl.gz

Rules:
- Idempotent MERGE on each type's id, with uniqueness constraints.
- An extracted item becomes a node only if its quote was located in the text; it is linked
  (n)-[:SUPPORTED_BY]->(:Evidence). Extracted relationships carry evidence_ids, source="Extracted".
- Every relationship has an edge_id (what explanations cite) and a source from ontology RELATION_SOURCE.
- Taxonomy edges (TYPICAL_FOR, FOR_ROLE/AS_ROLE targets) have source="Taxonomy"; RELATED_TO edges proposed by
  embeddings and confirmed by the LLM have source="LLM-inferred", inferred=true.
- `check()` fails if any extracted relationship lacks resolvable evidence or any extracted node has none.
"""

import argparse
import gzip
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

from eval.extraction_eval import find_span
from graph.normalize import Normalizer, OccupationNormalizer, clean, taxonomy

EXTRACTIONS = Path("data/processed/extractions")
SNAPSHOTS = Path("data/processed/graph_snapshots")
KEYS = {"Candidate": "candidate_id", "JobPosting": "job_id", "Company": "company_id", "Occupation": "onet_soc_code",
        "Skill": "skill_id", "Requirement": "req_id", "Experience": "exp_id", "Credential": "cred_id",
        "Project": "proj_id", "Location": "loc_id", "Evidence": "evidence_id"}
EXTRACTED_NODES = ["Skill", "Requirement", "Experience", "Credential", "Project", "Location"]


def h(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def driver():
    load_dotenv()
    return GraphDatabase.driver(os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                                auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
                                notifications_min_severity="OFF")   # "property does not exist yet" warnings


class GraphBatch:
    """Collects nodes and relationships in memory, then writes them with UNWIND + MERGE."""

    def __init__(self):
        self.nodes = defaultdict(dict)           # label -> key -> props
        self.rels = defaultdict(dict)            # (type, from_label, to_label) -> edge_id -> row

    def node(self, label: str, key: str, **props) -> str:
        cur = self.nodes[label].setdefault(key, {KEYS[label]: key})
        cur.update({k: v for k, v in props.items() if v is not None})
        return key

    def rel(self, rtype: str, a: tuple, b: tuple, source: str, evidence_ids=(), **props) -> str:
        edge_id = h(rtype, *a, *b)
        row = self.rels[(rtype, a[0], b[0])].setdefault(
            edge_id, {"a": a[1], "b": b[1], "props": {"edge_id": edge_id, "source": source, "evidence_ids": []}})
        row["props"]["evidence_ids"] = sorted(set(row["props"]["evidence_ids"]) | set(evidence_ids))
        row["props"].update({k: v for k, v in props.items() if v is not None})
        return edge_id

    def evidence(self, doc_id, doc_type, ev: dict, extractor, version, text) -> str | None:
        if not ev or ev.get("char_start") is None:
            return None
        s, e = ev["char_start"], ev["char_end"]
        eid = h(doc_id, s, e, extractor)
        self.node("Evidence", eid, doc_id=doc_id, doc_type=doc_type, char_start=s, char_end=e, text=text[s:e],
                  extractor=extractor, extractor_version=version, confidence=ev.get("confidence"))
        return eid

    def write(self, drv, batch: int = 2000):
        with drv.session() as s:
            for label, key in KEYS.items():
                s.run(f"CREATE CONSTRAINT {label.lower()}_key IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE")
            s.run("CREATE INDEX evidence_doc IF NOT EXISTS FOR (n:Evidence) ON (n.doc_id)")
            for label, rows in self.nodes.items():
                rows = list(rows.values())
                for i in range(0, len(rows), batch):
                    s.run(f"UNWIND $rows AS r MERGE (n:{label} {{{KEYS[label]}: r.{KEYS[label]}}}) SET n += r",
                          rows=rows[i:i + batch])
            for (rtype, la, lb), rows in self.rels.items():
                rows = list(rows.values())
                for i in range(0, len(rows), batch):
                    s.run(f"UNWIND $rows AS r MATCH (a:{la} {{{KEYS[la]}: r.a}}) MATCH (b:{lb} {{{KEYS[lb]}: r.b}}) "
                          f"MERGE (a)-[e:{rtype} {{edge_id: r.props.edge_id}}]->(b) SET e += r.props",
                          rows=rows[i:i + batch])


def loc_key(loc: dict) -> str:
    parts = [clean(loc.get(k) or "") for k in ("city", "state", "country")]
    return "remote" if loc.get("is_remote") and not any(parts) else "|".join(parts)


def company_key(name: str) -> str:
    return re.sub(r"\b(inc|llc|ltd|corp|corporation|co|company|group)\b\.?", "", clean(name)).strip(" ,.") or clean(name)


class Builder:
    def __init__(self, docs: pd.DataFrame, normalizer: Normalizer, occ: OccupationNormalizer):
        self.g = GraphBatch()
        self.docs = docs.set_index("doc_id")
        self.norm, self.occ = normalizer, occ
        self.tax = taxonomy().set_index("taxonomy_id")
        self.public = set()
        self.skill_mentions = []         # (doc_id, name, context, callback args) to normalize in one batch
        self.stats = defaultdict(int)

    def add_location(self, owner: tuple, loc: dict | None, ctx):
        if not loc:
            return
        doc_id, doc_type, extractor, version = ctx
        eid = self.g.evidence(doc_id, doc_type, loc.get("evidence"), extractor, version, self.docs.at[doc_id, "text"])
        if not eid:
            self.stats["ungrounded_skipped"] += 1
            return
        key = loc_key(loc)
        self.g.node("Location", key, city=loc.get("city"), state=loc.get("state"), country=loc.get("country"),
                    is_remote=bool(loc.get("is_remote")))
        self.g.rel("SUPPORTED_BY", ("Location", key), ("Evidence", eid), "Extracted", [eid])
        self.g.rel("LOCATED_IN", owner, ("Location", key), "Extracted", [eid], extractor=extractor)

    def add_doc(self, row: dict):
        p = row["parsed"]
        doc_id, extractor, version = p["doc_id"], p["extractor"], p["extractor_version"]
        text = self.docs.at[doc_id, "text"]
        is_job = "job" in p
        doc_type = "JobPosting" if is_job else "CV"
        ctx = (doc_id, doc_type, extractor, version)
        ev = lambda item: self.g.evidence(doc_id, doc_type, (item or {}).get("evidence"), extractor, version, text)
        if is_job:
            j = p["job"]
            owner = ("JobPosting", doc_id)
            self.g.node("JobPosting", doc_id, title=j.get("title"), seniority=j.get("seniority"),
                        employment_type=j.get("employment_type"), work_mode=j.get("work_mode"),
                        source=self.docs.at[doc_id, "source"])
            for field, rtype in (("company", "POSTED_BY"), ("title", "FOR_ROLE")):
                name = j.get(field)
                span = find_span(text, name) if name and name != "Unknown" else None
                if not span:
                    continue
                eid = self.g.evidence(doc_id, doc_type, {"char_start": span[0], "char_end": span[1], "confidence": 1.0},
                                      extractor, version, text)
                if rtype == "POSTED_BY":
                    ck = company_key(name)
                    self.g.node("Company", ck, name=name)
                    self.g.rel("POSTED_BY", owner, ("Company", ck), "Extracted", [eid], extractor=extractor)
                else:
                    soc, title, score = self.occ.normalize([name])[0]
                    if soc:
                        self.g.node("Occupation", soc, title=title)
                        self.g.rel("FOR_ROLE", owner, ("Occupation", soc), "Extracted", [eid], extractor=extractor,
                                   score=round(score, 3))
            self.add_location(owner, j.get("location"), ctx)
            for i, r in enumerate(p.get("requirements", [])):
                eid = ev(r)
                if not eid:
                    self.stats["ungrounded_skipped"] += 1
                    continue
                rk = f"{doc_id}:r{i}"
                self.g.node("Requirement", rk, text=r["text"], requirement_type=r["requirement_type"],
                            importance=r["importance"], min_years=r.get("min_years"),
                            education_level=r.get("education_level"))
                self.g.rel("SUPPORTED_BY", ("Requirement", rk), ("Evidence", eid), "Extracted", [eid])
                self.g.rel("HAS_REQUIREMENT", owner, ("Requirement", rk), "Extracted", [eid], extractor=extractor)
                for s in r.get("skills", []):
                    self.skill_mentions.append((("Requirement", rk), "TARGETS", s, None, eid, r["text"],
                                                {"importance": r["importance"]}, doc_id))
        else:
            c = p.get("candidate") or {}
            owner = ("Candidate", doc_id)
            self.g.node("Candidate", doc_id, headline=c.get("headline"), seniority=c.get("seniority"),
                        total_years_experience=c.get("total_years_experience"), source=self.docs.at[doc_id, "source"])
            self.add_location(owner, c.get("location"), ctx)
            for s in p.get("skills", []):
                eid = ev(s)
                if not eid:
                    self.stats["ungrounded_skipped"] += 1
                    continue
                self.skill_mentions.append((owner, "HAS_SKILL", s["name"], s["skill_type"], eid,
                                            s["evidence"]["quote"], {"proficiency": s.get("proficiency")}, doc_id))
            for i, x in enumerate(p.get("experiences", [])):
                eid = ev(x)
                if not eid:
                    self.stats["ungrounded_skipped"] += 1
                    continue
                xk = f"{doc_id}:e{i}"
                self.g.node("Experience", xk, title=x["title"], start_date=x.get("start_date"),
                            end_date=x.get("end_date"), is_current=x.get("is_current"),
                            employment_type=x.get("employment_type"), duration_years=duration(x))
                self.g.rel("SUPPORTED_BY", ("Experience", xk), ("Evidence", eid), "Extracted", [eid])
                self.g.rel("HELD", owner, ("Experience", xk), "Extracted", [eid], extractor=extractor)
                if x.get("company"):
                    ck = company_key(x["company"])
                    self.g.node("Company", ck, name=x["company"])
                    self.g.rel("AT", ("Experience", xk), ("Company", ck), "Extracted", [eid], extractor=extractor)
                soc, title, score = self.occ.normalize([x["title"]])[0]
                if soc:
                    self.g.node("Occupation", soc, title=title)
                    self.g.rel("AS_ROLE", ("Experience", xk), ("Occupation", soc), "Extracted", [eid],
                               extractor=extractor, score=round(score, 3))
                for s in x.get("skills_used", []):
                    self.skill_mentions.append((("Experience", xk), "USED", s, None, eid, x["title"], {}, doc_id))
            for kind, label, rtype in (("credentials", "Credential", "EARNED"), ("projects", "Project", "BUILT")):
                for i, x in enumerate(p.get(kind, [])):
                    eid = ev(x)
                    if not eid:
                        self.stats["ungrounded_skipped"] += 1
                        continue
                    k = f"{doc_id}:{label[0].lower()}{i}"
                    props = {kk: x.get(kk) for kk in ("name", "credential_type", "level", "field", "issuer", "year",
                                                       "description") if x.get(kk) is not None}
                    self.g.node(label, k, **props)
                    self.g.rel("SUPPORTED_BY", (label, k), ("Evidence", eid), "Extracted", [eid])
                    self.g.rel(rtype, owner, (label, k), "Extracted", [eid], extractor=extractor)
                    for s in x.get("skills_used", []):
                        self.skill_mentions.append(((label, k), "USED", s, None, eid, x.get("name", ""), {}, doc_id))
        self.stats[f"docs_{doc_type}"] += 1

    def finalize(self):
        """Normalize all skill mentions in one batch, then create Skill nodes and their edges."""
        names = [m[2] for m in self.skill_mentions]
        ctxs = [m[5] for m in self.skill_mentions]
        uniq = list(dict.fromkeys(names))
        private = any(self.docs.at[m[7], "source"] not in self.public for m in self.skill_mentions)
        matches = dict(zip(uniq, self.norm.normalize(uniq, [ctxs[names.index(u)] for u in uniq], private=private)))
        for owner, rtype, name, stype, eid, _, props, doc_id in self.skill_mentions:
            m = matches[name]
            sid = m.taxonomy_id or f"free:{clean(name)}"
            self.g.node("Skill", sid, canonical_name=m.canonical_name if m.taxonomy_id else clean(name),
                        skill_type=m.skill_type or stype, taxonomy_id=m.taxonomy_id, category=m.category,
                        hot_technology=m.hot_technology)
            self.g.rel("SUPPORTED_BY", ("Skill", sid), ("Evidence", eid), "Extracted", [eid])
            self.g.rel(rtype, owner, ("Skill", sid), "Extracted", [eid], surface=name, match_method=m.method,
                       match_score=m.score, **props)
            self.stats[f"skill_{m.method}"] += 1
        self.add_taxonomy_edges()

    def add_taxonomy_edges(self):
        """TYPICAL_FOR from O*NET for skills and occupations already in the graph."""
        occs = set(self.g.nodes["Occupation"])
        for sid in list(self.g.nodes["Skill"]):
            if sid in self.tax.index:
                r = self.tax.loc[sid]
                for soc in set(r.occupations) & occs:
                    self.g.rel("TYPICAL_FOR", ("Skill", sid), ("Occupation", soc), "Taxonomy",
                               hot_technology=bool(r.hot_technology))
                    self.stats["typical_for"] += 1


def duration(x: dict) -> float | None:
    years = lambda s: (int(s[:4]) + (int(s[5:7]) - 1) / 12 if len(s) >= 7 and s[5:7].isdigit() else int(s[:4])) \
        if s and s[:4].isdigit() else None
    a = years(x.get("start_date") or "")
    end = x.get("end_date") or ""
    b = 2026.8 if end.lower() in ("present", "current") or x.get("is_current") else years(end)
    return round(b - a, 2) if a is not None and b is not None and b >= a else None


def infer_related(drv, router=None, k: int = 5, min_cos: float = 0.70, batch: int = 25):
    """RELATED_TO between graph skills: embedding kNN proposes, the LLM (route infer_edges) confirms."""
    from graph.normalize import Normalizer
    with drv.session() as s:
        skills = s.run("MATCH (s:Skill) RETURN s.skill_id AS id, s.canonical_name AS name").data()
    if len(skills) < 2:
        return 0
    model, _ = Normalizer("modern")._embed_index()
    names = [x["name"] for x in skills]
    emb = model.encode(names, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    with np.errstate(all="ignore"):
        sims = emb @ emb.T
    np.fill_diagonal(sims, -1)
    pairs = {}
    for i in range(len(skills)):
        for j in np.argsort(-sims[i])[:k]:
            if sims[i, j] >= min_cos and names[i].lower() != names[j].lower():
                pairs[tuple(sorted((i, int(j))))] = float(sims[i, j])
    pairs = list(pairs.items())
    keep = []
    for n in range(0, len(pairs), batch):
        chunk = pairs[n:n + batch]
        if router is None:
            keep += [(p, w, "Embedding") for p, w in chunk]
            continue
        listing = "\n".join(f"{q}. {names[a]} <-> {names[b]}" for q, ((a, b), _) in enumerate(chunk))
        prompt = ("For each pair of skills, answer whether they are closely related (someone with one can "
                  "readily apply or learn the other, or they are commonly used together).\n" + listing +
                  '\nReply JSON {"related": [indices of related pairs]}.')
        try:
            reply = router.chat("infer_edges", [{"role": "user", "content": prompt}], private=False)
            ok = set(json.loads(reply.text).get("related", []))
        except Exception:
            ok = set()
        keep += [(p, w, "LLM-inferred") for q, (p, w) in enumerate(chunk) if q in ok]
    rows = [{"a": skills[a]["id"], "b": skills[b]["id"],
             "props": {"edge_id": h("RELATED_TO", skills[a]["id"], skills[b]["id"]), "source": src, "weight": w,
                       "inferred": True, "evidence_ids": []}} for (a, b), w, src in keep]
    with drv.session() as s:
        s.run("UNWIND $rows AS r MATCH (a:Skill {skill_id: r.a}) MATCH (b:Skill {skill_id: r.b}) "
              "MERGE (a)-[e:RELATED_TO {edge_id: r.props.edge_id}]->(b) SET e += r.props", rows=rows)
    return len(rows)


def louvain(drv) -> dict:
    with drv.session() as s:
        s.run("CALL gds.graph.drop('skills', false) YIELD graphName RETURN graphName").consume()
        s.run("CALL gds.graph.project('skills', 'Skill', {RELATED_TO: {orientation: 'UNDIRECTED', properties: 'weight'}})")
        r = s.run("CALL gds.louvain.write('skills', {writeProperty: 'community', relationshipWeightProperty: 'weight'}) "
                  "YIELD communityCount, modularity RETURN communityCount, modularity").single()
        s.run("CALL gds.graph.drop('skills', false) YIELD graphName RETURN graphName").consume()
    return dict(r)


def check(drv) -> dict:
    """Evidence audit: every extracted relationship and node must resolve to Evidence. Raises on failure."""
    q = {
        "extracted_rels_without_evidence":
            "MATCH ()-[r]->() WHERE r.source = 'Extracted' AND (r.evidence_ids IS NULL OR size(r.evidence_ids) = 0) "
            "RETURN count(r) AS n",
        "extracted_rels_with_dangling_evidence":
            "MATCH ()-[r]->() WHERE r.source = 'Extracted' UNWIND r.evidence_ids AS eid "
            "OPTIONAL MATCH (e:Evidence {evidence_id: eid}) WITH eid, e WHERE e IS NULL RETURN count(*) AS n",
        "extracted_nodes_without_evidence":
            "MATCH (n) WHERE any(l IN labels(n) WHERE l IN $labels) AND NOT (n)-[:SUPPORTED_BY]->(:Evidence) "
            "RETURN count(n) AS n",
        "rels_without_edge_id": "MATCH ()-[r]->() WHERE r.edge_id IS NULL RETURN count(r) AS n",
    }
    with drv.session() as s:
        out = {k: s.run(v, labels=EXTRACTED_NODES).single()["n"] for k, v in q.items()}
        out["nodes"] = {r["l"]: r["n"] for r in s.run("MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS n")}
        out["rels"] = {r["t"]: r["n"] for r in s.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n")}
    bad = {k: v for k, v in out.items() if isinstance(v, int) and v}
    if bad:
        raise AssertionError(f"evidence check failed: {bad}")
    return out


def snapshot(drv, tag: str) -> Path:
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOTS / f"{tag}.jsonl.gz"
    with drv.session() as s, gzip.open(path, "wt") as f:
        for r in s.run("MATCH (n) RETURN labels(n) AS labels, properties(n) AS props"):
            f.write(json.dumps({"node": r["labels"], "props": r["props"]}, default=str) + "\n")
        for r in s.run("MATCH (a)-[e]->(b) RETURN type(e) AS t, labels(a)[0] AS la, labels(b)[0] AS lb, "
                       "properties(a) AS pa, properties(b) AS pb, properties(e) AS props"):
            f.write(json.dumps({"rel": r["t"], "from": [r["la"], r["pa"].get(KEYS[r["la"]])],
                                "to": [r["lb"], r["pb"].get(KEYS[r["lb"]])], "props": r["props"]}, default=str) + "\n")
    return path


def load_extractions(names: list[str]) -> list[dict]:
    """Latest run0 row per doc across the given extractor dirs (earlier names win on conflicts)."""
    rows = {}
    for name in names:
        f = EXTRACTIONS / name / "run0.jsonl"
        if f.exists():
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    rows.setdefault(r["parsed"]["doc_id"], r)
    return list(rows.values())


def main():
    import yaml
    from router.router import Router
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractions", nargs="*", default=[])
    ap.add_argument("--wipe", action="store_true", help="delete the graph first")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="RELATED_TO from embeddings only")
    ap.add_argument("--snapshot", default="")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml"))
    drv = driver()
    if a.wipe:
        with drv.session() as s:
            s.run("MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS")
    if a.extractions:
        from extraction.run import load_docs
        rows = load_extractions(a.extractions)
        ids = [r["parsed"]["doc_id"] for r in rows]
        Path("data/interim").mkdir(parents=True, exist_ok=True)
        Path("data/interim/graph_ids.txt").write_text("\n".join(ids))
        docs = load_docs("ids:data/interim/graph_ids.txt")
        router = None if a.no_llm else Router(cfg)
        b = Builder(docs, Normalizer(cfg.get("graph", {}).get("normalizer", "modern"), router), OccupationNormalizer())
        b.public = set(cfg["public_sources"])
        for r in rows:
            b.add_doc(r)
        b.finalize()
        b.g.write(drv)
        print("loaded", dict(b.stats))
        print("RELATED_TO edges:", infer_related(drv, router))
        print("louvain:", louvain(drv))
    if a.check or a.extractions:
        print("check:", json.dumps(check(drv), indent=1))
    if a.snapshot:
        print("snapshot ->", snapshot(drv, a.snapshot))


if __name__ == "__main__":
    main()
