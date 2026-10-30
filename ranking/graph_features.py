"""Graph features for (candidate, job) pairs, read from Neo4j (R4 / RQ4).

The relevant subgraph (skill sets, requirements, occupations, RELATED_TO, Louvain communities) is pulled
once into memory; per-pair features are then cheap. A job or candidate missing from the graph gets
NaN features (LightGBM handles missing values), never a guess.
"""

from collections import defaultdict

import networkx as nx
import numpy as np

from graph.build import driver

MAX_PATH = 4
FEATURES = ["g_req_coverage", "g_pref_coverage", "g_n_required", "g_bridges", "g_mean_path", "g_exp_gap",
            "g_occ_match", "g_community_overlap"]


class GraphView:
    def __init__(self, drv=None):
        drv = drv or driver()
        with drv.session() as s:
            self.cand_skills = defaultdict(set)
            for r in s.run("MATCH (c:Candidate)-[:HAS_SKILL]->(k:Skill) RETURN c.candidate_id AS c, k.skill_id AS k"):
                self.cand_skills[r["c"]].add(r["k"])
            for r in s.run("MATCH (c:Candidate)-[:HELD]->(:Experience)-[:USED]->(k:Skill) "
                           "RETURN c.candidate_id AS c, k.skill_id AS k"):
                self.cand_skills[r["c"]].add(r["k"])
            self.cand_years = {}
            for r in s.run("MATCH (c:Candidate) OPTIONAL MATCH (c)-[:HELD]->(x:Experience) "
                           "RETURN c.candidate_id AS c, c.total_years_experience AS t, sum(x.duration_years) AS d"):
                self.cand_years[r["c"]] = r["t"] if r["t"] is not None else (r["d"] or None)
            self.cand_occ = defaultdict(set)
            for r in s.run("MATCH (c:Candidate)-[:HELD]->(:Experience)-[:AS_ROLE]->(o:Occupation) "
                           "RETURN c.candidate_id AS c, o.onet_soc_code AS o"):
                self.cand_occ[r["c"]].add(r["o"])
            self.job_req = defaultdict(lambda: {"Required": set(), "Preferred": set()})
            self.job_years = {}
            for r in s.run("MATCH (j:JobPosting)-[:HAS_REQUIREMENT]->(q:Requirement) "
                           "OPTIONAL MATCH (q)-[:TARGETS]->(k:Skill) "
                           "RETURN j.job_id AS j, q.importance AS imp, q.min_years AS y, collect(k.skill_id) AS ks"):
                self.job_req[r["j"]][r["imp"] or "Required"].update(r["ks"])
                if r["y"] is not None:
                    self.job_years[r["j"]] = max(self.job_years.get(r["j"], 0), r["y"])
            self.job_ids = {r["j"] for r in s.run("MATCH (j:JobPosting) RETURN j.job_id AS j")}
            self.cand_ids = {r["c"] for r in s.run("MATCH (c:Candidate) RETURN c.candidate_id AS c")}
            self.job_occ = defaultdict(set)
            for r in s.run("MATCH (j:JobPosting)-[:FOR_ROLE]->(o:Occupation) RETURN j.job_id AS j, o.onet_soc_code AS o"):
                self.job_occ[r["j"]].add(r["o"])
            self.related = nx.Graph()
            for r in s.run("MATCH (a:Skill)-[e:RELATED_TO]->(b:Skill) RETURN a.skill_id AS a, b.skill_id AS b, "
                           "e.edge_id AS id"):
                self.related.add_edge(r["a"], r["b"], edge_id=r["id"])
            self.community = {r["k"]: r["c"] for r in s.run(
                "MATCH (k:Skill) WHERE k.community IS NOT NULL RETURN k.skill_id AS k, k.community AS c")}

    def path_len(self, sources: set, target: str) -> int:
        if target in sources:
            return 0
        if target not in self.related:
            return MAX_PATH + 1
        lengths = nx.single_source_shortest_path_length(self.related, target, cutoff=MAX_PATH)
        hits = [lengths[s] for s in sources if s in lengths]
        return min(hits) if hits else MAX_PATH + 1

    def features(self, cand_id: str, job_id: str) -> dict:
        if cand_id not in self.cand_ids or job_id not in self.job_ids:
            return {f: np.nan for f in FEATURES}
        have = self.cand_skills.get(cand_id, set())
        req, pref = self.job_req[job_id]["Required"], self.job_req[job_id]["Preferred"]
        unmet = req - have
        paths = [self.path_len(have, k) for k in unmet]
        bridges = sum(1 for k in unmet if k in self.related and any(n in have for n in self.related.neighbors(k)))
        cy, jy = self.cand_years.get(cand_id), self.job_years.get(job_id)
        comm_c = {self.community[k] for k in have if k in self.community}
        comm_j = {self.community[k] for k in req | pref if k in self.community}
        return {
            "g_req_coverage": len(req & have) / len(req) if req else np.nan,
            "g_pref_coverage": len(pref & have) / len(pref) if pref else np.nan,
            "g_n_required": len(req),
            "g_bridges": bridges,
            "g_mean_path": float(np.mean(paths)) if paths else 0.0,
            "g_exp_gap": max(0.0, jy - cy) if cy is not None and jy is not None else np.nan,
            "g_occ_match": float(bool(self.cand_occ.get(cand_id, set()) & self.job_occ.get(job_id, set()))),
            "g_community_overlap": len(comm_c & comm_j) / len(comm_j) if comm_j else np.nan,
        }
