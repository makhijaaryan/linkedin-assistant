"""Interactive knowledge graph: the CV's neighbourhood (and one job's), coloured by ontology type; click a node to
see its edges and the Evidence spans they are SUPPORTED_BY. (Layout and colours follow Krish Naik's graph page.)"""

import html

import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

from ui import services as svc
from ui.components import page_header, require_profile

page_header("Knowledge graph", "Explore how your CV and a job connect: skills, requirements, roles and the evidence behind each link.")
p = require_profile()
if not p or not p.get("parsed"):
    st.stop()

COLORS = {"Candidate": "#7c3aed", "Skill": "#2563eb", "Occupation": "#059669", "JobPosting": "#d97706",
          "Company": "#6b7280", "Requirement": "#f59e0b", "Experience": "#db2777", "Credential": "#0891b2",
          "Location": "#64748b"}
c1, c2, c3 = st.columns([3, 2, 1])
labels = c1.multiselect("Node types", list(COLORS), default=["Candidate", "Skill", "Occupation", "JobPosting",
                                                             "Requirement", "Experience", "Credential"])
jobs = {r["doc_id"]: f"{r['rank']}. {r.get('title')}" for r in (st.session_state.get("matches") or {}).get("results", [])}
sel = st.session_state.get("selected_job")
job_id = c2.selectbox("Include job", [None] + list(jobs), format_func=lambda x: "— none —" if x is None else jobs[x],
                      index=list(jobs).index(sel) + 1 if sel in jobs else 0)
limit = c3.slider("Max nodes", 20, 300, 120)
if job_id and job_id not in st.session_state.get("explain", {}):
    st.caption("This job is added to the graph when you open “Why this job?” or Skill gap for it.")

g = svc.subgraph(p["cv_id"], job_id, labels + ["Candidate"], limit)
if not g["nodes"]:
    st.info("Nothing to draw yet for these node types.")
    st.stop()
shown = [l for l in COLORS if any(n["label"] == l for n in g["nodes"])]
st.markdown(" ".join(f"<span style='display:inline-block;margin:0 .9rem .3rem 0;font-size:.86rem'>"
                     f"<span style='display:inline-block;width:.75rem;height:.75rem;border-radius:50%;"
                     f"background:{COLORS[l]};margin-right:.35rem'></span>{html.escape(l)}</span>" for l in shown)
            + f"<span class='om-muted' style='font-size:.82rem'>· {len(g['nodes'])} nodes, {len(g['edges'])} links · "
              "dashed = inferred (RELATED_TO / taxonomy)</span>", unsafe_allow_html=True)


def name(n: dict) -> str:
    return str(n.get("name") or n.get("canonical_name") or n.get("title") or n.get("text") or n.get("candidate_id")
               or n.get("job_id") or n["label"])[:40]


nodes = [Node(id=n["id"], label=name(n), size=22 if n["label"] in ("Candidate", "JobPosting") else 12,
              color=COLORS.get(n["label"], "#999"), title=f"{n['label']}: {name(n)}") for n in g["nodes"]]
inferred = lambda e: e["type"] in ("RELATED_TO", "TYPICAL_FOR") or (e["props"] or {}).get("source") in ("embedding",
                                                                                                     "onet")
edges = [Edge(source=e["src"], target=e["dst"], title=e["type"].replace("_", " ").lower(),
              color="#c0c6cf" if inferred(e) else "#8a94a3", dashes=inferred(e)) for e in g["edges"]]
cfg = Config(height=680, directed=True, physics=True, hierarchical=False)
cfg.width = "100%"
if getattr(cfg, "groups", None) is None and hasattr(cfg, "groups"):
    del cfg.groups
left, right = st.columns([5, 2])
with left:
    clicked = agraph(nodes=nodes, edges=edges, config=cfg)
with right:
    st.subheader("Inspector")
    node = next((n for n in g["nodes"] if n["id"] == clicked), None)
    if not node:
        st.caption("Click a node to see its properties and the evidence behind its edges.")
    else:
        st.markdown(f"**{node['label']}** · {html.escape(name(node))}")
        props = {k: v for k, v in node.items() if k not in ("id", "label") and not isinstance(v, (list, dict))}
        st.json(props, expanded=False)
        ev = svc.node_evidence(node["id"])
        if len(ev):
            st.markdown("**Evidence (SUPPORTED_BY)**")
            for r in ev.itertuples():
                st.markdown(f"- *{r.relation.lower()}* → {html.escape(str(r.other))}  \n> “{html.escape(str(r.evidence)[:220])}”")
        else:
            st.caption("No evidence-backed edges on this node (taxonomy nodes come from O*NET).")
