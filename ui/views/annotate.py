"""Annotation: the human side of D30. Claude labels everything in its own columns; here you spot-check a fixed
random sample of each set, blind to Claude's label, and the page reports agreement (Cohen's κ). Explanation
faithfulness stays fully human. Every click is saved to the CSV immediately."""

import json

import numpy as np
import pandas as pd
import streamlit as st

from ui import services as svc

ANN = svc.ROOT / "data/processed/annotation"
SEED = 573
st.title("Annotation")
st.caption("Claude's labels stay hidden until you've judged an item. Your labels go in the human column of each "
           "file; Claude's are in the `llm_…` column. Results report Claude-judged labels plus your agreement (D30).")


def kappa(a: pd.Series, b: pd.Series) -> float | None:
    a, b = a.astype(str), b.astype(str)
    if len(a) < 5:
        return None
    cats = sorted(set(a) | set(b))
    po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    return None if pe == 1 else (po - pe) / (1 - pe)


def sample_ids(df: pd.DataFrame, n: int) -> list:
    """A fixed random sample (seeded by the file's own index): it doesn't move as Claude labels more rows."""
    order = np.random.default_rng(SEED).permutation(len(df))
    return sorted(df.index.to_numpy()[order[:n]].tolist())


def agreement(df, human, llm, idx):
    done = df.loc[idx]
    done = done[(done[human] != "") & (done[llm] != "")]
    k = kappa(done[human], done[llm])
    acc = float((done[human].astype(str) == done[llm].astype(str)).mean()) if len(done) else None
    c1, c2, c3 = st.columns(3)
    c1.metric("Checked", f"{int((df.loc[idx, human] != '').sum())} / {len(idx)}")
    c2.metric("Agreement with Claude", "—" if acc is None else f"{acc:.0%}")
    c3.metric("Cohen's κ", "—" if k is None else f"{k:.2f}")


def judge_csv(path, human, llm, n, show, choices, extra_filter=None):
    if not path.exists():
        st.info(f"{path.name} does not exist yet.")
        return
    df = pd.read_csv(path, dtype=str).fillna("")
    for col in (human, llm):
        if col not in df:
            df[col] = ""
    pool = df if extra_filter is None else df[extra_filter(df)]
    if (pool[llm] == "").all():
        st.info("Claude hasn't labeled this set yet; the spot-check opens once it has.")
        return
    idx = sample_ids(pool, n)
    agreement(df, human, llm, idx)
    todo = [i for i in idx if df.at[i, human] == "" and df.at[i, llm] != ""]
    if not todo:
        waiting = sum(df.at[i, llm] == "" for i in idx)
        if waiting:
            st.info(f"{waiting} sampled items are waiting for Claude's label.")
            return
        st.success("Spot-check complete.")
        diff = df.loc[idx][df.loc[idx, human] != df.loc[idx, llm]]
        if len(diff):
            with st.expander(f"{len(diff)} disagreements"):
                st.dataframe(diff, width="stretch")
        return
    i = todo[0]
    show(df.loc[i])
    cols = st.columns(len(choices))
    for (val, label), col in zip(choices, cols):
        if col.button(label, key=f"{path.name}-{i}-{val}", width="stretch"):
            df.at[i, human] = val
            df.to_csv(path, index=False)
            st.rerun()


tabs = st.tabs(["Relevance (≈120)", "Gold extraction (≈30)", "Dedupe (≈50)", "Graph mentions & edges",
                "Explanations (60, human)"])

with tabs[0]:
    res = svc.resumes_meta().set_index("doc_id")
    public = lambda d: d["profile_id"].map(res["source"]) == "kaggle_resume_dataset"

    def show_pair(r):
        l, rt = st.columns(2)
        with l:
            st.subheader(f"CV {r['profile_id']}")
            st.text_area("cv", res.at[r["profile_id"], "text"], height=480, label_visibility="collapsed",
                         key=f"cv-{r.name}")
        with rt:
            m = svc.job_meta(r["job_id"])
            st.subheader(f"{m['title']} — {m['company']}")
            st.text_area("job", svc.job_text(r["job_id"]), height=480, label_visibility="collapsed", key=f"job-{r.name}")
    st.caption("2 = strong fit (meets most required qualifications) · 1 = partial fit · 0 = not a fit. Judge stated "
               "qualifications only, not location or salary.")
    judge_csv(ANN / "relevance_pool.csv", "label", "llm_label", 120, show_pair,
              [("0", "0 · not a fit"), ("1", "1 · partial"), ("2", "2 · strong")], extra_filter=public)

with tabs[1]:
    gold_p, chk_p = ANN / "gold_extraction.json", ANN / "gold_spotcheck.json"
    if not gold_p.exists():
        st.info("Claude's gold labels aren't written yet.")
    else:
        gold = json.loads(gold_p.read_text())
        chk = json.loads(chk_p.read_text()) if chk_p.exists() else {}
        rng = np.random.default_rng(SEED)
        ids = [t["data"]["doc_id"] for t in gold]
        sample = sorted(rng.permutation(ids)[:30].tolist())
        kept = sum(len(v["kept"]) for v in chk.values())
        total = sum(len(v["kept"]) + len(v["removed"]) for v in chk.values())
        added = sum(len(v["added"]) for v in chk.values())
        c1, c2, c3 = st.columns(3)
        c1.metric("Docs checked", f"{len(chk)} / {len(sample)}")
        c2.metric("Claude's entities you kept", "—" if not total else f"{kept / total:.0%}")
        c3.metric("Entities you added", added)
        todo = [d for d in sample if d not in chk]
        if not todo:
            st.success("Spot-check complete.")
        else:
            task = next(t for t in gold if t["data"]["doc_id"] == todo[0])
            d = task["data"]
            ents = sorted({(r["value"]["labels"][0], r["value"]["text"]) for r in task["annotations"][-1]["result"]})
            st.markdown(f"**{d['doc_id']}** · {d['doc_type']}")
            l, r = st.columns([3, 2])
            l.text_area("text", d["text"], height=520, label_visibility="collapsed")
            with r:
                st.caption("Untick anything wrong; add anything missing as `Label: text`, one per line.")
                keep = {e: st.checkbox(f"{e[0]}: {e[1]}", value=True, key=f"g-{d['doc_id']}-{n}")
                        for n, e in enumerate(ents)}
                add = st.text_area("Missing entities", key=f"add-{d['doc_id']}", height=100)
                if st.button("Save and next", type="primary"):
                    chk[d["doc_id"]] = {"kept": [list(e) for e, v in keep.items() if v],
                                        "removed": [list(e) for e, v in keep.items() if not v],
                                        "added": [x.strip() for x in add.splitlines() if x.strip()]}
                    chk_p.write_text(json.dumps(chk, indent=1))
                    st.rerun()

with tabs[2]:
    def show_dup(r):
        st.caption(f"Estimated Jaccard {r['est_jaccard']} ({r['band']})")
        l, rt = st.columns(2)
        for col, s in ((l, "a"), (rt, "b")):
            with col:
                st.markdown(f"**{r[f'title_{s}']}** — {r[f'company_{s}']}")
                st.text_area(s, r[f"text_{s}"], height=380, label_visibility="collapsed", key=f"d{s}-{r.name}")
    st.caption("1 = the same posting (reposts, minor edits, same text at another location) · 0 = genuinely different.")
    judge_csv(ANN / "dedupe_handcheck_jobs.csv", "is_duplicate", "llm_is_duplicate", 50, show_dup,
              [("0", "0 · different"), ("1", "1 · duplicate")])

with tabs[3]:
    st.markdown("**Skill mentions:** pick the O\\*NET entry that is the same skill (or `none`). Claude's pick stays "
                "hidden until you choose.")
    mp = ANN / "skill_mentions_500.csv"
    if not mp.exists():
        st.info("The mention sheet is generated after the extraction runs (`python -m graph.annotate --make`).")
    else:
        m = pd.read_csv(mp, dtype=str).fillna("")
        idx = sample_ids(m, 50)
        done = [i for i in idx if m.at[i, "gold_taxonomy_id"] != ""]
        both = [i for i in done if m.at[i, "llm_gold_taxonomy_id"] != ""]
        agree = sum(m.at[i, "gold_taxonomy_id"] == m.at[i, "llm_gold_taxonomy_id"] for i in both)
        c1, c2 = st.columns(2)
        c1.metric("Checked", f"{len(done)} / {len(idx)}")
        c2.metric("Same pick as Claude", f"{agree / len(both):.0%}" if both else "—")
        todo = [i for i in idx if m.at[i, "gold_taxonomy_id"] == "" and m.at[i, "llm_gold_taxonomy_id"] != ""]
        if todo:
            r = m.loc[todo[0]]
            st.markdown(f"**Mention:** `{r['mention']}`  \n**Context:** {r['context']}")
            opts = ["none"] + [c.split(" = ", 1)[0].strip() for c in r["candidates"].split(" || ") if c.strip()]
            names = {c.split(" = ", 1)[0].strip(): c.split(" = ", 1)[-1] for c in r["candidates"].split(" || ")}
            pick = st.radio("Same skill as", opts, format_func=lambda o: o if o == "none" else f"{names.get(o, '')}  ({o})",
                            key=f"m-{todo[0]}")
            if st.button("Save", key=f"ms-{todo[0]}"):
                m.at[todo[0], "gold_taxonomy_id"] = pick
                m.to_csv(mp, index=False)
                st.rerun()
        elif idx:
            st.success("Mention spot-check complete." if done else "Waiting for Claude's labels.")
    st.divider()

    def show_edge(r):
        st.markdown(f"**{r.get('from_node', '')}** ({r.get('from_type', '')}) —*{r.get('rel', '')}*→ "
                    f"**{r.get('to_node', '')}** ({r.get('to_type', '')})")
        st.markdown(f"> {r.get('evidence_text', '')}")
    st.markdown("**Graph edges:** does the evidence span support the edge?")
    judge_csv(ANN / "edge_sample_100.csv", "is_correct", "llm_is_correct", 50, show_edge,
              [("0", "0 · not supported"), ("1", "1 · supported")])

with tabs[4]:
    sheet = ANN / "explanations_60.csv"

    def show_sent(r):
        st.markdown(f"**Sentence:** {r['text']}")
        st.markdown(f"**Facts it cites:** {r['facts_cited'] or '— (no citation)'}")
        st.caption(f"Pair {r['pair']} · method hidden")
    if not sheet.exists():
        st.info("The 60-explanation sheet is generated after the graph rebuild (`python -m eval.explanation_eval --make`).")
    else:
        df = pd.read_csv(sheet, dtype=str).fillna("")
        done = int((df["supported"] != "").sum())
        st.metric("Sentences judged", f"{done} / {len(df)}")
        todo = df.index[df["supported"] == ""].tolist()
        if todo:
            i = todo[0]
            show_sent(df.loc[i])
            c1, c2 = st.columns(2)
            for val, label, col in (("1", "Supported", c1), ("0", "Unsupported", c2)):
                if col.button(label, key=f"ex-{i}-{val}", width="stretch"):
                    df.at[i, "supported"] = val
                    df.to_csv(sheet, index=False)
                    st.rerun()
        else:
            st.success("All sentences judged.")
