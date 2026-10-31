"""Models & router: every route with where it runs (local / hosted / proprietary), a live model swap, the privacy
guard in action, and live cost / latency per model from the router log (the raw material of the RQ1 matrix)."""

import pandas as pd
import plotly.express as px
import streamlit as st

from router.router import PrivacyError, get_router
from ui import services as svc
from ui.components import badge_html, provider_kind

st.title("Models & router")
router = get_router()
cfg = router.cfg

st.subheader("Routes")
st.caption("The agent pipeline is fixed; each task is sent to the model on its route. A permanent change is one "
           "line in config.yaml.")
rows = []
for task, r in cfg["routes"].items():
    kind = provider_kind(cfg, r["provider"], r["model"])
    fb = r.get("fallback")
    rows.append(f"<tr><td><b>{task}</b></td><td>{r['provider']} / {r['model']}</td><td>{badge_html(kind, {'local': 'good', 'hosted': 'info', 'proprietary': 'warn'}[kind])}"
                f"</td><td>{(fb['provider'] + ' / ' + fb['model']) if fb else '—'}</td></tr>")
st.markdown("<table class='om-table'><tr><th>Task</th><th>Model</th><th>Runs on</th><th>Fallback</th></tr>"
            + "".join(rows) + "</table>", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1, st.container(border=True):
    st.markdown("**Live swap** (this session only)")
    task = st.selectbox("Task", list(cfg["routes"]))
    choices = [f"{m['provider']}/{m['model']}" for m in cfg["benchmark_models"]]
    new = st.selectbox("Model", choices)
    if st.button("Swap"):
        prov, mod = new.split("/", 1)
        cfg["routes"][task] = {"provider": prov, "model": mod}
        st.success(f"{task} → {new}. Private tasks will refuse it if it is hosted.")
with c2, st.container(border=True):
    st.markdown("**Privacy guard** (D2)")
    st.caption("A private CV may only reach providers marked private_ok (local Ollama). Try to send one to "
               "ASU Voyager:")
    if st.button("Send a private CV to a hosted model"):
        try:
            router.chat("test", [{"role": "user", "content": "hi"}], private=True, provider="asu", model="gpt-oss-120b")
            st.error("Not blocked — check config.yaml.")
        except PrivacyError as e:
            st.success(f"Blocked by the router: {e}")

st.subheader("Providers")
prov = []
for name, p in cfg["providers"].items():
    prov.append({"provider": name, "base_url": p["base_url"], "private_ok": p["private_ok"],
                 "enabled": p.get("enabled", True), "rpm limit": p.get("rpm"),
                 "used by": ", ".join(sorted({t for t, r in cfg["routes"].items() if r["provider"] == name})) or "—"})
st.dataframe(pd.DataFrame(prov), hide_index=True, width="stretch")

st.subheader("RQ1 benchmark progress")
st.caption("Hosted models: 120 public test docs × 3 runs. Local models: all 150 test docs × 1 run (D25).")
bp = svc.benchmark_progress()
st.dataframe(bp, hide_index=True, width="stretch",
             column_config={"progress": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})

st.subheader("Cost and latency (live calls from router/logs/calls.jsonl)")
df = svc.router_log()
if df.empty:
    st.info("No router calls logged yet.")
    st.stop()
live = df[~df["cached"] & df["task"].isin(["extract_job", "extract_cv"])]
agg = live.groupby(["provider", "model"]).agg(
    calls=("latency_s", "size"), p50_s=("latency_s", "median"), p95_s=("latency_s", lambda x: x.quantile(0.95)),
    prompt_tokens=("prompt_tokens", "mean"), completion_tokens=("completion_tokens", "mean"),
    usd=("cost_usd", lambda x: x.sum(min_count=1))).reset_index()
agg["usd_per_1k_calls"] = 1000 * agg["usd"] / agg["calls"]
agg["runs on"] = [provider_kind(cfg, p_, m_) for p_, m_ in zip(agg["provider"], agg["model"])]
st.dataframe(agg.round(3), hide_index=True, width="stretch")
st.caption("Extraction calls only (one call = one chunk). Hosted costs use list / indicative prices from "
           "config.yaml; ASU Voyager itself is free and local Ollama costs 0. Final per-document numbers with "
           "confidence intervals are on the Results page.")
plot = agg.dropna(subset=["p50_s"])
if len(plot):
    st.plotly_chart(px.scatter(plot, x="p50_s", y="usd_per_1k_calls", color="runs on", text="model", size="calls",
                               log_x=True, labels={"p50_s": "median latency per call (s, log)",
                                                   "usd_per_1k_calls": "$ per 1K calls (list price)"})
                    .update_traces(textposition="top center").update_layout(height=420), width="stretch")
