"""Claude's gold normalization labels (D30) for data/processed/annotation/skill_mentions_500.csv.

  python scripts/mentions.py show <start> <end>        # compact view: mention, context, candidates 0-4, classical pick
  python scripts/mentions.py set "i:k i:k ..."         # k = 0-4 (candidate), c (classical pick), n (none)
"""
import sys
import pandas as pd
P = "data/processed/annotation/skill_mentions_500.csv"
m = pd.read_csv(P, dtype=str).fillna("")
cands = lambda r: [c.split(" = ", 1) for c in r["candidates"].split(" || ") if c.strip()]
if sys.argv[1] == "show":
    for i in range(int(sys.argv[2]), min(int(sys.argv[3]), len(m))):
        r = m.loc[i]
        cs = " | ".join(f"{k}:{n[:38]}" for k, (_, n) in enumerate(cands(r)))
        print(f"[{i}] {r['mention']!r} ctx: {r['context'][:90]!r}\n     {cs} || c:{r['classical_name'][:38] or 'none'}")
else:
    for kv in sys.argv[2].split():
        i, k = kv.split(":")
        r = m.loc[int(i)]
        m.at[int(i), "llm_gold_taxonomy_id"] = ("none" if k == "n" else r["classical_id"] if k == "c"
                                                else cands(r)[int(k)][0])
    m.to_csv(P, index=False)
    print((m.llm_gold_taxonomy_id != "").sum(), "of", len(m), "labeled")
