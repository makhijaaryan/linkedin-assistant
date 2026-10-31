"""Write Claude's dedupe hand-check labels (D30): apply_dupes.py <jobs|resumes> "i:l i:l ..." (l = 0/1), or a range "a-b:l"."""
import sys
import pandas as pd
P = f"data/processed/annotation/dedupe_handcheck_{sys.argv[1]}.csv"
d = pd.read_csv(P, dtype=str).fillna("")
if "llm_is_duplicate" not in d:
    d["llm_is_duplicate"] = ""
for kv in sys.argv[2].split():
    k, l = kv.split(":")
    assert l in "01"
    lo, hi = (int(x) for x in k.split("-")) if "-" in k else (int(k), int(k))
    for i in range(lo, hi + 1):
        d.at[i, "llm_is_duplicate"] = l
d.to_csv(P, index=False)
print((d.llm_is_duplicate != "").sum(), "of", len(d), "labeled by Claude")
