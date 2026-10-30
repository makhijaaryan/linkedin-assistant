"""Write Claude's relevance labels: apply_labels.py "i:l i:l ..."  (row index in relevance_pool.csv : 0/1/2)."""
import sys
import pandas as pd
P = 'data/processed/annotation/relevance_pool.csv'
p = pd.read_csv(P, dtype=str).fillna('')
for kv in sys.argv[1].split():
    i, l = kv.split(':'); assert l in '012'; p.at[int(i), 'llm_label'] = l
p.to_csv(P, index=False)
print((p.llm_label != '').sum(), 'of', len(p), 'labeled by Claude')
