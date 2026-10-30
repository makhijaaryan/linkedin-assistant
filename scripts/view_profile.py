"""Print one public profile and its 20 pooled jobs compactly (for labeling). Usage: view_profile.py <n>"""
import re, sys
import pandas as pd
p = pd.read_csv('data/processed/annotation/relevance_pool.csv', dtype=str).fillna('')
cvs = pd.read_parquet('data/processed/resumes.parquet', columns=['doc_id', 'source', 'text']).set_index('doc_id')
jobs = pd.read_parquet('data/processed/jobs.parquet', columns=['doc_id', 'title', 'company', 'text']).set_index('doc_id')
pub = [x for x in p.profile_id.unique() if cvs.at[x, 'source'] == 'kaggle_resume_dataset']
pid = pub[int(sys.argv[1])]
print('PROFILE', pid, f'({sys.argv[1]}/{len(pub)})'); print(cvs.at[pid, 'text'][:1800]); print()
for i, r in p[p.profile_id == pid].iterrows():
    j = jobs.loc[r.job_id]; t = re.sub(r'\s+', ' ', j.text)
    m = re.search(r'(requirements|qualifications|what you|skills)', t, re.I)
    print(f'[{i}] {j.title} | {j.company}\n    ', t[m.start():m.start() + 450] if m else t[:450]); 
