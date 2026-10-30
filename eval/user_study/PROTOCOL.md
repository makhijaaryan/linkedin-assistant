# User Study Protocol (Phase 9)

**Question:** do cited LLM explanations (checked by the critic) help job seekers more than the template baseline?
**Design:** blind, within-subject A/B. Every participant rates both conditions on the same cases.

## Participants
- 15–20 adults (classmates, friends), recruited informally. No compensation.
- Check with the instructor/TA whether course projects need IRB review at ASU before recruiting. Informal
  course user studies with anonymous ratings are often exempt, but that is the course staff's call, not ours.

## Consent (read or show before starting)
> This is a CSE 573 course project study. You will read 6 short job-match explanations and rate them. It
> takes about 15 minutes. We record only your ratings, under a random participant number: no name, email
> or other personal data. The CVs shown are from a public, anonymized dataset. You can stop at any time and
> your ratings will be deleted. Do you agree to take part?

## Procedure
1. `python -m eval.user_study.study --packet` creates `packet.html` (6 cases, public Kaggle data only)
   and `packet_key.csv` (which position is which method; never show it to participants).
2. Give each participant the packet. For each case they read the CV and job, then rate
   **Explanation 1** and **Explanation 2** from 1 (not helpful or trustworthy) to 5 (very helpful and trustworthy).
3. Enter ratings into `eval/user_study/responses.csv` (columns: participant, case, position, rating),
   with participant as P01, P02, …
4. `python -m eval.user_study.study --analyze`, then report the results in `eval/results/user_study.json`.

## Analysis (pre-registered, `eval/STATS_PLAN.md`)
- Per-participant mean rating for each condition.
- Wilcoxon signed-rank test, B (LLM) vs. A (template), with rank-biserial r.
- Target mean ≥ 4.0: one-sample Wilcoxon of B against 4.0, plus a bootstrap 95% CI of the mean. The target
  counts as met only if the CI lower bound is ≥ 4.0.
