# STATE — 2026-10-15

branch main @ 63a0ea3 (+ this STATE commit), not pushed; server code at d209c69.
phase just done: research-do deploy E026 (pushed; server at 85e950e).

## What changed
- tools/static_table_search.py (new, 46f4985): coordinate-descent search for the best static table by system PDR
  (CRN reseeding, verified deterministic; fork pool; search_log.json + table_*.json)
- experiments/E026 card: status IMPLEMENTED, step-0 finding, noise, settings, usage
- avenues A034: (a) done (no change needed); codebase.md: tool line added

## Plan (D17)
E026 (A030) -> E027 (A034b, CPU, alongside) -> E028 (A028 DRL-OR-style) -> E029 (A025 diff reward).

## Active runs
E026: maddpg-E026-sp86 + maddpg-E026-dc on server (6 cpus each, 10 cores used; docker logs only; results host_data/results/E026_{sp,dc}).
Deltacom timing: 21.1 s/J (M=8, 8w, sp J0=74.54); printed '294 searchable entries' same as SP-86 -> check in analysis.

## Latest results
- E026 step 0: 'table' failure semantics already = learners' (kpath_cache rebuilt post-failure) -> E025 stands, no re-run.
- E026 noise (SP-86 3x, M=8, local): 14.7 s/J at 8 workers; J SE 0.63 pp; paired single-entry SE 0.08-0.69 pp.

## Open problem
Does Deltacom have static headroom that learners miss (E026 > 81.9)? If yes, A025 is retargeted and C4 reworded.

## EXACT next step
research-do status E026 (check ~10 h in: search_log.json via summarize); meanwhile research-do implement/deploy E027 on CPU (4 cores free).

## Read next
experiments/E026-system-static-table.md ('Implemented' block), infra.md, runs.md.
