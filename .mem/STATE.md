# STATE — 2026-10-15

branch main @ 63a0ea3 (+ this STATE commit), not pushed; server code at d209c69.
phase just done: research-do implement E026.

## What changed
- tools/static_table_search.py (new, 46f4985): coordinate-descent search for the best static table by system PDR
  (CRN reseeding, verified deterministic; fork pool; search_log.json + table_*.json)
- experiments/E026 card: status IMPLEMENTED, step-0 finding, noise, settings, usage
- avenues A034: (a) done (no change needed); codebase.md: tool line added

## Plan (D17)
E026 (A030) -> E027 (A034b, CPU, alongside) -> E028 (A028 DRL-OR-style) -> E029 (A025 diff reward).

## Active runs
none.

## Latest results
- E026 step 0: 'table' failure semantics already = learners' (kpath_cache rebuilt post-failure) -> E025 stands, no re-run.
- E026 noise (SP-86 3x, M=8, local): 14.7 s/J at 8 workers; J SE 0.63 pp; paired single-entry SE 0.08-0.69 pp.

## Open problem
Does Deltacom have static headroom that learners miss (E026 > 81.9)? If yes, A025 is retargeted and C4 reworded.

## EXACT next step
research-do deploy E026: git push; on server git pull; time Deltacom (configs/v2_deltacom.json, --load 8, --sweeps 0)
to set M/eps per card rule; launch SP-86 (--load 3 --episodes 16 --eps 0.5, starts sp / E025 table JSON / random:1 /
random:2, 8 workers) and Deltacom detached, CPU only (don't take all 16 cores). Then E027 on CPU alongside.
Paper deadline ~2027-01-14 (TNSM). ssh: pedroamaral@10.26.110.14 (alias 'server' not resolvable locally).

## Read next
experiments/E026-system-static-table.md ('Implemented' block), infra.md, runs.md.
