# E027 — Harden E025/K17: second table export + second eval traffic seed
Status: DONE, analysed 2026-10-17 (deployed 2026-10-17). Was PLANNED 2026-10-15. Avenue A034b; decision D17. CPU only; run alongside E026 (after E026 step 0 fixes 'table' failure semantics).

Hypothesis: K17's sign holds: own-flow table-for-all < ECMP on Deltacom 8x at n=0/4/8, with a 2nd export
(supervised_signal.py, different train seed) and a 2nd eval traffic seed (≠ 20240601, both rules paired on it).
Change: none expected. Only run E025's pipeline with new seeds (if step 0 changed semantics, SP-86 failure numbers re-run too).
Success criterion: Deltacom table - ECMP < 0 at n=0 for both exports x both eval seeds -> K17 confidence -> high.
  Any table >= ECMP -> K17 back to medium; report the spread.
Implemented bafe41b + b0581e1 (sha on server b0581e1): supervised_signal.py --train-seed (default 1000, unchanged; B uses 3000), env EVAL_TRAFFIC_SEED
  (runner _attack_episodes default; run_eval.sh passes it into docker), DOCKER_EXTRA in run_eval.sh. Smoke OK (SP-86 tiny export 14 agents/144 entries; env default read back).
Config: tools/run_E027.sh. Deltacom 8x only (SP-86 not re-run: E026 step 0 changed nothing). Table A = E025 export (train seed 1000), B = 3000.
  Cells (RULES="table random+flow", n=0/4/8, 20 eps): B@20240601, A@20240602, B@20240602; A@20240601 = E025. Output: host_data/results/v2_dc_lc/E027_<cell>/n_<k>/damage_ceiling.json.
  Fetch: summarize_run.py per E027_* dir (ECMP = random+flow row).
Deployed 2026-10-17 17:30 (server, 4 cpus, alongside E026).
Results (Deltacom 8x, PDR %, 20 paired eps; delta = table - ECMP, pp):
| export@eval seed | n=0 table/ECMP/delta | n=4 | n=8 |
| A@601 (=E025) | 78.0/80.4/-2.4 | 70.9/72.8/-1.9 | 64.6/66.7/-2.1 |
| B@601 | 77.9/80.4/-2.5 | 70.8/72.8/-2.0 | 64.6/66.7/-2.1 |
| A@602 | 78.9/81.6/-2.7 | 69.4/73.0/-3.5 | 62.5/65.2/-2.6 |
| B@602 | 78.8/81.6/-2.8 | 69.3/73.0/-3.7 | 62.5/65.2/-2.7 |
Sanity: ECMP identical across exports on each seed (80.402 @601 = E025; 81.584 @602) -> paired eval deterministic.
Interpretation: 12/12 cells table < ECMP, delta -1.9..-3.7 pp (> eval noise 0.5-1 pp; paired). Export spread tiny
  (A vs B <=0.12 pp) -> export estimation noise negligible, but the two exports are likely near-identical tables
  (same IPS data source), so this tests estimation noise, not a different own-flow optimum. Eval seed 602 shifts
  ECMP +1.2 pp and widens the gap (~-3 pp) -> sign robust to demand draw.
Verdict: hypothesis SUPPORTED. K17 (Deltacom part) -> high. SP-86 failure caveat closed by A034a (no change).
