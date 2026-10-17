# E027 — Harden E025/K17: second table export + second eval traffic seed
Status: RUNNING (deployed 2026-10-17). Was PLANNED 2026-10-15. Avenue A034b; decision D17. CPU only; run alongside E026 (after E026 step 0 fixes 'table' failure semantics).

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
Results:
Verdict:
