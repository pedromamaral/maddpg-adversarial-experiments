# E026b — Held-out eval of the E026 system tables + own-flow vs system gain of their changes
Status: IMPLEMENTED 2026-10-22 (sha 598d49d), not yet run. Avenues A030 (eval) + A035 (credit); decision D19; follows E026 (K11 revised, provisional).

Hypotheses:
  H1 (A030): on held-out eval (seed 20240601, 20 paired eps) the Deltacom 8x table_best stays >= 85 at n=0 (train J 87.4),
     and SP-86 3x table_best is within 1 pp of learners (96.2-97.4). Also report n=4/8 (optimised at n=0 only).
  H2 (A035): for each change accepted by the Deltacom search (search_log 'changed' [a,d,k_old,k_new]), the gain to the
     changing agent's OWN flows is <= 0 (or << system gain), while the system gain is large -> own-flow credit cannot
     discover the system table (mechanism for K17 / C4).

Change:
  1. Eval (no code): tools/run_eval.sh with TABLE_FILE=host_data/results/E026_{sp,dc}/table_best.json, n=0/4/8,
     RULES="table random+flow greedy greedy_stale2 sp", configs v2_pilot4_lc (LOAD=3) / v2_deltacom (LOAD=8).
  2. Credit (small script, e.g. tools/table_change_credit.py): for each accepted change along the best start's path,
     run the table before vs after the change on the same CRN eps (eval seeds, not train), and record system PDR and the
     changing agent's own-flow PDR (reuse own-flow accounting from tools/own_flow_effect.py / E023). Paired diffs + SE.
     Also the reverse: apply each change alone to the ECMP-like / learner baseline table if cheap.
Config: configs/v2_deltacom.json --load 8 (SP-86 optional).
Success / reading (eval noise <=1 pp; seed noise 1.3):
  - H1: Deltacom table >= 85 -> K11 revision confirmed (-> high), C4 reworded per D18. 81.9 < table < 85 -> headroom
    real but train-inflated; report eval number. table <= 81.9 -> E026 overfit; revert K11.
  - H2: own-flow gain <= 0 (within 1 SE) for the changes carrying most of the system gain -> supported; own-flow gain > 0
    for them -> own-flow credit could see them -> the miss is exploration/optimisation, not credit (redirects A025 -> A036).
Cost: CPU only, ~3 h eval + <1 h credit script.
Implementation: tools/table_change_credit.py (sha 598d49d; no cfg needed). Replays best start's accepted changes (asserts final
  == table_<start>.json), CRN eval seeds 20240601+e, per change: system / own-agent / own-entry(a,d) PDR paired diffs +-SE.
  Smoke OK on server (2 eps, 2 changes). Full run (docker on server, CPU):
  docker run --rm -v $R/host_data:/workspace/data -v $R/src:/workspace/src -v $R/tools:/workspace/tools -v $R/configs:/workspace/configs
   -w /workspace --entrypoint python3 maddpg-exp:latest tools/table_change_credit.py --config configs/v2_deltacom.json --load 8
   --search-log data/results/E026_dc/search_log.json --table-dir data/results/E026_dc --episodes 20 --workers 8 --out data/results/E026b/credit_dc.json
  (SP-86: configs/v2_pilot4_lc.json --load 3, E026_sp.) Eval part (H1) = run_eval.sh with TABLE_FILE, no code.
Results:
Interpretation:
Verdict:
