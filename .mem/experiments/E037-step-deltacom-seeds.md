# E037 — Step credit on Deltacom, 2 more seeds (A045, D24)
Status: IMPLEMENTED 2026-11-09 (not smoke-run: local venv has no numpy; first server run is the smoke), not deployed. Avenue A045. Follows E020 (1 seed, 80.9), K10.

Hypothesis H1: with 3 seeds, step MADDPG-LC on Deltacom 8x stays within noise of flow-credit LC, so credit has no detectable PDR effect there.
Change: none in src expected. Same recipe as E020, new seeds only.
Config: v2_dc_lc recipe. Check at implement: E020's exact command, seed and hotspot regime, to pick the matching comparator (E021 fix or E022 rhf) and report it in the card.
 Eval n=0/4/8, 20 paired eps, seed 20240601, least-loaded share as in E020.
Reading (noise: 3-5 pp Deltacom rhf; fixed-hotspot spread to be read from E021): judged on 3-seed mean and seed range.
 - step within noise of flow LC -> no credit effect detected at this power; C1 stays scoped to SP-86.
 - step clearly below flow LC (gap > noise, seed ranges separate) -> credit helps on Deltacom too; revisit D24 wording.
Limit: 3 seeds with this noise cannot show equality; say "not detected", not "no effect".
Cost: CPU, ~6 h per seed (estimate from E028 scale; confirm at implement). Runs alongside E030 if cores allow.

## Implementation (2026-11-09)
- E020 = configs/v2_deltacom.json, old pipeline (standalone_experiment_runner --phase train --variants LC-Simple), fixed hotspot, GPU, training.seed unset -> default 42; run dir v2_dc_lc. Comparator: E021 dc_ac_local_fix (flow-credit LC, 3 seeds, 80.2); rhf (E022) is not matched (trains on failures).
- No src change. configs/v2_deltacom_s{1,2}.json = v2_deltacom.json + training.seed 1/2 (only diff). Seeds: 42 (E020), 1, 2.
- tools/run_E037.sh: 2 parallel chains (train GPU 0 -> run_eval.sh policy/random+flow/greedy, n=0/4/8, 8x, 20 eps seed 20240601). Out host_data/results/v2_dc_lc_s{1,2}/E037_eval/n_*; marker ~/idqn/run_E037.done (2 lines).
- E020's eval was eval_8x (rules incl. stale/sp/worst); E037 reruns only policy/ECMP/greedy, n=0/4/8, so the s42 row for n=8 needs a rerun or is read from eval_8x if it has n=8 (check at fetch).
- Risk: both chains share GPU 0 (fine on 2080 Ti for this size, unverified). commit: see git log E037.
