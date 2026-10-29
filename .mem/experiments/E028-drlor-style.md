# E028 — DRL-OR-style learner in our simulator (G1)
Status: ANALYSED 2026-10-29 (deployed 2026-10-23, sha 3f02e7a; finished 10-24, fetched 10-28). Avenue A028; decisions D15, D17. KILL-3: no end-to-end port by wk 4 -> label "DRL-OR-style".

Hypothesis: a published-method-shaped learner (per-flow utility reward, SP pretraining, safe fallback) does not beat ECMP/static table
  in our pinned/stale regime -> consistent with C1-C4 (credit + static table, not architecture).
Port scope (NOT the Mininet testbed): per-flow outcome reward = flow credit (already idqn --credit flow); SP pretraining = --sp-pretrain N
  (first N episodes behave as k=0 while Q regresses outcomes, eps decay starts after); safe fallback = --fallback-margin m
  (use k=0 unless Q(best)-Q(k=0) > m, train and eval). Obs = our per-agent obs (DRL-OR's GNN link-state obs not ported: stated limitation).
Implemented 9ff3439 (src/idqn_control.py Policy.sp_only/fallback_margin, args; defaults off -> baselines unchanged). DQN learner only.
Smoke OK on server container (SP-86, 6 ep, 1 seed): runs, JSON has sp_pretrain/fallback_margin.
Planned cells (to decide at deploy): SP-86 3x (v2_pilot4_lc) + Deltacom 8x (v2_deltacom), 3 seeds, --train-random-hotspot --train-failures 0,2,4,6,8,
  e.g. --sp-pretrain 50 --fallback-margin {0.05}, eval-shuffle path/all ablation via --eval-only. Compare vs ECMP (random+flow), sp, E026 table.
Open for deploy: margin value(s), pretrain length, episode budget (750 default).
Deployed cells: Deltacom 8x (--batch 256 --updates 20) + SP-86 3x (defaults), seeds 0/1/2, --learner dqn --sp-pretrain 50 --fallback-margin 0.05,
  --train-random-hotspot --train-failures 0,2,4,6,8, eval failures 0,4,8 stale 0,2, 750 eps. CPU only (idqn_control has no GPU use), 2 cpus/container,
  3 seed chains parallel (dc then sp). Out: host_data/results/E028/{dc,sp}_s<seed>. Compare vs ECMP/greedy/sp/E026 table in results/E026b_*.json.
  No margin sweep / eval-shuffle ablation yet (add after main cells).

## Results (in-script eval, 3 seeds, mean (min-max); results/E028_{dc,sp}.json)
| | n=0 | n=4 | n=8 | least-loaded vs chance (n=0) |
|---|---|---|---|---|
| Deltacom 8x E028 | 79.5 (78.5-80.7) | 72.5 | 66.0 | 39.2 vs 43.2 (below) |
| Deltacom ECMP / greedy / E026 table | 80.4 / 84.9 / 87.35 | 72.8 / 76.5 / 77.6 | 66.7 / 68.9 / 68.0 | |
| SP-86 3x E028 | 94.5 (94.2-94.9) | 92.3 | 86.5 | 48.1 vs 40.3 (above) |
| SP-86 ECMP / greedy / E026 table | 91.4 / 95.7 / 96.98 | 89.7 / 93.6 / 91.5 | 81.7 / 89.1 / 83.8 | |
| SP-86 flow-credit DQN (K06/K14) | 96.2-96.9 | 92.7-93.6 | 86.7-87.8 | |
stale2 costs <=0.6 pp everywhere. Train-curve last-episode PDR spread wide (DC 70-88, SP 89-95) -> eval is the number.
Not logged: share of decisions where fallback overrode k=0 (deviation rate). Not run: shuffle ablation, margin sweep.

## Interpretation
- Deltacom: ~ECMP (-0.9 pp, inside 3-5 pp Deltacom seed noise) at n=0/4/8, below chance on least-loaded -> same as every learner (K10); misses the 7 pp static headroom (K11).
- SP-86: +3.1 pp over ECMP but -1.7..-2.4 pp below plain flow-credit DQN at n=0 (> 1.3 pp noise); under failures ~flow-credit learners (-0.4..-1.3) and above the n=0-optimised table.
  Likely cause: SP pretraining + fallback margin bias toward k=0 (SP = 75.8 at 3x) -> fewer off-SP deviations. Unverified (no deviation rate).
- Least-loaded above chance on SP-86 is NOT adaptivity evidence without a shuffle ablation (a static table aligned with demand also scores above chance).

## Verdict
H (DRL-OR-style does not beat ECMP/static table in our regime): SUPPORTED. Deltacom = ECMP; SP-86 < static table and < our flow-credit learners.
Its components (SP pretraining, safe fallback) cost PDR on SP-86 rather than help -> consistent with C1/C2/C4 (credit decides; add-ons do not open the Deltacom headroom).
Inconclusive for C3 (no shuffle ablation) and for the cause of the SP-86 gap (no deviation rate, no margin=0 cell). Label: "DRL-OR-style" (KILL-3: DQN learner, our obs, no GNN/MAPPO).
