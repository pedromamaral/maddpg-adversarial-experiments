# E028 — DRL-OR-style learner in our simulator (G1)
Status: IMPLEMENTED 2026-10-17 (not deployed). Avenue A028; decisions D15, D17. KILL-3: no end-to-end port by wk 4 -> label "DRL-OR-style".

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
