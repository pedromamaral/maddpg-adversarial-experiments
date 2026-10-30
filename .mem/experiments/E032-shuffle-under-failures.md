# E032 — Shuffle ablation under failures (A039 + A028 remainder)
Status: ANALYSED 2026-10-30 (D22; K21). Implemented 2026-10-29 (D21). Avenues A039, A028. Follows K20 (E031), K19 (E028), K14 (E018), K08.

Hypotheses:
 H1 (A039) SP-86 3x failure-trained DQN (E018, K14 96.7/93.6/87.8) uses current state under failures: shuffling path telemetry / whole obs
    costs > 1 pp at n=4 and n=8 (vs < 1 pp at n=0, K08), and its extracted static table (argmax with obs fixed/averaged) evaluated under
    failures falls below the DQN by > 1 pp.
 H2 (A028) DRL-OR-style learner (E028) is static like ours: shuffle < 1 pp at n=0/4/8 on both topologies; fallback overrides k=0 on a
    low share of decisions on SP-86 (explains ~2 pp below plain DQN, K19).
Change: eval-only, no training. Load E018 SP-86 checkpoints (verify they exist on server; else retrain 3 seeds, rhf + train failures, CPU) and
  E028 {dc,sp}_s{0,1,2} checkpoints; run eval with shuffle {none, path, all} at n=0/4/8 (seed 20240601, 20 paired eps).
  Add logging: fallback deviation rate (share of decisions with Q(best)-Q(k=0) > margin). Extracted table: reuse the table-export path (E025)
  for the DQN at each failure count, evaluate via run_eval.sh TABLE_FILE.
Config: v2_pilot4_lc 3x (H1, H2); v2_deltacom 8x (H2). 3 seeds each.
Success / reading (noise: seed 1.3 pp, eval <=1 pp):
 - H1 supported: shuffle cost > 1 pp at n=4/8 AND extracted table < DQN by > 1 pp -> C3 = 'static table in normal operation, failure-reactive'.
 - H1 refuted: shuffle <= 1 pp at n=8 and extracted table ~ DQN -> learner found a better failure-robust table than local search (K20 caveat);
   C3 fully static; run A038 rule next.
 - Mixed (shuffle cost but table ~ DQN, or vice versa): report, inspect per-failure-count.
 - H2: report shuffle deltas + deviation rate; deviation rate feeds the margin=0 decision.
Cost: CPU, eval-only, hours. Small code: deviation-rate counter, shuffle at n>0 if not already supported.
Implemented (c296401, src/idqn_control.py): eval JSON now has argmax_off_sp + fallback_override per fail<n> (existing --eval-shuffle path|all, --eval-failures, --fallback-margin, --eval-only already work at n>0).
 Smoke OK on server (E018 fail s0, shuffle path, n=0/4): off-SP 53.8%/52.8%, fallback(0.05) 14.0%/12.9%. Smoke dir host_data/results/E032_smoke (server, root-owned).
 Checkpoints present: idqn_p4_fail/qnet_seed{0,1,2}.pt, E028/{dc,sp}_s{0,1,2}/qnet_seed*.pt.
 Table export: idqn_control.py --eval-only --export-tables writes table_idqn_seed<s>_fail<n>.json = argmax_k of Q averaged over held-out episodes (seed EVAL_SEED+1000, n failures) -> TABLE_FILE rule. Smoke OK (14 agents x 21 dests, k in 0..2).
 Launcher: tools/run_E032.sh (3 seed chains: p4f [base+export, path, all at n=0,4,8] -> table evals via run_eval.sh run v2_p4_lc RULES 'table sp' -> E028 sp -> E028 dc, fallback 0.05). Out host_data/results/E032/{p4f,sp,dc}_s<seed>, evals EVAL_NAME=E032_tab_p4f_s*_n*. Done marker ~/idqn/run_E032.done.
 Commit 3 (see git log E032). Next: research-do deploy E032.
Results: (FINISHED/FETCHED 2026-10-30; 3 seeds, 20 paired eps seed 20240601; deltas = shuffle - base, paired per seed)
 | learner | n | base PDR | d path (per seed) | d all (per seed) | LL-chance pp | fallback reset |
 | SP-86 fail-DQN (E018) | 0 | 96.47 | -0.29 (-0.47/+0.16/-0.57) | -0.22 (-0.61/-0.22/+0.16) | +11..13 | — |
 | | 4 | 93.62 | -1.28 (-1.25/-1.68/-0.92) | -1.38 (-1.64/-1.71/-0.79) | +11 | — |
 | | 8 | 87.82 | -2.19 (-2.50/-1.82/-2.24) | -2.69 (-2.52/-3.06/-2.49) | +12..14 | — |
 | SP-86 DRL-OR-style (E028) | 0 | 93.74 | +0.18 | -0.14 | +4..7 | 16.8% |
 | | 4 | 92.28 | -0.43 | -0.82 | +7..9 | 15.2% |
 | | 8 | 86.53 | -0.74 | -1.75 (-2.22/-1.41/-1.61) | +8..10 | 13.3% |
 | Deltacom DRL-OR-style (E028) | 0/4/8 | 79.47/72.52/65.96 | -0.04/-0.30/-0.33 | +0.30/-0.14/-0.07 | -1..-6 (below chance) | 16.6/16.4/16.4% |
 Extracted table of fail-DQN (argmax of mean Q over held-out eps at same n), SP-86: 95.84/88.88/83.19 (seeds 95.2-96.9/88.7-89.1/82.5-84.4)
  = DQN -0.6/-4.7/-4.6; vs ECMP 91.4/89.7/81.7, E026b best table 96.98/91.5/83.8, greedy 95.7/93.6/89.1.
 argmax_off_sp ~48% (fail-DQN), ~53-55% (E028 SP), ~58-60% (E028 DC); fallback_override = share of decisions the 0.05 margin reset to k=0.
 Raw: host_data/results/E032/{p4f,sp,dc}_s*, results/E032.json, results/E032_tab.json. E028 n=0 SP 93.7 here vs 94.5 in-script (eval-noise level).
Interpretation:
 - Fail-DQN is static at n=0 (K08 holds) but uses path telemetry under failures: shuffle cost grows with n (0.3 -> 1.3 -> 2.2-2.7), 3/3 seeds,
   ~80-90% of the cost from path telemetry alone. Its own frozen table loses 4.7 pp at n=4/8 (below ECMP at n=4), and the best searched
   table (E026b/E031) is still 2.1/4.0 below it -> the failure-time headroom (K20) is reached through state use, not a better table.
 - Caveat: argmax-of-mean-Q is one static policy, not the optimal one; whether "path telemetry" = failure signal (path down/rerouted) or
   load is not separated (what the agent reads is untested). k renumbering after failures makes any table index-relative.
 - E028 on Deltacom is fully static at every n (|d| <= 0.33) and below chance on least-loaded -> table learner at ~ECMP (K19).
   E028 on SP-86: static at n=0/4, mild failure-time use at n=8 (all-obs -1.75, 3/3 seeds); path-only below 1 pp.
 - Fallback margin resets 13-17% of all decisions (~30% of off-SP argmaxes) to k=0 on both topologies: a plausible cause of the ~2 pp
   SP-86 gap to plain DQN (K19), not yet causal (needs margin=0).
Verdict:
 H1 SUPPORTED (n=4 shuffle just above the 1 pp bar, n=8 clear; table gap clear at both) -> C3 = 'static table in normal operation,
   failure-reactive when failure-trained' (D22).
 H2 SUPPORTED on Deltacom (static at all n); PARTLY REFUTED on SP-86 at n=8 (all-obs -1.75). Deviation rate 13-17% -> margin=0 cell justified.
