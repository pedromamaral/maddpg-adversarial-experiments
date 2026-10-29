# E032 — Shuffle ablation under failures (A039 + A028 remainder)
Status: PARTIALLY IMPLEMENTED 2026-10-29 (D21). Avenues A039, A028. Follows K20 (E031), K19 (E028), K14 (E018), K08.

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
 NOT done: extracted-table eval for E018 at each failure count (tools/supervised_signal.py --export-table exists; check it takes a failure-trained net + obs averaging) -> next implement step; then script tools/run_E032.sh (eval-only loop: shuffle none/path/all x n=0,4,8 x seeds).
Results:
Interpretation:
Verdict:
