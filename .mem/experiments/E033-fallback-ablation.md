# E033 — DRL-OR-style fallback ablation (A028 remainder, D22)
Status: IMPLEMENTED 2026-10-30. Avenue A028. Follows K19 (E028), E032 (fallback resets 13-17% of decisions).

Hypothesis H1: the 0.05 safe-fallback margin causes the SP-86 gap of E028 (93.7-94.5 at n=0) to plain flow-credit DQN (E018 fail-DQN 96.5).
 A (eval-only, no fallback on E028 checkpoints) separates deployment-time cost; B (retrain without fallback, keep SP pretrain)
 separates training-time cost (fallback also shapes the behaviour policy during training: src/idqn_control.py:264-268 runs in training too).
Change: none in src. margin None (flag omitted) == margin 0 except exact ties. Launcher tools/run_E033.sh.
Config: A: E028 {sp,dc}_s{0,1,2} checkpoints, v2_pilot4_lc 3x + v2_deltacom 8x, eval n=0/4/8, 20 paired eps seed 20240601
 (paired with E032 base = with fallback). B: v2_pilot4_lc, --learner dqn --train-random-hotspot --train-failures 0,2,4,6,8
 --sp-pretrain 50, 750 eps, eval n=0/4/8 stale 0/2, 3 seeds.
Success / reading (noise: seed 1.3 pp, eval <=1 pp):
 - A recovers >1 pp on SP-86 (toward 96.5) -> gap is deployment-time fallback.
 - A ~0 but B ~ E018 (within 1.3 pp) -> gap is fallback during training (exploration/behaviour policy), not pretraining.
 - A ~0 and B ~ E028 -> gap is the SP pretraining (or noise); fallback innocent.
 - Deltacom A: expected ~0 (static, ~ECMP); report.
Cost: CPU only. A ~1 h/seed; B ~E028 sp cell time per seed. 3 parallel chains, 4 cpus each.
Out: host_data/results/E033/{nofb_sp,nofb_dc,train_sp}_s<seed>; done marker ~/idqn/run_E033.done.
Results:
Interpretation:
Verdict:
