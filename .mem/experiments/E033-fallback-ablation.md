# E033 — DRL-OR-style fallback ablation (A028 remainder, D22)
Status: ANALYSED 2026-11-04 (K19 edited). Avenue A028. Follows K19 (E028), E032 (fallback resets 13-17% of decisions).

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
Implemented 38bce0f (tools/run_E033.sh, no src change). Smoke OK on server 2026-10-30 (host_data/results/E033_smoke, root-owned):
 A eval-only 2 eps n=0 PDR 95.85, fallback 0.0%; B 3 eps train + eval runs, outputs idqn_seed0.json + qnet. Deployed 2026-10-30 18:43 UTC on server (setsid nohup bash tools/run_E033.sh; log ~/idqn/run_E033.out, per-container ~/idqn/maddpg-E033-*.log).
Results (results/E033.json, 3 seeds, mean (min-max); fallback_override 0.0 in every cell):
| cell | n=0 | n=4 | n=8 |
|---|---|---|---|
| A SP-86 nofb (E028 ckpts, paired) | 94.74 (94.38-95.03) | 92.49 | 86.16 |
| E032 base SP-86 (same ckpts, fallback, paired) | 93.74 (92.77-94.39) | 92.28 | 86.53 |
| delta A SP | +1.00 | +0.21 | -0.37 |
| A Deltacom nofb / delta vs E032 base | 80.29 / +0.82 | 72.98 / +0.46 | 66.23 / +0.28 |
| B SP-86 retrain no fallback (in-script eval) | 96.05 (95.49-96.36) | 92.16 | 86.07 |
| E028 SP-86 in-script (fallback) | 94.5 | 92.3 | 86.5 |
| E018 fail-DQN in-script (K14) / E032 paired | 96.7 / 96.47 | 93.6 / 93.62 | 87.8 / 87.82 |
B stale2: 95.57/91.73/85.72. B least-loaded 50.9 vs chance 40.4 (n=0); argmax off SP ~50%.
Interpretation:
- n=0: deployment fallback costs ~1.0 pp (A, paired, at the eval-noise edge; per-seed signs not checked). Retraining without it (B)
  gives 96.05 = E018 within 0.4-0.65 pp (< seed noise 1.3): the whole SP-86 n=0 gap of E028 is the fallback (deployment ~1 pp +
  training-time remainder ~0.5-1 pp, the latter cross-protocol A paired vs B in-script). SP pretraining costs nothing at n=0.
- n=4/8: fallback has no effect (A |d|<=0.4, B = E028 within 0.4). B stays -1.4/-1.7 below E018 (just above seed noise, cross-run):
  residual failure-time gap is not the fallback; candidate = SP pretraining weakens the learned failure reaction (K21), unverified.
- Deltacom: fallback removal +0.3..+0.8, all within noise; still ~ECMP (80.4/72.8/66.7). K19's Deltacom reading unchanged.
Verdict: H1 supported at n=0 (fallback, both deployment and training); refuted at n=4/8 (fallback innocent; residual ~1.5 pp
 vs E018 is inconclusive: SP pretraining or noise). Analysed 2026-11-04.
