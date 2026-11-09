# E035 — Telemetry lie gate for Paper 2 (A042b, T7)
Status: DONE 2026-11-06 00:45 (margin reruns @df8fe17), analysed 2026-11-09, code 5599288. Smoke OK on DQN and MADDPG, lie_zero_grad 0. Avenue A042. Launcher tools/run_E034.sh (runs after E034; paired base = E034 base, E032 p4f_s*/idqn_seed*_base.json for E018).

Hypothesis H1: failure-trained learners can be damaged through path telemetry, more so under failures than at n=0.
Change: --eval-lie + --lie-eps (Policy._apply_lie, src/idqn_control.py). Threat model A004 (path_util slots only, clamped, all agents = upper bound).
Config: E018 (p4f), ac_local_rhf (lc), ac_central_rhf (cc), SP-86 3x, n=0/4/8, 20 paired eps, 3 seeds.
 lure · repel (full control) · margin eps 0.3 (one-step gradient, lie_zero_grad = masking check) · random eps 0.3 and 1.0 (budget-matched controls). Ceiling + greedy-under-lure proxy: worst rule at n=0/4/8.
PASS (A042): best lie costs >=3 pp beyond noise and beyond its budget-matched random, at some n, on DQN and on MADDPG.
FAIL: attacks stay dropped; one discussion paragraph in Paper 1. PASS: literature check, then scope after Paper 1 is submitted.
Limits: gate only, SP-86 only, no iterated or learned attacker, so a FAIL on margin with zero-gradient share high is not evidence of robustness; lure/repel still count.

## Results (2026-11-09; delta pp vs paired unattacked base, mean of 3 seeds (s0 s1 s2); f = share of decisions flipped)
| victim | n | base | lure | repel | margin 0.3 | random 0.3 | random 1.0 |
|---|---|---|---|---|---|---|---|
| DQN E018 | 0 | 96.47 | -7.44 (-6.4 -8.0 -7.9) f.60 | -7.47 | -1.81 f.59 | -0.71 f.41 | -3.39 f.56 |
| DQN E018 | 4 | 93.62 | -9.52 (-8.8 -10.0 -9.8) | -6.00 | -1.92 | -0.65 | -3.13 |
| DQN E018 | 8 | 87.82 | -10.96 (-10.1 -12.4 -10.4) | -8.24 | -2.74 | -1.02 | -4.24 |
| MADDPG LC rhf | 0 | 96.27 | -6.50 (-4.7 -7.1 -7.7) f.46 | -3.59 | -0.66 f.38 | +0.22 f.21 | -1.05 f.37 |
| MADDPG LC rhf | 4 | 92.90 | -6.85 (-5.0 -8.0 -7.6) | -3.61 | -0.48 | +0.01 | -1.52 |
| MADDPG LC rhf | 8 | 86.67 | -7.40 (-6.2 -8.2 -7.8) | -4.29 | -1.10 | -0.03 | -2.08 |
| MADDPG CC rhf | 0 | 97.02 | -9.81 (-11.3 -6.8 -11.3) f.42 | -4.92 | -1.01 f.38 | -0.11 f.19 | -1.40 f.34 |
| MADDPG CC rhf | 4 | 92.97 | -9.73 (-10.1 -11.3 -7.7) | -6.22 | -0.59 | +0.30 | -1.69 |
| MADDPG CC rhf | 8 | 86.72 | -10.52 (-9.9 -13.6 -8.1) | -8.92 | -1.34 | +0.28 | -1.84 |
Margin lie_zero_grad = 0.00 everywhere (no masking). Worst-rule ceiling 67.06/64.74/59.49 (n=0/4/8): lure reaches ~25-35% of the ceiling damage.
Bases: E032 p4f (DQN), E034 base (MADDPG).

## Interpretation
- Gate: lure beats its budget-matched control (random 1.0) by 4.1/6.4/6.7 pp (DQN), 5.5/5.3/5.3 (LC), 8.4/8.0/8.7 (CC) at n=0/4/8, every seed >=3 pp. PASS on DQN and MADDPG.
- Not "immune in normal operation": lure costs 6.5-9.8 pp at n=0, where shuffle costs <1 pp (K08). The policies ignore natural telemetry variation but respond to extreme (out-of-distribution) path_util values.
- Growth with failures only for DQN (lure -7.4 -> -11.0); MADDPG roughly flat (LC -6.5 -> -7.4, CC -9.8 -> -10.5).
- Small-budget one-step margin attack is weak (0.5-2.7 pp, 0.5-1.7 beyond random 0.3) despite flipping 37-59% of decisions: most flipped decisions carry no packets or swap near-equivalent paths. Zero-gradient share 0, so weakness is not gradient masking.
- Threat model is an upper bound: all agents' path_util lied to, full value control for lure/repel.
## Verdict
Gate A042 PASS. H1 partly supported: damage through path telemetry is large and targeted on all three learners, but already at n=0; "more under failures" holds for DQN only. Paper 2 narrative "immune normally, exposed in failures" is refuted; the defensible story is "static on realised telemetry, yet steerable by out-of-range telemetry".
