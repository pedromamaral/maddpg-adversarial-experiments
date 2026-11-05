# E035 — Telemetry lie gate for Paper 2 (A042b, T7)
Status: RUNNING after E034 in the same chains, code 5599288. Smoke OK on DQN and MADDPG, lie_zero_grad 0. Avenue A042. Launcher tools/run_E034.sh (runs after E034; paired base = E034 base, E032 p4f_s*/idqn_seed*_base.json for E018).

Hypothesis H1: failure-trained learners can be damaged through path telemetry, more so under failures than at n=0.
Change: --eval-lie + --lie-eps (Policy._apply_lie, src/idqn_control.py). Threat model A004 (path_util slots only, clamped, all agents = upper bound).
Config: E018 (p4f), ac_local_rhf (lc), ac_central_rhf (cc), SP-86 3x, n=0/4/8, 20 paired eps, 3 seeds.
 lure · repel (full control) · margin eps 0.3 (one-step gradient, lie_zero_grad = masking check) · random eps 0.3 and 1.0 (budget-matched controls). Ceiling + greedy-under-lure proxy: worst rule at n=0/4/8.
PASS (A042): best lie costs >=3 pp beyond noise and beyond its budget-matched random, at some n, on DQN and on MADDPG.
FAIL: attacks stay dropped; one discussion paragraph in Paper 1. PASS: literature check, then scope after Paper 1 is submitted.
Limits: gate only, SP-86 only, no iterated or learned attacker, so a FAIL on margin with zero-gradient share high is not evidence of robustness; lure/repel still count.
