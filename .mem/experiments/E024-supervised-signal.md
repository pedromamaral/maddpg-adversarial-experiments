# E024 — Is the own-flow state signal learnable? (supervised check)
Status: written, NOT committed, NOT run (2026-10-14). Avenue A024; gates A025 (difference reward) per D11.

Question: can plain supervised regression extract a better-than-table path choice from the agents'
observations, scored on the decision's own-flow delivered share?
Data: applied contested decisions under ECMP (pinned uniform), obs at decision step, flow delivered share r;
train/test on disjoint traffic seeds.
Estimator: IPS on held-out decisions, V(pi) = mean(r * n * 1[pi(x)==chosen]); bootstrap SE.
Rules: ecmp (mean r), greedy (least U_i), table (best path per agent,dest), util (regressor on K path
utils + ids), full (learner Q-net shape on full obs + agent id).

Run (both topologies):
  python tools/supervised_signal.py --config configs/v2_pilot4_lc.json --train-episodes 150
  python tools/supervised_signal.py --config configs/v2_deltacom.json  --train-episodes 150
(inside maddpg-exp:latest with src mounted at /workspace/src — see infra.md)

Decision rule: if util/full >> table on Deltacom (beyond 2 SE) -> signal learnable -> run A025
(difference reward, MADDPG LC/CC, 3 seeds, ~10 h). If util/full ~ table -> state signal not learnable from
the observation; supports C4 as is; write it into the paper.
Results: TODO -> results/E024.json
