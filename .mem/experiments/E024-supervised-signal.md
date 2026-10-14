# E024 — Is the own-flow state signal learnable? (supervised check)
Status: DONE 2026-10-14 (analysed); ran from 16:04 UTC, sha 6e29524 (smoke 3/2 eps ok; ~/idqn/E024_smoke.log). Avenue A024; gates A025 (difference reward) per D11.

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
Launch: ~/idqn/run_E024.sh; containers maddpg-E024-{v2_pilot4_lc,v2_deltacom}; done marker ~/idqn/run_E024.done
Output: host_data/results/supervised_signal/{v2_pilot4_lc,v2_deltacom}.json (stdout also in ~/idqn/maddpg-E024-*.log)
Results: results/E024.json (runtimes 40 min SP-86, 72 min Deltacom; ~35k train / ~9.3k test decisions each)
  own-flow delivery % (IPS, bootstrap SE ~1.4; ecmp SE ~0.2):
  | rule   | SP-86  | Deltacom |
  | ecmp   | 93.29  | 88.18 |
  | greedy | 100.72 | 94.07 |
  | table  | 99.61  | 95.75 |
  | util   | 99.25  | 96.68 |
  | full   | 99.58  | 94.79 |
  agree with greedy (table/full): SP-86 57.7/57.0 %, Deltacom 25.0/37.9 %
Interpretation:
  - Deltacom util-table +0.9, full-table -1.0 pp: both far inside 2 SE (~2.8-3.9 pp) -> no learnable
    state signal beyond the table, at this power (cannot resolve < ~3 pp).
  - Key: greedy (fresh current state) does NOT beat the table on own-flow (94.1 vs 95.8). So on the
    own-flow objective there is no state-dependent headroom to learn; the table captures it all
    (+7.6 pp vs ECMP on Deltacom, +6.3 on SP-86).
  - Yet system PDR on Deltacom: learners ~ECMP (K10) while greedy +4.5. The table's large own-flow
    gain does not become system gain; greedy's system gain is not own-flow. -> the gap is an
    EXTERNALITY / objective mismatch of per-decision own-flow credit, not observation learnability.
  - Caveats: off-policy (ECMP background); own-flow under all-table background unmeasured; IPS
    unbounded (greedy SP-86 100.7 is within SE of <=100); K12 used a different own-flow measure.
Verdict: H "current-state signal learnable beyond the table from obs (Deltacom)" -> REFUTED at this
  power (literal decision rule -> C4 holds). Mechanism revised: objective (own-flow) has no state
  headroom; the missing gain is system-level -> re-motivates A025 (difference reward), see K16.
