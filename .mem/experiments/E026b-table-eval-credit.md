# E026b — Held-out eval of the E026 system tables + own-flow vs system gain of their changes
Status: DONE + ANALYSED 2026-10-23 (sha 598d49d / run c748412). H1 supported; H2 mixed (path-level supported on Deltacom, per-change inconclusive). Avenues A030 (eval) + A035 (credit); decision D19; follows E026 (K11 revised, provisional).

Hypotheses:
  H1 (A030): on held-out eval (seed 20240601, 20 paired eps) the Deltacom 8x table_best stays >= 85 at n=0 (train J 87.4),
     and SP-86 3x table_best is within 1 pp of learners (96.2-97.4). Also report n=4/8 (optimised at n=0 only).
  H2 (A035): for each change accepted by the Deltacom search (search_log 'changed' [a,d,k_old,k_new]), the gain to the
     changing agent's OWN flows is <= 0 (or << system gain), while the system gain is large -> own-flow credit cannot
     discover the system table (mechanism for K17 / C4).

Change:
  1. Eval (no code): tools/run_eval.sh with TABLE_FILE=host_data/results/E026_{sp,dc}/table_best.json, n=0/4/8,
     RULES="table random+flow greedy greedy_stale2 sp", configs v2_pilot4_lc (LOAD=3) / v2_deltacom (LOAD=8).
  2. Credit (small script, e.g. tools/table_change_credit.py): for each accepted change along the best start's path,
     run the table before vs after the change on the same CRN eps (eval seeds, not train), and record system PDR and the
     changing agent's own-flow PDR (reuse own-flow accounting from tools/own_flow_effect.py / E023). Paired diffs + SE.
     Also the reverse: apply each change alone to the ECMP-like / learner baseline table if cheap.
Config: configs/v2_deltacom.json --load 8 (SP-86 optional).
Success / reading (eval noise <=1 pp; seed noise 1.3):
  - H1: Deltacom table >= 85 -> K11 revision confirmed (-> high), C4 reworded per D18. 81.9 < table < 85 -> headroom
    real but train-inflated; report eval number. table <= 81.9 -> E026 overfit; revert K11.
  - H2: own-flow gain <= 0 (within 1 SE) for the changes carrying most of the system gain -> supported; own-flow gain > 0
    for them -> own-flow credit could see them -> the miss is exploration/optimisation, not credit (redirects A025 -> A036).
Cost: CPU only, ~3 h eval + <1 h credit script.
Implementation: tools/table_change_credit.py (sha 598d49d; no cfg needed). Replays best start's accepted changes (asserts final
  == table_<start>.json), CRN eval seeds 20240601+e, per change: system / own-agent / own-entry(a,d) PDR paired diffs +-SE.
  Smoke OK on server (2 eps, 2 changes). Full run (docker on server, CPU):
  docker run --rm -v $R/host_data:/workspace/data -v $R/src:/workspace/src -v $R/tools:/workspace/tools -v $R/configs:/workspace/configs
   -w /workspace --entrypoint python3 maddpg-exp:latest tools/table_change_credit.py --config configs/v2_deltacom.json --load 8
   --search-log data/results/E026_dc/search_log.json --table-dir data/results/E026_dc --episodes 20 --workers 8 --out data/results/E026b/credit_dc.json
  (SP-86: configs/v2_pilot4_lc.json --load 3, E026_sp.) Eval part (H1) = run_eval.sh with TABLE_FILE, no code.
Results (held-out eval, seed 20240601, 20 paired eps; table optimised at n=0 only; eval noise <=1 pp, seed noise 1.3):
  H1  PDR n=0/4/8      table             ECMP(random+flow)   greedy            greedy_stale2     sp
  Deltacom 8x          87.35/77.62/68.00  80.40/72.80/66.75   84.90/76.46/68.86  84.07/75.80/68.47  74.4/68.2/63.3
  SP-86 3x             96.98/91.54/83.78  91.42/89.73/81.70   95.69/93.64/89.06  93.75/90.67/86.07  75.8/75.7/70.7
  (SP-86 learners n=0 96.2-97.4 from E026; train J 97.0 -> held-out 96.98; Deltacom train 87.4 -> held-out 87.35: no overfit.)
  H2  Deltacom, best-start path, 4 accepted changes [agent,dest,k_old,k_new]; paired diffs pp (+-SE), CRN eval eps:
   i  change     system        own-agent      own-entry     search_gain
   1  [5,8,2,1]  +2.30 (.16)   -19.56 (.79)   -26.83 (1.20) 1.95
   2  [4,7,2,0]  +3.27 (.21)   +0.87 (.38)    -2.35 (.64)  3.45
   3  [0,7,2,1]  +1.15 (.08)   -1.61 (.35)    -3.41 (.68)  1.36
   4  [5,7,1,0]  +2.64 (.17)  +19.18 (.84)   +20.55 (1.25) 2.71
   sum           +9.36 (=78.3->87.7)  own-agent sum -1.1 (agent 5 net -0.4, agent 4 +0.9, agent 0 -1.6)
  SP-86, 7 changes: system +7.6/+4.2/+4.4/+1.5/+1.6/+0.4/+0.4 (sum 20.1); own-agent +3.8/+28.5/+12.1/+1.1/+1.5/+1.9/-0.03 (all >= 0).
  Search-gain (M=16 train) agrees with held-out system gain within 0.4 pp for every change (no search overfit).
Interpretation:
  H1 SUPPORTED. Deltacom table 87.35 >= 85 held-out, +7.0 over ECMP, +2.5 over greedy: the static headroom is real and K11 revision stands.
     SP-86 table 96.98 = learners' band (within 1 pp): C3 holds. BUT the n=0-optimised table decays under failures:
     Deltacom +4.8/+1.3 over ECMP at n=4/8 (below greedy 76.5 at n=4, ~= greedy at n=8); SP-86 table +1.8/+2.1 over ECMP and
     -2.1/-5.3 vs greedy at n=4/8. So 'demand-robust' (C3) does not extend to failure-robust; failure-aware search (A037) is the open gap.
  H2 Deltacom: NOT clean. Per change: #1 and #3 own<0 (supports), #2 own +0.9 (2.3 SE, tiny vs +3.3 system, ~supports), #4 own +19 (refutes: own-flow
     sees it). But #1 (-19.6 own) and #4 (+19.2 own, same agent 5) cancel: agent 5 net -0.4 own for +4.9 system; they are a coordinated
     pair (#4's own gain is conditional on #1 having diverted load), so own-flow credit would have to accept a -20 pp own excursion first
     (it never would: greedy hill-climb on own-flow rejects #1). Path-level: own-agent sum -1.1 vs system +9.4 -> own-flow credit is blind
     to the table on net. Only 4 changes, one start path, order-dependent => qualitative, not a statistic.
     SP-86 control: all own gains >= 0 (6/7 > 2 SE) and system ~ own sign -> own-flow credit IS aligned there (consistent with E023 +9.6, E025 SP table ~ learners).
     The contrast SP-86 aligned / Deltacom misaligned is the clean finding; it fits K17 (externality only on Deltacom).
  Not a refutation of the other reading: the miss may be optimisation/exploration (change #1 needs a non-greedy step), which A036 (warm-start) separates.
Verdict: H1 supported (C3 on SP-86; K11 revision -> high for n=0). H2 inconclusive per change, supported at path level on Deltacom (a coordinated pair
  with offsetting own-flow effects), with SP-86 as aligned control. Mechanism for C4 = own-flow credit cannot take the first step of a coordinated
  system change; wording 'cannot discover' -> 'does not reach by greedy own-flow ascent'.
