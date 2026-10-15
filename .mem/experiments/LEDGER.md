# Experiment ledger  (id | date | sha | change | params | key result | verdict | avenue)
# results dir = host_data/results/<dir> (local copy + server). Reconstructed 2026-10-14 from _import.md + git.

E001 | 2026-09 | <=1764cd3 | v1 MADDPG 7 variants, step reward (reward_fix) | 2x hotspot | PDR 73.1-87.3 vs random 90.0, greedy 97.2 (ceil2x) | negative: static tables | A001
E002 | 2026-09-29..10-02 | 2e2e58e,76f9674 | v1 attacks util-only, telemetry_reliance, policy_attribution | 2x | shuffle <0.75 pp; static 82-99% | victims ignore obs | A003,A004
E003 | 2026-10-02 | d1bd248 | learnability probe (tools/learnability_probe.py) | bandit | 63% -> 82% (factored+softmax) -> 91-92% (+entropy) | root cause found | A005
E004 | 2026-10-02 | d19b15a | pilot 1 v2_pilot_lc / v2_pilot_cc (learning fix) | 100 ep, 2x | LC 89.0, CC 86.8 (98% static) | < random 90.0 | A005
E005 | 2026-10-03 | 699bd60 | pilot 2 v2_p2_lc_e01/e05, v2_p2_cc | entropy .01/.05; obs critic | LC 89.1/88.4, CC 85.6 | no gain | A007,A008
E006 | 2026-10-03 | f45ba9c | reward_alignment.py | — | greedy 22.75>random 20.01>sp 16.40>worst 10.93 | reward aligned | A006
E007 | 2026-10-05 | 0889fb1 | pilot 3 v2_p3_lc_w05/w10, v2_p3_cc | credit mixed w | LC 91.3/91.2 (n=0), CC 85.3 | rule not met | A009
E008 | 2026-10-05 | 34954cf | eval_realism (v2_pilot_lc/eval_realism) | +flow pinning, stale | random 90.0->81.9, greedy 97.2->89.2 | realism adopted | A010
E009 | 2026-10-06 | 573994b | granularity sweep (granularity/k*.log) | k, 2x/3x, 5 eps | k=4 3x: SP 74.9, ECMP 90.8, greedy 96.3 | chose k=4 3x | A011
E010 | 2026-10-06 | 4ded6c9 | pilot 4 v2_p4_lc (realistic) + eval_3x + flow_choice | pinned k=4 3x | LC 92.7; least-loaded 41.7% (chance) | step credit fails | A012
E011 | 2026-10-07 | 9edf460 | idqn_p4_step (DQN, step credit) | 3 seeds | 91.1 (88.0-93.1) | step fails for DQN too | A013
E012 | 2026-10-06 | a690385 | idqn_p4 (DQN, per-decision credit) | 3 seeds | 96.9 n=0; shiftA 90.2 | learns (table) | A013
E013 | 2026-10-07 | d03e5cb | ac_local_p4, ac_central_p4 (MADDPG per-decision) | 3 seeds | LC 97.3, CC 97.4 | credit decides | A014
E014 | 2026-10-07 | 3b4a754 | idqn_p4_rh, idqn_p4_rhf (changing hotspot [+failures]) | 3 seeds | rh shiftA/B 95.9/95.7 | generalises | A016
E015 | 2026-10-08 | 38ca713 | ac_{local,central}{,_duel}_rhf | 3 seeds | 96.2-96.9/92.7-93.0/86.7-87.0 (n=0/4/8) | arch secondary | A015,A016
E016 | 2026-10-08 | 38ca713 | ac_*_gnn_rhf (original GNN) | 3 seeds | LC-GNN 90.9; frozen 4/6 | collapse | A017
E017 | 2026-10-09 | dc7efab | ac_*_gnns_rhf (stable GNN) | 3 seeds | LC 95.5, CC 96.3; nbr shuffle 0 pp | ok, no gain | A018
E018 | 2026-10-07 | 9edf460 | idqn_p4_fail (failures in training) | 3 seeds | 96.7/93.6/87.8 vs 96.9/92.1/84.3 | helps | A021
E019 | 2026-10-12 | 63eb5b1 | Deltacom calibration (granularity_dc, load*.log) | ttl 50; 3/5/8/12x | 8x chosen (3x: all rules ~96) | setup | A022
E020 | 2026-10-12 | 63eb5b1 | v2_dc_lc (step MADDPG-LC, Deltacom 8x) + rules | 1 seed | 80.9 vs ECMP 80.4, greedy 84.9 | ~ECMP | A022
E021 | 2026-10-12 | 63eb5b1 | dc_ac_{local,central}_fix | 3 seeds | LC 80.2, CC 79.8 | ~ECMP | A022
E022 | 2026-10-12 | 63eb5b1 | dc_ac_{local,central}_rhf | 3 seeds | LC 80.3, CC 79.8 (seeds 75.4-83.3) | ~ECMP | A022
E023 | 2026-10-13 | 5b068de | own_flow_effect.py (own_flow/*.json, server only) | ECMP natural exp. | SP-86 +9.6 pp; Deltacom +3.3 pp | see K12 | A023
E024 | 2026-10-14 | 6e29524 | supervised_signal.py on v2_pilot4_lc + v2_deltacom | 150 train eps, IPS own-flow | Deltacom ecmp 88.2, greedy 94.1, table 95.8, util 96.7, full 94.8 (SE~1.4); SP-86 all rules 99.3-100.7 vs ecmp 93.3 | no state signal beyond table; greedy <= table on own-flow (K16) | A024
E025 | 2026-10-15 | d209c69 | table rule (E024 own-flow table) for all agents, run_eval.sh v2_deltacom 8x + v2_pilot4_lc 3x | 1 export, 20 paired eps | Deltacom table 78.0/70.9/64.6 vs ECMP 80.4/72.8/66.7, greedy 84.9; SP-86 table 95.3/90.2/82.7 vs learners 96.2-97.4 (n=0) | externality confirmed: own-flow table < ECMP system (K17) | A029
