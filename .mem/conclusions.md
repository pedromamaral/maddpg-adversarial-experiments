# Conclusions — what we currently believe (PDR %, n = link failures; 3x = SP-86 realistic regime)

## Reference numbers
REF | greedy SP-86 3x n=0/4/8 | 95.7/93.6/89.1 | E010 (v2_p4_lc/eval_3x)
REF | greedy_stale2 / stale4 SP-86 3x n=0 | 93.7 / 92.1 | E010
REF | ECMP (random+flow) SP-86 3x n=0/4/8 | 91.4/89.7/81.7 | E010
REF | SP SP-86 3x n=0 | 75.8 | E010
REF | least-loaded applied SP-86 3x | greedy 100%, stale2 83%, stale4 71%, ECMP 38%, chance ~41% | E010 flow_choice_3x.json
REF | Deltacom 8x n=0 | greedy 84.9, stale2 84.1, ECMP 80.4, SP 74.4 | E020
REF | v1 2x (old regime) | policies 73.1-87.3; random 90.0, greedy 97.2, SP 73.8 | E001 ceil2x
REF | seed noise | <=1.3 pp train hotspot; <=2.3 pp shifts; 3-5 pp unstable GNN / Deltacom rhf | E012-E022
REF | eval noise | 20 paired eps; 5-ep sweeps differ ~0.5-1 pp from 20-ep | E009

## Conclusions  (id | statement | evidence | confidence | date)
K01 | v1 MADDPG never learned to route: static share 82-99%, least-loaded 21-42% vs chance 42-46%, saturated 77-99%, telemetry shuffle <0.5 pp (6/7) | E001,E002 | high | 2026-10-02
K02 | Root cause in probe: scalar joint critic resolves choice 63% vs 55% chance; sigmoid actor saturates; actor capable (98% supervised) | E003 | high | 2026-10-02
K03 | Reward is aligned with PDR (greedy 22.75 > random 20.01 > sp 16.40 > worst 10.93 /ep) but greedy beats random by only ~12% of return | E006 | high | 2026-10-03
K04 | Traffic is sparse: ~1.8 flows/step, 0.6% of 294 decisions live, 19% steps empty (orig.); ~3% at k=4 3x | E004,E009 | high | 2026-10-05
K05 | Credit, not algorithm, decides learning: step MADDPG-LC 92.7, step DQN 91.1; per-decision DQN 96.9, MADDPG LC 97.3 / CC 97.4 (seeds within 0.2) | E010-E013 | high | 2026-10-08
K06 | With per-decision credit, LC/CC x simple/duelling (rhf) all 96.2-96.9 (n=0), 92.7-93.0 (n=4), 86.7-87.0 (n=8) -> architecture secondary | E015 | high | 2026-10-09
K07 | Original GNN collapses (LC-GNN 90.9); stable GNN 95.5 LC / 96.3 CC; neighbour shuffle 0 pp | E016,E017 | high | 2026-10-10
K08 | Learned = demand-robust static table: whole-obs shuffle <1 pp for every learner | E012-E017 | high | 2026-10-10
K09 | Fixed-hotspot tables fall below ECMP on shift A (DQN 90.2, LC 92.8, CC 89.5 vs 93.5); changing-hotspot training generalises (DQN rh 95.9/95.7 A/B) | E012-E015 | high | 2026-10-09
K10 | Deltacom 8x: every learner ~ECMP (step LC 80.9, fix LC 80.2/CC 79.8, rhf 80.3/79.8 vs ECMP 80.4); least-loaded at/below chance; shifts below ECMP (86-87.5 vs 88.1) | E020-E022 | high | 2026-10-13
K11 | Learners capture static headroom only: SP-86 has it, Deltacom does not (no fixed rule beats ECMP; greedy's 4.5 pp needs current state) | E019-E022 | medium | 2026-10-13
K12 | Own-flow effect: SP-86 +9.6 pp (SE 0.5), own gain 5.9 >= system 4.3 (externality not the explanation); Deltacom +3.3 pp (SE 0.7), own gain 1.9 = ~42% of system 4.5 | E023 | medium | 2026-10-13
K13 | Realism: per-flow pinning costs random 8.2 pp, greedy 8.0 pp; finer flows (k>=8) make ECMP ~optimal | E008,E009 | high | 2026-10-06
K14 | Failure-augmented training helps DQN under failures (96.7/93.6/87.8 vs 96.9/92.1/84.3) | E018 | medium | 2026-10-09
K15 | Old P1 claims and all v1 attack results (P2, theses) describe frozen tables; FGSM gradient-masked; legacy all-feature threat model pre-2026-09-29 | E001,E002 | high | 2026-10-02
K16 | Own-flow objective has no state-dependent headroom beyond a static table: IPS own-flow Deltacom ecmp 88.2, table 95.8, greedy 94.1, util 96.7, full 94.8 (SE ~1.4); SP-86 all rules 99.3-100.7 vs 93.3. Greedy's system gain on Deltacom (K10) is therefore externality, not own-flow -> per-decision own-flow credit cannot reach it; supports C4, reframes C5 | E024 | medium (off-policy, ECMP background, ~3 pp resolution) | 2026-10-14
