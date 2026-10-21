# Avenue registry — one line per idea (status: proposed·active·tried·dropped·parked·needed-for-paper)
# id | status | idea | claim | experiments | conclusion | date

A001 | tried | v1 MADDPG, 7 variants (LC/CC x simple/duelling x GNN), step reward | old P1 | E001 | never learned to route: static tables (K01) | 2026-10-02
A002 | dropped | old Paper 1 claims: critic dominance, failure robustness, GNN cost | old P1 | E001 | withdrawn — compared frozen tables | 2026-10-02
A003 | dropped | Paper 2: observation attacks (FGSM, iterated, logit, learned adversary) on v1 victims | P2 | E002 | on hold: victims ignored observations (robustness = insensitivity); strategy-plan: AVENUE #5 RED until a state-using learner exists (revisit if KILL-2 passes) | 2026-10-14
A004 | tried | utilisation-only threat model (attacker reads path_util slots 33-93 only) | P2 | E002 | now the only threat model; all pre-2026-09-29 attack numbers legacy | 2026-09-29
A005 | tried | learnability probe: factored critic + block-softmax actor (+entropy) | C1 | E003 | 63%->82%->91-92% on bandit; did not transfer to real net (E004) | 2026-10-02
A006 | dropped | reward misspecification hypothesis | C1 | E006 | rejected: reward ranks rules like PDR (greedy>random>sp>worst) | 2026-10-03
A007 | tried | actor entropy bonus 0.01/0.05 | C1 | E005 | no gain in real network (88.4-89.1 vs 89.0) | 2026-10-05
A008 | tried | observation-based central critic | C1 | E005 | no gain (85.6 vs v1 87.3) | 2026-10-05
A009 | tried | per-agent credit (reward.credit mixed, w=0.5/1.0) | C1 | E007 | +~1-2 pp, still ~random, least-loaded at chance | 2026-10-06
A010 | tried | realistic baselines: per-flow pinning (ECMP), stale telemetry greedy | setup | E008 | pinning costs random 8.2 pp, greedy 8.0 pp | 2026-10-05
A011 | tried | traffic granularity sweep (k x finer flows) | setup | E009 | finer flows make ECMP ~optimal; chose k=4 at 3x | 2026-10-06
A012 | tried | realistic regime pilot (pinned, k=4, 3x) step-reward MADDPG-LC | C1 | E010 | 92.7; least-loaded at chance (41.7%) | 2026-10-07
A013 | tried | independent DQN as positive control (step vs per-decision credit) | C1 | E011,E012 | step 91.1, per-decision 96.9 | 2026-10-07
A014 | tried | MADDPG with per-decision flow credit (--learner ac, LC/CC) | C1 | E013 | LC 97.3, CC 97.4 -> credit decides learning | 2026-10-08
A015 | tried | real per-destination duelling critic (V_d + A_dk - mean A) | C2 | E015 | no effect; old duelling was algebraically the simple head | 2026-10-09
A016 | tried | changing hotspot + failures training (rhf) | C2,C3 | E014,E015 | generalises to shifts; architecture spread <1 pp | 2026-10-09
A017 | tried | GNN encoder (original) | C2 | E016 | collapsed: 7-25x encoding spread, frozen decisions 4/6 runs | 2026-10-09
A018 | tried | stable GNN (LayerNorm, averaged step, lr/10) | C2 | E017 | 95.5 LC / 96.3 CC; neighbour shuffle 0 pp | 2026-10-10
A019 | dropped | duelling-GNN, 7-variant table in new paper | C2 | - | architecture secondary once credit fixed (Pedro 2026-10-08) | 2026-10-08
A020 | tried | whole-observation / telemetry shuffle ablation | C3 | E012-E017,E021,E022 | <1 pp for every learner -> static table | 2026-10-10
A021 | tried | failure-augmented training for DQN | C3 | E018 | helps under failures: 93.6/87.8 vs 92.1/84.3 (n=4/8) | 2026-10-09
A022 | tried | second topology Deltacom (provider-edge conversion, ttl 50, 8x) | C4 | E019-E022 | every learner ~ECMP; greedy +4.5 pp needs current state | 2026-10-13
A023 | tried | own-flow diagnostic (ECMP natural experiment) | C4,C5 | E023 | externality not the explanation on SP-86; ~58% on Deltacom | 2026-10-13
A024 | tried | supervised signal check (IPS own-flow value of greedy/table/util/full regressors) | C5 | E024 | tried: table captures all own-flow headroom; greedy <= table on own-flow; util/full ~ table (K16) | 2026-10-14
A025 | proposed | difference reward (own outcome minus drops caused on shared links; proxy from per-LSP + interface discards) | C5 | E029 (after E028) | E024: own-flow has no state headroom -> remaining gain is externality, which is exactly what a difference reward targets; gated on A029/E025 (D14) -> gate OPEN by E025/K17 (D16); still subject to KILL-2 | 2026-10-15
A026 | dropped | re-run Paper 2 attacks on per-decision-credit learners | P2 | - | pointless while learners ignore state (K08); strategy-plan: do-not-pursue until a state-using learner exists | 2026-10-14
A027 | tried | redo lit-graph drl-routing-adversarial (academic-research + research-strategist in Cowork): nearest works, reported numbers, DOIs | novelty, G1-G3 | - | done 2026-10-14: lit-graph + strategy-plan rev 0 (digest in strategy.md); novelty narrowed to credit-scheme comparison + state-use ablations + static/state split | 2026-10-14
A028 | needed-for-paper | re-run >=1 published MARL/DRL routing method in our simulator (pinned, stale, 3x/8x) vs ECMP/greedy | G1, C1-C4 | E028 (after E026) | target DRL-OR (github netlab-lcy/DRL-OR, MIT): port per-flow utility reward + obs + SP pretraining + safe fallback into our simulator (pinned, stale, SP-86 3x + Deltacom 8x, 3 seeds, shuffle ablation); NOT its Mininet testbed. KILL-3: if no end-to-end port by wk 4 -> "DRL-OR-style" learner | 2026-10-14
A029 | tried | deploy E024 own-flow table as every agent's rule; system PDR Deltacom 8x + SP-86 3x | C4,C5 | E025 | Deltacom table-for-all 78.0 < ECMP 80.4 (all n): externality confirmed, A025 gate open (K17) | 2026-10-15
A030 | active | stochastic-optimal static table (DOTE-style): optimise one static per-flow table by SGD over sampled demands in the simulator; eval on same 20 paired eps, both topologies | C3, C4 | E026 | strategy-plan AVENUE #6 (GREEN, ~1 wk); KILL-4: beats learners by >1 pp -> C3 = "learners approximate the robust table imperfectly"; no public DOTE code | 2026-10-14 || E026 2026-10-21: SP-86 best 97.0 ~ learners (C3 ok); Deltacom 87.4 >> ECMP 80.4 (K11 revised). Next: E026b held-out eval of table_best n=0/4/8.
A031 | dropped | train Teal / MAGNNETO now | G1 | - | strategy-plan: GPU contention on shared server, off-path (MLU/fluid setting, not comparable) | 2026-10-14
A032 | parked | field-wide state-use audit of published DRL-routing methods (strategy-plan AVENUE #2) | next paper | - | next paper (CoNEXT/INFOCOM/ToN 2027); reuses Paper 1 ablation method | 2026-10-14
A034 | tried | harden E025/K17: (a) 'table' rule behaviour when chosen path failed (step 0 of E026); (b) 2nd table export + 2nd eval traffic seed | K17, C4 | E026 step 0, E027 | (a) no change (failures rebuild kpath_cache; table and learners fall back to k=0); (b) E027: 12/12 cells table < ECMP by 1.9-3.7 pp, export spread <=0.12 pp -> K17 high | 2026-10-17
