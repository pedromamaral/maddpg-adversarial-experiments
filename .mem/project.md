# Project — MADDPG routing on a service-provider network (updated 2026-11-05)

Repo: maddpg-adversarial-experiments. Multi-agent RL routing (14 PE agents x 21 destinations,
K paths each) on a real 86-node SP topology ("SP-86"), second topology Deltacom (113 nodes).

## Research question
What do MARL routing agents actually learn, and what makes them learn at all?
Goal: publish. Paper 1 first; everything else is judged by whether a Paper 1 claim needs it.
(Original question — robustness of MADDPG routing to observation attacks — is ON HOLD: the
v1 victims never used their observations (K01). Flow-credit learners do learn, but use state only
under failures (K08, K21), so Paper 2 is gated on A042, see Outputs.)

## Target claims (Paper 1 v2, paper/paper1_v2/paper1_v2.tex)
- C1 Credit assignment, not algorithm/architecture, decides whether MARL routing learns
  (step reward: no learning; per-decision flow credit: learns with DQN and MADDPG on SP-86). [supported K05]
  Necessary, not sufficient: on Deltacom flow credit does not beat step or ECMP (K10; step 1 seed until E037). D24.
- C2 Once credit is fixed, architecture (LC/CC critic, simple/duelling, GNN) is secondary. [K06,K07]
- C3 What is learned is a static routing table in normal operation, not load-adaptive routing; demand-robust when trained on changing demand (K09), and untested for the searched table (E036); failure-trained learners add a learned failure reaction, shown for DQN and MADDPG LC/CC on SP-86 (D22, K21, E034); none on Deltacom (K22). [K08, K21; A030 vs a stochastic-optimal
  static table — KILL-4: >1 pp gap -> 'learners approximate the robust table imperfectly']
- C4 Learners reach static headroom only when own-flow credit points at it. SP-86: learners ~ best
  static table (K11). Deltacom: a static table reaches 87.35 vs ECMP 80.4, yet learners stay ~80.3
  (K10, K11, K19). Mechanism tentative: own-flow-optimal table is selfish (K17), own-flow ascent
  misses coordinated changes (K18, low-medium); E030 tests it. Table is an n=0 result, it decays under failures.
  [Reworded 2026-11-05: 'Deltacom has no static headroom' was refuted by E026/E026b, see D20.]
- C5 (open, optional for TNSM) Own-flow credit has no state headroom beyond the table; the remaining
  Deltacom gain is system-level (externality) -> E025 confirmed (K17: own-flow table-for-all < ECMP) -> difference reward (A025).
  KILL-2 PASS = Deltacom 8x 3 seeds > 80.4 by >=1.5 pp AND telemetry shuffle >1 pp; also report share of the 80.4 -> E026 system-table gap closed (D19); FAIL -> open problem.

## Metrics
- PDR % (end-to-end packet delivery ratio), 20 paired eval episodes (traffic seed 20240601),
  at 0..8 random link failures (n=0/4/8 headline). Realistic regime = 3x load SP-86, 8x Deltacom.
- least_loaded share of applied contested decisions vs chance (adaptivity); regret.
- static share, saturation (attribution); shuffle ablation delta (observation use).

## Success criteria
- "Learns": PDR clearly > ECMP (random+flow) AND least_loaded above chance.
- Adaptive: shuffle of telemetry/whole obs costs > seed noise (~1 pp).
- Reference ceiling: greedy (least-utilised, fresh telemetry); realistic ceiling greedy_stale2.

## Literature
- lit-graph topic: drl-routing-adversarial (redone 2026-10-14; digest + kill criteria in strategy.md).

## Outputs
- Paper 1 v2 -> IEEE TNSM (markboth in tex), target ~2027-01-14.
- Paper 2 (attacks, paper/paper2_robustness.tex) on hold. The v1 manuscript cannot be salvaged (K15).
  Restart only if gate A042 passes, as a new study on flow-credit failure-trained learners.
- Two MSc theses (students/) — handed off, built on v1 results (see archive).
