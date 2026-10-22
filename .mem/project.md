# Project — MADDPG routing on a service-provider network (updated 2026-10-14)

Repo: maddpg-adversarial-experiments. Multi-agent RL routing (14 PE agents x 21 destinations,
K paths each) on a real 86-node SP topology ("SP-86"), second topology Deltacom (113 nodes).

## Research question
What do MARL routing agents actually learn, and what makes them learn at all?
(Original question — robustness of MADDPG routing to observation attacks — is ON HOLD: the
v1 victims never used their observations, see K01.)

## Target claims (Paper 1 v2, paper/paper1_v2/paper1_v2.tex)
- C1 Credit assignment, not algorithm/architecture, decides whether MARL routing learns
  (step reward: no learning; per-decision flow credit: learns with DQN and MADDPG). [supported K05]
- C2 Once credit is fixed, architecture (LC/CC critic, simple/duelling, GNN) is secondary. [K06,K07]
- C3 What is learned is a demand-robust static routing table, not adaptive routing. [K08; A030 vs a stochastic-optimal
  static table — KILL-4: >1 pp gap -> 'learners approximate the robust table imperfectly']
- C4 Learners capture only static (demand-structure) headroom: beat ECMP where a fixed rule
  can (SP-86), match ECMP where it cannot (Deltacom). [K10,K11,K16,K17 supported]
- C5 (open, reframed by K16) Own-flow credit has no state headroom beyond the table; the remaining
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
- Paper 1 v2 -> IEEE TNSM (markboth in tex). Paper 2 (attacks, paper/paper2_robustness.tex) on hold.
- Two MSc theses (students/) — handed off, built on v1 results (see archive).
