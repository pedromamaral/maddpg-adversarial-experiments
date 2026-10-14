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
- C3 What is learned is a demand-robust static routing table, not adaptive routing. [K08]
- C4 Learners capture only static (demand-structure) headroom: beat ECMP where a fixed rule
  can (SP-86), match ECMP where it cannot (Deltacom). [K10,K11 — needs supervised check E024]
- C5 (open) The current-state signal is/isn't learnable from the observation -> difference reward?

## Metrics
- PDR % (end-to-end packet delivery ratio), 20 paired eval episodes (traffic seed 20240601),
  at 0..8 random link failures (n=0/4/8 headline). Realistic regime = 3x load SP-86, 8x Deltacom.
- least_loaded share of applied contested decisions vs chance (adaptivity); regret.
- static share, saturation (attribution); shuffle ablation delta (observation use).

## Success criteria
- "Learns": PDR clearly > ECMP (random+flow) AND least_loaded above chance.
- Adaptive: shuffle of telemetry/whole obs costs > seed noise (~1 pp).
- Reference ceiling: greedy (least-utilised, fresh telemetry); realistic ceiling greedy_stale2.

## Outputs
- Paper 1 v2 -> IEEE TNSM (markboth in tex). Paper 2 (attacks, paper/paper2_robustness.tex) on hold.
- Two MSc theses (students/) — handed off, built on v1 results (see archive).
