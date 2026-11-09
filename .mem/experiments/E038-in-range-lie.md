# E038 — In-range telemetry lie (A048, D27)
Status: SKELETON 2026-11-09, not implemented. Avenue A048. Follows E035 (K24).

Hypothesis H1: the learners are static over realised telemetry; only out-of-range values steer them. A lure/repel clamped to the realised per-slot range costs <1 pp at n=0.
 H0: in-range lies still cost >2 pp at n=0 -> the policies read state inside the normal range; C3 "static" must be weakened (F5).
Change: extend Policy._apply_lie (src/idqn_control.py) with a range clamp: lie values limited to [p5, p95] of each path_util slot, taken from the unattacked rollout of the same checkpoint, n and eval seed (collect first, then attack; or a short pre-pass). Flag e.g. --lie-range 5,95. Keep E035 behaviour as default.
Arms: lure_in, repel_in (clamped), random_in (uniform within [p5,p95], budget-matched control). Reuse E035 out-of-range numbers and E032/E034 bases.
Config: victims as E035: E018 DQN (p4f), MADDPG LC rhf, CC rhf; SP-86 3x; n=0/4/8; 20 paired eps, seed 20240601; 3 seeds. Report lie_flip.
Reading (eval noise ~1 pp): n=0 lure_in and repel_in within 1 pp of random_in on all learners -> H1, C3 wording "static over realised telemetry" measured. >2 pp beyond random_in on any learner -> H0 for that learner. 1-2 pp -> inconclusive, report as is.
Also n=4/8: in-range lies beyond random_in there = the failure reaction (K21) is steerable inside the normal range (Paper 2 note).
Cost: eval-only, ~30 lines, ~3 h CPU (E035 scale).
