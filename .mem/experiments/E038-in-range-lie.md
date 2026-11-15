# E038 — In-range telemetry lie (A048, D27)
Status: DONE 2026-11-12 20:46 (@8993bcc), analysed 2026-11-15. Avenue A048. Follows E035 (K24).

Hypothesis H1: the learners are static over realised telemetry; only out-of-range values steer them. A lure/repel clamped to the realised per-slot range costs <1 pp at n=0.
 H0: in-range lies still cost >2 pp at n=0 -> the policies read state inside the normal range; C3 "static" must be weakened (F5).
Change: extend Policy._apply_lie (src/idqn_control.py) with a range clamp: lie values limited to [p5, p95] of each path_util slot, taken from the unattacked rollout of the same checkpoint, n and eval seed (collect first, then attack; or a short pre-pass). Flag e.g. --lie-range 5,95. Keep E035 behaviour as default.
Arms: lure_in, repel_in (clamped), random_in (uniform within [p5,p95], budget-matched control). Reuse E035 out-of-range numbers and E032/E034 bases.
Config: victims as E035: E018 DQN (p4f), MADDPG LC rhf, CC rhf; SP-86 3x; n=0/4/8; 20 paired eps, seed 20240601; 3 seeds. Report lie_flip.
Reading (eval noise ~1 pp): n=0 lure_in and repel_in within 1 pp of random_in on all learners -> H1, C3 wording "static over realised telemetry" measured. >2 pp beyond random_in on any learner -> H0 for that learner. 1-2 pp -> inconclusive, report as is.
Also n=4/8: in-range lies beyond random_in there = the failure reaction (K21) is steerable inside the normal range (Paper 2 note).
Cost: eval-only, ~30 lines, ~3 h CPU (E035 scale).

## Implementation (73eba58)
- src/idqn_control.py: --lie-range lo,hi (needs --eval-lie and --eval-only). Per eval condition (n) a pre-pass rolls the unattacked policy with the base seeds (same RNG reset, no lie), collects the path_util slots (Policy.collect), and sets Policy.lie_lo/lie_hi = per (agent, slot) percentiles; seeds are reset again before the attacked rollout. _apply_lie then clips lure/repel (and margin) to [lo,hi]; random with a range = uniform inside [lo,hi] (replaces the +-eps step). Output adds lie_width (mean hi-lo) and result['lie_range']. Without the flag nothing changes (lo/hi None, collect None).
- tools/run_E038.sh: nets p4f (E018), lc, cc x seeds 0-2 x {lure,repel,random}_in, 1 container per cell evaluating n=0/4/8; 3 chains (per seed) at 1.5 cpus. Checkpoints copied from host_data/results/E035/<net>_s<seed>/. Out host_data/results/E038/<net>_s<seed>/<prefix>_seed<s>_{lure,repel,random}_in.json; markers ~/idqn/run_E038.done (3 seed lines + 'E038 all done').
- Bases: E032 p4f / E034 base (same as E035). Compare with E035 out-of-range: lure, repel, random10.
- Risks: slot value percentiles from the unattacked rollout use only that rollout (lies shift the state later); slots with no traffic have lo=hi=0 so lies there are inert (reported via lie_width).

## Results (2026-11-15; SP-86 3x, 20 paired eps, mean of 3 seeds; delta pp vs unattacked base E032 p4f / E034 lc, cc at n=0/4/8)
| learner | base | lure_in | repel_in | random_in | lure_in - random_in (per seed, n=0) | repel_in - random_in (per seed, n=0) |
|---|---|---|---|---|---|---|
| DQN E018 (p4f) | 96.47/93.62/87.82 | -5.0/-8.4/-9.5 | -4.7/-4.7/-7.1 | -1.0/-1.8/-3.1 | -4.0 -3.7 -4.4 | -3.7 -3.0 -4.4 |
| MADDPG LC rhf | 96.27/92.91/86.67 | -5.7/-6.4/-7.0 | -2.4/-3.1/-4.4 | -0.8/-1.1/-1.6 | -4.7 -6.5 -3.6 | -2.2 -1.3 -1.4 |
| MADDPG CC rhf | 97.03/92.97/86.72 | -8.0/-6.5/-8.0 | -4.0/-4.7/-6.4 | -0.7/-1.0/-1.3 | -10.3 -5.4 -6.1 | -3.2 -4.6 -2.2 |
Beyond random_in at n=4/8 (mean): lure -6.6/-6.5 (DQN), -5.4/-5.5 (LC), -5.5/-6.7 (CC); repel -2.9/-4.1, -2.0/-2.8, -3.7/-5.1; per seed at n=4/8, all 18 lure cells are -4.2 or lower and all 18 repel cells -1.9 or lower.
Decisions flipped at n=0 (lure/repel/random_in): DQN 0.53/0.53/0.39, LC 0.38/0.31/0.23, CC 0.34/0.29/0.21. lie_width (mean p95-p5 per slot) 0.54-0.56 at n=0, 0.68 at n=8; zero-gradient 0.
vs E035 out-of-range lure at n=0 (-7.4 DQN, -6.5 LC, -9.8 CC): the in-range lure keeps ~68% / ~88% / ~81% of the damage.
## Interpretation
- Smoke passed in the run itself (non-zero width, flips, 27/27 cells with n=0/4/8).
- Lure with every lied value inside the slot's own [p5,p95] costs 5-8 pp at n=0, 3.6-10.3 pp beyond the in-range random control on 9/9 seed cells: far above eval noise (~1 pp) and above the 2 pp H0 line for all three learners.
- Repel: DQN and CC > 2 pp beyond random on every seed; LC 1.3-2.2 pp (inconclusive for LC repel alone).
- random_in flips 21-39% of decisions and costs ~1 pp at n=0; lure flips only ~1.4x as many and costs 5-8x more. Damage follows where the flips point, not how many.
- Caveat: the clamp is per slot. Each value is in its normal range, but the joint pattern (congested path at its p5, the others at p95) need not be a realised state; realised ranges are wide (width ~0.55). So: 'values within their normal range', NOT 'in-distribution states'.
- Reconciles with K08: shuffling realised telemetry costs <1 pp because realised patterns rarely move the argmax where it matters; the policy is still a function of path telemetry.
## Verdict
H1 refuted, H0 supported (high for lure, 3 learners x 3 seeds; repel: DQN, CC yes, LC inconclusive). 'Only out-of-range telemetry steers the learners' is false. C3/F5 wording: static in effect over realised telemetry (shuffle), yet steerable by adversarial path telemetry even with each value inside its normal range (D30). n=4/8: in-range lies also beyond control -> failure-time behaviour steerable in range (Paper 2 note).
