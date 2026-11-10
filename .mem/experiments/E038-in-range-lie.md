# E038 — In-range telemetry lie (A048, D27)
Status: IMPLEMENTED 2026-11-10 @73eba58, not smoke-run (local venv has no numpy: first server run is the smoke), not deployed. Avenue A048. Follows E035 (K24).

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
