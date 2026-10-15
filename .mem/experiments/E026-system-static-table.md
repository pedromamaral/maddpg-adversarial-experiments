# E026 — Best static table by SYSTEM PDR (simulator-only local search)
Status: PLANNED 2026-10-15. Avenue A030 (+A034a as step 0); decision D17; follows E025 (K17); gates KILL-4, tests K11.

Hypothesis:
  - SP-86 3x: the best system-objective static table ~ learners (96.2-97.4 at n=0) -> C3 holds (learners ~ the
    robust table).
  - Deltacom 8x: the best static table ~ ECMP (80.4) -> K11 holds (no static headroom; greedy's 4.5 pp needs state).
  Unlike E025 (own-flow-optimal table, which is selfish per K17), the objective here is system PDR.

Change (minimal, behind a switch; baselines unchanged):
  0. Step 0 (A034a): check how the 'table' rule (runner._rule_action, E025) handles a failed chosen path
     (drop? fallback to path 0? next live path?). Make it match the learners' failure semantics (masked argmax over live
     paths) and record what it did before. Re-run E025 SP-86 n=4/8 only if the semantics changed.
  1. tools/static_table_search.py: table T = {agent: {dest: k}} over all (agent,dest) pairs (SP-86: 14x21=294 entries;
     Deltacom: same agent layout per config). The objective J(T) = mean system PDR over M TRAIN traffic episodes
     (seeds disjoint from eval seed 20240601), with common random numbers across candidates. Uses the existing 'table'
     rule via TABLE_FILE or an in-process call, NOT the eval path.
  2. Search: first-improvement coordinate descent. Visit entries in decreasing order of traffic volume. For each, try
     every other k; accept if dJ > eps. Repeat sweeps until no accept or the budget runs out. Starts: SP table, E025
     own-flow table, the learners' argmax table (best seed), plus 1-2 random starts. Keep the best by J.
  3. Failures: optimise at n=0. Optionally a 2nd variant optimised on a mix of n in {0,4} if the budget allows.
     Always report n=0/4/8.
  4. Log per sweep: J, number of accepts, changed entries. Final table JSON is saved next to the results.
Config: configs/v2_pilot4_lc.json (LOAD=3), configs/v2_deltacom.json (LOAD=8). M and eps to be set in implement after
  timing one episode (target: J's SE << 1 pp; eps ~ 2 SE of the paired difference).
Eval: tools/run_eval.sh, 20 paired eps, seed 20240601, n=0/4/8, RULES="table random+flow greedy greedy_stale2 sp"
  with TABLE_FILE = searched table. Held-out eval -> also report train-vs-eval gap (overfit check).
Success criterion / reading (seed noise ~1.3 pp; eval noise <=1 pp):
  - SP-86: |table - best learner (97.4)| <= 1 -> C3 holds. table > learners + 1 -> KILL-4: "learners approximate the
    robust table imperfectly". table < learners - 1 -> search under-converged (more sweeps/starts) or learners exploit state
    (contradicts K08) -> investigate before claiming.
  - Deltacom: table <= 81.9 (ECMP + 1.5) -> K11 holds; A025 targets state use as planned.
    table > 81.9 -> K11 revised: Deltacom HAS static headroom, which learners miss under selfish own-flow credit (K17).
    A025 is retargeted at finding that table, and C4 is reworded.
  - train J >> eval by > 1 pp -> overfit to train demands; raise M.
Cost: 0 GPU; CPU-days (parallelise candidates over the 16 cores; don't starve other users).

Results:
Interpretation:
Verdict:
