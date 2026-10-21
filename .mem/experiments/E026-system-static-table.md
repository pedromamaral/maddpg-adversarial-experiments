# E026 — Best static table by SYSTEM PDR (simulator-only local search)
Status: ANALYSED 2026-10-21 (search only; held-out eval pending), sha 46f4985. Avenue A030 (+A034a as step 0); decision D17; follows E025 (K17); gates KILL-4, tests K11.

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

Implemented (2026-10-15):
  Step 0 (A034a): NO CHANGE needed. After failures the runner rebuilds kpath_cache on the post-failure graph
    (runner:1112/1209/2346), so k always indexes a LIVE path; k >= n_live -> k=0 for the 'table' rule (runner:2060) and
    effectively k=0 for learners (_next_hop_for_transit falls through, env:1276). Semantics identical -> no E025 re-run.
    Caveat for analysis: after failures index k may be a different physical path than at n=0 (renumbering), for
    table and learners alike; learners see utils, the table does not.
  tools/static_table_search.py: J over M paired train eps (seed0=7000.., reseeded per ep = CRN, verified
    deterministic), fork pool over episodes, entries by traffic volume, first-improvement, eps on paired mean gain.
    Logs search_log.json (per sweep J, accepts, changed [a,d,k_old,k_new], gains, SE); table_<start>.json, table_best.json.
  Smoke/noise (SP-86 3x, local Mac, 8 workers, M=8): ~8 s/episode, 14.7 s/J; J SE 0.63 pp; paired single-entry
    gain SE 0.08-0.69 pp. Landscape steep (single changes from a random start: -7.8..+0.7 pp).
  Setting: SP-86 M=16, eps=0.5 pp (~1.5-2 SE paired at M=16), --failures 0. 294 entries x ~3 alt ~ 880 J/sweep
    -> ~4-7 h/sweep at 8 workers; use --max-evals as budget. Deltacom: time with --sweeps 0 at deploy, same rule.
  Starts: sp, E025 table JSON (host_data/results/<E025 run>/table*.json), random:1, random:2; learner-argmax table
    needs an export (not built; optional).
  Usage: python tools/static_table_search.py --config configs/v2_pilot4_lc.json --load 3 --episodes 16 --eps 0.5
    --workers 8 --start sp --start <E025.json> --start random:1 --out-dir host_data/results/E026_sp
Results (2026-10-21; train J = system PDR, M=16 CRN train eps seed0=7000, n=0, eps 0.5 pp; ALL starts converged = last sweep 0 accepts):
  start        | SP-86 J0 -> J (sweeps, accepts/sweep) | Deltacom J0 -> J (sweeps, accepts/sweep)
  sp           | 75.6 -> 97.0 (4; 5,1,1,0)             | 74.5 -> 87.1 (2; 3,0)
  E025 table   | 95.0 -> 96.1 (2; 1,0)                 | 77.9 -> 87.4 (3; 3,1,0)   <- dc best
  random:1     | 89.3 -> 95.2 (3; 2,2,0)               | 78.3 -> 86.9 (2; 2,0)
  random:2     | 90.6 -> 95.4 (2; 4,0)                 | 78.1 -> 86.9 (3; 2,1,0)
  J0 SE 0.38-0.62 pp. Calibration: E025 table train J0 95.0 / 77.9 vs its eval 95.3 / 78.0 (E025) -> train J is on the
  eval scale for a FIXED table (<=0.3 pp).
  Not yet run: held-out eval of table_best (run_eval.sh TABLE_FILE, seed 20240601, n=0/4/8) -> E026b.
Interpretation:
  - Deltacom: every start (incl. SP 74.5 and the selfish E025 table) reaches 86.9-87.4 (spread 0.55 pp ~ 1 SE) with only
    2-4 accepted changes, i.e. a few high-traffic entries carry +9..+13 pp. Best 87.4 is +7.0 over ECMP 80.4, +6.8 over
    learners 80.3-80.9, and +2.5 over greedy 84.9 (which uses current state). Far beyond the 81.9 threshold and any
    noise (seed 1.3, eval <=1 pp). Winner's-curse inflation is bounded by the few accepts (<= ~4 x 0.5 pp worst case)
    and cannot close a 7 pp gap. => Deltacom HAS large static headroom at n=0; K11's "no fixed rule beats ECMP" was
    true only for the rules we tried (SP, ECMP, own-flow table). Learners miss it -> fits K17 (own-flow credit).
  - SP-86: best 97.0 (sp start) vs learners 96.2-97.4 at n=0 -> within 1 pp -> C3 holds. Local optima differ by up to
    1.8 pp (95.2-97.0) and the E025 start stalled at 96.1 -> search landscape is multi-modal; 97.0 is a lower bound on
    the best static table, so 'learners ~ best table' is 'learners >= best FOUND table - 0.8'.
  - Caveats: (1) train eps, not eval protocol -> held-out eval needed before the paper uses any number (rule: paper numbers
    from paper1_tables.py); (2) optimised at n=0 only — failure robustness unknown (E025 table lost 2.5-4 pp under n=4/8);
    (3) index k renumbers after failures (step 0 note).
Verdict (provisional until E026b held-out eval):
  - SP-86 (best table ~ learners): SUPPORTED (|97.0 - 96.2..97.4| <= 1) -> C3 holds on SP-86.
  - Deltacom (best table ~ ECMP, K11): REFUTED (87.4 >> 81.9). K11 revised; C4 must be reworded; per card, A025 is
    retargeted at reaching this table (system-credit), and greedy's edge is NOT evidence of state use.
