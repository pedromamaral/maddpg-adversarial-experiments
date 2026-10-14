# STATE — 2026-10-14 (init of .mem/)

branch main @ 5b068de (+ uncommitted tools/supervised_signal.py)
phase just done: memory init (research-memory init) from import-archive.md + git + results.

## Active runs
none — server idle (checked 2026-10-14 12:58 UTC; GPU 0 %).

## Latest results
- E013 per-decision MADDPG SP-86 3x: LC 97.3 / CC 97.4 vs ECMP 91.4, greedy 95.7 (K05)
- E015-E017 architecture secondary; E012-E017 shuffle <1 pp -> static table (K06-K08)
- E020-E022 Deltacom 8x: all learners ~ECMP 80.4 vs greedy 84.9 (K10)
- E023 own-flow effect SP-86 +9.6 pp, Deltacom +3.3 pp (~42 % of system gain) (K12)

## Open problem
Is the current-state signal learnable from the observation on Deltacom (greedy's 4.5 pp)?
Best hypothesis: weak own-flow signal (+3.3 pp) + externality ~58 % -> per-decision credit can't see it.

## EXACT next step (research-do)
1. Commit + push tools/supervised_signal.py; `git pull` on server.
2. Run E024 on server (docker pattern in infra.md), both configs:
   python3 tools/supervised_signal.py --config configs/v2_pilot4_lc.json --train-episodes 150
   python3 tools/supervised_signal.py --config configs/v2_deltacom.json --train-episodes 150
3. Fetch JSON, summarise into results/E024.json; then research-think applies decision rule in
   experiments/E024-supervised-signal.md (-> A025 difference reward or write C4 as is).
Parallel (Pedro): agree reframing for pending paper edits (D13).

## Read next
experiments/E024-supervised-signal.md, infra.md, tools/supervised_signal.py (args at top).
