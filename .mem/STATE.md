# STATE — 2026-11-04

branch main @ a3ec0ff (clean, pushed). Server code at 38bce0f. Server idle (E033 done, fetched 11-04).
phase this session: research-think analyze E032 -> research-do implement + deploy + status E033.
Note: ssh alias "server" is unresolved locally; use pedroamaral@10.26.110.14 (add `< /dev/null` to ssh launch cmds or they hang).

## What changed
- E032 analysed: card verdict, LEDGER line, K21 new, K08/K19/K20 edited, A039 done, A038 deprioritised, D22, C3 reworded (project.md)
- tools/run_E033.sh (new, 38bce0f; no src change) + experiments/E033-fallback-ablation.md (card)
- runs.md: E033 row (done 10-30 23:12); container column rewritten plain (braces/parens broke tools/runs_status.sh)

## Latest results
E032 (K21): SP-86 fail-trained DQN static at n=0, failure-reactive at n=4/8 (shuffle -1.3/-2.7 pp, own frozen table -4.7/-4.6).
E028 (K19): DRL-OR-style ~ECMP on Deltacom (79.5); SP-86 ~2 pp below flow-credit DQN; fallback 0.05 resets 13-17% of decisions.
E031 (K20): no failure-robust static table from local search.

## Active runs
E033: FINISHED 10-30, FETCHED 11-04 -> .mem/results/E033.json (61 metrics, seeds 0-2); raw host_data/results/E033/ local+server.

## Open problem
Does the fallback (deployment vs training) or the SP pretraining cause E028's SP-86 2 pp gap? (E033)
Is Deltacom's missed headroom credit or optimisation? (A036/E030). Is K21's failure reaction a failure flag or load reading? (open)

## EXACT next step
research-think analyze E033 (compare nofb cells vs E032 base, train_sp vs E028 sp / E018). Then E030 (A036) -> E029 (A025).

## Read next
experiments/E033-fallback-ablation.md, results/E032.json (base with fallback), conclusions.md K19/K21, decisions.md D22.
