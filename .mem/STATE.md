# STATE — 2026-11-05

branch main @ 65ebf93 + this state commit (local, NOT pushed; origin behind). Server code at 38bce0f. Server idle.
phase this session: research-do fetch E033 -> research-think analyze E033.
Note: ssh alias "server" is unresolved locally; use pedroamaral@10.26.110.14 (needs VPN, else port 22 times out; add `< /dev/null` to ssh launch cmds or they hang). macOS has no `timeout`.

## What changed
- results/E033.json + host_data/results/E033 (fetched, 9 JSON)
- experiments/E033-fallback-ablation.md (results, interpretation, verdict), LEDGER line
- conclusions.md K19 edited; avenues.md A028 done for paper (optional SP-pretrain-off cell); runs.md E033 row
- no new D#: direction unchanged

## Latest results
E033 (K19): SP-86 n=0 gap of E028 = 0.05 fallback (eval-only removal +1.00; retrain without it 96.05 vs E018 96.7). n=4/8 fallback no effect (|d|<=0.4), retrain -1.4/-1.7 vs E018. Deltacom +0.3..+0.8, still ~ECMP.
E032 (K21): SP-86 fail-trained DQN static at n=0, failure-reactive at n=4/8 (shuffle -1.3/-2.7 pp, own frozen table -4.7/-4.6).
E031 (K20): no failure-robust static table from local search.

## Active runs
None.

## Open problem
Is Deltacom's missed headroom credit or optimisation? (A036/E030; best hypothesis untested: drift back to ~80 = credit undoes the optimum, stays >=86 = exploration.)
Is K21's failure reaction a failure flag or load reading? (open)
Is the E033 residual at n=4/8 vs E018 SP pretraining or noise? (just above seed noise, cross-run; optional)
Unchecked: per-seed signs of the E033 +1.00 (card notes it).

## Task list to publish (2026-11-05, D23, confirmed by Pedro)
Paper 1: T1 E030 + SP-86 control (A036 design note) · T2 A042a n=8 shuffle + own table for E015 MADDPG and E014 DQN rhf (K22) · T3 paper-plan (outline; Pedro decides D13) · T4 paper1_tables.py entries for the TODO rows in paper.md, incl. E033 per-seed signs · T5 draft missing results/discussion · T6 Pedro's voice, bib, submit ~2027-01-14.
Paper 2 gate: T7 A042b telemetry lie, eval-only, alongside T1-T2 · T8 only if PASS: lit check in Cowork, then decide scope after Paper 1 is submitted.

## EXACT next step
research-do implement E030 (T1) + A042 eval script (T2, T7).

## Read next
paper.md, avenues.md A036/A042, decisions.md D20/D22/D23, conclusions.md K11/K18/K21.
