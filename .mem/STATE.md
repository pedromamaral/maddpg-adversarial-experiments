# STATE — 2026-11-05

branch main @ fd80333, origin in sync; this session's .mem/ edits are NOT committed. Server code at df8fe17 (E035 margin fix; E030 and E034 unaffected).
phase this session: research-think paper-plan (T3 of D23), done; Pedro agreed D24.
Note: ssh alias "server" is unresolved locally; use pedroamaral@10.26.110.14 (needs VPN, else port 22 times out; add `< /dev/null` to ssh launch cmds or they hang). macOS has no `timeout`.

## What changed
- paper.md: rewritten (framing proposal, claim map, outline, contradicted sentences F1-F4, E030/E034 branches, T4 list)
- avenues.md: A044 -> E036 (table on shift A/B, eval-only), A045 -> E037 (step Deltacom 2 more seeds), both active
- decisions.md: D24 (framing, title, experiment list reopened), D13 resolved; project.md: C1, C3 wording
- experiments/E036, E037: card skeletons
- no tex, code or K# changes

## Latest results
E033 (K19): SP-86 n=0 gap of E028 = 0.05 fallback (eval-only removal +1.00; retrain without it 96.05 vs E018 96.7). n=4/8 fallback no effect.
E032 (K21): SP-86 fail-trained DQN static at n=0, failure-reactive at n=4/8 (shuffle -1.3/-2.7 pp, own frozen table -4.7/-4.6).
E031 (K20): no failure-robust static table from local search.

## Active runs
Per runs.md, updated 11-05 22:05 outside this session (not verified here): E030 running (spW ~00:00, dcC ~02:30 11-06); E034 done 19:51, fetch next; E035 running with crashed cells, see its row. Check with tools/runs_status.sh.

## Open problem
Is Deltacom's missed headroom credit or optimisation? (E030; drift to ~80 = credit undoes the optimum, stays >=86 = search/exploration.)
Is the ~7 pp Deltacom headroom one-matrix traffic engineering? (E036; the searched table was never evaluated on other demand.)
Is K21's reaction shared across learners? (E034; read DQN rhf first, E018 is fixed-hotspot.) Failure flag or load reading: open.
Unchecked: per-seed signs of the E033 +1.00 (T4).

## Task list to publish (D23)
Paper 1: T1 E030 running · T2 E034 done, fetch next · T2b E036, E037 to implement (D24) · T3 paper-plan done (D24) · T4 paper1_tables.py entries listed in paper.md · T5 draft results/discussion, fix F1-F4 · T6 Pedro's voice, bib, submit ~2027-01-14.
Paper 2 gate: T7 E035 running · T8 only if PASS: lit check in Cowork, then decide scope after Paper 1 is submitted.

## EXACT next step
1. research-do implement E036, then E037 (cards in experiments/); deploy when cores allow alongside E030.
2. When runs finish: research-do fetch E030 + E034 + E035 + E036 + E037, then research-think analyze (E034, E030, E036, E037, E035); apply branches in paper.md.
3. Then research-do figures (T4).

## Read next
codebase.md, infra.md, runs.md, experiments/E036-table-on-shifts.md, experiments/E037-step-deltacom-seeds.md, decisions.md D24.
