# STATE — 2026-11-09

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
E035 (K24, D26): gate A042 PASS. Lure on path_util costs 6.5-11 pp (>=4 beyond random) on DQN and MADDPG, already at n=0; failure growth DQN-only. Paper 2 after Paper 1 submission, new narrative.
E030 (K23): Deltacom warm start on the E026 table falls 86.3 -> ~80 in 50 eps, ends = cold start = ECMP (80.3-80.5); SP-86 control holds 97. Credit undoes the optimum.
D25 (11-09): GNN dropped from Paper 1 (current encoder near-passthrough); link-aware GNN parked as A046. T4: remove GNN rows from paper1_tables.py.
E034 (K21 revised): failure reaction shared by DQN rhf and MADDPG LC/CC rhf on SP-86: n=8 shuffle -0.7..-1.3 pp (17/18 cells <0), own table -4.4..-4.8 pp at n=8; ~half E018 on shuffle. C3 updated in project.md and paper.md.
E033 (K19): SP-86 n=0 gap of E028 = 0.05 fallback (eval-only removal +1.00; retrain without it 96.05 vs E018 96.7). n=4/8 fallback no effect.
E032 (K21): SP-86 fail-trained DQN static at n=0, failure-reactive at n=4/8 (shuffle -1.3/-2.7 pp, own frozen table -4.7/-4.6).
E031 (K20): no failure-robust static table from local search.

## Active runs
E036 running since 11-09 (server, E036_* containers); done when ~/idqn/run_E036.done has "E036 all done".
E037 running since 11-09 00:14 @236a6f1 (maddpg-E037-train-s{1,2}, ~6 h/seed, then E037_eval_*); ~/idqn/run_E037.done (2 lines).

## Open problem
Deltacom's missed headroom is credit, not search (E030, K23).
Is the ~7 pp Deltacom headroom one-matrix traffic engineering? (E036; the searched table was never evaluated on other demand.)
K21 reaction is shared across learners (E034). Failure flag or load reading: open.
Unchecked: per-seed signs of the E033 +1.00 (T4).

## Task list to publish (D23)
Paper 1: T1 E030 analysed (credit undoes the optimum, K23) · T2 E034 analysed (reaction shared) · T2b E036, E037 fetched, analyse (D24); E038, E039 to implement (D27) · T3 paper-plan done (D24) · T4 paper1_tables.py entries listed in paper.md · T5 draft results/discussion, fix F1-F4 · T6 Pedro's voice, bib, submit ~2027-01-14.
Paper 2 gate: T7 E035 analysed: PASS (D26) · T8 only if PASS: lit check in Cowork, then decide scope after Paper 1 is submitted.

## EXACT next step
1. E036 + E037 fetched 11-10 (.mem/results/E036.json, E037.json): research-think analyze E036, then E037.
1b. D27: research-do implement E038 (in-range lie, eval-only), then E039 (MADDPG LC warm start); deploy both. Cards in experiments/.
2. E030, E034, E035 fetched 11-07 (.mem/results/E03{0,4,5}.json). Not in those summaries, read from host_data/results: E030 track (PDR + table_agree per episode), E034 own-table evals (v2_p4_lc/E034_tab_*), E035 lie_flip/lie_zero_grad and worst ceiling (v2_p4_lc/E035_worst_n*). Next: research-think analyze E030, then E035; E036/E037 after they run.
3. Then research-do figures (T4).

## Read next
codebase.md, infra.md, runs.md, experiments/E036-table-on-shifts.md, experiments/E037-step-deltacom-seeds.md, decisions.md D24.
