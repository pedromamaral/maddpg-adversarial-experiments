# E031 — Failure-aware static table search (A037)
Status: ANALYSED 2026-10-29 (ran 2026-10-23..25, sha 3f02e7a; launch tools/run_E031.sh). Avenue A037; decisions D19, D20; follows E026/E026b (K11, K18).

Hypothesis: a static table searched with J over n in {0,4} link failures (restart from E026 table_best) keeps its n=0 headroom and gains robustness under failures.
Change: tools/static_table_search.py --failures 0,4 --start E026_{dc,sp}/table_best.json --episodes 16 --eps 0.5 (same M, eps as E026), then held-out
  eval via run_eval.sh TABLE_FILE (n=0/4/8, seed 20240601, rules table/random+flow/greedy), configs v2_deltacom 8x / v2_pilot4_lc 3x.
Success / reading (E026b baseline: Deltacom table-ECMP +7.0/+4.8/+1.3, SP-86 +5.6/+1.8/+2.1; seed noise 1.3, eval noise <=1 pp):
  - falsified (A037): new table - ECMP < 1 pp at n=4 AND n=8 -> no failure-robust static table; greedy stays the failure-time rule.
  - supported: n=4 gain over ECMP clearly above E026b (+4.8 dc / +1.8 sp) with n=0 loss <= 1 pp; n=8 reported as transfer (not optimised).
  - Note: failure index k renumbers after failures (K11 caveat), so a table optimised at n=4 is one specific failure sample set (CRN seeds 7000..).
Cost: CPU only, 5 workers; E026 took 4-7 h/sweep for SP-86 from scratch, a warm restart should need 1-2 sweeps. Run alongside E028.
Results (results/E031_{dc,sp,eval_dc,eval_sp}.json; held-out = 20 paired eps, seed 20240601):
Search: Deltacom J(n in {0,4}) 81.72 -> 81.72, 0 changes accepted (1 sweep, 589 evals) -> table = E026 table_best.
        SP-86 J 94.80 -> 96.01 (+1.2 train, 2 sweeps, 1176 evals).
| held-out n=0/4/8 | table E031 | table E026b | ECMP | greedy |
|---|---|---|---|---|
| Deltacom 8x | 87.35 / 77.62 / 68.00 | identical | 80.40 / 72.80 / 66.75 | 84.90 / 76.46 / 68.86 |
| SP-86 3x | 97.28 / 91.92 / 83.52 | 96.98 / 91.5 / 83.8 | 91.42 / 89.73 / 81.70 | 95.69 / 93.64 / 89.06 |
| SP-86 failure-trained DQN (K14) | 96.7 / 93.6 / 87.8 | | | |
Interpretation:
- E026's n=0 table is already a local optimum of the failure-aware objective (Deltacom), or improves only on the training failure samples (SP-86:
  +1.2 train -> +0.3/+0.4/-0.3 held-out, all < 1 pp eval noise = CRN overfit to the fixed failure sets).
- Single-change local search gives no static table that is more failure-robust. On SP-86 greedy stays +1.7/+5.5 above the best table at n=4/8,
  and the failure-trained DQN (K14) is also above it (+1.7/+4.3) -> under failures there is headroom that our static tables do not reach.
  Caveats: path index k renumbers after failures, so 'static table' is not the same object under failures; local search (single changes, warm start)
  is not a global optimum; K14 DQN is a different eval run (same protocol).
- Deltacom: table >= greedy at n=4 (+1.2), ~greedy at n=8 (-0.9): failure headroom over the table is small there.
Verdict: H REFUTED in its robustness part (no held-out gain at n=4 over E026b; n=0 kept). Formal falsification branch (table-ECMP < 1 pp at n=4 AND n=8)
  not triggered: the E026 table was already above ECMP under failures. A037 closed. Opens: is the failure-time gap (SP-86) state use? -> shuffle under failures.
