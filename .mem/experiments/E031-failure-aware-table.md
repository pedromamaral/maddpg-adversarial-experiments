# E031 — Failure-aware static table search (A037)
Status: IMPLEMENTED 2026-10-23 (launch script tools/run_E031.sh; no src/tool change). Avenue A037; decisions D19, D20; follows E026/E026b (K11, K18).

Hypothesis: a static table searched with J over n in {0,4} link failures (restart from E026 table_best) keeps its n=0 headroom and gains robustness under failures.
Change: tools/static_table_search.py --failures 0,4 --start E026_{dc,sp}/table_best.json --episodes 16 --eps 0.5 (same M, eps as E026), then held-out
  eval via run_eval.sh TABLE_FILE (n=0/4/8, seed 20240601, rules table/random+flow/greedy), configs v2_deltacom 8x / v2_pilot4_lc 3x.
Success / reading (E026b baseline: Deltacom table-ECMP +7.0/+4.8/+1.3, SP-86 +5.6/+1.8/+2.1; seed noise 1.3, eval noise <=1 pp):
  - falsified (A037): new table - ECMP < 1 pp at n=4 AND n=8 -> no failure-robust static table; greedy stays the failure-time rule.
  - supported: n=4 gain over ECMP clearly above E026b (+4.8 dc / +1.8 sp) with n=0 loss <= 1 pp; n=8 reported as transfer (not optimised).
  - Note: failure index k renumbers after failures (K11 caveat), so a table optimised at n=4 is one specific failure sample set (CRN seeds 7000..).
Cost: CPU only, 5 workers; E026 took 4-7 h/sweep for SP-86 from scratch, a warm restart should need 1-2 sweeps. Run alongside E028.
Results:
Interpretation:
Verdict:
