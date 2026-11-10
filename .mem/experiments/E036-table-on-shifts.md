# E036 — Searched static table on unseen demand (A044, D24)
Status: DONE 2026-11-09 02:03 @438c148, analysed 2026-11-10. Avenue A044. Follows E026b (K11).

Hypothesis H1: the E026 tables were optimised on one hotspot and lose their gain on other demand, as fixed-hotspot learners do (K09).
 H0: the gain over ECMP survives on shifts A and B, so the Deltacom headroom is not one-matrix traffic engineering.
Change: none in src expected. Eval-only: run_eval.sh with TABLE_FILE = E026 table_best (confirm path; same files as E026b).
Config: configs/v2_pilot4_lc_shift{A,B}.json (3x), configs/v2_deltacom_shift{A,B}.json (8x), n=0/4/8, 20 paired eps, seed 20240601.
 Check at implement: ECMP, greedy and learner shift cells must share this traffic seed and n set (existing eval_shiftA/B dirs); rerun rules if not.
Reading (noise <=2.3 pp on shifts, REF), n=0 headline, per topology and shift:
 - table >= ECMP + 2.3 on both shifts -> headroom holds across demand; H0.
 - table within 2.3 of ECMP or below it on a shift -> H1; C4 says "on the training demand".
 - Also report table vs changing-hotspot learners on each shift (is the learned table more demand-robust than the searched one?).
Limit: one searched table per topology, so no seed spread for the table itself.
Cost: CPU only, 2 topologies x 2 shifts x 3 n (ETA at implement).

## Implementation (881ff02)
- No src change. tools/run_E036.sh: run_eval.sh with TABLE_FILE=E026_{sp,dc}/table_best.json, RULES="table random+flow greedy policy", configs/v2_{pilot4_lc,deltacom}_shift{A,B}.json, LOAD 3/8, n=0/4/8, 20 eps, default seed 20240601.
- Out: host_data/results/{v2_p4_lc,v2_dc_lc}/E036_{sp,dc}_shift{A,B}/n_*/damage_ceiling.json; done marker ~/idqn/run_E036.done (5 lines).
- Rules rerun in the same call (old eval_shiftA/B have no n=8 and no ECMP row), so table, ECMP, greedy and learner are paired. policy = LC-Simple of v2_p4_lc / v2_dc_lc, same as eval_shiftA/B.
- 2 parallel chains (dc A->B, sp A->B), 3 ceilings each at --cpus 1.5 -> up to 9 cpus. Smoke: bash -n only; inputs verified locally (table files); models are server-only, check at deploy.

## Results (2026-11-10; 20 paired eps, seed 20240601; one table per topology)
| topology | shift | n | table | ECMP | greedy | table - ECMP |
|---|---|---|---|---|---|---|
| Deltacom 8x | A | 0/4/8 | 82.51/79.01/77.85 | 88.13/84.48/82.81 | 90.41/86.68/84.68 | -5.62/-5.47/-4.96 |
| Deltacom 8x | B | 0/4/8 | 82.06/74.66/72.62 | 89.66/78.85/76.69 | 90.27/80.44/78.21 | -7.60/-4.19/-4.07 |
| SP-86 3x | A | 0/4/8 | 83.19/81.62/78.89 | 93.55/91.26/87.48 | 96.75/94.97/91.77 | -10.36/-9.64/-8.59 |
| SP-86 3x | B | 0/4/8 | 84.99/83.35/78.47 | 92.84/90.80/84.73 | 95.74/93.03/88.00 | -7.85/-7.45/-6.26 |
On the training demand (E026b): Deltacom table +6.95 over ECMP, SP-86 +5.6.
Pairing check: ECMP shift A = 93.55 (SP-86) and 88.13 (Deltacom), equal to K09/K10 references.
Policy row = step-credit LC-Simple of v2_p4_lc / v2_dc_lc (E010 / E020), not a flow-credit learner: SP-86 89.64/91.90 (A/B, n=0), Deltacom 88.00/87.54. Not used for the claim.

## Interpretation
- 12/12 cells: the searched table is 4-10 pp below ECMP on unseen demand, far beyond the 2.3 pp shift noise. The gain is specific to the demand it was searched on.
- It is more demand-specific than learned fixed-hotspot tables: SP-86 shift A 83.2 vs learned 89.5-92.8 (K09), ECMP 93.5.
- Changing-hotspot learners stay near or above ECMP on shifts (SP-86 DQN rh 95.9/95.7; Deltacom rhf 86-87.5 vs 88.1, K09/K10), i.e. 12 pp (SP-86) and ~5 pp (Deltacom) above the searched table there.
- So the Deltacom "7 pp headroom" is headroom for a known, fixed demand matrix. Whether a demand-robust static table beats ECMP on Deltacom is unknown (no search over several hotspots was run).
- K23 is unaffected: E030 trained and evaluated on the table's own demand.
Limits: one table per topology, 2 shifts, single eval seed.
## Verdict
H1 supported (high for "does not transfer"): headroom is one-matrix traffic engineering. C4 and F4_headroom say "on the training demand"; "demand-robust" is not used for the searched table.
