# Paper plan (2026-10-14)

Paper 1 v2: "Credit Assignment, not Architecture: What MARL Routing Agents Learn in a Service-Provider
Network" — paper/paper1_v2/paper1_v2.tex (IEEEtran journal, TNSM). Bib: Routing_v2.bib.
All numbers: tools/paper1_tables.py -> tables/ figures/ numbers.json (never hand-type numbers).

## Claims -> evidence -> status
| claim | table/figure | experiments | status |
| v1 step-reward MADDPG does not route (C1) | T1_v1 | E001,E002 | done |
| realism of baselines | T2_realism | E008 | done |
| granularity / operating point | T3, F1_granularity | E009 | done |
| pilots fail (C1) | T4_pilots | E004,E005,E007,E010 | done |
| credit decides learning (C1) | T5_credit, F2_credit | E011-E013 | done |
| architecture secondary (C2) | T6_architecture | E015-E017 | done |
| static table (C3) | T7_ablation | E012-E017 | done |
| generalisation / shifts (C3) | T8, F3_generalisation | E012-E015 | done |
| second topology Deltacom (C4) | — (to add) | E019-E022 | TODO, awaits D13 |
| why state unused: own-flow + supervised (C4,C5) | — (to add) | E023,E024 | TODO |

## Pending (D13, await Pedro's agreement on reframing)
- "Second topology" subsection (Deltacom table + calibration); method text for conversion/ttl/8x.
- Qualify "beats ECMP by 1.5-5 pp" as SP-86-specific; discussion of why state is unused.
- Pedro rewrites text in own voice; needs Routing.bib, author photos, acknowledgement placeholders.

Paper 2 (paper/paper2_robustness.tex, roadmap paper/PAPER2_ROADMAP.md): ON HOLD (D03).
