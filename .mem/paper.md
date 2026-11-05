# Paper plan (evidence map refreshed 2026-11-05; outline still to do in research-think paper-plan)

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
| static table at n=0 (C3) | T7_ablation | E012-E017 | done |
| generalisation / shifts (C3) | T8, F3_generalisation | E012-E015 | done |
| failure-trained learners react to failures (C3) | — (to add) | E018,E032 (K14,K21) | TODO; one DQN only, E015 check pending (A042a) |
| best static table vs learners (C3,C4) | — (to add) | E026,E026b,E031 (K11,K20) | TODO |
| second topology Deltacom (C4) | — (to add) | E019-E022 (K10) | TODO, awaits D13 |
| why learners miss Deltacom headroom (C4) | — (to add) | E023-E025,E027,E026b (K12,K16-K18); E030 pending | weak until E030 |
| published-method-shaped learner (G1) | — (to add) | E028,E032,E033 (K19) | TODO |

## Pending (D13, await Pedro's agreement on reframing)
- Manuscript has no results subsection beyond the static table (last: "What is learned", line 494) and
  mentions Deltacom on one line only. All rows marked TODO need tex, tables and tools/paper1_tables.py entries.
- "Second topology" subsection (Deltacom table + calibration); method text for conversion/ttl/8x.
- Qualify "beats ECMP by 1.5-5 pp" as SP-86-specific; discussion of why state is unused.
- "Demand-robust static table" wording must carry the failure caveat (D20, D22).
- Pedro rewrites text in own voice; needs Routing.bib, author photos, acknowledgement placeholders.

Paper 2 (paper/paper2_robustness.tex, roadmap paper/PAPER2_ROADMAP.md): ON HOLD (D03); gate = A042.
