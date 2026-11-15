# E040 — Credit effect on a second public topology (A043 retargeted, D29)
Status: stage 1 DONE 2026-11-12 18:53 (@dd10c82), analysed 2026-11-15: no go at the gridded loads; stage 1b (finer loads) proposed, stage 2 not started. Avenue A043. Staged: stage 2 only on a go from stage 1.

Hypothesis H1: per-decision flow credit beats step credit and ECMP wherever own-flow gain is aligned with system gain (as SP-86, K12, K18), so the SP-86 credit effect (K05) reappears on a public topology selected for alignment.
 H0: on an aligned topology with headroom flow credit still ~ step ~ ECMP -> C1's credit effect is an SP-86 fact; title and abstract scoped to it.
Stage 1 (screen, rules only, no training): 3-4 Topology Zoo graphs via tools/zoo_to_sp.py (choice at implement: public, PE/agent count and K-path diversity comparable to SP-86 / Deltacom). Per graph: calibrate load so ECMP PDR is in the 80-92 band (as 3x SP-86, 8x Deltacom); then (a) ECMP, greedy, greedy_stale2, SP at n=0 (tools/run_eval.sh, 20 paired eps); (b) own-flow effect and own vs system gain (tools/own_flow_effect.py, K12 method). Deltacom and SP-86 rows = the calibration of the predictor (own/system 0.42 and >=1).
 Go: at least one graph with greedy - ECMP >= 3 pp AND own gain >= system gain. Prediction written in this card BEFORE stage 2: aligned -> learns; a misaligned graph (if one has headroom) -> ~ECMP.
 No-go: no graph passes -> stop; the screen table goes to the paper as scope evidence (how common the aligned regime is); C1 stays scoped to SP-86.
Stage 2 (train, on the go graph): step MADDPG-LC vs flow MADDPG-LC (and flow DQN if cores allow), fixed hotspot, n=0 training, 750 eps, 3 seeds, eval n=0/4/8 20 paired eps + whole-obs shuffle at n=0. Same pipelines as E037 (step) and E021 (flow).
Reading (seed noise <=1.3 pp, unknown on the new graph: take the 3-seed spread): flow - step and flow - ECMP both > max(1.3 pp, 2x seed spread) -> H1, C1 holds on two topologies and the alignment predictor is confirmed once. Both within noise -> H0. One of the two -> report as is, no widening of C1.
Risks: own_flow_effect is medium confidence (K12; own_flow/ results are server-only, rsync them); alignment measured under ECMP background, not under the learned policy; one confirmed prediction is not a law; load calibration is a free parameter (fix the ECMP band before looking at greedy).
Cost: stage 1 ~2-3 days CPU (estimate, mostly conversion + calibration); stage 2 6-9 runs x ~6 h, 2 cpus each. Deadline 2027-01-14: stage 1 decision by ~11-20, else drop.

## Stage 1 implementation (2026-11-12, 90018a9)
- Graphs (chosen at implement, size/degree near Deltacom 113/2.85): Interoute 110/146, GtsCe 149/193, TataNld 145/186, Geant2012 40/61 (mesh, deg 3.05). Source: GitHub mirror sk2/topologyzoo (topology-zoo.org unreachable); graphml in host_data/topo_check/zoo_E040 (gitignored). Topologies configs/topologies/<g>_sp.txt via tools/zoo_to_sp.py; configs configs/v2_zoo_<g>.json = v2_deltacom.json (ttl 50, same hotspot) with the topology swapped.
- tools/own_flow_effect.py --load (override; default unchanged). One run per (graph, load) gives ECMP, greedy, own gain, system gain, own share.
- tools/run_E040.sh: loads 3 5 8 12 16 24 per graph (one chain each, 1.5 cpus) + calibration chain sp86_L3, deltacom_L8 (same code, replaces the server-only own_flow/). Out host_data/results/E040/<g>_L<load>.json; markers ~/idqn/run_E040.done (5 lines + 'all done').
- tools/e040_screen.py: load = ECMP in [80,92] closest to 86 (ECMP only); GO = greedy-ECMP >= 3 AND own_share >= 1. Not checked: greedy_stale2 / SP rows (add with run_eval.sh at the chosen load if needed).
- Smoke (local, Geant2012 5x, 1 ep): runs; ECMP 98.25 (out of band, hence loads up to 24x).

## Stage 1 results (2026-11-15; tools/e040_screen.py; 20 paired eps; own = own-flow gain over contested flows, share = own / system gain)
| graph | load (rule) | ECMP | greedy | greedy-ECMP | own | share | verdict |
|---|---|---|---|---|---|---|---|
| SP-86 (calib) | 3 | 91.42 | 95.69 | 4.28 | 5.91 | 1.38 | reference: learns |
| Deltacom (calib) | 8 | 80.40 | 84.89 | 4.49 | 1.87 | 0.42 | reference: ~ECMP |
| Interoute | 5 | 88.66 | 91.99 | 3.33 | 1.99 | 0.60 | no-go (misaligned) |
| Geant2012 | 8 | 91.92 | 93.70 | 1.78 | 4.83 | 2.71 | no-go (headroom < 3) |
| TataNld | 5 | 91.39 | 92.76 | 1.36 | 3.86 | 2.84 | no-go (headroom < 3) |
| GtsCe | 5 | 88.88 | 89.28 | 0.40 | 1.10 | 2.75 | no-go (no headroom at any load: <= 1.1 pp) |
Neighbouring loads (ECMP / gain / share): Geant2012 12x 76.5 / 3.90 / 2.97; TataNld 8x 74.1 / 3.94 / 3.69; Interoute 8x 71.8 / 6.40 / 0.96, 12x 56.2 / 6.98 / 1.59.
## Interpretation
- Calibration reproduces K12 with the same code (SP-86 own 5.91 vs system 4.28; Deltacom 1.87 vs 4.49): pipeline valid.
- By the pre-registered rule no graph is a go. But the grid was too coarse for the rule: it asks for the load with ECMP closest to 86, and Geant2012 and TataNld have only one in-band load, at the band's upper edge (91.4-91.9), with the next grid load already below 80. Their headroom grows fast in between (1.8 -> 3.9, 1.4 -> 3.9). The rule's operating point was not measured for these two.
- The share is strongly load-dependent (Interoute 0.53 -> 3.95 from 3x to 24x), so 'alignment' is a property of (topology, load), not of the topology; and share > 1 by up to 7.8x shows own and system gain are not on one scale (own is per contested flow). The predictor rests on two calibration points: medium-low confidence.
- Interoute is a second Deltacom-like point in band (headroom 3.3, share 0.60). GtsCe has no headroom at any load.
## Verdict (stage 1)
Inconclusive for the go/no-go: no go at the gridded loads; the rule's target load (ECMP ~86) is unmeasured for Geant2012 and TataNld. H1/H0 untested (no training).
Proposed stage 1b (same rule, no change of criteria): Geant2012 at 9, 10, 11x and TataNld at 6, 7x (5 runs, ~1 h each); pick ECMP closest to 86; GO if greedy-ECMP >= 3 and share >= 1. Then stop or stage 2 as in the card. Decision by ~11-20 unchanged.
