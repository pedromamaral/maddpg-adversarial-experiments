# E040 — Credit effect on a second public topology (A043 retargeted, D29)
Status: stage 1 IMPLEMENTED 2026-11-12 @90018a9 (tools/run_E040.sh, tools/e040_screen.py), not deployed. Avenue A043. Staged: stage 2 only on a go from stage 1.

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
