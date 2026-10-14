# Decisions  (D# | decision | why | rejected alternatives | date)

D01 | Utilisation-only attacker is the only threat model; all-feature attacker deleted | telemetry cannot reach timestep/queue/one-hot features | keep both as options (Pedro: delete) | 2026-09-29
D02 | Fix learning and retrain rather than publish v1 | v1 victims never learned (K01) | publish v1 with caveats | 2026-10-02
D03 | Paper 2 (attacks) and theses re-measurement on hold | victims ignore observations | re-attack v1 | 2026-10-02
D04 | Switch to realistic regime ("option 2"): pin_paths_per_flow, k=4 (flow_count 84, flow_packet_size 0.0075), 3x load — configs/v2_pilot4_lc.json | unpinned random was an unrealistically strong baseline (K13) | keep 2x unpinned | 2026-10-06
D05 | Decision rule for retrain: LC must beat random/ECMP (>= ~93%) AND least-loaded above chance | avoid retraining frozen tables | — | 2026-10-05
D06 | Shift configs: shiftA hotspot sites 4-7 -> CS5,CS6; shiftB sites 1-4 -> CS3,CS4 | test demand robustness | — | 2026-10-07
D07 | Drop the 7-variant comparison from the new paper; one GNN version (stable) with LC and CC; no duelling-GNN | architecture secondary once credit fixed (K06) | full 7-variant grid | 2026-10-08
D08 | Paper 1 rewritten around credit assignment: paper/paper1_v2/paper1_v2.tex; every number from tools/paper1_tables.py | old claims withdrawn | patch old paper | 2026-10-11
D09 | Second topology = Deltacom (flat, 6% bridges, 113 nodes) via tools/zoo_to_sp.py: farthest-point 14 PEs, betweenness capacities 300-1000, traffic.ttl=50, load 8x | GtsCe/Cogentco less suitable; TTL 15 dropped 27% of packets; at 3x all rules ~96% | GtsCe, Cogentco, 3x | 2026-10-12
D10 | Rewards are training-only (CTDE): need not be measurable online (simulator / digital twin) | enables difference-reward proxies | — | 2026-10-13
D11 | Run supervised-signal check (E024) before any difference-reward experiment (A025) | ~10 h of GPU only if the signal is learnable | go straight to difference reward | 2026-10-13
D12 | Never delete exited training containers or other projects' images on the server | Pedro said no | clean-up | 2026-10-07
D13 | Pending paper edits (Deltacom subsection, qualify "beats ECMP by 1.5-5 pp" as SP-86-specific, conversion/ttl/8x method text, own-flow discussion) await Pedro's agreement on reframing | reframing changes C4 | — | 2026-10-13
