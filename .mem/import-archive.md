# Import from the Claude Code session of 2026-09-29 .. 2026-10-14 (not recorded in code)

## Conclusions (PDR % unless stated; 3x = realistic regime: pinned flows, k=4, 3x hotspot on SP-86)
- v1 MADDPG (7 variants, step reward, sigmoid actor, joint critic) never learned to route: static share 82-99 %, least-loaded 21-42 % vs chance 42-46 %, saturated 77-99 %, telemetry shuffle changes PDR < 0.5 pp (6/7); 2x PDR 73.1-87.3 vs random 90.0, greedy 97.2, SP 73.8 (policy_attribution.json, ceil2x).
- Old Paper 1 claims (critic dominance, failure robustness, GNN cost) withdrawn: they compared frozen tables.
- Learnability probe (tools/learnability_probe.py): joint critic resolves best path 63 % vs 55 % chance; softmax+factored 82 %, +entropy 91-92 %; actor supervised 98 %.
- Sparsity: original traffic ~1.8 flows/step, 0.6 % of 294 decisions live per step, 19 % of steps empty.
- Reward is aligned (tools/reward_alignment.py: greedy 22.75 > random 20.01 > sp 16.40 > worst 10.93 return/ep) but greedy beats random by only ~12 % of return.
- Step-reward pilots 1-4 (v2_pilot_*, v2_p2_*, v2_p3_*, v2_p4_lc) never beat random/ECMP clearly nor chose least-loaded above chance; best LC 91.3 at 2x (per-agent credit w=0.5).
- Baseline realism (v2_pilot_lc/eval_realism*): pinning per flow costs random 8.2 pp (90.0->81.9), greedy 8.0 (97.2->89.2); greedy with 2/4-step-old telemetry 92.2/86.4.
- Granularity (granularity/k*.log): finer flows make ECMP ~optimal (2x, k>=8: ECMP 97.3 vs greedy 98.3); chosen point k=4 3x: SP 74.9, ECMP 90.8, greedy 96.3 (5-ep sweep).
- Rules at 3x, 20 paired eps (v2_p4_lc/eval_3x): greedy 95.7/93.6/89.1 (n=0/4/8), greedy_stale2 93.7, stale4 92.1, ECMP 91.4/89.7/81.7, SP 75.8.
- Applied-decision least-loaded (v2_p4_lc/flow_choice_3x.json): greedy 100 %, stale2 83 %, stale4 71 %, ECMP 38 %, chance ~41 %; pilot 4 policy 41.7 % (at chance).
- Algorithm x credit (SP-86, 3x, fixed hotspot): MADDPG-LC step 92.7; DQN step (idqn_p4_step) 91.1 (seeds 88.0-93.1); DQN per-decision (idqn_p4) 96.9; MADDPG per-decision LC 97.3, CC 97.4 (ac_*_p4, seeds within 0.2) -> credit, not algorithm, decides learning.
- Architecture with per-decision credit (changing hotspot + failures, *_rhf): LC/CC x simple/duelling all 96.2-96.9 (n=0), 92.7-93.0 (n=4), 86.7-87.0 (n=8); shift A 95.5-96.1, B 95.4-96.4 vs greedy 96.8/95.7, ECMP 93.5/92.8 -> architecture secondary.
- GNN (original encoder) collapsed: encodings 7-25x raw spread, 57-75 % zeros, frozen decisions in 4/6 runs (LC-GNN 90.9); stable GNN (LayerNorm, averaged step, lr/10) 95.5 (LC) / 96.3 (CC), neighbour shuffle costs 0 pp.
- What is learned = demand-robust static table: whole-observation shuffle costs < 1 pp for every learner; fixed-hotspot tables fall below ECMP on shift A (DQN 90.2, MADDPG-LC 92.8, CC 89.5 vs ECMP 93.5); changing-hotspot training generalises (DQN rh 95.9/95.7 on A/B).
- Failure training helps DQN: idqn_p4_fail 96.7/93.6/87.8 vs idqn_p4 96.9/92.1/84.3.
- Deltacom (second topology, 8x): every learner ~ECMP: step MADDPG-LC 80.9, per-decision fix LC 80.2 / CC 79.8, rhf LC 80.3 / CC 79.8 vs ECMP 80.4, greedy 84.9, greedy_stale2 84.1, SP 74.4; least-loaded at/below chance; shuffle <= 0.5 pp; shifts below ECMP (A 86-87.5 vs 88.1).
- Interpretation: learners capture static (demand-structure) headroom only; SP-86 has it, Deltacom does not (no fixed rule beats ECMP there; greedy's 4.5 pp needs current state).
- Own-flow diagnostic (own_flow/*.json, ECMP natural experiment): SP-86 own-flow effect +9.6 pp (SE 0.5), own gain 5.9 >= system gain 4.3 (externality NOT the explanation); Deltacom +3.3 pp (SE 0.7), own gain 1.9 = ~42 % of system gain 4.5 (externality ~58 %), own loss non-monotone in utilisation gap.

## Ideas tried / rejected
- Reward misspecification hypothesis: rejected (reward_alignment ranks rules like PDR).
- Entropy bonus 0.01/0.05: no gain in real network.
- Observation-based central critic (pilot 2 CC): no gain.
- Per-agent credit w=0.5/1.0 (pilot 3): +2 pp, still ~random.
- Simple vs duelling: old duelling was algebraically the simple head (scalar Q with action as input); replaced by per-destination V_d + A_dk - mean A; no effect either way.
- 7-variant comparison dropped from new paper (Pedro, 2026-10-08): architecture secondary once credit fixed.
- One GNN version suffices; GNN tested with LC and CC critics (critic could interact with neighbour info); duelling-GNN not run.
- Difference reward: proposed (own outcome minus drops caused on shared links); only worth it if supervised check shows the Deltacom own-flow signal is learnable; exact counterfactual not measurable, proxy from per-LSP + interface discard counters is.
- Rewards are training-only (CTDE): need not be measurable online if trained in simulator/digital twin.
- Old leaderboard/robustness and Paper 2 (attacks on v1 victims) on hold: victims did not use observations.

## Decisions
- Fix learning and retrain rather than publish v1 (2026-10-02); decision rule then switched to realistic regime (option 2) after realism results.
- Realistic regime: pin_paths_per_flow, k=4 (flow_count 84, flow_packet_size 0.0075), 3x load (configs/v2_pilot4_lc.json).
- Shift configs: shiftA sites 4-7 -> CS5,CS6; shiftB sites 1-4 -> CS3,CS4.
- Paper 1 rewritten around credit assignment (paper/paper1_v2/paper1_v2.tex; tables/figures from tools/paper1_tables.py).
- Second topology = Deltacom (flat, 6 % bridges, 113 nodes) over GtsCe/Cogentco; converter tools/zoo_to_sp.py (farthest-point 14 PEs, betweenness capacities 300-1000); traffic.ttl=50 (TTL 15 dropped 27 % of packets); load 8x chosen (3x: all rules ~96 %).
- Do not delete exited training containers or other projects' images (Pedro said no, 2026-10-07).

## Running / state (checked 2026-10-14 12:58 UTC)
- Server pedroamaral@10.26.110.14 (NOVA VPN only): idle - no containers, no waiter scripts, load 0.00, GPU 0 % / 74 MiB; rebooted ~1 day earlier (uptime 1 day). The other user's gnn-adversarial-nids job is no longer running.
- Server repo clean at 5b068de (own_flow_effect.py); all campaigns finished and their results are on disk (own_flow/, dc_ac_*, v2_dc_lc, ac_*, idqn_*).
- tools/supervised_signal.py written locally, uncommitted, not yet pushed or run (step (a) below).
- Results copied locally to host_data/results (gitignored); server layout same path.
- Docker image maddpg-exp:latest; learners via src/idqn_control.py; waiters scripts in ~/idqn/run_*.sh, completion logs ~/idqn/*.done.

## Baselines / noise
- Learned results: 3 seeds; seed spread typically <= 1.3 pp on training hotspot, up to 2.3 pp on shifts, up to 3-5 pp for unstable GNN / Deltacom rhf (75.4-83.3).
- All evaluations paired (traffic seed 20240601 inside _attack_episodes); 20 eps per point; granularity sweeps 5 eps (differ ~0.5-1 pp from 20-ep evals).
- tools/run_eval.sh evaluates at 2x unless LOAD is set.

## Open problems / next steps
- (a) Run tools/supervised_signal.py on configs/v2_pilot4_lc.json and configs/v2_deltacom.json: IPS own-flow delivery of greedy / table / util / full regressors on held-out ECMP decisions -> is the state signal learnable?
- (b) If learnable on Deltacom: difference-reward experiment (MADDPG LC/CC, 3 seeds, ~10 h).
- Pending paper edits (await Pedro's agreement on reframing): add "Second topology" subsection (Deltacom table + calibration), qualify "beats ECMP by 1.5-5 pp" as SP-86-specific, method text for conversion/ttl/8x, discussion of why state is unused (own-flow diagnostic).
- Pedro to rewrite paper text in own voice; needs Routing.bib, author photos, acknowledgement placeholders.
