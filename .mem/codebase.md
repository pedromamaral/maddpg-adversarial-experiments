# Codebase map (2026-10-14)

## Entry points
- src/idqn_control.py (517) — CURRENT learners: independent DQN (default) or --learner ac (MADDPG with
  per-decision flow credit; --critic local|central, --duelling, --gnn [--gnn-stable]); --credit flow|step;
  training on hotspot / --train-random-hotspot / --train-failures; eval at failures 0..8, --eval-stale,
  --eval-shuffle path|all|neighbours. Output: <out>/<prefix>_seed<s>[_<tag>].json
  {config,seed,credit,learner,critic,history[...],eval:{fail<n>[_stale<d>]:{pdr,least_loaded,chance,regret,contested}}}
  Classes: QNet :65, ActorCritic :77, Policy :222; score_decisions :267; main :284.
- src/standalone_experiment_runner.py (2985) — v1/step-reward pipeline: train / evaluate / attack
  (--phase, --variants LC-Simple|CC-Duelling-GNN...). Rule labels parsed in _parse_rule_spec :2237
  (policy, random, greedy, greedy_stale<d>, sp, worst, '+flow' = pinned per flow, policy_stale<d>).
  Eval output n_<k>/damage_ceiling.json {<rule>:{mean_end_to_end_pdr,std_...,pdr_series,...}, _meta}.
  Training: phase1_training_results.json {<variant>:{rewards,pkt_losses,best_checkpoint_*,...}}.
- src/maddpg_clean/network_environment.py (1723) — NetworkEngine, topology, traffic (pin_paths_per_flow,
  ttl, hotspot), path_util_slots (obs slots 33-93).
- src/maddpg_clean/maddpg_implementation.py (1128) — v1 MADDPG Agent/Critic, learn_action_projection
  options (actor_head block_softmax, critic_head factored, actor_entropy, central_state).
- src/attack_framework/ — improved_fgsm_attack.py (FGSM/PGD/logit, util-only), learned_adversary.py. (P2, on hold)

## Configs (configs/*.json)
- v2_pilot4_lc.json = SP-86 realistic regime (pinned, k=4, 3x); _shiftA/_shiftB variants.
- v2_deltacom.json (+_shiftA/_shiftB), topology configs/topologies/deltacom_sp.txt, traffic.ttl 50, 8x.
- v2_pilot_*, v2_pilot2_*, v2_pilot3_* = historical pilots. probe_* / *_attack_* = v1/P2.

## Tools (tools/)
- paper1_tables.py — regenerates paper/paper1_v2/{tables/T*.tex, figures/F*.pdf, numbers.json} from host_data/results.
- run_eval.sh, compare_eval.py, flow_choice_check.py, policy_attribution.py — evaluation/diagnostics.
- learnability_probe.py, reward_alignment.py, credit_check.py, traffic_granularity.py, own_flow_effect.py,
  supervised_signal.py (uncommitted) — diagnostics behind K02-K13.
- zoo_to_sp.py (Topology Zoo -> provider-edge model), topo_invariants.py, plot_*.py.
- summarize_run.py -> .mem/results/E###.json; runs_status.sh (memory tooling).

## Add an experiment
1. config in configs/ (or flags on idqn_control.py); 2. commit+push+pull on server; 3. campaign script
in ~/idqn/run_<name>.sh using the docker pattern in infra.md; 4. row in .mem/runs.md + LEDGER line;
5. rsync JSON back; tools/summarize_run.py <dir> --id E###.
