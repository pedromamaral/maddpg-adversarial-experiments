#!/usr/bin/env python3
"""Which traffic granularity separates the routing rules under per-flow pinning?

The base traffic (flow_count 21, flow_packet_size 0.03) has ~2 concurrent
elephant flows, so <1 % of the agents' decisions carry a packet in any step.
Splitting it into k times as many flows of 1/k the rate keeps the offered load
identical (flow-mode drops depend only on bandwidth) while giving more
decisions traffic.

For one k and the given loads, runs the fixed rules with every flow pinned to
the path its first packet got (ECMP-like) on the 2x-hotspot evaluation traffic
and reports end-to-end PDR plus the share of (agent, destination) decisions
that route a packet per step.

    python tools/traffic_granularity.py --k 8 --loads 2 3 --episodes 5
"""
import argparse
import json
import os
import random
import sys
import tempfile

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--src', default=os.path.join(os.path.dirname(__file__), '..', 'src'))
ap.add_argument('--config', default='configs/v2_pilot_config.json')
ap.add_argument('--k', type=int, default=1)
ap.add_argument('--loads', type=float, nargs='+', default=[2.0])
ap.add_argument('--episodes', type=int, default=5)
ap.add_argument('--rules', nargs='+',
                default=['sp', 'random', 'greedy', 'greedy_stale2', 'greedy_stale4', 'worst'])
args = ap.parse_args()
sys.path.insert(0, args.src)
sys.path.insert(0, os.path.join(args.src, 'maddpg_clean'))

from standalone_experiment_runner import StandaloneExperimentRunner  # noqa: E402

cfg = json.load(open(args.config))
tr = cfg['traffic']
tr['flow_count'] = int(tr['flow_count']) * args.k
tr['flow_packet_size'] = float(tr['flow_packet_size']) / args.k
tr['pin_paths_per_flow'] = True
tmp = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
json.dump(cfg, tmp)
tmp.close()

runner = StandaloneExperimentRunner(tmp.name, 0, tempfile.mkdtemp())
ae = runner.config.get('attack_eval', {})
env = runner._make_attack_env(ae.get('hotspot') or None)
eng = env.engine
hosts, tr_idx = eng.get_all_hosts(), eng.trainable_host_indices
T = runner.config['training']['timesteps_per_episode']
n_dec = len(tr_idx) * eng.n_destinations
assert eng.pin_paths_per_flow


class _NoPolicy:   # _attack_episodes only reads these from the agent object for rules
    n_agents, n_actions, agents = eng.n_total_hosts, eng.n_actions, [None]


for load in args.loads:
    # traffic: share of decisions with a packet, under the random rule
    random.seed(1); np.random.seed(1)
    rr, active, flows = random.Random(1), [], []
    for _ in range(2):
        eng.topology.restore_intact()
        eng.reset_with_load(offered_load_factor=load)
        for _ in range(T):
            acts = [runner._rule_action(hosts[i], 'random', eng, rr) for i in tr_idx]
            env.step(runner._build_full_actions(acts, eng.n_total_hosts, tr_idx, eng.n_actions))
            flows.append(len(eng._active_flows))
            active.append(len({(f['src'], f['dst']) for f in eng._active_flows}))
    row = {'k': args.k, 'load': load, 'flows_per_step': round(float(np.mean(flows)), 2),
           'active_decisions_pct': round(100 * float(np.mean(active)) / n_dec, 2)}
    for label in args.rules:
        rule, _, stale = runner._parse_rule_spec(label)
        random.seed(20240601); np.random.seed(20240601)
        res = runner._attack_episodes(_NoPolicy(), env, args.episodes, T, attack=False,
                                      offered_load_factor=load, routing_rule=rule,
                                      stale_steps=stale)
        row[label] = round(res['mean_end_to_end_pdr'], 2)
    print(json.dumps(row), flush=True)
os.unlink(tmp.name)
