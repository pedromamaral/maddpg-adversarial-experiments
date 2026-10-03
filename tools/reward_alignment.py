#!/usr/bin/env python3
"""Does the training reward rank routing policies the way the evaluation does?

The policies are trained on _shared_flow_reward (per step: delivered / forwarding
events, minus drops / forwarding events, minus mean and variance of link
utilisation), but judged on end-to-end delivery per injected packet. If the
reward prefers a policy that delivers less, no learner can be expected to find
the better one.

Runs the fixed routing rules (greedy, random, sp, worst) on identical traffic at
the evaluation operating point (2x hotspot, no failures) and reports, per rule:
the training return per episode with its components, and the end-to-end PDR.

    python tools/reward_alignment.py --episodes 5
"""
import argparse
import os
import random
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--src', default=os.path.join(os.path.dirname(__file__), '..', 'src'))
ap.add_argument('--config', default='reward_fix_full_config.json')
ap.add_argument('--results', default='data/results/reward_fix')
ap.add_argument('--episodes', type=int, default=5)
args = ap.parse_args()
sys.path.insert(0, args.src)
sys.path.insert(0, os.path.join(args.src, 'maddpg_clean'))

from standalone_experiment_runner import StandaloneExperimentRunner, _shared_flow_reward  # noqa: E402

runner = StandaloneExperimentRunner(args.config, 0, args.results)
reward_cfg = runner.config.get('reward', {})
ae = runner.config.get('attack_eval', {})
env = runner._make_attack_env(ae.get('hotspot') or None)
eng = env.engine
hosts, tr = eng.get_all_hosts(), eng.trainable_host_indices
T = runner.config['training']['timesteps_per_episode']
COMP = ['delivery_rate', 'packet_loss_rate', 'network_utilization', 'util_variance', 'backlog_packets']

print(f"reward config: { {k: reward_cfg.get(k) for k in ('delivery_weight', 'drop_penalty', 'mean_util_weight', 'var_util_penalty', 'backlog_penalty')} }")
print(f"{'rule':<7} {'return/ep':>9} {'PDR %':>6} | mean per step: " + " ".join(f"{c[:12]:>12}" for c in COMP) + "  hops/delivered")
rows = {}
for rule in ('greedy', 'random', 'sp', 'worst'):
    random.seed(20240601); np.random.seed(20240601)
    rule_rng = random.Random(20240601 + 999)
    rets, pdrs, comps, hops = [], [], {c: [] for c in COMP}, []
    for _ in range(args.episodes):
        eng.topology.restore_intact()
        eng.reset_with_load(offered_load_factor=float(ae.get('offered_load_factor', 2.0)))
        ret = 0.0
        for _ in range(T):
            acts = [runner._rule_action(hosts[i], rule, eng, rule_rng) for i in tr]
            _, _, info = env.step(runner._build_full_actions(acts, eng.n_total_hosts, tr, eng.n_actions))
            ret += _shared_flow_reward(info, reward_cfg)
            for c in COMP:
                comps[c].append(float(info.get(c, 0.0)))
        st = env.get_stats()
        rets.append(ret); pdrs.append(float(st.get('end_to_end_pdr', 0.0)))
        hops.append(float(st.get('hops_mean', 0.0)))
    rows[rule] = (np.mean(rets), np.mean(pdrs))
    print(f"{rule:<7} {np.mean(rets):9.2f} {np.mean(pdrs):6.1f} | " +
          " ".join(f"{np.mean(comps[c]):12.4f}" for c in COMP) + f"  {np.mean(hops):5.2f}")

by_ret = sorted(rows, key=lambda r: -rows[r][0])
by_pdr = sorted(rows, key=lambda r: -rows[r][1])
print(f"\nranked by training return: {' > '.join(by_ret)}")
print(f"ranked by end-to-end PDR:  {' > '.join(by_pdr)}")
print("ALIGNED" if by_ret == by_pdr else "MISALIGNED: the training reward does not rank policies like the evaluation")
