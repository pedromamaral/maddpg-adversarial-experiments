#!/usr/bin/env python3
"""Does per-agent credit (reward.credit = 'mixed') reach the agent that earned it?

Half the agents route greedily (least-utilised path), the other half take the
most congested path, on the evaluation traffic (2x hotspot, no failures). Under
the shared team reward both halves receive the same number every step, so a
learner cannot tell which routing was better. With per-agent credit, each agent's
reward includes the outcome of the packets whose path it chose; it should be
clearly higher for the greedy half.

Also checks that the attribution is consistent with the global counts
(attributed forwarding events, deliveries and drops never exceed the step totals).

    python tools/credit_check.py --episodes 3
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
ap.add_argument('--episodes', type=int, default=3)
args = ap.parse_args()
sys.path.insert(0, args.src)
sys.path.insert(0, os.path.join(args.src, 'maddpg_clean'))

from standalone_experiment_runner import (StandaloneExperimentRunner,  # noqa: E402
                                          _shared_flow_reward, _credited_rewards)

runner = StandaloneExperimentRunner(args.config, 0, args.results)
reward_cfg = dict(runner.config.get('reward', {}))
ae = runner.config.get('attack_eval', {})
env = runner._make_attack_env(ae.get('hotspot') or None)
eng = env.engine
hosts, tr = eng.get_all_hosts(), eng.trainable_host_indices
agent_hosts = [hosts[i] for i in tr]
rule = ['greedy' if j % 2 == 0 else 'worst' for j in range(len(tr))]
T = runner.config['training']['timesteps_per_episode']

random.seed(20240601); np.random.seed(20240601)
rule_rng = random.Random(1)
per = {w: [] for w in (0.0, 0.5, 1.0)}       # per step: list of per-agent rewards
consistent, attributed = True, []
for _ in range(args.episodes):
    eng.topology.restore_intact()
    eng.reset_with_load(offered_load_factor=float(ae.get('offered_load_factor', 2.0)))
    for _ in range(T):
        acts = [runner._rule_action(hosts[i], rule[j], eng, rule_rng) for j, i in enumerate(tr)]
        _, _, info = env.step(runner._build_full_actions(acts, eng.n_total_hosts, tr, eng.n_actions))
        st = info['decider_stats']
        fwd, dlv, drp = (sum(v[k] for v in st.values()) for k in range(3))
        consistent &= fwd <= info['packets_sent'] and dlv <= info['packets_delivered'] \
            and drp <= info['packets_dropped'] and min(min(v) for v in st.values()) >= 0 if st else True
        if info['packets_delivered']:
            attributed.append(dlv / info['packets_delivered'])
        r_shared = _shared_flow_reward(info, reward_cfg)
        for w in per:
            cfg = dict(reward_cfg, credit='mixed' if w > 0 else 'shared', local_weight=w)
            per[w].append(_credited_rewards(info, cfg, agent_hosts, r_shared))

g = [j for j, r in enumerate(rule) if r == 'greedy']
b = [j for j, r in enumerate(rule) if r == 'worst']
print(f"attribution consistent with step totals: {consistent}; "
      f"share of deliveries attributed to an agent: {100 * np.mean(attributed):.1f}%")
print(f"{'credit':<16} {'greedy agents':>14} {'worst agents':>13} {'gap':>8}  greedy half higher on % of steps")
for w, rows in per.items():
    R = np.array(rows)                          # [steps, agents]
    rg, rb = R[:, g].mean(1), R[:, b].mean(1)
    tag = 'shared' if w == 0 else f'mixed w={w:g}'
    print(f"{tag:<16} {rg.mean():14.4f} {rb.mean():13.4f} {rg.mean() - rb.mean():+8.4f}  {100 * (rg > rb).mean():5.1f}%")
