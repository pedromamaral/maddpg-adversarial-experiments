#!/usr/bin/env python3
"""How much of greedy's gain does a decision's own flow see?

Per-decision credit rewards a path decision with the delivery of its own flow.
If choosing the least-loaded path mostly protects *other* flows (the ones that
later find a link full), that reward carries little signal for state-dependent
routing, however good the learner.

Under ECMP (random path, pinned per flow) the choice is randomised, so comparing
the flows that landed on a least-utilised path with those that did not gives a
causal estimate of the benefit to the flow itself. We compare it with the
system-level gain of greedy over ECMP on the same traffic.

Reports, per configuration:
  own-flow effect   E[r | least-loaded] - E[r | not], r = delivered share of the flow
  own-flow gain     what the flows themselves would gain if every contested decision
                    that ECMP got wrong had chosen a least-loaded path
                    (share of such decisions x own-flow effect), in PDR points
  system gain       PDR(greedy) - PDR(ECMP), paired traffic
  own share         own-flow gain / system gain
plus the own-flow effect by the utilisation gap between the chosen and the best path.

    python tools/own_flow_effect.py --config configs/v2_pilot4_lc.json --episodes 20
"""
import argparse
import json
import os
import random
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--src', default=os.path.join(os.path.dirname(__file__), '..', 'src'))
ap.add_argument('--config', required=True)
ap.add_argument('--episodes', type=int, default=20)
ap.add_argument('--out', default=None)
args = ap.parse_args()
sys.path.insert(0, args.src)
sys.path.insert(0, os.path.join(args.src, 'maddpg_clean'))

from standalone_experiment_runner import StandaloneExperimentRunner  # noqa: E402

SEED, TAIL = 20240601, 20
runner = StandaloneExperimentRunner(args.config, 0, '/tmp/own_flow')
ae = runner.config.get('attack_eval', {})
load = float(ae.get('offered_load_factor', 2.0))
T = runner.config['training']['timesteps_per_episode']
env = runner._make_attack_env(ae.get('hotspot') or None)
eng = env.engine
assert eng.pin_paths_default, "needs traffic.pin_paths_per_flow"
hosts, tr = eng.get_all_hosts(), eng.trainable_host_indices


def run(rule):
    """Per applied decision: (least-loaded?, utilisation gap, own delivered share); and PDR."""
    random.seed(SEED); np.random.seed(SEED)
    rng = random.Random(SEED + 999)
    rows, pdrs = [], []
    for _ in range(args.episodes):
        eng.topology.restore_intact()
        eng.reset_with_load(offered_load_factor=load)
        eng.decision_log = []
        for _ in range(T):
            acts = [runner._rule_action(hosts[i], rule, eng, rng) for i in tr]
            env.step(runner._build_full_actions(acts, eng.n_total_hosts, tr, eng.n_actions))
        pdrs.append(float(env.get_stats().get('end_to_end_pdr', 0.0)))
        for d in eng.decision_log:
            u = np.asarray(d['utils'])
            if (d['flow_id'] is None or d['step'] > T - TAIL or d['n_distinct'] < 2
                    or u.max() - u.min() <= 1e-9 or d['k'] >= len(u)):
                continue
            dl, dr = eng.flow_outcomes.get(d['flow_id'], (0, 0))
            if dl + dr == 0:
                continue
            rows.append((float(u[d['k']] <= u.min() + 1e-9), float(u[d['k']] - u.min()), dl / (dl + dr)))
        eng.decision_log = None
    return np.array(rows), float(np.mean(pdrs))


ecmp, pdr_ecmp = run('random')
_, pdr_greedy = run('greedy')
ll, gap, r = ecmp[:, 0], ecmp[:, 1], ecmp[:, 2]
eff = r[ll == 1].mean() - r[ll == 0].mean()
n_wrong = int((ll == 0).sum())
# own-flow gain in PDR points: contested-but-wrong decisions as a share of all flows placed, times the effect
own_gain = 100 * eff * n_wrong / len(r)   # per contested flow, in points of that flow's delivery
sys_gain = pdr_greedy - pdr_ecmp
se = np.sqrt(r[ll == 1].var() / max(1, (ll == 1).sum()) + r[ll == 0].var() / max(1, (ll == 0).sum()))
bins = [0, 0.05, 0.1, 0.2, 0.4, 1.01]
by_gap = []
for lo, hi in zip(bins[:-1], bins[1:]):
    sel = (ll == 0) & (gap > lo) & (gap <= hi)
    if sel.sum() >= 20:
        by_gap.append({'gap': f'({lo:g},{hi:g}]', 'n': int(sel.sum()),
                       'own_loss_pp': round(100 * (r[ll == 1].mean() - r[sel].mean()), 2)})
res = {'config': args.config, 'load': load, 'episodes': args.episodes, 'contested_decisions': len(r),
       'least_loaded_share': round(float(ll.mean()), 3),
       'own_flow_effect_pp': round(100 * eff, 2), 'own_flow_effect_se_pp': round(100 * se, 2),
       'mean_own_delivery_ll': round(100 * r[ll == 1].mean(), 2),
       'mean_own_delivery_not_ll': round(100 * r[ll == 0].mean(), 2),
       'own_gain_pp_contested': round(own_gain, 2),
       'pdr_ecmp': round(pdr_ecmp, 2), 'pdr_greedy': round(pdr_greedy, 2), 'system_gain_pp': round(sys_gain, 2),
       'own_share_of_system_gain': round(own_gain / sys_gain, 3) if sys_gain > 0 else None,
       'by_gap': by_gap}
print(json.dumps(res, indent=1))
if args.out:
    json.dump(res, open(args.out, 'w'), indent=1)
