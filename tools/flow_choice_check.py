#!/usr/bin/env python3
"""Does a trained policy choose the least-loaded path where its choice is applied?

tools/policy_attribution.py scores every live (agent, destination) output. With
per-flow pinning (traffic.pin_paths_per_flow) only the decision at a flow's first
packet is applied; later packets follow the pinned path whatever the policy says.
This tool scores exactly the applied decisions, through NetworkEngine.decision_log,
against the utilisation the agent observed at the start of that step.

For the policy and the reference rules (random = ECMP under pinning, greedy and
stale greedy) on identical traffic, per failure level it reports:

  decisions    applied path decisions (flows placed)
  contested    share with >= 2 distinct candidate paths of different utilisation
  least-loaded among contested decisions, share choosing a minimum-utilisation path
  chance       the share a uniform choice over the K slots would reach
  regret       mean (chosen - minimum) bottleneck utilisation over contested decisions
  spread       mean (maximum - minimum) utilisation over contested decisions
  PDR          end-to-end delivery

Greedy on fresh utilisation should score ~100 % least-loaded and random ~chance:
both check the metric.

    python tools/flow_choice_check.py --config configs/v2_pilot4_lc.json \
        --results data/results/v2_p4_lc --variant LC-Simple --load 3 --failures 0,4
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
ap.add_argument('--results', required=True)
ap.add_argument('--variant', default='LC-Simple')
ap.add_argument('--load', type=float, default=None, help='default: attack_eval.offered_load_factor')
ap.add_argument('--failures', default='0,4')
ap.add_argument('--episodes', type=int, default=10)
ap.add_argument('--schemes', default='policy,random,greedy,greedy_stale2,greedy_stale4')
ap.add_argument('--out', default=None, help='JSON output (default: <results>/flow_choice.json)')
args = ap.parse_args()
sys.path.insert(0, args.src)
sys.path.insert(0, os.path.join(args.src, 'maddpg_clean'))

import torch  # noqa: E402
from standalone_experiment_runner import StandaloneExperimentRunner  # noqa: E402

TRAFFIC_SEED = 20240601

runner = StandaloneExperimentRunner(args.config, 0, args.results)
ae = runner.config.get('attack_eval', {})
load = args.load if args.load is not None else float(ae.get('offered_load_factor', 2.0))
T = runner.config['training']['timesteps_per_episode']
env = runner._make_attack_env(ae.get('hotspot') or None)
eng = env.engine
vcfg = next(x for x in runner.config['variants'] if x['name'] == args.variant)
maddpg, _, _ = runner._make_variant(vcfg)
runner._load_variant_checkpoint(maddpg, args.variant)


def score(log):
    contested = [d for d in log if d['n_distinct'] >= 2 and max(d['utils']) - min(d['utils']) > 1e-9]
    if not contested:
        return {'decisions': len(log), 'contested': 0.0}
    hit, chance, regret, spread = [], [], [], []
    for d in contested:
        u = np.asarray(d['utils'])
        lo = u.min()
        hit.append(float(u[d['k']] <= lo + 1e-9) if d['k'] < len(u) else 0.0)
        chance.append(float((u <= lo + 1e-9).sum()) / len(u))
        regret.append(float(u[min(d['k'], len(u) - 1)] - lo))
        spread.append(float(u.max() - lo))
    return {'decisions': len(log), 'contested': len(contested) / len(log),
            'least_loaded': float(np.mean(hit)), 'chance': float(np.mean(chance)),
            'regret': float(np.mean(regret)), 'spread': float(np.mean(spread))}


out = {'config': args.config, 'variant': args.variant, 'load': load,
       'pinned': bool(eng.pin_paths_default)}
print(f"{args.variant} at {load:g}x, flows pinned: {eng.pin_paths_default}, {args.episodes} episodes")
print(f"{'fail':>4} {'scheme':<14} {'PDR':>6} {'decisions':>9} {'contested':>9} "
      f"{'least-loaded':>12} {'chance':>6} {'regret':>7} {'spread':>7}")
for nf in (int(x) for x in args.failures.split(',')):
    for label in args.schemes.split(','):
        rule, pin, stale = runner._parse_rule_spec(label)
        random.seed(TRAFFIC_SEED); np.random.seed(TRAFFIC_SEED); torch.manual_seed(TRAFFIC_SEED)
        eng.decision_log = []
        res = runner._attack_episodes(maddpg, env, args.episodes, T, attack=False,
                                      offered_load_factor=load, routing_rule=rule,
                                      n_link_failures=nf, pin_per_flow=pin, stale_steps=stale)
        r = score(eng.decision_log)
        eng.decision_log = None
        r['pdr'] = res['mean_end_to_end_pdr']
        out.setdefault(f"fail{nf}", {})[label] = r
        if 'least_loaded' in r:
            print(f"{nf:>4} {label:<14} {r['pdr']:6.2f} {r['decisions']:9d} {100 * r['contested']:8.1f}% "
                  f"{100 * r['least_loaded']:11.1f}% {100 * r['chance']:5.1f}% "
                  f"{r['regret']:7.3f} {r['spread']:7.3f}", flush=True)
        else:
            print(f"{nf:>4} {label:<14} {r['pdr']:6.2f} {r['decisions']:9d}  no contested decisions", flush=True)
json.dump(out, open(args.out or os.path.join(args.results, 'flow_choice.json'), 'w'), indent=1)
