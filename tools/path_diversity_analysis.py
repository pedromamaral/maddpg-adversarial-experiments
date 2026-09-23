#!/usr/bin/env python3
"""Test the mechanism behind the robustness result: does path diversity explain it?

The thesis argument is that K-path redundancy absorbs adversarial decision flips --
a flipped decision moves a flow to a different candidate path, which usually also
delivers. That is asserted on one topology. This script tests it from the inside,
using the variation that already exists across (agent, destination) pairs.

The lever is _build_distinct_kpaths: when a pair has fewer than K genuinely
distinct routes it PADS the list by repeating the shortest path,

    while distinct and len(distinct) < k:
        distinct.append(distinct[0])

so some pairs really offer K alternatives and others offer 1 or 2. Two consequences
are measurable without retraining anything:

  1. For a padded pair, flipping the decision between two duplicate slots routes the
     packet down the SAME hops. The flip is real at the decision layer and a no-op in
     the network -- a direct, mechanical explanation for "many flips, no damage".

  2. If redundancy is what protects, the adversarial effect should be concentrated on
     LOW-diversity destinations, where there is no alternative to absorb the reroute.

Outputs, per variant:
  * the distribution of genuine path diversity over (agent, destination) pairs
  * the share of FGSM decision flips that are effective (change the hop sequence)
    versus vacuous (land on a duplicate slot)
  * flip rate and effective-flip rate broken down by diversity level

Usage (inside the container, from the repo root):
    python tools/path_diversity_analysis.py --variant CC-Simple \
        --results-dir data/results/reward_fix --json-out data/results/diversity/CC-Simple.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'maddpg_clean'))

from standalone_experiment_runner import StandaloneExperimentRunner  # noqa: E402


def path_diversity(engine, host, dst):
    """Number of genuinely distinct candidate paths for this (agent, destination).

    The cache holds exactly K entries; duplicates are padding, so the count of
    unique hop-tuples is the real number of routes the agent can choose between.
    """
    paths = engine.topology.kpath_cache.get((host, dst), [])
    return len({tuple(p) for p in paths}), paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='reward_fix_full_config.json')
    ap.add_argument('--variant', default='CC-Simple')
    ap.add_argument('--results-dir', default='data/results/reward_fix')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--epsilon', type=float, default=0.3)
    ap.add_argument('--warmup', type=int, default=24)
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    runner = StandaloneExperimentRunner(args.config, args.gpu, args.results_dir)
    vcfg = next(v for v in runner.config['variants'] if v['name'] == args.variant)
    maddpg, _, _ = runner._make_variant(vcfg)
    runner._load_variant_checkpoint(maddpg, args.variant)

    attack_eval = runner.config.get('attack_eval', {})
    load = float(attack_eval.get('offered_load_factor', 2.0))
    env = runner._make_attack_env(attack_eval.get('hotspot') or None)
    engine = env.engine

    hosts = engine.get_all_hosts()
    trainable = engine.trainable_host_indices
    dests = engine.topology.access_nodes
    n_dest, k_paths = len(dests), max(1, maddpg.n_actions // len(dests))
    device = maddpg.agents[0].actor.device

    fw = runner.attack_framework
    fw.epsilon = args.epsilon
    fw.attack_type = 'packet_loss'
    fw.n_steps = 1

    # ── 1. Structural diversity, independent of any attack ──────────────────
    div = {}
    for ti, topo_idx in enumerate(trainable):
        h = hosts[topo_idx]
        for di, d in enumerate(dests):
            n_unique, paths = path_diversity(engine, h, d)
            div[(ti, di)] = (n_unique, [tuple(p) for p in paths])
    hist = Counter(v[0] for v in div.values())
    total_pairs = len(div)
    print(f"variant={args.variant}  agents={len(trainable)}  destinations={n_dest}  K={k_paths}")
    print("\nGenuine path diversity over (agent, destination) pairs:")
    for lvl in sorted(hist):
        print(f"  {lvl} distinct path(s): {hist[lvl]:5d} pairs  ({hist[lvl]/total_pairs*100:5.1f}%)")

    # ── 2. Are the flips effective? ─────────────────────────────────────────
    engine.reset_with_load(offered_load_factor=load)
    states = env.reset()
    n_total = engine.n_total_hosts

    flips_by_div = defaultdict(int)       # decision changed
    eff_by_div = defaultdict(int)         # ... and the hop sequence changed too
    seen_by_div = defaultdict(int)
    for t in range(args.warmup + args.steps):
        with torch.no_grad():
            t_states = [states[i] for i in trainable]
            acts = maddpg.choose_action(t_states)
            actions = runner._build_full_actions(acts, n_total, trainable, maddpg.n_actions)

        if t >= args.warmup:
            for ti, topo_idx in enumerate(trainable):
                clean_obs = np.asarray(states[topo_idx], dtype=np.float32)
                adv_obs = fw.generate_adversarial_state(
                    state=clean_obs, agent_network=maddpg.agents[ti],
                    network_engine=engine, agent_index=ti)
                with torch.no_grad():
                    def dec(o):
                        x = torch.tensor(np.asarray(o, dtype=np.float32),
                                         device=device).unsqueeze(0)
                        out = maddpg.agents[ti].actor(x).squeeze(0).cpu().numpy()
                        return out[:n_dest * k_paths].reshape(n_dest, k_paths).argmax(axis=1)
                    c_idx, a_idx = dec(clean_obs), dec(adv_obs)
                for di in range(n_dest):
                    lvl, paths = div[(ti, di)]
                    seen_by_div[lvl] += 1
                    if c_idx[di] == a_idx[di]:
                        continue
                    flips_by_div[lvl] += 1
                    # Effective only if the two chosen slots hold different hops.
                    if (c_idx[di] < len(paths) and a_idx[di] < len(paths)
                            and paths[c_idx[di]] != paths[a_idx[di]]):
                        eff_by_div[lvl] += 1
        states, _, _ = env.step(actions)

    print("\nFGSM decision flips, by genuine path diversity:")
    print(f"  {'diversity':>9}  {'decisions':>10}  {'flips':>8}  {'flip%':>7}  "
          f"{'effective':>9}  {'eff%offlips':>11}")
    rows = []
    tot_f = tot_e = 0
    for lvl in sorted(seen_by_div):
        s, f, e = seen_by_div[lvl], flips_by_div[lvl], eff_by_div[lvl]
        tot_f += f; tot_e += e
        print(f"  {lvl:>9}  {s:>10}  {f:>8}  {f/max(1,s)*100:6.2f}%  {e:>9}  "
              f"{e/max(1,f)*100:10.1f}%")
        rows.append({'diversity': lvl, 'decisions': s, 'flips': f, 'effective_flips': e,
                     'flip_rate': f / max(1, s), 'effective_share': e / max(1, f)})
    print(f"\n  overall: {tot_f} flips, {tot_e} effective "
          f"({tot_e/max(1,tot_f)*100:.1f}%) — the rest land on duplicate path slots "
          f"and route identically.")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or '.', exist_ok=True)
        json.dump({'variant': args.variant, 'epsilon': args.epsilon,
                   'n_agents': len(trainable), 'n_destinations': n_dest, 'k_paths': k_paths,
                   'diversity_histogram': {str(k): v for k, v in hist.items()},
                   'by_diversity': rows,
                   'total_flips': tot_f, 'total_effective': tot_e},
                  open(args.json_out, 'w'), indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == '__main__':
    main()
