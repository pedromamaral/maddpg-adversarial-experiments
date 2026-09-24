#!/usr/bin/env python3
"""Why does the gradient attack not steer traffic onto worse paths?

path_diversity_analysis.py found that only ~34-47% of effective flips move traffic
onto a more congested path. Two different explanations would produce that, and they
imply very different things for the paper:

  (1) THERE WAS NOTHING WORSE TO STEER TOWARD. If the alternatives are about as
      loaded as the chosen path, no attack can do harm by rerouting, and the
      robustness is a property of the network.

  (2) WORSE PATHS EXISTED BUT THE OBJECTIVE DID NOT AIM AT THEM. The attack
      maximises L = sum_k pi_k * w_k with w_k = sigmoid((u_k - 0.5) * 10), and the
      actor emits INDEPENDENT sigmoids pi_k = sigmoid(z_k), not a softmax. So
      dL/dpi_k = w_k > 0 for every path, including the one already chosen: the
      objective asks to raise all outputs, weighted by congestion, and contains no
      term that pushes the current decision down. If that is what is happening, the
      robustness is partly an artefact of the attack.

Measured on real mid-episode observations of a frozen victim, per (agent,
destination) decision with at least two genuinely distinct candidate paths:

  * headroom         u_max - u_chosen: did a strictly more congested path exist?
  * load level       share of candidate paths above the 0.5 knee of the weight
  * weight spread    max w - min w within a destination's distinct paths: how much
                     the objective can even tell the paths apart
  * saturation       share of chosen outputs with pi > 0.99 (sigmoid gradient ~ 0)
  * misalignment     after the attack, did pi_chosen go UP? (the objective rewards
                     that, and doing it defends the current decision)

Usage (inside the container, from the repo root):
    python tools/gradient_signal_analysis.py --variant CC-Simple \
        --results-dir data/results/reward_fix --json-out data/results/gradsignal/CC-Simple.json
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'maddpg_clean'))

from standalone_experiment_runner import StandaloneExperimentRunner  # noqa: E402


def weight(u):
    return 1.0 / (1.0 + math.exp(-(u - 0.5) * 10.0))


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
    n_dest, K = len(dests), max(1, maddpg.n_actions // len(dests))
    device = maddpg.agents[0].actor.device
    u0 = engine.max_neighbors + 3 + n_dest + 3      # start of the K-path util block

    fw = runner.attack_framework
    fw.epsilon, fw.attack_type, fw.n_steps = args.epsilon, 'packet_loss', 1

    def outputs(ti, obs):
        with torch.no_grad():
            x = torch.tensor(np.asarray(obs, dtype=np.float32), device=device).unsqueeze(0)
            return maddpg.agents[ti].actor(x).squeeze(0).cpu().numpy()

    engine.reset_with_load(offered_load_factor=load)
    states = env.reset()
    n_total = engine.n_total_hosts

    rec = []   # one record per decision with >= 2 distinct paths
    all_u = []
    for t in range(args.warmup + args.steps):
        with torch.no_grad():
            acts = maddpg.choose_action([states[i] for i in trainable])
            actions = runner._build_full_actions(acts, n_total, trainable, maddpg.n_actions)
        if t >= args.warmup:
            for ti, topo_idx in enumerate(trainable):
                clean = np.asarray(states[topo_idx], dtype=np.float32)
                adv = fw.generate_adversarial_state(
                    state=clean, agent_network=maddpg.agents[ti],
                    network_engine=engine, agent_index=ti)
                pc, pa = outputs(ti, clean), outputs(ti, adv)
                for di, d in enumerate(dests):
                    paths = engine.topology.kpath_cache.get((hosts[topo_idx], d), [])
                    # First slot of each genuinely distinct route.
                    seen, distinct = set(), []
                    for k, p in enumerate(paths[:K]):
                        if tuple(p) not in seen:
                            seen.add(tuple(p)); distinct.append(k)
                    idx = [u0 + di * K + k for k in distinct]
                    if len(distinct) < 2 or max(idx) >= len(clean):
                        continue
                    u = {k: float(clean[u0 + di * K + k]) for k in distinct}
                    all_u.extend(u.values())
                    block_c = pc[di * K:(di + 1) * K]
                    block_a = pa[di * K:(di + 1) * K]
                    kc = int(np.argmax(block_c))
                    ka = int(np.argmax(block_a))
                    if kc not in u:            # chosen a padded duplicate slot
                        kc = next(k for k in distinct
                                  if tuple(paths[k]) == tuple(paths[kc]))
                    kw = max(distinct, key=lambda k: u[k])
                    w = [weight(u[k]) for k in distinct]
                    rec.append({
                        'headroom': max(u.values()) - u[kc],
                        'worse_exists': max(u.values()) > u[kc] + 1e-9,
                        'weight_spread': max(w) - min(w),
                        'chosen_saturated': block_c[kc] > 0.99,
                        'chosen_went_up': block_a[kc] > block_c[kc],
                        'worst_went_up': block_a[kw] > block_c[kw],
                        'worst_rel_gain': (block_a[kw] - block_c[kw])
                                          - (block_a[kc] - block_c[kc]),
                        'flipped': ka != kc,
                        'hit_worst': ka == kw and kw != kc,
                        'chosen_is_worst': kc == kw,
                    })
        states, _, _ = env.step(actions)

    n = len(rec)
    if not n:
        print("no decisions with >= 2 distinct paths"); return
    f = lambda key: sum(1 for r in rec if r[key]) / n * 100
    wr = [r for r in rec if r['worse_exists']]
    mean = lambda xs: sum(xs) / len(xs) if xs else float('nan')

    print(f"variant={args.variant}  decisions with >=2 distinct paths: {n}\n")
    print("(1) Was there anything worse to steer toward?")
    print(f"    a strictly more congested path existed   {f('worse_exists'):5.1f}%")
    print(f"    policy already on the most congested     {f('chosen_is_worst'):5.1f}%")
    print(f"    mean headroom when one existed            {mean([r['headroom'] for r in wr]):.3f}"
          f"   (utilisation units, 0-1)")
    print(f"    candidate paths above the 0.5 knee       "
          f"{sum(1 for x in all_u if x > 0.5) / max(1, len(all_u)) * 100:5.1f}%"
          f"   (median util {sorted(all_u)[len(all_u)//2]:.3f})")

    print("\n(2) Could the objective even tell the paths apart?")
    print(f"    mean congestion-weight spread in a block  {mean([r['weight_spread'] for r in rec]):.3f}"
          f"   (1.0 = fully distinguishable, 0 = identical)")
    print(f"    blocks with weight spread < 0.05          "
          f"{sum(1 for r in rec if r['weight_spread'] < 0.05) / n * 100:5.1f}%")
    print(f"    chosen output saturated (pi > 0.99)       {f('chosen_saturated'):5.1f}%")

    print("\n(3) Did the attack push in the right direction? (decisions with headroom)")
    if wr:
        g = lambda key: sum(1 for r in wr if r[key]) / len(wr) * 100
        print(f"    chosen path's output went UP             {g('chosen_went_up'):5.1f}%"
              f"   <- reinforces the current decision")
        print(f"    worst path's output went UP              {g('worst_went_up'):5.1f}%")
        print(f"    worst gained on chosen (mean rel. gain)   {mean([r['worst_rel_gain'] for r in wr]):+.4f}")
        print(f"    decision flipped                          {g('flipped'):5.1f}%")
        print(f"    ... and landed on the most congested      {g('hit_worst'):5.1f}%")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or '.', exist_ok=True)
        json.dump({'variant': args.variant, 'n_decisions': n,
                   'worse_exists_pct': f('worse_exists'),
                   'chosen_is_worst_pct': f('chosen_is_worst'),
                   'mean_headroom': mean([r['headroom'] for r in wr]),
                   'share_paths_above_knee': sum(1 for x in all_u if x > 0.5) / max(1, len(all_u)),
                   'median_util': sorted(all_u)[len(all_u) // 2],
                   'mean_weight_spread': mean([r['weight_spread'] for r in rec]),
                   'chosen_saturated_pct': f('chosen_saturated'),
                   'chosen_went_up_pct': (sum(1 for r in wr if r['chosen_went_up']) / len(wr) * 100) if wr else None,
                   'worst_went_up_pct': (sum(1 for r in wr if r['worst_went_up']) / len(wr) * 100) if wr else None,
                   'mean_worst_rel_gain': mean([r['worst_rel_gain'] for r in wr]),
                   'flipped_pct': (sum(1 for r in wr if r['flipped']) / len(wr) * 100) if wr else None,
                   'hit_worst_pct': (sum(1 for r in wr if r['hit_worst']) / len(wr) * 100) if wr else None},
                  open(args.json_out, 'w'), indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == '__main__':
    main()
