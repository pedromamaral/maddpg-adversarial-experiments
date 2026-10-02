#!/usr/bin/env python3
"""What do the victims' routing decisions depend on?

Records real observations and decisions of each frozen victim on clean episodes
(2x hotspot, the attack operating point), on the intact topology and with random
link failures, then measures:

  static share     how often a (agent, destination) decision equals that pair's
                   most frequent path slot: 100 % = a fixed routing table
  slot mix         share of decisions on path slot k0 (shortest), k1, k2
  greedy agreement among decisions with >= 2 distinct candidate paths, how often
                   the chosen path has the lowest observed utilisation, against
                   the rate a uniform choice of slot would reach (chance)
  U spread         mean (max - min) observed utilisation across those candidates:
                   how much there is to choose between
  saturation       share of decisions whose chosen sigmoid output exceeds 0.99
  importance       permutation importance per observation group: shuffle that
                   group across time steps (same shuffle for every agent, so a
                   step stays internally consistent) and count changed decisions.
                   'all' shuffles whole observations: how much decisions vary
                   with the state at all

Decisions are counted on 'live' (agent, destination) pairs, those whose
destination flag says packets for it are queued, since only those route traffic.

    python tools/policy_attribution.py --failures 0,2,4 --episodes 3
"""
import argparse
import json
import os
import random
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--src', default=os.path.join(os.path.dirname(__file__), '..', 'src'))
ap.add_argument('--config', default='reward_fix_full_config.json')
ap.add_argument('--results', default='data/results/reward_fix')
ap.add_argument('--variants', default='CC-Simple,CC-Duelling,LC-Simple,LC-Duelling,'
                                      'CC-Simple-GNN,CC-Duelling-GNN,LC-Duelling-GNN')
ap.add_argument('--failures', default='0,2,4')
ap.add_argument('--episodes', type=int, default=3)
ap.add_argument('--repeats', type=int, default=2, help='permutations per group')
ap.add_argument('--out', default='data/results/policy_attribution.json')
args = ap.parse_args()
sys.path.insert(0, args.src)
sys.path.insert(0, os.path.join(args.src, 'maddpg_clean'))

import torch  # noqa: E402
from standalone_experiment_runner import StandaloneExperimentRunner  # noqa: E402

TRAFFIC_SEED = 20240601  # as in _attack_episodes, so episode 1 matches the probes


def feature_groups(engine):
    mn, nd, S = engine.max_neighbors, engine.n_destinations, engine.state_dims
    g = {'nbr_bandwidth': range(0, mn), 'queue': [mn, mn + 1], 'timestep': [mn + 2],
         'dest_flags': range(mn + 3, mn + 3 + nd), 'adj_util': range(mn + 3 + nd, mn + 6 + nd),
         'path_util': engine.path_util_slots}
    g = {k: list(v) for k, v in g.items()}
    assert sorted(i for v in g.values() for i in v) == list(range(S)), "groups must tile the observation"
    return g


def collect(runner, maddpg, env, n_fail, n_eps, load, t_per_ep):
    """Observations X [T, A, S], decisions D [T, A, nd], chosen-output values,
    and distinct-path counts [T, A, nd] over n_eps clean episodes."""
    eng = env.engine
    hosts, tr = eng.get_all_hosts(), eng.trainable_host_indices
    nd, K = eng.n_destinations, maddpg.n_actions // eng.n_destinations
    random.seed(TRAFFIC_SEED); np.random.seed(TRAFFIC_SEED); torch.manual_seed(TRAFFIC_SEED)
    X, D, V, NDIST = [], [], [], []
    for _ in range(n_eps):
        eng.topology.restore_intact()
        eng.reset_with_load(offered_load_factor=load)
        if n_fail:
            runner._inject_failures(eng, n_fail)
            eng.topology.refresh_path_cache()
        ndist = np.array([[len({tuple(p) for p in eng.topology.kpath_cache.get((hosts[i], dst), [])[:K]})
                           for dst in eng.topology.access_nodes] for i in tr])
        states = [eng.get_state(h) for h in hosts]
        for _ in range(t_per_ep):
            obs = [np.asarray(states[i], dtype=np.float32) for i in tr]
            with torch.no_grad():
                acts = maddpg.choose_action(obs)
            out = np.stack([np.asarray(a).reshape(nd, K) for a in acts])
            X.append(np.stack(obs)); D.append(out.argmax(2)); V.append(out.max(2)); NDIST.append(ndist)
            states, _, _ = env.step(runner._build_full_actions(acts, eng.n_total_hosts, tr, maddpg.n_actions))
    eng.topology.restore_intact()
    return np.array(X), np.array(D), np.array(V), np.array(NDIST)


def decide(maddpg, X, nd, K):
    out = []
    with torch.no_grad():
        for t in range(X.shape[0]):
            acts = maddpg.choose_action(list(X[t]))
            out.append(np.stack([np.asarray(a).reshape(nd, K).argmax(1) for a in acts]))
    return np.array(out)


def analyse(maddpg, eng, X, D, V, NDIST, repeats):
    nd, S = eng.n_destinations, eng.state_dims
    K = maddpg.n_actions // nd
    mn, s0 = eng.max_neighbors, eng.path_util_slots[0]
    live = X[:, :, mn + 3:mn + 3 + nd] > 0.5                       # [T, A, nd]
    r = {'n_decisions': int(D.size), 'live_share': float(live.mean())}

    mode = np.zeros(D.shape[1:], dtype=int)                        # per (agent, destination)
    for a in range(D.shape[1]):
        for d in range(nd):
            mode[a, d] = np.bincount(D[:, a, d], minlength=K).argmax()
    r['static_share_live'] = float((D == mode)[live].mean())
    r['slot_mix_live'] = [float((D[live] == k).mean()) for k in range(K)]
    r['saturation_live'] = float((V[live] > 0.99).mean())

    # greedy agreement on live decisions with >= 2 distinct paths and all K slots observed
    full = np.array([s0 + d * K + K - 1 < S for d in range(nd)])
    U = np.full(D.shape + (K,), np.nan)
    for d in np.where(full)[0]:
        U[:, :, d, :] = X[:, :, s0 + d * K:s0 + (d + 1) * K]
    m = live & (NDIST >= 2) & full[None, None, :]
    Um = U[m]                                                       # [n, K]
    ch = D[m]
    umin = Um.min(1)
    at_min = Um <= umin[:, None] + 1e-9
    r['greedy_agreement'] = float(at_min[np.arange(len(ch)), ch].mean())
    r['greedy_chance'] = float(at_min.mean())
    r['u_spread'] = float((Um.max(1) - umin).mean())
    r['n_choice_decisions'] = int(m.sum())

    rng = np.random.default_rng(0)
    groups = feature_groups(eng)
    groups['all'] = list(range(S))
    imp = {}
    for g, idx in groups.items():
        fl = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, :, idx] = X[rng.permutation(X.shape[0])][:, :, idx]
            fl.append(float((decide(maddpg, Xp, nd, K) != D)[live].mean()))
        imp[g] = float(np.mean(fl))
    r['importance_live'] = imp
    return r


def main():
    runner = StandaloneExperimentRunner(args.config, 0, args.results)
    ae = runner.config.get('attack_eval', {})
    load = float(ae.get('offered_load_factor', 2.0))
    t_per_ep = runner.config['training']['timesteps_per_episode']
    env = runner._make_attack_env(ae.get('hotspot') or None)
    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for v in args.variants.split(','):
        vcfg = next(x for x in runner.config['variants'] if x['name'] == v)
        maddpg, _, _ = runner._make_variant(vcfg)
        runner._load_variant_checkpoint(maddpg, v)
        for nf in (int(x) for x in args.failures.split(',')):
            X, D, V, NDIST = collect(runner, maddpg, env, nf, args.episodes, load, t_per_ep)
            r = analyse(maddpg, env.engine, X, D, V, NDIST, args.repeats)
            out.setdefault(v, {})[f"fail{nf}"] = r
            json.dump(out, open(args.out, 'w'), indent=1)
            imp = ' '.join(f"{g}={100 * x:.1f}" for g, x in r['importance_live'].items())
            print(f"{v:<16} fail{nf}: static {100 * r['static_share_live']:.1f}%  "
                  f"slots {[round(100 * s) for s in r['slot_mix_live']]}  "
                  f"greedy {100 * r['greedy_agreement']:.1f}% (chance {100 * r['greedy_chance']:.1f}%)  "
                  f"Uspread {r['u_spread']:.3f}  sat {100 * r['saturation_live']:.0f}%  | flips% {imp}",
                  flush=True)


if __name__ == '__main__':
    main()
