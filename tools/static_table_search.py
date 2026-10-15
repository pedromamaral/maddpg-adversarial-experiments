#!/usr/bin/env python3
"""Best static routing table by SYSTEM PDR (E026), simulator-only local search.

A table T = {agent: {dest: k}} fixes, for every (agent, destination), the index of
the K-path the agent uses (same file format and semantics as the 'table' rule,
StandaloneExperimentRunner._rule_action: k beyond the live paths -> k=0; after link
failures the K paths are rebuilt on the post-failure graph, as in evaluation).

J(T) = mean end-to-end PDR over M TRAIN episodes; episode e is reseeded with
seed0 + e for every candidate (common random numbers), so candidates are compared
on paired traffic. Train seeds must stay disjoint from the eval seed 20240601.

Search: first-improvement coordinate descent. Entries are visited by decreasing
traffic volume (decisions per (agent, dest) under the start table); for each, every
other live k is tried and accepted if the paired gain mean(dJ) > eps (pp). Sweeps
repeat until a sweep accepts nothing or the evaluation budget runs out. Each start
is searched in turn; the best table by J is kept.

    python tools/static_table_search.py --config configs/v2_pilot4_lc.json --load 3 \\
        --episodes 8 --eps 0.3 --start sp --start host_data/.../table.json --start random:1 \\
        --workers 8 --out-dir host_data/results/E026_sp

--sweeps 0 only evaluates the starts (use it to time one J and to size M / eps).
Writes <out-dir>/table_best.json, table_<start>.json and search_log.json.
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--src', default=os.path.join(os.path.dirname(__file__), '..', 'src'))
ap.add_argument('--config', required=True)
ap.add_argument('--load', type=float, default=None, help='offered load (default: attack_eval)')
ap.add_argument('--episodes', type=int, default=8, help='M train episodes per J')
ap.add_argument('--seed0', type=int, default=7000, help='train seeds seed0..seed0+M-1')
ap.add_argument('--failures', default='0', help='link failures, cycled over episodes, e.g. 0 or 0,4')
ap.add_argument('--eps', type=float, default=0.3, help='accept threshold on the paired gain, pp')
ap.add_argument('--start', action='append', default=[],
                help="sp | random:<seed> | path to a table JSON; repeatable (default: sp)")
ap.add_argument('--sweeps', type=int, default=20, help='max sweeps per start (0 = evaluate starts only)')
ap.add_argument('--max-evals', type=int, default=100000, help='J evaluations per start')
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--out-dir', required=True)
args = ap.parse_args()
assert args.seed0 > 20240601 or args.seed0 + args.episodes <= 20240601, "train seeds overlap the eval seed"
sys.path.insert(0, args.src)
sys.path.insert(0, os.path.join(args.src, 'maddpg_clean'))

from standalone_experiment_runner import StandaloneExperimentRunner  # noqa: E402

runner = StandaloneExperimentRunner(args.config, 0, '/tmp/static_table_search')
ae = runner.config.get('attack_eval', {})
load = args.load if args.load is not None else float(ae.get('offered_load_factor', 2.0))
T_STEPS = runner.config['training']['timesteps_per_episode']
env = runner._make_attack_env(ae.get('hotspot') or None)
eng = env.engine
hosts, tr = eng.get_all_hosts(), eng.trainable_host_indices
A, ND = len(tr), eng.n_destinations
K = eng.n_actions // ND
access = list(eng.topology.access_nodes)
FAILS = [int(x) for x in args.failures.split(',')]
SEEDS = [args.seed0 + e for e in range(args.episodes)]


def n_live(host, dst):
    return min(len(eng.topology.kpath_cache.get((host, dst), [])), K)


def table_actions(tab):
    """One-hot action per agent from the table on the current kpath_cache (as the 'table' rule)."""
    acts = []
    for a, i in enumerate(tr):
        v = np.zeros(eng.n_actions, dtype=np.float32)
        for d, dst in enumerate(access):
            k = tab[a][d]
            v[d * K + (k if k < n_live(hosts[i], dst) else 0)] = 1.0
        acts.append(v)
    return runner._build_full_actions(acts, eng.n_total_hosts, tr, eng.n_actions)


def episode(job):
    """PDR of one paired train episode; optionally the decision counts per (agent, dest)."""
    tab, e, count = job
    seed = SEEDS[e]
    random.seed(seed); np.random.seed(seed)
    eng.topology.restore_intact()
    eng.reset_with_load(offered_load_factor=load)
    nf = FAILS[e % len(FAILS)]
    if nf:
        runner._inject_failures(eng, nf)
        eng.topology.refresh_path_cache()
    full = table_actions(tab)
    eng.decision_log = [] if count else None
    for _ in range(T_STEPS):
        env.step(full)
    pdr = float(env.get_stats().get('end_to_end_pdr', 0.0))
    counts = None
    if count:
        agent_of = {hosts[i]: a for a, i in enumerate(tr)}
        dix = {d: j for j, d in enumerate(access)}
        counts = {}
        for r in eng.decision_log:
            if r['host'] in agent_of:
                key = (agent_of[r['host']], dix[r['dst']])
                counts[key] = counts.get(key, 0) + 1
        eng.decision_log = None
    return pdr, counts


def evaluate(pool, tab, count=False):
    out = pool.map(episode, [(tab, e, count) for e in range(args.episodes)])
    return np.array([p for p, _ in out]), [c for _, c in out]


def load_start(spec):
    if spec == 'sp':
        return [[0] * ND for _ in range(A)]
    if spec.startswith('random:'):
        rng = random.Random(int(spec.split(':')[1]))
        return [[rng.randrange(max(1, n_live(hosts[i], dst))) for dst in access] for i in tr]
    f = json.load(open(spec))
    return [[int(f.get(str(a), {}).get(str(d), 0)) for d in range(ND)] for a in range(A)]


def to_json(tab):
    return {str(a): {str(d): int(tab[a][d]) for d in range(ND)} for a in range(A)}


def search(pool, spec, log):
    tab = load_start(spec)
    t0 = time.time()
    J, counts = evaluate(pool, tab, count=True)
    t_eval = time.time() - t0
    vol = {}
    for c in counts:
        for key, n in c.items():
            vol[key] = vol.get(key, 0) + n
    live = {(a, d): n_live(hosts[i], dst) for a, i in enumerate(tr) for d, dst in enumerate(access)}
    order = sorted((key for key in live if live[key] > 1), key=lambda key: -vol.get(key, 0))
    entry = {'start': spec, 'J0': round(float(J.mean()), 3),
             'J0_se': round(float(J.std(ddof=1) / np.sqrt(len(J))), 3) if len(J) > 1 else None,
             'sec_per_J': round(t_eval, 1), 'entries': len(order),
             'entries_with_traffic': sum(1 for key in order if vol.get(key, 0) > 0), 'sweeps': []}
    print(f"[{spec}] J0={J.mean():.2f} ({t_eval:.1f}s per J, {len(order)} searchable entries)", flush=True)
    n_evals = 1
    for sweep in range(args.sweeps):
        accepted, gains, diffs_se = [], [], []
        for (a, d) in order:
            if n_evals >= args.max_evals:
                break
            for k in range(live[(a, d)]):
                if k == tab[a][d] or n_evals >= args.max_evals:
                    continue
                cand = [row[:] for row in tab]
                cand[a][d] = k
                Jc, _ = evaluate(pool, cand)
                n_evals += 1
                diff = Jc - J
                if diff.mean() > args.eps:
                    accepted.append([a, d, tab[a][d], k])
                    gains.append(round(float(diff.mean()), 3))
                    diffs_se.append(round(float(diff.std(ddof=1) / np.sqrt(len(diff))), 3) if len(diff) > 1 else None)
                    tab, J = cand, Jc
                    break
        entry['sweeps'].append({'sweep': sweep, 'J': round(float(J.mean()), 3), 'accepts': len(accepted),
                                'changed': accepted, 'gains_pp': gains, 'gain_se_pp': diffs_se,
                                'evals': n_evals, 'elapsed_s': round(time.time() - t0, 1)})
        print(f"[{spec}] sweep {sweep}: J={J.mean():.2f} accepts={len(accepted)} evals={n_evals} "
              f"t={time.time() - t0:.0f}s", flush=True)
        json.dump(log, open(os.path.join(args.out_dir, 'search_log.json'), 'w'), indent=1)
        if not accepted or n_evals >= args.max_evals:
            break
    entry['J'] = round(float(J.mean()), 3)
    entry['evals'] = n_evals
    return tab, float(J.mean()), entry


if __name__ == '__main__':
    os.makedirs(args.out_dir, exist_ok=True)
    starts = args.start or ['sp']
    log = {'config': args.config, 'load': load, 'episodes': args.episodes, 'seed0': args.seed0,
           'failures': FAILS, 'eps': args.eps, 'K': K, 'agents': A, 'destinations': ND, 'starts': []}
    best = (None, -1.0, None)
    with mp.get_context('fork').Pool(args.workers) as pool:
        for spec in starts:
            tab, J, entry = search(pool, spec, log)
            log['starts'].append(entry)
            name = os.path.basename(spec).replace('.json', '').replace(':', '')
            json.dump(to_json(tab), open(os.path.join(args.out_dir, f'table_{name}.json'), 'w'))
            if J > best[1]:
                best = (tab, J, spec)
            json.dump(log, open(os.path.join(args.out_dir, 'search_log.json'), 'w'), indent=1)
    log['best'] = {'start': best[2], 'J': round(best[1], 3)}
    json.dump(to_json(best[0]), open(os.path.join(args.out_dir, 'table_best.json'), 'w'))
    json.dump(log, open(os.path.join(args.out_dir, 'search_log.json'), 'w'), indent=1)
    print(f"best: {best[2]} J={best[1]:.2f}", flush=True)
