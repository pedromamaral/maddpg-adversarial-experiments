#!/usr/bin/env python3
"""System vs own-flow gain of each accepted static-table change (E026b, H2 / A035).

Replays one start of a static_table_search.py search_log (default: the best start),
change by change. For change i the table before (T_{i-1}) and after (T_i) are rolled
out on the same CRN eval episodes (seed 20240601+e, disjoint from the search's train
seeds). Per episode we record
  system PDR                      env end_to_end_pdr
  own PDR  (agent a, all dests)   delivered share of the flows the changing agent placed
  entry PDR (agent a, dest d)     same, restricted to the changed (a, d) entry
Paired diffs (after - before) are reported as mean +- SE over episodes, in pp.
If own-flow credit could see a change, own/entry gain would be > 0 and track the system gain.

    python tools/table_change_credit.py --config configs/v2_deltacom.json --load 8 \\
        --search-log host_data/results/E026_dc/search_log.json \\
        --table-dir host_data/results/E026_dc --episodes 20 --workers 8 \\
        --out .mem/results/E026b_credit_dc.json

The start table of a file start (e.g. E025's) is read from its spec path, resolved
against --data-root ('data/' in the log -> host_data/); 'sp' and 'random:<seed>' are rebuilt.
The replay is checked against table_<start>.json in --table-dir.
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--src', default=os.path.join(os.path.dirname(__file__), '..', 'src'))
ap.add_argument('--config', required=True)
ap.add_argument('--load', type=float, default=None)
ap.add_argument('--search-log', required=True)
ap.add_argument('--table-dir', required=True, help='dir holding table_<start>.json for the replay check')
ap.add_argument('--start', default=None, help="start spec in the log (default: log['best']['start'])")
ap.add_argument('--episodes', type=int, default=20)
ap.add_argument('--seed0', type=int, default=20240601)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--failures', type=int, default=0)
ap.add_argument('--max-changes', type=int, default=None, help='smoke test: replay only the first n changes')
ap.add_argument('--out', default=None)
args = ap.parse_args()
assert args.seed0 + args.episodes <= 7000 or args.seed0 > 7100, "eval seeds overlap train seeds 7000+"
sys.path.insert(0, args.src)
sys.path.insert(0, os.path.join(args.src, 'maddpg_clean'))

from standalone_experiment_runner import StandaloneExperimentRunner  # noqa: E402

runner = StandaloneExperimentRunner(args.config, 0, '/tmp/table_change_credit')
ae = runner.config.get('attack_eval', {})
load = args.load if args.load is not None else float(ae.get('offered_load_factor', 2.0))
T_STEPS = runner.config['training']['timesteps_per_episode']
env = runner._make_attack_env(ae.get('hotspot') or None)
eng = env.engine
hosts, tr = eng.get_all_hosts(), eng.trainable_host_indices
A, ND = len(tr), eng.n_destinations
K = eng.n_actions // ND
access = list(eng.topology.access_nodes)


def n_live(host, dst):
    return min(len(eng.topology.kpath_cache.get((host, dst), [])), K)


def table_actions(tab):
    acts = []
    for a, i in enumerate(tr):
        v = np.zeros(eng.n_actions, dtype=np.float32)
        for d, dst in enumerate(access):
            k = tab[a][d]
            v[d * K + (k if k < n_live(hosts[i], dst) else 0)] = 1.0
        acts.append(v)
    return runner._build_full_actions(acts, eng.n_total_hosts, tr, eng.n_actions)


def share(flows):
    dl = sum(eng.flow_outcomes.get(f, (0, 0))[0] for f in flows)
    dr = sum(eng.flow_outcomes.get(f, (0, 0))[1] for f in flows)
    return 100.0 * dl / (dl + dr) if dl + dr else float('nan')


def episode(job):
    """(system PDR, [own PDR of agent a, entry PDR of (a, d)] per watched (a, d)) for one table/episode."""
    tab, e, watch = job
    seed = args.seed0 + e
    random.seed(seed); np.random.seed(seed)
    eng.topology.restore_intact()
    eng.reset_with_load(offered_load_factor=load)
    if args.failures:
        runner._inject_failures(eng, args.failures)
        eng.topology.refresh_path_cache()
    full = table_actions(tab)
    eng.decision_log = []
    for _ in range(T_STEPS):
        env.step(full)
    pdr = float(env.get_stats().get('end_to_end_pdr', 0.0))
    agent_of = {hosts[i]: a for a, i in enumerate(tr)}
    dix = {d: j for j, d in enumerate(access)}
    own, ent = {}, {}
    for r in eng.decision_log:
        if r['flow_id'] is None or r['host'] not in agent_of:
            continue
        a = agent_of[r['host']]
        own.setdefault(a, set()).add(r['flow_id'])
        ent.setdefault((a, dix[r['dst']]), set()).add(r['flow_id'])
    eng.decision_log = None
    return pdr, [(share(own.get(a, ())), share(ent.get((a, d), ()))) for a, d in watch]


def load_start(spec):
    if spec == 'sp':
        return [[0] * ND for _ in range(A)]
    if spec.startswith('random:'):
        rng = random.Random(int(spec.split(':')[1]))
        return [[rng.randrange(max(1, n_live(hosts[i], dst))) for dst in access] for i in tr]
    path = spec.replace('data/', 'host_data/', 1) if not os.path.exists(spec) else spec
    f = json.load(open(path))
    return [[int(f.get(str(a), {}).get(str(d), 0)) for d in range(ND)] for a in range(A)]


def paired(x, y):
    d = np.asarray(y) - np.asarray(x)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return None, None
    return round(float(d.mean()), 3), round(float(d.std(ddof=1) / np.sqrt(len(d))), 3)


if __name__ == '__main__':
    log = json.load(open(args.search_log))
    spec = args.start or log['best']['start']
    entry = next(s for s in log['starts'] if s['start'] == spec)
    changes = [(c, g) for sw in entry['sweeps'] for c, g in zip(sw['changed'], sw['gains_pp'])]
    if args.max_changes:
        changes = changes[:args.max_changes]
    tab = load_start(spec)
    tables = [tab]
    for (a, d, k_old, k_new), _ in changes:
        assert tables[-1][a][d] == k_old, f"replay mismatch at change {[a, d, k_old, k_new]}"
        nxt = [row[:] for row in tables[-1]]
        nxt[a][d] = k_new
        tables.append(nxt)
    name = os.path.basename(spec).replace('.json', '').replace(':', '')
    ref = json.load(open(os.path.join(args.table_dir, f'table_{name}.json')))
    assert args.max_changes or all(int(ref[str(a)][str(d)]) == tables[-1][a][d] for a in range(A) for d in range(ND)), \
        "replayed final table != table_<start>.json"
    watch_all = [(c[0], c[1]) for c, _ in changes]
    with mp.get_context('fork').Pool(args.workers) as pool:
        res = [pool.map(episode, [(t, e, watch_all) for e in range(args.episodes)]) for t in tables]
    sys_pdr = [np.array([r[0] for r in rr]) for rr in res]
    rows = []
    for i, ((a, d, k_old, k_new), g_search) in enumerate(changes):
        own = [np.array([r[1][i][0] for r in rr]) for rr in (res[i], res[i + 1])]
        ent = [np.array([r[1][i][1] for r in rr]) for rr in (res[i], res[i + 1])]
        sm, ss = paired(sys_pdr[i], sys_pdr[i + 1])
        om, os_ = paired(own[0], own[1])
        em, es = paired(ent[0], ent[1])
        rows.append({'i': i + 1, 'change': [a, d, k_old, k_new], 'search_gain_pp': g_search,
                     'system_gain_pp': sm, 'system_se': ss, 'own_gain_pp': om, 'own_se': os_,
                     'entry_gain_pp': em, 'entry_se': es})
        print(f"{i + 1:2d} {[a, d, k_old, k_new]} sys {sm:+.2f}±{ss:.2f}  own {om:+.2f}±{os_:.2f}  "
              f"entry {em if em is None else format(em, '+.2f')}±{es}", flush=True)
    out = {'config': args.config, 'load': load, 'start': spec, 'episodes': args.episodes, 'seed0': args.seed0,
           'pdr_start': round(float(sys_pdr[0].mean()), 3), 'pdr_end': round(float(sys_pdr[-1].mean()), 3),
           'changes': rows}
    print(f"PDR start {out['pdr_start']} -> end {out['pdr_end']}")
    if args.out:
        json.dump(out, open(args.out, 'w'), indent=1)
