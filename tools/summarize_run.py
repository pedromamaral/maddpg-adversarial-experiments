#!/usr/bin/env python3
"""Summarise one results directory into .mem/results/<ID>.json and print a short table.

Reads JSON only, never logs. Understands the three result formats in host_data/results:

  idqn_control.py       <prefix>_seed<s>[_<tag>].json  with eval.fail<n>[_stale<d>].{pdr, least_loaded, chance, ...}
  run_eval.sh           <eval_dir>/n_<k>/damage_ceiling.json  with <rule>.mean_end_to_end_pdr
  supervised_signal.py  <config>.json  with <rule>.{own_delivery_pct, se} and agree_greedy
  static_table_search.py search_log.json with starts[].{start, J0, J, evals, sweeps} (works on partial logs)

Learner metrics are averaged over seeds (min-max kept); rule rows are per n.

    python tools/summarize_run.py host_data/results/ac_local_p4 --id E013
    python tools/summarize_run.py host_data/results/v2_p4_lc --id E010 --no-write
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import subprocess
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('run_dir')
ap.add_argument('--id', required=True, help='experiment id, e.g. E013')
ap.add_argument('--host', default='server')
ap.add_argument('--no-write', action='store_true')
ap.add_argument('--rows', type=int, default=15)
args = ap.parse_args()

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
run = os.path.normpath(args.run_dir)
metrics = {}            # name -> {final, best, best_step, min, max, n}
configs, seeds = set(), set()


def add(name, values):
    values = [v for v in values if v is not None]
    if not values:
        return
    metrics[name] = {'final': round(st.mean(values), 3), 'best': round(max(values), 3),
                     'best_step': None, 'min': round(min(values), 3),
                     'max': round(max(values), 3), 'n': len(values)}


# idqn_control.py learner files
learner = defaultdict(list)
for f in sorted(glob.glob(os.path.join(run, '**', '*_seed*.json'), recursive=True)):
    if os.path.basename(f).startswith('table_'):
        continue
    m = re.match(r'(.+)_seed(\d+)(?:_(.+))?\.json$', os.path.basename(f))
    if not m:
        continue
    d = json.load(open(f))
    if 'eval' not in d:
        continue
    prefix, seed, tag = m.group(1), m.group(2), m.group(3) or 'train'
    sub = re.sub(r'_s\d+$', '', os.path.relpath(os.path.dirname(f), run))   # E032: cell dirs <name>_s<seed>
    if sub != '.':
        prefix = f'{sub}/{prefix}'
    configs.add(d.get('config', '?'))
    seeds.add(int(seed))
    for cond, e in d['eval'].items():
        for k in ('pdr', 'least_loaded', 'chance', 'argmax_off_sp', 'fallback_override'):
            if k in e:
                learner[f'{prefix}/{tag}/{cond}/{k}'].append(e[k] * (100 if k != 'pdr' else 1))
    hist = d.get('history') or []
    if hist and 'pdr' in hist[-1]:
        learner[f'{prefix}/train_curve/pdr_last'].append(hist[-1]['pdr'])
for k, v in learner.items():
    add(k, v)

# run_eval.sh rule/policy evaluations
for f in sorted(glob.glob(os.path.join(run, '**', 'damage_ceiling.json'), recursive=True)):
    rel = os.path.relpath(os.path.dirname(f), run)
    d = json.load(open(f))
    for rule, r in d.items():
        if rule.startswith('_') or not isinstance(r, dict):
            continue
        if 'mean_end_to_end_pdr' in r:
            add(f'{rel}/{rule}/pdr', [r['mean_end_to_end_pdr']])

# supervised_signal.py: <config>.json with <rule>.{own_delivery_pct, se} and agree_greedy
for f in sorted(glob.glob(os.path.join(run, '*.json'))):
    d = json.load(open(f))
    if not isinstance(d, dict) or 'agree_greedy' not in d:
        continue
    stem = os.path.splitext(os.path.basename(f))[0]
    configs.add(d.get('config', '?'))
    for rule, r in d.items():
        if isinstance(r, dict) and 'own_delivery_pct' in r:
            add(f'{stem}/{rule}/own_delivery_pct', [r['own_delivery_pct']])
            metrics[f'{stem}/{rule}/own_delivery_pct']['se'] = r['se']
    for rule, v in d['agree_greedy'].items():
        add(f'{stem}/agree_greedy/{rule}', [100 * v])
    for k in ('train_decisions', 'test_decisions'):
        if k in d:
            add(f'{stem}/{k}', [d[k]])

# static_table_search.py: search_log.json, one metric set per start (system PDR, %)
sl = os.path.join(run, 'search_log.json')
if os.path.exists(sl):
    d = json.load(open(sl))
    configs.add(d.get('config', '?'))
    for s in d.get('starts', []):
        name = os.path.basename(s['start']).replace('.json', '')
        add(f'search/{name}/J0', [s.get('J0')])
        add(f'search/{name}/J', [s.get('J')])
        if s.get('J') is not None and s.get('J0') is not None:
            add(f'search/{name}/gain_pp', [s['J'] - s['J0']])
        add(f'search/{name}/evals', [s.get('evals')])
        add(f'search/{name}/sweeps', [len(s.get('sweeps', []))])
        if s.get('J0_se') is not None:
            metrics[f'search/{name}/J0']['se'] = s['J0_se']

# step-reward training curves
tr = os.path.join(run, 'phase1_training_results.json')
if os.path.exists(tr):
    for v, r in json.load(open(tr)).items():
        if isinstance(r, dict) and r.get('best_checkpoint_metric') is not None:
            metrics[f'{v}/train/{r.get("best_checkpoint_metric_name", "metric")}'] = {
                'final': r.get('final_reward'), 'best': r['best_checkpoint_metric'],
                'best_step': r.get('best_checkpoint_epoch')}

try:
    sha = subprocess.check_output(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'], text=True).strip()
except Exception:
    sha = '?'
files = glob.glob(os.path.join(run, '**', '*.json'), recursive=True)
started = min((os.path.getmtime(f) for f in files), default=None)
out = {'id': args.id, 'sha': sha, 'config': sorted(configs) or None, 'host': args.host,
       'started': None if started is None else __import__('datetime').datetime.fromtimestamp(started).strftime('%Y-%m-%d %H:%M'),
       'status': 'finished' if metrics else 'empty', 'seeds': sorted(seeds),
       'metrics': metrics, 'artifacts': os.path.relpath(run, ROOT)}

if not args.no_write:
    dst = os.path.join(ROOT, '.mem', 'results', f'{args.id}.json')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(out, open(dst, 'w'), indent=1)

# Short table: headline metrics first (fail0/fail4/fail8 PDR, n_0 rules)
keys = sorted(metrics, key=lambda k: (not re.search(r'(fail0|fail4|fail8|n_0)/', k) and not k.endswith('/pdr'), k))
keys = [k for k in keys if k.endswith('/pdr') or 'least_loaded' in k or '/train/' in k
        or k.startswith('search/') or k.endswith('/own_delivery_pct') or '/agree_greedy/' in k]
print(f"{args.id} {out['artifacts']} sha={sha} seeds={out['seeds']} metrics={len(metrics)}")
for k in keys[:args.rows - 1]:
    m = metrics[k]
    rng = f" [{m['min']:.1f}-{m['max']:.1f}]" if m.get('n', 1) > 1 else ''
    se = f" ±{m['se']:.2f}" if 'se' in m else ''
    print(f"  {k:<48} {m['final']:>7.2f}{rng}{se}")
