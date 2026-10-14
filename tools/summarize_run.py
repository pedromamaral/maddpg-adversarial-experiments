#!/usr/bin/env python3
"""Summarise one results directory into .mem/results/<ID>.json and print a short table.

Reads JSON only, never logs. Understands the two result formats in host_data/results:

  idqn_control.py   <prefix>_seed<s>[_<tag>].json  with eval.fail<n>[_stale<d>].{pdr, least_loaded, chance, ...}
  run_eval.sh       <eval_dir>/n_<k>/damage_ceiling.json  with <rule>.mean_end_to_end_pdr

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
for f in sorted(glob.glob(os.path.join(run, '*_seed*.json'))):
    m = re.match(r'(.+)_seed(\d+)(?:_(.+))?\.json$', os.path.basename(f))
    if not m:
        continue
    d = json.load(open(f))
    if 'eval' not in d:
        continue
    prefix, seed, tag = m.group(1), m.group(2), m.group(3) or 'train'
    configs.add(d.get('config', '?'))
    seeds.add(int(seed))
    for cond, e in d['eval'].items():
        for k in ('pdr', 'least_loaded', 'chance'):
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
keys = [k for k in keys if k.endswith('/pdr') or 'least_loaded' in k or '/train/' in k]
print(f"{args.id} {out['artifacts']} sha={sha} seeds={out['seeds']} metrics={len(metrics)}")
for k in keys[:args.rows - 1]:
    m = metrics[k]
    rng = f" [{m['min']:.1f}-{m['max']:.1f}]" if m.get('n', 1) > 1 else ''
    print(f"  {k:<48} {m['final']:>7.2f}{rng}")
