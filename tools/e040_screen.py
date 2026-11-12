#!/usr/bin/env python3
"""E040 stage 1 screen (D29): per graph pick the operating load from ECMP PDR only, then apply the go rule.

Load rule (fixed before looking at greedy): among loads with ECMP PDR in [80, 92], the one closest to 86;
none in band -> graph out. Go rule: greedy - ECMP >= 3 pp AND own_share_of_system_gain >= 1 (own gain >= system gain).
Calibration rows (sp86_L3, deltacom_L8) are printed as reference (expected: SP-86 aligned, Deltacom ~0.42).

    python tools/e040_screen.py host_data/results/E040 [--json .mem/results/E040_screen.json]
"""
import glob, json, os, re, sys

d = sys.argv[1]
rows = {}
for f in glob.glob(os.path.join(d, '*_L*.json')):
    m = re.match(r'(.+)_L([\d.]+)\.json$', os.path.basename(f))
    rows.setdefault(m.group(1), []).append((float(m.group(2)), json.load(open(f))))
out = []
print(f"{'graph':10s} {'load':>4s} {'ECMP':>6s} {'greedy':>6s} {'g-E':>5s} {'own':>5s} {'share':>6s}  verdict")
for g in sorted(rows):
    cand = [(L, r) for L, r in rows[g] if 80 <= r['pdr_ecmp'] <= 92]
    calib = g in ('sp86', 'deltacom')
    if not cand and not calib:
        print(f"{g:10s} ECMP out of band at all loads: " + ', '.join(f"{L:g}x {r['pdr_ecmp']}" for L, r in sorted(rows[g])))
        out.append({'graph': g, 'verdict': 'no load in band'})
        continue
    L, r = min(cand or rows[g], key=lambda x: abs(x[1]['pdr_ecmp'] - 86))
    share = r['own_share_of_system_gain']
    go = r['system_gain_pp'] >= 3 and share is not None and share >= 1
    v = 'calibration' if calib else ('GO' if go else 'no-go')
    print(f"{g:10s} {L:4g} {r['pdr_ecmp']:6.2f} {r['pdr_greedy']:6.2f} {r['system_gain_pp']:5.2f} "
          f"{r['own_gain_pp_contested']:5.2f} {share if share is not None else float('nan'):6.2f}  {v}")
    out.append({'graph': g, 'load': L, 'verdict': v, **{k: r[k] for k in (
        'pdr_ecmp', 'pdr_greedy', 'system_gain_pp', 'own_gain_pp_contested', 'own_share_of_system_gain',
        'own_flow_effect_pp', 'own_flow_effect_se_pp')}})
if '--json' in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)
