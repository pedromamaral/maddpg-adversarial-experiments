#!/usr/bin/env python3
"""Regenerate every table and figure of Paper 1 (credit-assignment version) from results.

Reads only result files under host_data/results (copied from the server with
rsync, JSON only) and writes:

  paper/paper1_v2/tables/T*.tex   LaTeX table bodies, \\input by paper1_v2.tex
  paper/paper1_v2/figures/F*.pdf  figures (needs matplotlib; skipped without it)
  paper/paper1_v2/numbers.json    every number quoted in the text, for checking

All policies and rules at a given load and failure count were evaluated on the
same traffic (StandaloneExperimentRunner._attack_episodes reseeds the traffic
with seed 20240601), so differences are paired. Learned rows are means over the
three training seeds; the per-seed values are kept in numbers.json.

    .venv-plots/bin/python tools/paper1_tables.py
"""
import glob
import json
import os
import statistics as st
import sys

R = os.path.join(os.path.dirname(__file__), '..', 'host_data', 'results')
OUT = os.path.join(os.path.dirname(__file__), '..', 'paper', 'paper1_v2')
NUM = {}


def load(path):
    with open(os.path.join(R, path)) as f:
        return json.load(f)


def ceiling(run, n, sub='eval'):
    """{rule or 'policy': PDR} of a run_eval.sh damage-ceiling file."""
    p = os.path.join(R, run, sub, f'n_{n}', 'damage_ceiling.json')
    if not os.path.exists(p):
        return {}
    d = load(os.path.relpath(p, R))
    return {k: v['mean_end_to_end_pdr'] for k, v in d.items() if k != '_meta'}


def seeds(run, tag='', key='fail0', field='pdr'):
    """Per-seed values of an idqn_control result ({prefix}_seed<s>{tag}.json)."""
    vals = []
    for f in sorted(glob.glob(os.path.join(R, run, f'*_seed?{tag}.json'))):
        e = json.load(open(f))['eval'].get(key)
        if e and field in e:
            vals.append(e[field])
    return vals


def mean(v):
    return st.mean(v) if v else None


def fmt(x, nd=1, pct=False):
    if x is None:
        return '--'
    return f'{100 * x:.{nd}f}' if pct else f'{x:.{nd}f}'


def write_table(name, header, rows, note=None):
    os.makedirs(os.path.join(OUT, 'tables'), exist_ok=True)
    lines = [r'\toprule', ' & '.join(header) + r' \\', r'\midrule']
    for r in rows:
        lines.append(r'\midrule' if r == 'MID' else ' & '.join(r) + r' \\')
    lines.append(r'\bottomrule')
    if note:
        lines.append('% ' + note)
    with open(os.path.join(OUT, 'tables', f'{name}.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ── T1: the v1 policies (step reward, sigmoid actor, joint critic) ───────────────
attr = load('policy_attribution.json')
tel = load('telemetry_reliance_summary.json')
rows = []
for v in ['CC-Simple', 'CC-Duelling', 'LC-Simple', 'LC-Duelling',
          'CC-Simple-GNN', 'CC-Duelling-GNN', 'LC-Duelling-GNN']:
    a = attr[v]['fail0']
    pdr = load(f'ceil2x/{v}/damage_ceiling.json')['policy']['mean_end_to_end_pdr']
    drop = tel[v]['ablate_shuffle']['drop_pp']   # PDR lost when path telemetry is shuffled (pp)
    NUM[f'v1/{v}'] = dict(pdr=pdr, static=a['static_share_live'], ll=a['greedy_agreement'],
                          chance=a['greedy_chance'], sat=a['saturation_live'], shuffle_drop=drop)
    rows.append([v, fmt(pdr), fmt(a['static_share_live'], 0, True), fmt(a['greedy_agreement'], 0, True),
                 fmt(a['greedy_chance'], 0, True), fmt(a['saturation_live'], 0, True), f'{drop:.2f}'])
rules2x = {k: v['mean_end_to_end_pdr'] for k, v in load('ceil2x/CC-Simple/damage_ceiling.json').items()
           if k != '_meta'}
NUM['rules_2x_k1_packet'] = rules2x
rows += ['MID'] + [[n, fmt(rules2x[k]), '--', '--', '--', '--', '--']
                   for n, k in [('Greedy', 'greedy'), ('Random (per packet)', 'random'),
                                ('Shortest path', 'sp'), ('Worst path', 'worst')]]
write_table('T1_v1', ['Policy', 'PDR', 'Static', 'Least-loaded', 'Chance', 'Saturated', 'Shuffle loss'], rows)

# ── T2: baseline realism (2x, original traffic granularity) ──────────────────────
re0 = {**ceiling('v2_pilot_lc', 0, 'eval_realism'), **ceiling('v2_pilot_lc', 0, 'eval_realism2')}
re4 = {**ceiling('v2_pilot_lc', 4, 'eval_realism'), **ceiling('v2_pilot_lc', 4, 'eval_realism2')}
NUM['realism_2x'] = {'n0': re0, 'n4': re4}
spec = [('Random', 'per packet', 'random'), ('Random (ECMP)', 'per flow', 'random+flow'),
        ('Greedy', 'per packet', 'greedy'), ('Greedy', 'per flow', 'greedy+flow'),
        ('Greedy, 1 step old', 'per packet', 'greedy_stale1'),
        ('Greedy, 2 steps old', 'per packet', 'greedy_stale2'),
        ('Greedy, 4 steps old', 'per packet', 'greedy_stale4'),
        ('Greedy, 8 steps old', 'per packet', 'greedy_stale8'),
        ('Greedy, 2 steps old', 'per flow', 'greedy_stale2+flow'),
        ('Greedy, 4 steps old', 'per flow', 'greedy_stale4+flow')]
write_table('T2_realism', ['Rule', 'Path choice', '$n=0$', '$n=4$'],
            [[a, b, fmt(re0.get(k)), fmt(re4.get(k))] for a, b, k in spec])

# ── T3: traffic granularity sweep (pinned flows) ─────────────────────────────────
gran = []
for f in sorted(glob.glob(os.path.join(R, 'granularity', 'k*.log'))):
    gran += [json.loads(l) for l in open(f) if l.startswith('{')]
gran.sort(key=lambda r: (r['load'], r['k']))
NUM['granularity'] = gran
rows = []
for g in gran:
    rows.append([f"{g['load']:g}$\\times$", str(g['k']), f"{g['flows_per_step']:.1f}",
                 f"{g['active_decisions_pct']:.1f}", fmt(g['sp']), fmt(g['random']), fmt(g['greedy']),
                 fmt(g['greedy_stale2']), fmt(g['greedy_stale4']), fmt(g['worst'])])
write_table('T3_granularity', ['Load', '$k$', 'Flows/step', 'Live (\\%)', 'SP', 'ECMP', 'Greedy',
                               'Greedy$_{2}$', 'Greedy$_{4}$', 'Worst'], rows)

# ── T4: step-reward attempts (pilots) ────────────────────────────────────────────
def attribution(run, var):
    p = os.path.join(R, run, 'eval', f'{var}_attribution.json')
    if not os.path.exists(p):
        return None, None
    a = json.load(open(p))[var]['fail0']
    return a['greedy_agreement'], a['greedy_chance']


pil = [('v1 local critic (sigmoid, joint critic)', None, 'LC-Simple'),
       ('Softmax actor + factored critic, LC', 'v2_pilot_lc', 'LC-Simple'),
       ('Softmax actor + factored critic, CC', 'v2_pilot_cc', 'CC-Simple'),
       ('+ entropy 0.01, LC', 'v2_p2_lc_e01', 'LC-Simple'),
       ('+ entropy 0.05, LC', 'v2_p2_lc_e05', 'LC-Simple'),
       ('+ observation critic, CC', 'v2_p2_cc', 'CC-Simple'),
       ('+ per-agent credit $w{=}0.5$, LC', 'v2_p3_lc_w05', 'LC-Simple'),
       ('+ per-agent credit $w{=}1$, LC', 'v2_p3_lc_w10', 'LC-Simple'),
       ('+ per-agent credit $w{=}0.5$, CC', 'v2_p3_cc', 'CC-Simple')]
rows = []
for name, run, var in pil:
    if run is None:
        p0 = load('ceil2x/LC-Simple/damage_ceiling.json')['policy']['mean_end_to_end_pdr']
        p8 = None
        ll, ch = NUM['v1/LC-Simple']['ll'], NUM['v1/LC-Simple']['chance']
    else:
        p0, p8 = ceiling(run, 0).get('policy'), ceiling(run, 8).get('policy')
        ll, ch = attribution(run, var)
    NUM[f'pilot/{run or "v1"}'] = dict(pdr0=p0, pdr8=p8, ll=ll, chance=ch)
    rows.append([name, fmt(p0), fmt(p8), fmt(ll, 0, True), fmt(ch, 0, True)])
r0, r8 = ceiling('v2_pilot_lc', 0), ceiling('v2_pilot_lc', 8)
rows += ['MID', ['Random (per packet)', fmt(r0['random']), fmt(r8['random']), '--', '--'],
         ['Greedy', fmt(r0['greedy']), fmt(r8['greedy']), '--', '--']]
write_table('T4_pilots', ['Configuration', '$n=0$', '$n=8$', 'Least-loaded', 'Chance'], rows)

# ── realistic regime (3x, pinned, k=4): rules ────────────────────────────────────
NS = [0, 2, 4, 6, 8]
rules3 = {n: ceiling('v2_p4_lc', n, 'eval_3x') for n in NS}
NUM['rules_3x'] = rules3
fc = load('v2_p4_lc/flow_choice_3x.json')
NUM['flow_choice_3x'] = fc


def learner(run, tag=''):
    out = {}
    for n in NS:
        v = seeds(run, tag, f'fail{n}')
        if v:
            out[n] = v
    for k in ('fail0_stale2', 'fail4_stale2'):
        v = seeds(run, tag, k)
        if v:
            out[k] = v
    ll = seeds(run, tag, 'fail0', 'least_loaded')
    ch = seeds(run, tag, 'fail0', 'chance')
    out['ll'], out['chance'] = mean(ll), mean(ch)
    return out


L = {k: learner(r) for k, r in [
    ('dqn_step', 'idqn_p4_step'), ('dqn_flow', 'idqn_p4'), ('dqn_fail', 'idqn_p4_fail'),
    ('dqn_rh', 'idqn_p4_rh'), ('dqn_rhf', 'idqn_p4_rhf'),
    ('ac_lc_fix', 'ac_local_p4'), ('ac_cc_fix', 'ac_central_p4'),
    ('ac_lc', 'ac_local_rhf'), ('ac_cc', 'ac_central_rhf'),
    ('ac_lc_duel', 'ac_local_duel_rhf'), ('ac_cc_duel', 'ac_central_duel_rhf'),
    ('ac_lc_gnn', 'ac_local_gnns_rhf'), ('ac_cc_gnn', 'ac_central_gnns_rhf'),
    ('ac_lc_gnn_orig', 'ac_local_gnn_rhf'), ('ac_cc_gnn_orig', 'ac_central_gnn_rhf')]}
NUM['learners'] = L


def m(d, k):
    return mean(d.get(k, []))


# ── T5: algorithm x credit (fixed hotspot, no failures in training) ──────────────
p4 = {n: rules3[n].get('policy') for n in NS}
rows = [['MADDPG-LC', 'per-agent step', fmt(p4[0]), fmt(p4[4]), fmt(p4[8]),
         fmt(fc['fail0']['policy']['least_loaded'], 0, True), fmt(fc['fail0']['policy']['chance'], 0, True)],
        ['DQN', 'per-agent step', fmt(m(L['dqn_step'], 0)), fmt(m(L['dqn_step'], 4)), fmt(m(L['dqn_step'], 8)),
         fmt(L['dqn_step']['ll'], 0, True), fmt(L['dqn_step']['chance'], 0, True)]]
for name, k in [('MADDPG-LC', 'ac_lc_fix'), ('MADDPG-CC', 'ac_cc_fix'), ('DQN', 'dqn_flow')]:
    rows.append([name, 'per decision', fmt(m(L[k], 0)), fmt(m(L[k], 4)), fmt(m(L[k], 8)),
                 fmt(L[k]['ll'], 0, True), fmt(L[k]['chance'], 0, True)])
rows.append('MID')
for name, k in [('Greedy', 'greedy'), ('Greedy, 2 steps old', 'greedy_stale2'),
                ('Greedy, 4 steps old', 'greedy_stale4'), ('ECMP', 'random'), ('Shortest path', 'sp')]:
    fck = {'greedy': 'greedy', 'greedy_stale2': 'greedy_stale2', 'greedy_stale4': 'greedy_stale4',
           'random': 'random'}.get(k)
    llr = fc['fail0'].get(fck, {}) if fck else {}
    rows.append([name, '--', fmt(rules3[0].get(k)), fmt(rules3[4].get(k)), fmt(rules3[8].get(k)),
                 fmt(llr.get('least_loaded'), 0, True), fmt(llr.get('chance'), 0, True)])
write_table('T5_credit', ['Learner', 'Credit', '$n=0$', '$n=4$', '$n=8$', 'Least-loaded', 'Chance'], rows)

# ── T6: architecture (changing hotspot + failures in training) ───────────────────
shA = {n: ceiling('v2_p4_lc', n, 'eval_shiftA') for n in (0, 4)}
shB = {n: ceiling('v2_p4_lc', n, 'eval_shiftB') for n in (0, 4)}
NUM['rules_shift'] = {'A': shA, 'B': shB}
rows = []
for name, k, run in [('LC, simple', 'ac_lc', 'ac_local_rhf'), ('CC, simple', 'ac_cc', 'ac_central_rhf'),
                     ('LC, duelling', 'ac_lc_duel', 'ac_local_duel_rhf'),
                     ('CC, duelling', 'ac_cc_duel', 'ac_central_duel_rhf'),
                     ('GNN + LC', 'ac_lc_gnn', 'ac_local_gnns_rhf'), ('GNN + CC', 'ac_cc_gnn', 'ac_central_gnns_rhf'),
                     ('GNN + LC, unstable', 'ac_lc_gnn_orig', 'ac_local_gnn_rhf'),
                     ('GNN + CC, unstable', 'ac_cc_gnn_orig', 'ac_central_gnn_rhf'),
                     ('DQN (control)', 'dqn_rhf', 'idqn_p4_rhf')]:
    d = L[k]
    a0, a4 = mean(seeds(run, '_shiftA', 'fail0')), mean(seeds(run, '_shiftA', 'fail4'))
    b0, b4 = mean(seeds(run, '_shiftB', 'fail0')), mean(seeds(run, '_shiftB', 'fail4'))
    NUM[f'shift/{k}'] = dict(A0=a0, A4=a4, B0=b0, B4=b4)
    rows.append([name, fmt(m(d, 0)), fmt(m(d, 4)), fmt(m(d, 8)), fmt(m(d, 'fail0_stale2')),
                 fmt(a0), fmt(a4), fmt(b0), fmt(b4)])
rows.append('MID')
for name, k in [('Greedy', 'greedy'), ('Greedy, 2 steps old', 'greedy_stale2'), ('ECMP', 'random'),
                ('Shortest path', 'sp')]:
    rows.append([name, fmt(rules3[0].get(k)), fmt(rules3[4].get(k)), fmt(rules3[8].get(k)),
                 '--',
                 fmt(shA[0].get(k)), fmt(shA[4].get(k)), fmt(shB[0].get(k)), fmt(shB[4].get(k))])
write_table('T6_architecture', ['MADDPG variant', '$n=0$', '$n=4$', '$n=8$', 'Stale$_2$',
                                'A, $0$', 'A, $4$', 'B, $0$', 'B, $4$'], rows)

# ── T7: what is learned (observation ablations) ──────────────────────────────────
rows = []
for name, k, run in [('DQN, fixed hotspot', 'dqn_flow', 'idqn_p4'),
                     ('DQN, changing hotspot', 'dqn_rh', 'idqn_p4_rh'),
                     ('MADDPG-LC', 'ac_lc', 'ac_local_rhf'), ('MADDPG-CC', 'ac_cc', 'ac_central_rhf'),
                     ('MADDPG-CC, duelling', 'ac_cc_duel', 'ac_central_duel_rhf'),
                     ('GNN + CC', 'ac_cc_gnn', 'ac_central_gnns_rhf')]:
    real0, real4 = m(L[k], 0), m(L[k], 4)
    sa0, sa4 = mean(seeds(run, '_shuffled_all', 'fail0')), mean(seeds(run, '_shuffled_all', 'fail4'))
    sp0 = mean(seeds(run, '_shuffled', 'fail0'))
    sn0 = mean(seeds(run, '_shuffled_neighbours', 'fail0'))
    lls = mean(seeds(run, '_shuffled_all', 'fail0', 'least_loaded'))
    NUM[f'ablation/{k}'] = dict(real0=real0, real4=real4, all0=sa0, all4=sa4, path0=sp0, nb0=sn0,
                               ll=L[k]['ll'], ll_shuffled=lls, chance=L[k]['chance'])
    rows.append([name, fmt(real0), fmt(sp0), fmt(sn0), fmt(sa0), fmt(real4), fmt(sa4),
                 fmt(L[k]['ll'], 0, True), fmt(lls, 0, True), fmt(L[k]['chance'], 0, True)])
write_table('T7_ablation', [r' & \multicolumn{4}{c}{$n=0$} & \multicolumn{2}{c}{$n=4$} & \multicolumn{3}{c}{Least-loaded} \\ Learner', 'Real', 'Path$^{s}$', 'Neigh.$^{s}$', 'All$^{s}$', 'Real', 'All$^{s}$',
                            'LL', 'LL$^{s}$', 'Chance'], rows)

# ── T8: generalisation to unseen demand ──────────────────────────────────────────
rows = []
for name, k, run in [('DQN, fixed hotspot', 'dqn_flow', 'idqn_p4'),
                     ('DQN, fixed hotspot + failures', 'dqn_fail', 'idqn_p4_fail'),
                     ('DQN, changing hotspot', 'dqn_rh', 'idqn_p4_rh'),
                     ('DQN, changing hotspot + failures', 'dqn_rhf', 'idqn_p4_rhf'),
                     ('MADDPG-LC, fixed hotspot', 'ac_lc_fix', 'ac_local_p4'),
                     ('MADDPG-CC, fixed hotspot', 'ac_cc_fix', 'ac_central_p4'),
                     ('MADDPG-LC, changing hotspot + failures', 'ac_lc', 'ac_local_rhf'),
                     ('MADDPG-CC, changing hotspot + failures', 'ac_cc', 'ac_central_rhf')]:
    vals = [m(L[k], 0), m(L[k], 4)] + [mean(seeds(run, t, f'fail{n}')) for t in ('_shiftA', '_shiftB')
                                        for n in (0, 4)]
    NUM[f'generalisation/{k}'] = vals
    rows.append([name] + [fmt(x) for x in vals])
rows.append('MID')
for name, k in [('Greedy', 'greedy'), ('Greedy, 2 steps old', 'greedy_stale2'), ('ECMP', 'random'),
                ('MADDPG-LC, step reward', 'policy')]:
    rows.append([name, fmt(rules3[0].get(k)), fmt(rules3[4].get(k)), fmt(shA[0].get(k)), fmt(shA[4].get(k)),
                 fmt(shB[0].get(k)), fmt(shB[4].get(k))])
write_table('T8_generalisation', ['Learner and training demand', 'Orig., $0$', 'Orig., $4$', 'A, $0$',
                                  'A, $4$', 'B, $0$', 'B, $4$'], rows)

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'numbers.json'), 'w') as f:
    json.dump(NUM, f, indent=1, default=str)

# ── figures ──────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print('matplotlib not available: tables and numbers written, figures skipped')
    sys.exit(0)
os.makedirs(os.path.join(OUT, 'figures'), exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'pdf.fonttype': 42})

# F1: granularity
fig, axs = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
for ax, load_ in zip(axs, (2.0, 3.0)):
    g = [r for r in gran if r['load'] == load_]
    ks = [r['k'] for r in g]
    for key, lab, sty in [('greedy', 'Greedy', 'C0-o'), ('greedy_stale4', 'Greedy, 4 steps old', 'C0--^'),
                          ('random', 'ECMP', 'C1-s'), ('sp', 'Shortest path', 'k:d')]:
        ax.plot(ks, [r[key] for r in g], sty, label=lab, ms=4)
    ax.set_xscale('log', base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel('flow granularity $k$'); ax.set_title(f'{load_:g}$\\times$ load')
axs[0].set_ylabel('PDR (%)'); axs[0].legend(fontsize=7, loc='lower right')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'figures', 'F1_granularity.pdf')); plt.close(fig)

# F2: algorithm x credit vs failures
fig, ax = plt.subplots(figsize=(3.5, 2.8))
series = [('Greedy', [rules3[n]['greedy'] for n in NS], 'C0-o'),
          ('Greedy, 2 steps old', [rules3[n]['greedy_stale2'] for n in NS], 'C0--^'),
          ('ECMP', [rules3[n]['random'] for n in NS], 'C1-s'),
          ('MADDPG-LC, step reward', [p4[n] for n in NS], 'C3:x'),
          ('DQN, step reward', [m(L['dqn_step'], n) for n in NS], 'C7:+')]
for lab, k, sty in [('MADDPG-LC, per decision', 'ac_lc_fix', 'C2-D'), ('DQN, per decision', 'dqn_flow', 'C4-v')]:
    if L[k].get(0):
        series.append((lab, [m(L[k], n) for n in NS], sty))
for lab, ys, sty in series:
    ax.plot(NS, ys, sty, label=lab, ms=4)
ax.set_xlabel('failed links $n$'); ax.set_ylabel('PDR (%)'); ax.legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'figures', 'F2_credit.pdf')); plt.close(fig)

# F3: generalisation (n=0)
fig, ax = plt.subplots(figsize=(7, 2.6))
groups = [('DQN\nfixed', 'dqn_flow', 'idqn_p4'), ('DQN\nchanging', 'dqn_rh', 'idqn_p4_rh'),
          ('MADDPG-CC\nchanging+fail', 'ac_cc', 'ac_central_rhf')]
labels, vals = [], []
for lab, k, run in groups:
    labels.append(lab)
    vals.append([m(L[k], 0), mean(seeds(run, '_shiftA', 'fail0')), mean(seeds(run, '_shiftB', 'fail0'))])
for lab, key in [('Greedy', 'greedy'), ('Greedy\n2 steps old', 'greedy_stale2'), ('ECMP', 'random')]:
    labels.append(lab)
    vals.append([rules3[0][key], shA[0].get(key), shB[0].get(key)])
import numpy as np
x = np.arange(len(labels))
for i, (cond, col) in enumerate([('trained pattern', 'C0'), ('shift A', 'C1'), ('shift B', 'C2')]):
    ax.bar(x + (i - 1) * 0.27, [v[i] for v in vals], 0.27, label=cond, color=col)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7); ax.set_ylim(85, 98)
ax.set_ylabel('PDR (%)'); ax.legend(fontsize=7, ncol=3, loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'figures', 'F3_generalisation.pdf')); plt.close(fig)
print('tables, figures and numbers written to', os.path.relpath(OUT))
