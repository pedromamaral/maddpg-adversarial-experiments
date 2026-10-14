#!/usr/bin/env python3
"""Is the own-flow value of a path choice learnable from the agents' observations?

The learners with per-decision credit end up as fixed routing tables. Before
blaming the reinforcement learning, we ask whether plain supervised regression
can extract a better-than-table path choice from the same observations and the
same reward (the delivered share of the decision's own flow).

Data: applied, contested path decisions logged under ECMP (uniformly random path,
pinned per flow), with the deciding agent's observation at that step and the
flow's delivered share r. Training and test episodes use disjoint traffic seeds.

Because ECMP picks among the n available paths uniformly, the own-flow delivery
of any decision rule pi can be estimated without bias on the held-out decisions
by inverse propensity scoring:  V(pi) = mean( r * n * 1[pi(x) == chosen path] ).

Rules compared (each fitted on the training decisions only):
  ecmp      the logging policy itself (mean r)
  greedy    least-utilised path (the U_i slots of the observation)
  table     best path per (agent, destination) by mean r in training
  util      small regressor on the destination's K path utilisations (+ ids)
  full      the learners' Q-network shape on the full observation (+ agent id)
with bootstrap standard errors.

    python tools/supervised_signal.py --config configs/v2_deltacom.json --train-episodes 150
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument('--src', default=os.path.join(os.path.dirname(__file__), '..', 'src'))
ap.add_argument('--config', required=True)
ap.add_argument('--train-episodes', type=int, default=150)
ap.add_argument('--test-episodes', type=int, default=40)
ap.add_argument('--epochs', type=int, default=40)
ap.add_argument('--out', default=None)
args = ap.parse_args()
sys.path.insert(0, args.src)
sys.path.insert(0, os.path.join(args.src, 'maddpg_clean'))
torch.set_num_threads(4)

from standalone_experiment_runner import StandaloneExperimentRunner  # noqa: E402

TAIL = 20
runner = StandaloneExperimentRunner(args.config, 0, '/tmp/supsig')
ae = runner.config.get('attack_eval', {})
load = float(ae.get('offered_load_factor', 2.0))
T = runner.config['training']['timesteps_per_episode']
env = runner._make_attack_env(ae.get('hotspot') or None)
eng = env.engine
assert eng.pin_paths_default, "needs traffic.pin_paths_per_flow"
hosts, tr = eng.get_all_hosts(), eng.trainable_host_indices
A, nd = len(tr), eng.n_destinations
K = eng.n_actions // nd
agent_of = {hosts[i]: a for a, i in enumerate(tr)}
dst_idx = {d: j for j, d in enumerate(eng.topology.access_nodes)}


def collect(n_eps, seed):
    random.seed(seed); np.random.seed(seed)
    rng = random.Random(seed + 999)
    X, rows = [], []
    for _ in range(n_eps):
        eng.topology.restore_intact()
        eng.reset_with_load(offered_load_factor=load)
        eng.decision_log = []
        states, seen = [eng.get_state(h) for h in hosts], {}
        for _ in range(T):
            seen[eng.time_step + 1] = [states[i] for i in tr]
            acts = [runner._rule_action(hosts[i], 'random', eng, rng) for i in tr]
            states, _, _ = env.step(runner._build_full_actions(acts, eng.n_total_hosts, tr, eng.n_actions))
        for d in eng.decision_log:
            u = np.asarray(d['utils'])
            if (d['flow_id'] is None or d['step'] > T - TAIL or d['host'] not in agent_of
                    or d['n_distinct'] < 2 or u.max() - u.min() <= 1e-9 or d['k'] >= len(u)):
                continue
            dl, dr = eng.flow_outcomes.get(d['flow_id'], (0, 0))
            if dl + dr == 0:
                continue
            a = agent_of[d['host']]
            X.append(np.asarray(seen[d['step']][a], dtype=np.float32))
            uu = np.full(K, 1.0, dtype=np.float32); uu[:len(u)] = u
            rows.append((a, dst_idx[d['dst']], d['k'], len(u), dl / (dl + dr), uu))
        eng.decision_log = None
    return np.stack(X), rows


Xtr, Rtr = collect(args.train_episodes, 1000)
Xte, Rte = collect(args.test_episodes, 2000)


def cols(R):
    a = np.array([r[0] for r in R]); d = np.array([r[1] for r in R]); k = np.array([r[2] for r in R])
    n = np.array([r[3] for r in R]); y = np.array([r[4] for r in R], dtype=np.float32)
    u = np.stack([r[5] for r in R])
    return a, d, k, n, y, u


atr, dtr, ktr, ntr, ytr, utr = cols(Rtr)
ate, dte, kte, nte, yte, ute = cols(Rte)


def ips(choice):
    """Per-decision IPS terms of a rule's choices on the test decisions."""
    return yte * nte * (choice == kte)


def boot(v, B=500):
    rng = np.random.default_rng(0)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return float(np.mean(v)), float(np.std(m))


res = {'config': args.config, 'load': load, 'train_decisions': len(ytr), 'test_decisions': len(yte)}
masked = lambda q, n: np.where(np.arange(K)[None, :] < n[:, None], q, -np.inf)

# ecmp and greedy
res['ecmp'] = boot(yte)
res['greedy'] = boot(ips(np.argmin(masked(-ute, nte) * -1 if False else np.where(np.arange(K)[None, :] < nte[:, None], ute, np.inf), 1)))

# table: best path per (agent, destination), shrunk towards the destination mean
tab = {}
for a, d, k, y in zip(atr, dtr, ktr, ytr):
    tab.setdefault((a, d), [[0.0, 0] for _ in range(K)])
    tab[(a, d)][k][0] += y; tab[(a, d)][k][1] += 1
prior = float(ytr.mean())
choice = []
for a, d, n in zip(ate, dte, nte):
    s = tab.get((a, d))
    if s is None:
        choice.append(0); continue
    means = [(s[k][0] + 5 * prior) / (s[k][1] + 5) for k in range(n)]
    choice.append(int(np.argmax(means)))
res['table'] = boot(ips(np.array(choice)))


def fit(inp_tr, inp_te, out_dim, sel_tr, sel_te, hidden=(256, 128)):
    """Regress r on the chosen output; return the test argmax over available paths."""
    torch.manual_seed(0)
    layers, dim = [], inp_tr.shape[1]
    for h in hidden:
        layers += [nn.Linear(dim, h), nn.ReLU()]; dim = h
    net = nn.Sequential(*layers, nn.Linear(dim, out_dim))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xt, st_, yt = torch.as_tensor(inp_tr), torch.as_tensor(sel_tr), torch.as_tensor(ytr)
    for _ in range(args.epochs):
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), 512):
            b = perm[i:i + 512]
            q = net(xt[b]).gather(1, st_[b].unsqueeze(1)).squeeze(1)
            loss = ((q - yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        q = net(torch.as_tensor(inp_te)).numpy()
    return q


eyeA, eyeD = np.eye(A, dtype=np.float32), np.eye(nd, dtype=np.float32)
# util: the destination's K path utilisations plus agent and destination ids -> K values
f_tr = np.concatenate([utr, eyeA[atr], eyeD[dtr]], 1)
f_te = np.concatenate([ute, eyeA[ate], eyeD[dte]], 1)
q = fit(f_tr, f_te, K, ktr, None, hidden=(64, 64))
res['util'] = boot(ips(np.argmax(masked(q, nte), 1)))
# full: observation plus agent id -> nd*K values, as the learners' Q-network
g_tr = np.concatenate([Xtr, eyeA[atr]], 1)
g_te = np.concatenate([Xte, eyeA[ate]], 1)
q = fit(g_tr, g_te, nd * K, dtr * K + ktr, None)
qd = q.reshape(-1, nd, K)[np.arange(len(dte)), dte]
res['full'] = boot(ips(np.argmax(masked(qd, nte), 1)))
# how often the fitted rules agree with greedy on the test decisions
gch = np.argmin(np.where(np.arange(K)[None, :] < nte[:, None], ute, np.inf), 1)
res['agree_greedy'] = {'table': float((np.array(choice) == gch).mean()),
                       'full': float((np.argmax(masked(qd, nte), 1) == gch).mean())}
for k in ('ecmp', 'greedy', 'table', 'util', 'full'):
    res[k] = {'own_delivery_pct': round(100 * res[k][0], 2), 'se': round(100 * res[k][1], 2)}
print(json.dumps(res, indent=1))
if args.out:
    json.dump(res, open(args.out, 'w'), indent=1)
