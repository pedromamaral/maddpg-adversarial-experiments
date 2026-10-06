#!/usr/bin/env python3
"""Positive control for the MADDPG routing agents: independent value-based learners.

The same 14 PE agents, observations, traffic and simulator as the MADDPG pilots,
but each agent picks, per destination, the path with the highest learned value
Q(o_i, agent_i)[d, k]; there is no actor and no gradient through a critic.
Decentralised training and execution: one Q-network shared by the agents (with an
agent one-hot), each acting on its own observation only.

Credit is per decision. With per-flow pinning (traffic.pin_paths_per_flow) only a
flow's first decision is applied, so each applied decision is a sample
(o_i, agent, destination, path) -> share of the flow's packets delivered, learned
by regression (a contextual bandit, gamma = 0). Decisions read from
NetworkEngine.decision_log, outcomes from NetworkEngine.flow_outcomes.

After training, evaluates the greedy (epsilon = 0) policy like tools/run_eval.sh
(PDR across failure levels, through _attack_episodes) and scores its applied
decisions like tools/flow_choice_check.py.

    python src/idqn_control.py --config configs/v2_pilot4_lc.json \
        --out data/results/idqn_p4 --episodes 750 --seed 0
"""
import argparse
import json
import os
import random
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maddpg_clean'))
from standalone_experiment_runner import StandaloneExperimentRunner  # noqa: E402

EVAL_SEED = 20240601      # the evaluation traffic seed of _attack_episodes
TAIL_STEPS = 20           # decisions this close to the episode end have unfinished flows


class QNet(nn.Module):
    def __init__(self, obs_dim, n_agents, n_dest, k):
        super().__init__()
        self.n_dest, self.k = n_dest, k
        self.net = nn.Sequential(nn.Linear(obs_dim + n_agents, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, n_dest * k))

    def forward(self, x):
        return self.net(x).view(-1, self.n_dest, self.k)


class Policy:
    """What _attack_episodes needs from an agent object, for evaluation."""

    def __init__(self, q, n_agents, n_dest, k, eps=0.0):
        self.q, self.n_agents, self.n_dest, self.k = q, n_agents, n_dest, k
        self.n_actions = n_dest * k
        self.agents = [None] * n_agents
        self.eps = eps
        self.eye = torch.eye(n_agents)

    def inputs(self, states):
        return torch.cat([torch.as_tensor(np.asarray(states), dtype=torch.float32), self.eye], 1)

    def choose_action(self, states):
        with torch.no_grad():
            q = self.q(self.inputs(states)).numpy()
        choice = q.argmax(2)                                          # [agents, dest]
        if self.eps > 0:
            explore = np.random.rand(*choice.shape) < self.eps
            choice = np.where(explore, np.random.randint(self.k, size=choice.shape), choice)
        out = []
        for a in range(self.n_agents):
            v = np.zeros(self.n_actions, dtype=np.float32)
            v[np.arange(self.n_dest) * self.k + choice[a]] = 1.0
            out.append(v)
        return out


def score_decisions(log):
    """Least-loaded share on contested applied decisions (as tools/flow_choice_check.py)."""
    hit, chance, regret = [], [], []
    for d in log:
        u = np.asarray(d['utils'])
        if d['n_distinct'] < 2 or u.max() - u.min() <= 1e-9 or d['k'] >= len(u):
            continue
        lo = u.min()
        hit.append(float(u[d['k']] <= lo + 1e-9))
        chance.append(float((u <= lo + 1e-9).sum()) / len(u))
        regret.append(float(u[d['k']] - lo))
    if not hit:
        return {}
    return {'least_loaded': float(np.mean(hit)), 'chance': float(np.mean(chance)),
            'regret': float(np.mean(regret)), 'contested': len(hit)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--episodes', type=int, default=750, help='MADDPG pilots: 150 epochs x 5')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--updates', type=int, default=50, help='gradient steps per episode')
    ap.add_argument('--eps-start', type=float, default=1.0)
    ap.add_argument('--eps-end', type=float, default=0.05)
    ap.add_argument('--eps-episodes', type=int, default=250, help='MADDPG pilots: 50 epochs')
    ap.add_argument('--eval-episodes', type=int, default=20)
    ap.add_argument('--eval-failures', default='0,2,4,6,8')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(2)

    runner = StandaloneExperimentRunner(args.config, 0, args.out)
    cfg = runner.config
    ae = cfg.get('attack_eval', {})
    train_load = float(cfg['training'].get('offered_load_factor', 2.0))
    eval_load = float(ae.get('offered_load_factor', 2.0))
    T = cfg['training']['timesteps_per_episode']
    env = runner._make_attack_env(ae.get('hotspot') or None)
    eng = env.engine
    assert eng.pin_paths_default, "per-decision credit needs traffic.pin_paths_per_flow"
    hosts, tr = eng.get_all_hosts(), eng.trainable_host_indices
    A, nd = len(tr), eng.n_destinations
    K = eng.n_actions // nd
    agent_of = {hosts[i]: a for a, i in enumerate(tr)}
    dst_idx = {d: j for j, d in enumerate(eng.topology.access_nodes)}

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    q = QNet(eng.state_dims, A, nd, K)
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)
    pol = Policy(q, A, nd, K)
    buf = deque(maxlen=200_000)        # (state, agent, dest, path, reward)
    history, t0 = [], time.time()

    for ep in range(args.episodes):
        pol.eps = max(args.eps_end, args.eps_start
                      - (args.eps_start - args.eps_end) * ep / max(1, args.eps_episodes))
        eng.topology.restore_intact()
        eng.reset_with_load(offered_load_factor=train_load)
        eng.decision_log = []
        states = [eng.get_state(h) for h in hosts]
        seen = {}                                   # engine step -> agents' observations
        sent = delivered = 0
        for _ in range(T):
            t_states = [states[i] for i in tr]
            seen[eng.time_step + 1] = t_states
            acts = pol.choose_action(t_states)
            states, _, info = env.step(runner._build_full_actions(acts, eng.n_total_hosts, tr, eng.n_actions))
            sent += info['packets_sent']
            delivered += info.get('packets_delivered', 0)
        rewards = []
        for d in eng.decision_log:
            if d['flow_id'] is None or d['step'] > T - TAIL_STEPS or d['host'] not in agent_of:
                continue
            dl, dr = eng.flow_outcomes.get(d['flow_id'], (0, 0))
            if dl + dr == 0:
                continue
            a = agent_of[d['host']]
            r = dl / (dl + dr)
            buf.append((seen[d['step']][a], a, dst_idx[d['dst']], d['k'], r))
            rewards.append(r)
        stats = score_decisions(eng.decision_log)
        eng.decision_log = None
        pdr = float(env.get_stats().get('end_to_end_pdr', 0.0))

        losses = []
        if len(buf) >= args.batch:
            for _ in range(args.updates):
                batch = random.sample(buf, args.batch)
                s = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
                a = torch.as_tensor([b[1] for b in batch])
                x = torch.cat([s, pol.eye[a]], 1)
                qd = q(x)[torch.arange(args.batch), [b[2] for b in batch], [b[3] for b in batch]]
                loss = ((qd - torch.as_tensor([b[4] for b in batch], dtype=torch.float32)) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
        history.append({'episode': ep, 'eps': pol.eps, 'pdr': pdr,
                        'flow_reward': float(np.mean(rewards)) if rewards else None,
                        'loss': float(np.mean(losses)) if losses else None, **stats})
        if ep % 25 == 0 or ep == args.episodes - 1:
            print(f"[IDQN] ep {ep:4d}  eps={pol.eps:.2f}  PDR={pdr:6.2f}  "
                  f"flow reward={history[-1]['flow_reward'] or 0:.3f}  "
                  f"least-loaded={100 * stats.get('least_loaded', 0):5.1f}% "
                  f"(chance {100 * stats.get('chance', 0):4.1f}%)  "
                  f"loss={history[-1]['loss'] or 0:.4f}  buffer={len(buf)}  {time.time() - t0:6.0f}s",
                  flush=True)
    torch.save(q.state_dict(), os.path.join(args.out, f'qnet_seed{args.seed}.pt'))

    # evaluation: greedy policy, as tools/run_eval.sh (PDR) and flow_choice_check.py (choices)
    pol.eps = 0.0
    result = {'config': args.config, 'seed': args.seed, 'train_load': train_load,
              'eval_load': eval_load, 'episodes': args.episodes, 'history': history, 'eval': {}}
    for nf in (int(x) for x in args.eval_failures.split(',')):
        random.seed(EVAL_SEED); np.random.seed(EVAL_SEED); torch.manual_seed(EVAL_SEED)
        eng.decision_log = []
        res = runner._attack_episodes(pol, env, args.eval_episodes, T, attack=False,
                                      offered_load_factor=eval_load, n_link_failures=nf)
        stats = score_decisions(eng.decision_log)
        eng.decision_log = None
        result['eval'][f'fail{nf}'] = {'pdr': res['mean_end_to_end_pdr'], **stats}
        print(f"[EVAL] fail {nf}: PDR={res['mean_end_to_end_pdr']:6.2f}  "
              f"least-loaded={100 * stats.get('least_loaded', 0):5.1f}% "
              f"(chance {100 * stats.get('chance', 0):4.1f}%)  regret={stats.get('regret', 0):.3f}",
              flush=True)
        json.dump(result, open(os.path.join(args.out, f'idqn_seed{args.seed}.json'), 'w'), indent=1)
    print("[IDQN] done", flush=True)


if __name__ == '__main__':
    main()
