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

--credit step is the cross-over with MADDPG's reward: Q-learning (gamma = 0.95)
on the per-agent step reward the MADDPG pilots train on (_credited_rewards, i.e.
reward.credit / local_weight of the config), every destination's choice credited
with the agent's step reward, a factored Q per destination with a soft-updated
target network: the value-based twin of the factored LC critic.

--train-failures draws a random number of link failures per training episode.

After training, evaluates the greedy (epsilon = 0) policy like tools/run_eval.sh
(PDR across failure levels, through _attack_episodes) and scores its applied
decisions like tools/flow_choice_check.py; --eval-stale adds rollouts on
telemetry d steps old (policy_stale<d>). --eval-only re-evaluates a saved network.

    python src/idqn_control.py --config configs/v2_pilot4_lc.json \
        --out data/results/idqn_p4 --episodes 750 --seed 0
"""
import copy
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
from standalone_experiment_runner import (StandaloneExperimentRunner,  # noqa: E402
                                          _shared_flow_reward, _credited_rewards)

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
        self.shuffle_slots = None   # set: path telemetry from a random earlier step
        self._past = []
        self._rng = np.random.default_rng(0)   # own stream: failures and traffic unchanged
        self.n_actions = n_dest * k
        self.agents = [None] * n_agents
        self.eps = eps
        self.eye = torch.eye(n_agents)

    def inputs(self, states):
        return torch.cat([torch.as_tensor(np.asarray(states), dtype=torch.float32), self.eye], 1)

    def choose_action(self, states):
        if self.shuffle_slots is not None:
            # Telemetry ablation: each agent reads the per-path utilisation of a random
            # earlier step of the episode (realistic values, no link to the present).
            sl = self.shuffle_slots
            cur = np.asarray(states, dtype=np.float32)
            self._past.append(cur[:, sl].copy())
            states = cur.copy()
            for a in range(self.n_agents):
                states[a, sl] = self._past[self._rng.integers(len(self._past))][a]
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
    ap.add_argument('--eval-stale', default='0', help='telemetry ages to evaluate, e.g. 0,2,4')
    ap.add_argument('--eval-only', action='store_true', help='load qnet_seed<seed>.pt from --out')
    ap.add_argument('--eval-shuffle', action='store_true',
                    help='telemetry ablation: path utilisation from a random earlier step')
    ap.add_argument('--credit', choices=['flow', 'step'], default='flow')
    ap.add_argument('--gamma', type=float, default=0.95, help='--credit step only')
    ap.add_argument('--tau', type=float, default=0.005, help='--credit step: target update')
    ap.add_argument('--train-failures', default='0', help='e.g. 0,2,4,6,8: drawn per episode')
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
    q_target = copy.deepcopy(q)
    buf = deque(maxlen=200_000)   # flow: (state, agent, dest, path, reward)
    #                               step: (state, agent, paths[nd], reward, next state, done)
    reward_cfg = cfg.get('reward', {})
    agent_hosts = [hosts[i] for i in tr]
    train_failures = [int(x) for x in args.train_failures.split(',')]
    history, t0 = [], time.time()
    tag = f'seed{args.seed}'
    if args.eval_only:
        q.load_state_dict(torch.load(os.path.join(args.out, f'qnet_{tag}.pt')))
        args.episodes = 0

    for ep in range(args.episodes):
        pol.eps = max(args.eps_end, args.eps_start
                      - (args.eps_start - args.eps_end) * ep / max(1, args.eps_episodes))
        eng.topology.restore_intact()
        eng.reset_with_load(offered_load_factor=train_load)
        nf = random.choice(train_failures)
        if nf:
            runner._inject_failures(eng, nf)
            eng.topology.refresh_path_cache()
        eng.decision_log = []
        states = [eng.get_state(h) for h in hosts]
        seen = {}                                   # engine step -> agents' observations
        rewards = []
        for t in range(T):
            t_states = [states[i] for i in tr]
            seen[eng.time_step + 1] = t_states
            acts = pol.choose_action(t_states)
            states, _, info = env.step(runner._build_full_actions(acts, eng.n_total_hosts, tr, eng.n_actions))
            if args.credit == 'step':
                r_agents = _credited_rewards(info, reward_cfg, agent_hosts,
                                             _shared_flow_reward(info, reward_cfg))
                for a in range(A):
                    buf.append((t_states[a], a, np.argmax(acts[a].reshape(nd, K), 1), r_agents[a],
                                states[tr[a]], float(t == T - 1)))
                rewards.append(float(np.mean(r_agents)))
        for d in eng.decision_log:
            if args.credit == 'step':
                break
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
                r = torch.as_tensor([b[3 if args.credit == 'step' else 4] for b in batch],
                                    dtype=torch.float32)
                if args.credit == 'flow':
                    qd = q(x)[torch.arange(args.batch), [b[2] for b in batch], [b[3] for b in batch]]
                    loss = ((qd - r) ** 2).mean()
                else:
                    paths = torch.as_tensor(np.stack([b[2] for b in batch]))           # [B, nd]
                    qd = q(x).gather(2, paths.unsqueeze(2)).squeeze(2)                 # [B, nd]
                    s2 = torch.as_tensor(np.stack([b[4] for b in batch]), dtype=torch.float32)
                    done = torch.as_tensor([b[5] for b in batch], dtype=torch.float32)
                    with torch.no_grad():
                        nxt = q_target(torch.cat([s2, pol.eye[a]], 1)).max(2).values   # [B, nd]
                    y = r.unsqueeze(1) + args.gamma * (1 - done).unsqueeze(1) * nxt
                    loss = ((qd - y) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
                if args.credit == 'step':
                    with torch.no_grad():
                        for pt, p in zip(q_target.parameters(), q.parameters()):
                            pt.mul_(1 - args.tau).add_(args.tau * p)
        history.append({'episode': ep, 'eps': pol.eps, 'pdr': pdr, 'train_failures': nf,
                        'flow_reward': float(np.mean(rewards)) if rewards else None,
                        'loss': float(np.mean(losses)) if losses else None, **stats})
        if ep % 25 == 0 or ep == args.episodes - 1:
            print(f"[IDQN] ep {ep:4d}  eps={pol.eps:.2f}  PDR={pdr:6.2f}  "
                  f"flow reward={history[-1]['flow_reward'] or 0:.3f}  "
                  f"least-loaded={100 * stats.get('least_loaded', 0):5.1f}% "
                  f"(chance {100 * stats.get('chance', 0):4.1f}%)  "
                  f"loss={history[-1]['loss'] or 0:.4f}  buffer={len(buf)}  {time.time() - t0:6.0f}s",
                  flush=True)
    if not args.eval_only:
        torch.save(q.state_dict(), os.path.join(args.out, f'qnet_{tag}.pt'))

    # evaluation: greedy policy, as tools/run_eval.sh (PDR) and flow_choice_check.py (choices)
    pol.eps = 0.0
    result = {'config': args.config, 'seed': args.seed, 'credit': args.credit,
              'train_failures': args.train_failures, 'train_load': train_load,
              'eval_load': eval_load, 'episodes': args.episodes, 'history': history, 'eval': {}}
    suffix = ('_shuffled' if args.eval_shuffle else '_eval') if args.eval_only else ''
    out_json = os.path.join(args.out, f'idqn_{tag}{suffix}.json')
    if args.eval_shuffle:
        pol.shuffle_slots = eng.path_util_slots
    for stale in (int(x) for x in args.eval_stale.split(',')):
        for nf in (int(x) for x in args.eval_failures.split(',')):
            random.seed(EVAL_SEED); np.random.seed(EVAL_SEED); torch.manual_seed(EVAL_SEED)
            pol._past = []   # NB: kept across the episodes of one condition, so 'earlier' spans them
            eng.decision_log = []
            res = runner._attack_episodes(pol, env, args.eval_episodes, T, attack=False,
                                          offered_load_factor=eval_load, n_link_failures=nf,
                                          stale_steps=stale)
            stats = score_decisions(eng.decision_log)
            eng.decision_log = None
            key = f'fail{nf}' + (f'_stale{stale}' if stale else '')
            result['eval'][key] = {'pdr': res['mean_end_to_end_pdr'], **stats}
            print(f"[EVAL] {key}: PDR={res['mean_end_to_end_pdr']:6.2f}  "
                  f"least-loaded={100 * stats.get('least_loaded', 0):5.1f}% "
                  f"(chance {100 * stats.get('chance', 0):4.1f}%)  regret={stats.get('regret', 0):.3f}",
                  flush=True)
            json.dump(result, open(out_json, 'w'), indent=1)
    print("[IDQN] done", flush=True)


if __name__ == '__main__':
    main()
