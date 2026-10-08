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
--train-random-hotspot draws a new hotspot per training episode (4 random sites'
BS and MECS as hot sources, 2 random CS as hot destinations, the shape of the
evaluation hotspot), so that no single static routing table fits all episodes.

--learner ac is MADDPG with per-decision credit: MADDPG's own ActorNetwork
(block_softmax) and factored CriticNetwork, one per agent, with the actor trained
through the critic exactly as MADDPG.learn does (straight-through one-hot of the
per-destination softmax, entropy bonus); only the critic's target changes, to each
applied decision's flow outcome. --critic local reads the agent's observation (LC);
--critic central reads all 14 agents' observations (CC, the 'observations'
central state of the v2 pilots).

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


class ActorCritic:
    """MADDPG's actor and factored critic per agent, critic fitted to flow outcomes.

    duelling: the critic's per-destination duelling head (CriticNetwork, factored).
    gnn: MADDPG's GNNProcessor encodes all agents' observations jointly over the
    full topology (transit switches as relay nodes) before the actors; as in
    MADDPG.learn, the critics read the encoding detached and the GNN is trained
    through the actor loss.
    """

    def __init__(self, n_agents, obs_dim, n_dest, k, critic, alpha, beta, entropy,
                 duelling=False, gnn_adjacency=None, gnn_n_relay=0):
        from maddpg_implementation import ActorNetwork, CriticNetwork, GNNProcessor
        self.n_agents, self.n_dest, self.k, self.central = n_agents, n_dest, k, critic == 'central'
        self.entropy = entropy
        c_in = n_agents * obs_dim if self.central else obs_dim
        self.actors = [ActorNetwork(obs_dim, 256, 128, n_dest * k, f'a{i}', '/tmp', device='cpu',
                                    head='block_softmax', block=k) for i in range(n_agents)]
        self.critics = [CriticNetwork(c_in, 256, 128, 1, n_dest * k, f'c{i}', '/tmp',
                                      action_input_dims=n_dest * k, head='factored', block=k,
                                      network_type='duelling_q_network' if duelling
                                      else 'simple_q_network').cpu()
                        for i in range(n_agents)]
        self.a_opt = [torch.optim.Adam(a.parameters(), lr=alpha) for a in self.actors]
        self.c_opt = [torch.optim.Adam(c.parameters(), lr=beta) for c in self.critics]
        self.gnn = None
        if gnn_adjacency is not None:
            self.gnn = GNNProcessor(obs_dim, 64, n_agents, adjacency=gnn_adjacency,
                                    n_relay_nodes=gnn_n_relay).cpu()
            self.gnn.device = torch.device('cpu')
            assert self.gnn.available, "torch_geometric is required for --gnn"
            self.g_opt = torch.optim.Adam(self.gnn.parameters(), lr=alpha)

    @property
    def needs_all_obs(self):
        return self.central or self.gnn is not None

    def encode(self, all_obs):
        """[A, obs] numpy -> per-agent actor inputs [A, obs] (no grad)."""
        if self.gnn is None:
            return torch.as_tensor(all_obs, dtype=torch.float32)
        return torch.as_tensor(np.stack(self.gnn.process_observations(list(all_obs))))

    def scores(self, states, others=None):
        """Per-path actor outputs [A, n_dest, k]. others: a past snapshot of all
        observations; agent a then sees its own current observation and the others'
        from the snapshot (the neighbour-shuffle ablation, GNN only)."""
        with torch.no_grad():
            if others is None:
                x = self.encode(states)
                out = [self.actors[a](x[a]) for a in range(self.n_agents)]
            else:
                out = []
                for a in range(self.n_agents):
                    mix = np.array(others, dtype=np.float32)
                    mix[a] = states[a]
                    out.append(self.actors[a](self.encode(mix)[a]))
            return torch.stack(out).view(self.n_agents, self.n_dest, self.k)

    def update(self, a, batch):
        own = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
        B = len(batch)
        if self.needs_all_obs:
            allo = torch.as_tensor(np.stack([b[1] for b in batch]), dtype=torch.float32).view(
                B, self.n_agents, -1)
        if self.gnn is not None:
            enc = self.gnn.process_batch([allo[:, j] for j in range(self.n_agents)])
            actor_in = enc[a]
            critic_in = (torch.cat([e.detach() for e in enc], 1) if self.central
                         else enc[a].detach())
        else:
            actor_in = own
            critic_in = allo.reshape(B, -1) if self.central else own
        idx = torch.as_tensor([b[2] * self.k + b[3] for b in batch])
        r = torch.as_tensor([b[4] for b in batch], dtype=torch.float32)
        critic, actor = self.critics[a], self.actors[a]
        q = critic.decision_values(critic_in).gather(1, idx.unsqueeze(1)).squeeze(1)
        c_loss = ((q - r) ** 2).mean()
        self.c_opt[a].zero_grad(); c_loss.backward(); self.c_opt[a].step()
        # actor: as MADDPG.learn with actor_mode st_onehot and the factored critic
        soft = actor(actor_in)
        blocks = soft.view(B, self.n_dest, self.k)
        hard = torch.zeros_like(blocks).scatter_(2, blocks.argmax(2, keepdim=True), 1.0).view_as(soft)
        act = hard + soft - soft.detach()
        a_loss = -critic(critic_in, act).mean()
        if self.entropy > 0:
            a_loss = a_loss - self.entropy * -(blocks * blocks.clamp_min(1e-12).log()).sum(-1).mean()
        self.a_opt[a].zero_grad()
        if self.gnn is not None:
            self.g_opt.zero_grad()
        a_loss.backward()
        self.a_opt[a].step()
        if self.gnn is not None:
            self.g_opt.step()
        return c_loss.item()

    def state_dict(self):
        sd = {'actors': [x.state_dict() for x in self.actors],
              'critics': [x.state_dict() for x in self.critics]}
        if self.gnn is not None:
            sd['gnn'] = self.gnn.state_dict()
        return sd

    def load_state_dict(self, sd):
        for x, d in zip(self.actors, sd['actors']):
            x.load_state_dict(d)
        for x, d in zip(self.critics, sd['critics']):
            x.load_state_dict(d)
        if self.gnn is not None:
            self.gnn.load_state_dict(sd['gnn'])


class Policy:
    """What _attack_episodes needs from an agent object, for evaluation."""

    def __init__(self, score, n_agents, n_dest, k, eps=0.0):
        self.score, self.n_agents, self.n_dest, self.k = score, n_agents, n_dest, k
        self.shuffle_slots = None   # set: path telemetry from a random earlier step
        self.shuffle_neighbours = False   # GNN: other agents' observations from a random earlier step
        self._past = []
        self._rng = np.random.default_rng(0)   # own stream: failures and traffic unchanged
        self.n_actions = n_dest * k
        self.agents = [None] * n_agents
        self.eps = eps

    def choose_action(self, states):
        if self.shuffle_neighbours:
            cur = np.asarray(states, dtype=np.float32)
            self._past.append(cur.copy())
            q = self.score(cur, others=self._past[self._rng.integers(len(self._past))]).numpy()
            return self._to_actions(q)
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
            q = self.score(np.asarray(states, dtype=np.float32)).numpy()
        return self._to_actions(q)

    def _to_actions(self, q):
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
    ap.add_argument('--eval-shuffle', choices=['path', 'all', 'neighbours'], default=None,
                    help='ablation: path utilisation (path) or the whole observation (all) '
                         'from a random earlier step')
    ap.add_argument('--credit', choices=['flow', 'step'], default='flow')
    ap.add_argument('--gamma', type=float, default=0.95, help='--credit step only')
    ap.add_argument('--tau', type=float, default=0.005, help='--credit step: target update')
    ap.add_argument('--train-failures', default='0', help='e.g. 0,2,4,6,8: drawn per episode')
    ap.add_argument('--train-random-hotspot', action='store_true')
    ap.add_argument('--eval-tag', default=None, help='--eval-only: output <prefix>_seed<s>_<tag>.json')
    ap.add_argument('--learner', choices=['dqn', 'ac'], default='dqn')
    ap.add_argument('--critic', choices=['local', 'central'], default='local', help='--learner ac')
    ap.add_argument('--entropy', type=float, default=0.01, help='--learner ac (pilot 4: 0.01)')
    ap.add_argument('--duelling', action='store_true', help='--learner ac: per-destination duelling critic')
    ap.add_argument('--gnn', action='store_true', help='--learner ac: GNN encoder before the actors')
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
    eye = torch.eye(A)
    if args.learner == 'ac':
        assert args.credit == 'flow', "--learner ac implements per-decision (flow) credit"
        nets = next(v for v in cfg['variants'] if v['name'] == 'LC-Simple')   # pilots' learning rates
        gnn_adj, gnn_relay = None, 0
        if args.gnn:   # as StandaloneExperimentRunner._make_variant: full graph, switches relay
            agents_h = [hosts[i] for i in tr]
            order = agents_h + [h for h in hosts if h not in set(agents_h)]
            pos = {h: i for i, h in enumerate(order)}
            gnn_adj = [[pos[nb] for nb in eng.topology.get_neighbors(h) if nb in pos] for h in order]
            gnn_relay = len(order) - A
        ac = ActorCritic(A, eng.state_dims, nd, K, args.critic, float(nets.get('alpha', 3e-4)),
                         float(nets.get('beta', 3e-4)), args.entropy, duelling=args.duelling,
                         gnn_adjacency=gnn_adj, gnn_n_relay=gnn_relay)
        pol = Policy(ac.scores, A, nd, K)
        prefix = f"ac_{args.critic}" + ('_duel' if args.duelling else '') + ('_gnn' if args.gnn else '')
        ckpt = f'{prefix}_seed{args.seed}.pt'
        bufs = [deque(maxlen=20_000) for _ in range(A)]   # (obs, central obs, dest, path, reward)
    else:
        q = QNet(eng.state_dims, A, nd, K)
        opt = torch.optim.Adam(q.parameters(), lr=args.lr)
        pol = Policy(lambda st: q(torch.cat([torch.as_tensor(st), eye], 1)), A, nd, K)
        q_target = copy.deepcopy(q)
        prefix, ckpt = 'idqn', f'qnet_seed{args.seed}.pt'
        bufs = None
    buf = deque(maxlen=200_000)   # flow: (state, agent, dest, path, reward)
    #                               step: (state, agent, paths[nd], reward, next state, done)
    reward_cfg = cfg.get('reward', {})
    agent_hosts = [hosts[i] for i in tr]
    train_failures = [int(x) for x in args.train_failures.split(',')]
    history, t0 = [], time.time()
    tag = f'seed{args.seed}'
    access = list(eng.topology.access_nodes)
    sites = sorted({h[2:] for h in access if h.startswith('BS') and 'MECS' + h[2:] in access})
    cs_nodes = [h for h in access if h.startswith('CS')]
    eval_hot = (list(eng._skew_hot_srcs), list(eng._skew_hot_dsts))
    if args.eval_only:
        (ac if args.learner == 'ac' else q).load_state_dict(torch.load(os.path.join(args.out, ckpt)))
        args.episodes = 0

    for ep in range(args.episodes):
        pol.eps = max(args.eps_end, args.eps_start
                      - (args.eps_start - args.eps_end) * ep / max(1, args.eps_episodes))
        eng.topology.restore_intact()
        if args.train_random_hotspot:
            hot = random.sample(sites, 4)
            eng._skew_hot_srcs = [f'BS{i}' for i in hot] + [f'MECS{i}' for i in hot]
            eng._skew_hot_dsts = random.sample(cs_nodes, 2)
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
            if bufs is not None:
                central = np.concatenate(seen[d['step']]).astype(np.float16) if ac.needs_all_obs else None
                bufs[a].append((seen[d['step']][a], central, dst_idx[d['dst']], d['k'], r))
            else:
                buf.append((seen[d['step']][a], a, dst_idx[d['dst']], d['k'], r))
            rewards.append(r)
        stats = score_decisions(eng.decision_log)
        eng.decision_log = None
        pdr = float(env.get_stats().get('end_to_end_pdr', 0.0))

        losses = []
        if bufs is not None:
            for _ in range(args.updates):
                for a in range(A):
                    if len(bufs[a]) >= args.batch:
                        losses.append(ac.update(a, random.sample(bufs[a], args.batch)))
        elif len(buf) >= args.batch:
            for _ in range(args.updates):
                batch = random.sample(buf, args.batch)
                s = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
                a = torch.as_tensor([b[1] for b in batch])
                x = torch.cat([s, eye[a]], 1)
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
                        nxt = q_target(torch.cat([s2, eye[a]], 1)).max(2).values   # [B, nd]
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
            print(f"[{prefix.upper()}] ep {ep:4d}  eps={pol.eps:.2f}  PDR={pdr:6.2f}  "
                  f"flow reward={history[-1]['flow_reward'] or 0:.3f}  "
                  f"least-loaded={100 * stats.get('least_loaded', 0):5.1f}% "
                  f"(chance {100 * stats.get('chance', 0):4.1f}%)  "
                  f"loss={history[-1]['loss'] or 0:.4f}  "
                  f"buffer={sum(map(len, bufs)) if bufs is not None else len(buf)}  {time.time() - t0:6.0f}s",
                  flush=True)
    if not args.eval_only:
        torch.save((ac if args.learner == 'ac' else q).state_dict(), os.path.join(args.out, ckpt))

    # evaluation: greedy policy, as tools/run_eval.sh (PDR) and flow_choice_check.py (choices)
    eng._skew_hot_srcs, eng._skew_hot_dsts = list(eval_hot[0]), list(eval_hot[1])
    pol.eps = 0.0
    result = {'config': args.config, 'seed': args.seed, 'credit': args.credit,
              'learner': args.learner, 'critic': args.critic if args.learner == 'ac' else None,
              'train_random_hotspot': args.train_random_hotspot,
              'eval_hotspot': {'hot_srcs': eval_hot[0], 'hot_dsts': eval_hot[1]},
              'train_failures': args.train_failures, 'train_load': train_load,
              'eval_load': eval_load, 'episodes': args.episodes, 'history': history, 'eval': {}}
    suffix = (f'_shuffled_{args.eval_shuffle}' if args.eval_shuffle else '_eval') if args.eval_only else ''
    if args.eval_only and args.eval_tag:
        suffix = f'_{args.eval_tag}'
    out_json = os.path.join(args.out, f'{prefix}_{tag}{suffix}.json')
    if args.eval_shuffle == 'neighbours':
        assert args.learner == 'ac' and args.gnn, "neighbour shuffle needs the GNN actor"
        pol.shuffle_neighbours = True
    elif args.eval_shuffle:
        pol.shuffle_slots = (eng.path_util_slots if args.eval_shuffle == 'path'
                             else list(range(eng.state_dims)))
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
    print(f"[{prefix.upper()}] done", flush=True)


if __name__ == '__main__':
    main()
