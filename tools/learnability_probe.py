#!/usr/bin/env python3
"""Can the training machinery learn to read path utilisation at all?

The trained victims behave as near-static routing tables that ignore their
utilisation telemetry. This probe separates "the actor-critic machinery cannot
learn the mapping" from "the training reward does not teach it": it gives the
real Agent / Critic / MADDPG.learn() code (canonical hyper-parameters,
straight-through one-hot actor update, block-argmax critic targets, epsilon-greedy
exploration per destination block) the easiest possible version of the task.

  contexts  real 94-dim observations of the 14 agents, recorded at 2x hotspot
            under random routing (so utilisations vary)
  reward    immediate, per step: mean over destinations of
            (lowest candidate utilisation - utilisation of the chosen path),
            0 for the greedy choice; transitions are terminal, so no bootstrapping
  measure   on held-out contexts: share of decisions on a least-utilised path
            (chance ~55 % here, from ties), mean regret, static share, saturation;
            at the end, whether the critic resolves single-destination choices
            and where its action-gradient pushes the actor

Variants change one piece of the machinery at a time, through the production
options of MADDPG (training.learn_action_projection.actor_head / critic_head):
--critic factored (Q = mean over decisions of a per-(decision, path) value),
--head block_softmax (one softmax per destination instead of independent
sigmoids), and --supervised (the same actor network trained on greedy labels).

Findings, 3000 updates (2 Oct 2026):
  as trained (joint critic, sigmoid head) 57 % agreement, 91 % static, 80 % saturated;
      the critic resolves single-destination choices at ~63 % (chance 55 %), and
      ~3e-7 of gradient reaches the actor's logits
  supervised, same actor network          98 %: capacity is not the problem
  factored critic, sigmoid head           65 %: critic now 91 % per destination,
      but independent sigmoids follow absolute Q levels, not differences, and saturate
  joint critic, softmax head              62 %: the critic is the bottleneck
  factored critic, softmax head           83 % and rising, regret -73 %, 57 % static
Both fixes are needed: a critic that credits each destination's choice, and an
actor whose outputs per destination compete (the gradient then follows the
advantage of one path over the others).

    python tools/learnability_probe.py --init scratch --updates 3000      # v1: fails
    python tools/learnability_probe.py --critic factored --head block_softmax  # the fix
    python tools/learnability_probe.py --init trained   # start from CC-Simple's actor
"""
import argparse
import os
import random
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--src', default=os.path.join(os.path.dirname(__file__), '..', 'src'))
ap.add_argument('--config', default='reward_fix_full_config.json')
ap.add_argument('--results', default='data/results/reward_fix')
ap.add_argument('--init', default='scratch', choices=['scratch', 'trained'])
ap.add_argument('--actor-mode', default=None, help="override learn_action_projection.actor_mode")
ap.add_argument('--episodes', type=int, default=4, help='rollouts recorded for contexts')
ap.add_argument('--updates', type=int, default=3000)
ap.add_argument('--steps-per-update', type=int, default=4, help='contexts acted on per update')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--supervised', action='store_true',
                help='capacity check: train the same actor network with cross-entropy on greedy labels')
ap.add_argument('--critic', default='joint', choices=['joint', 'factored'],
                help="critic head: 'joint' (v1), Q(s, a) from concat(s, a); 'factored', "
                     "Q(s, a) = mean over decisions of q(s)[chosen path]")
ap.add_argument('--head', default='sigmoid', choices=['sigmoid', 'block_softmax'],
                help="actor head: 'sigmoid' (v1) or one softmax per destination")
args = ap.parse_args()
sys.path.insert(0, args.src)
sys.path.insert(0, os.path.join(args.src, 'maddpg_clean'))

import torch  # noqa: E402
from standalone_experiment_runner import StandaloneExperimentRunner  # noqa: E402
from maddpg_implementation import MADDPG  # noqa: E402


def _logits(actor, s):
    return actor.action_out(torch.relu(actor.fc2(torch.relu(actor.fc1(s)))))


random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
runner = StandaloneExperimentRunner(args.config, 0, args.results)
cfg_t = runner.config['training']
vcfg = next(v for v in runner.config['variants'] if v['name'] == 'CC-Simple')
env = runner._make_attack_env(runner.config['attack_eval'].get('hotspot'))
eng = env.engine
hosts, tr = eng.get_all_hosts(), eng.trainable_host_indices
nd, S = eng.n_destinations, eng.state_dims
NA = eng.n_actions
K = NA // nd
slots = eng.path_util_slots
s0 = slots[0]
FULL = [d for d in range(nd) if s0 + (d + 1) * K <= S]       # destinations with all K slots observed

# ---- 1. contexts: real observations under random routing ----------------------
rng = random.Random(args.seed)
X = []
for _ in range(args.episodes):
    eng.reset_with_load(offered_load_factor=2.0)
    states = [eng.get_state(h) for h in hosts]
    for _ in range(cfg_t['timesteps_per_episode']):
        X.extend(np.asarray(states[i], dtype=np.float32) for i in tr)
        acts = [runner._rule_action(hosts[i], 'random', eng, rng) for i in tr]
        states, _, _ = env.step(runner._build_full_actions(acts, eng.n_total_hosts, tr, NA))
X = np.stack(X)
perm = np.random.permutation(len(X))
X_train, X_test = X[perm[: int(0.8 * len(X))]], X[perm[int(0.8 * len(X)):]]
U = lambda x: np.stack([x[:, s0 + d * K: s0 + (d + 1) * K] for d in FULL], 1)   # [n, |FULL|, K]


def reward(x, a):
    """Mean over full destinations of (min_k U - U_chosen): 0 when greedy."""
    u = U(x[None])[0]
    ch = a.reshape(nd, K)[FULL].argmax(1)
    return float((u.min(1) - u[np.arange(len(FULL)), ch]).mean())


# ---- 2. the real learning machinery, one agent, local critic -------------------
proj = cfg_t.get('learn_action_projection', {})
m = MADDPG(actor_dims=[S], critic_dims=[S], n_agents=1, n_actions=NA,
           chkpt_dir='/tmp/learnability_probe', alpha=vcfg['alpha'], beta=vcfg['beta'],
           fc1=vcfg['fc1'], fc2=vcfg['fc2'], gamma=cfg_t['gamma'], tau=cfg_t['tau'],
           critic_type='local_critic', network_type='simple_q_network',
           critic_target_mode=proj.get('critic_target_mode', 'block_argmax_onehot'),
           actor_mode=args.actor_mode or proj.get('actor_mode', 'st_onehot'),
           actor_head=args.head, critic_head=args.critic, decision_block=K)
agent = m.agents[0]
if args.init == 'trained':
    p = os.path.join(args.results, 'models', 'CC-Simple', 'agent_0', 'agent_0_actor_best.pth')
    agent.actor.load_state_dict(torch.load(p, weights_only=True))
    agent.update_network_parameters(tau=1.0)
block = int(cfg_t['exploration']['decision_block_size'])
eps0, eps1 = cfg_t['exploration']['epsilon_start'], cfg_t['exploration']['epsilon_end']


def evaluate(x):
    with torch.no_grad():
        out = torch.sigmoid(_logits(agent.actor, torch.tensor(x, device=agent.actor.device)))
        out = out.cpu().numpy().reshape(len(x), nd, K)
    ch = out[:, FULL].argmax(2)                                      # [n, |FULL|]
    u = U(x)
    at_min = u <= u.min(2, keepdims=True) + 1e-9
    agree = at_min[np.arange(len(x))[:, None], np.arange(len(FULL))[None], ch].mean()
    chance = at_min.mean()
    regret = (u[np.arange(len(x))[:, None], np.arange(len(FULL))[None], ch] - u.min(2)).mean()
    mode = np.array([np.bincount(ch[:, j], minlength=K).argmax() for j in range(len(FULL))])
    static = (ch == mode[None]).mean()
    sat = (out[:, FULL].max(2) > 0.99).mean()
    return agree, chance, regret, static, sat


def report(tag):
    a, c, r, s, sat = evaluate(X_test)
    print(f"{tag:<14} greedy-agreement {100 * a:5.1f}% (chance {100 * c:4.1f}%)  regret {r:.3f}  "
          f"static {100 * s:5.1f}%  saturated {100 * sat:5.1f}%", flush=True)


def onehot(choice):
    """[n, nd] slot indices -> [n, NA] one-hot action vectors."""
    a = np.zeros((len(choice), nd, K), dtype=np.float32)
    a[np.arange(len(choice))[:, None], np.arange(nd)[None], choice] = 1.0
    return a.reshape(len(choice), NA)


def greedy_choice(x):
    ch = np.zeros((len(x), nd), dtype=int)
    ch[:, FULL] = U(x).argmin(2)
    return ch


def diagnose(x):
    """Does the critic rank actions correctly, and how much gradient reaches the actor?"""
    dev = agent.actor.device
    xt = torch.tensor(x, device=dev)
    with torch.no_grad():
        logits = _logits(agent.actor, xt)
        pol = logits.cpu().numpy().reshape(len(x), nd, K).argmax(2)
        gre = pol.copy()                       # differ from the policy only where scored
        gre[:, FULL] = U(x).argmin(2)
        rnd = np.random.randint(K, size=(len(x), nd))
        q = {k: agent.critic(xt, torch.tensor(onehot(c), device=dev)).squeeze(1).cpu().numpy()
             for k, c in (('greedy', gre), ('policy', pol), ('random', rnd))}
    r_true = {k: np.array([reward(xi, ai) for xi, ai in zip(x, onehot(c))])
              for k, c in (('greedy', gre), ('policy', pol), ('random', rnd))}
    fid = np.corrcoef(q['random'], r_true['random'])[0, 1]
    print(f"  critic Q  greedy {q['greedy'].mean():+.3f}  policy {q['policy'].mean():+.3f}  "
          f"random {q['random'].mean():+.3f}   (true reward {r_true['greedy'].mean():+.3f} / "
          f"{r_true['policy'].mean():+.3f} / {r_true['random'].mean():+.3f})")
    print(f"  critic ranks greedy above policy on {100 * (q['greedy'] > q['policy']).mean():.1f}% of contexts; "
          f"corr(Q, true reward) on random actions = {fid:.3f}")
    # actor gradient, as learn() computes it: straight-through one-hot into the critic
    z = logits.detach().clone().requires_grad_(True)
    soft = torch.sigmoid(z)
    head_out = soft if args.head == 'sigmoid' else torch.softmax(z.view(len(x), nd, K), -1).view_as(z)
    st = m._project_actions(head_out, decision_block_size=block, mode='block_argmax_onehot', straight_through=True)
    (-agent.critic(xt, st).mean()).backward()
    gz = z.grad.abs().cpu().numpy().reshape(len(x), nd, K)
    chosen = soft.detach().cpu().numpy().reshape(len(x), nd, K).max(2) > 0.99
    print(f"  mean |dLoss/dlogit|: saturated decisions {gz.max(2)[chosen].mean():.2e}, "
          f"unsaturated {gz.max(2)[~chosen].mean() if (~chosen).any() else float('nan'):.2e}")

    # Per destination: the critic's own choice (Q with that block switched to each
    # path, the rest at the policy's action) versus the path its action-gradient
    # pushes the actor towards (largest dQ/da in the block, at the policy's action).
    n = min(len(x), 256)
    base = onehot(pol[:n])
    a_pol = torch.tensor(base, device=dev, requires_grad=True)
    agent.critic(xt[:n], a_pol).sum().backward()
    g = a_pol.grad.cpu().numpy().reshape(n, nd, K)[:, FULL]
    grad_choice = g.argmax(2)
    crit_choice = np.zeros((n, len(FULL)), dtype=int)
    with torch.no_grad():
        for j, d in enumerate(FULL):
            qs = []
            for k in range(K):
                a = base.reshape(n, nd, K).copy()
                a[:, d] = 0.0
                a[:, d, k] = 1.0
                qs.append(agent.critic(xt[:n], torch.tensor(a.reshape(n, NA), device=dev)).squeeze(1).cpu().numpy())
            crit_choice[:, j] = np.stack(qs, 1).argmax(1)
    u = U(x[:n])
    at_min = u <= u.min(2, keepdims=True) + 1e-9
    idx = (np.arange(n)[:, None], np.arange(len(FULL))[None])

    def static(c):
        return np.mean([(c[:, j] == np.bincount(c[:, j], minlength=K).argmax()).mean() for j in range(c.shape[1])])

    print(f"  per destination, picks a least-utilised path: critic's own choice "
          f"{100 * at_min[idx + (crit_choice,)].mean():.1f}% (static {100 * static(crit_choice):.0f}%), "
          f"gradient's choice {100 * at_min[idx + (grad_choice,)].mean():.1f}% (static {100 * static(grad_choice):.0f}%), "
          f"chance {100 * at_min.mean():.1f}%")


print(f"contexts: {len(X_train)} train / {len(X_test)} test; actor_mode={m.actor_mode}; init={args.init}")
report("before")

if args.supervised:
    opt = torch.optim.Adam(agent.actor.parameters(), lr=vcfg['alpha'])
    dev = agent.actor.device
    for u_i in range(1, args.updates + 1):
        xb = X_train[np.random.randint(len(X_train), size=cfg_t['batch_size'])]
        xt = torch.tensor(xb, device=dev)
        z = _logits(agent.actor, xt).view(len(xb), nd, K)[:, FULL]
        y = torch.tensor(greedy_choice(xb)[:, FULL], device=dev)
        loss = torch.nn.functional.cross_entropy(z.reshape(-1, K), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if u_i % max(1, args.updates // 6) == 0:
            report(f"sup {u_i}")
    sys.exit(0)

for u_i in range(1, args.updates + 1):
    eps = eps0 + (eps1 - eps0) * min(1.0, u_i / (0.25 * args.updates))
    for _ in range(args.steps_per_update):
        x = X_train[np.random.randint(len(X_train))]
        pa, ea = agent.choose_action(x, training=True, epsilon=eps, decision_block_size=block)
        m.store_transition([x], [ea], [reward(x, ea)], [x], [True])
    m.learn(batch_size=cfg_t['batch_size'], decision_block_size=block)
    if u_i % max(1, args.updates // 6) == 0:
        report(f"update {u_i}")
diagnose(X_test[:1024])
