#!/usr/bin/env python3
"""Diagnose whether PGD is genuinely no stronger than FGSM, or merely mis-tuned.

Background
----------
Across every probe run to date, 10-step PGD flips FEWER decisions than single-step
FGSM (0.164-0.224 vs 0.258-0.259). That is backwards: PGD with the same epsilon
ball is a strictly larger search space than FGSM, so a properly tuned PGD should
never do less damage. Paper 2's central negative result ("the attack is not
under-powered at the decision level") rests on the myopic attack being competent,
so this anomaly needs an explanation before more PGD numbers are generated.

Hypothesis under test: BUDGET UNDER-SPEND.
FGSM takes one step of size epsilon, so every coordinate lands exactly on the
+/-epsilon boundary -- the full budget is always spent. PGD takes n_steps of size
step_alpha; the runner sets step_alpha = epsilon / n_steps * 2.5 (0.075 for
epsilon=0.3, n=10). If the gradient sign oscillates between steps, those small
signed steps partially cancel and the final perturbation can sit well inside the
ball, spending less than the full budget and thus flipping fewer decisions.

Note the attack objective is LINEAR in the actor's outputs: _per_action_util
returns a DETACHED tensor, so the congestion weights sigmoid((u-0.5)*10) are
constants w.r.t. the perturbation. Objective saturation is therefore not a
candidate explanation; vanishing gradient can still arise from the actor's own
output sigmoid, which this script measures directly.

What it measures, per (n_steps, step_alpha) configuration, on real observations:
  * objective gain          -- did PGD actually climb the loss it optimises?
  * per-step gradient norm  -- is the gradient vanishing?
  * budget spend            -- mean |delta| / epsilon, and the fraction of
                               coordinates pinned to the boundary
  * decision flip rate      -- the outcome metric the paper reports

Usage (inside the container, from the repo root):
    python tools/pgd_diagnostic.py --config reward_fix_full_config.json \
        --variant CC-Simple --results-dir data/results/reward_fix
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'maddpg_clean'))

from standalone_experiment_runner import StandaloneExperimentRunner  # noqa: E402


def collect_observations(runner, maddpg, env, load, n_warmup, n_collect):
    """Roll the frozen policy forward and return realistic mid-episode states.

    Warm-up matters: at t=0 every queue is empty, so the k-path utilisation block
    is all zeros and the congestion weights carry no signal. Observations are
    taken only after the network has built up backlog.
    """
    env.engine.reset_with_load(offered_load_factor=load)
    states = env.reset() if hasattr(env, 'reset') else env.engine.get_all_states()
    trainable = getattr(env.engine, 'trainable_host_indices', None)
    n_total = getattr(env.engine, 'n_total_hosts', maddpg.n_agents)
    n_actions = maddpg.n_actions

    collected = []
    for t in range(n_warmup + n_collect):
        with torch.no_grad():
            t_states = [states[i] for i in trainable]
            t_actions = maddpg.choose_action(t_states)
            actions = runner._build_full_actions(t_actions, n_total, trainable, n_actions)
        if t >= n_warmup:
            # Store (maddpg_agent_index, observation) pairs.
            for ti, topo_idx in enumerate(trainable):
                collected.append((ti, np.asarray(states[topo_idx], dtype=np.float32)))
        states, _, _ = env.step(actions)
    return collected


def instrumented_pgd(framework, agent, engine, state, agent_index, epsilon,
                     n_steps, step_alpha, freeze_weights=False,
                     momentum=0.0, random_start=False):
    """Replicate generate_adversarial_state's PGD loop with instrumentation.

    Mirrors improved_fgsm_attack.generate_adversarial_state exactly (sign step,
    projection into the L-inf ball, domain constraints) but records the objective
    and gradient norm at every step. Non-GNN variants only: the actor is applied
    directly, matching _actor_probs' non-GNN branch.
    """
    device = framework.device
    orig_np = np.asarray(state, dtype=np.float32)
    orig = torch.tensor(orig_np, device=device).unsqueeze(0)
    start = orig.clone()
    if random_start:
        # Standard PGD random restart: begin at a uniform point in the eps-ball.
        start = start + torch.empty_like(start).uniform_(-epsilon, epsilon)
        start = framework._apply_domain_constraints(start, None)
    adv = start.clone().requires_grad_(True)
    grad_accum = torch.zeros_like(orig)

    objectives, grad_norms = [], []
    was_training = agent.actor.training
    agent.actor.eval()

    # The shipped objective reads its congestion weights out of the state it is
    # handed -- which during PGD is the ALREADY-PERTURBED observation. The weights
    # therefore move under the attacker every step. freeze_weights pins them to the
    # clean observation once, making the objective stationary across iterations.
    frozen_w = None
    if freeze_weights:
        with torch.no_grad():
            u = framework._per_action_util(orig, agent.actor(orig), engine)
            frozen_w = None if u is None else torch.sigmoid((u - 0.5) * 10.0)

    def objective(cur):
        probs = agent.actor(cur)
        if frozen_w is None:
            return framework._packet_loss_objective(cur, probs, engine, agent_index)
        L = frozen_w.shape[-1]
        return torch.sum(probs[:, :L] * frozen_w)

    try:
        with torch.enable_grad():
            for _step in range(n_steps):
                if adv.grad is not None:
                    adv.grad.zero_()
                loss = objective(adv)
                loss.backward()
                objectives.append(float(loss.detach()))
                grad_norms.append(float(adv.grad.data.norm()))
                with torch.no_grad():
                    g = adv.grad.data
                    if momentum > 0.0:
                        # MI-FGSM: accumulate the L1-normalised gradient so a sign
                        # that oscillates between steps cancels in the accumulator
                        # instead of undoing the previous step's displacement.
                        grad_accum.mul_(momentum).add_(g / (g.abs().sum() + 1e-12))
                        g = grad_accum
                    adv = adv + step_alpha * torch.sign(g)
                    adv = orig + torch.clamp(adv - orig, -epsilon, epsilon)
                    adv = framework._apply_domain_constraints(adv, None)
                adv = adv.detach().requires_grad_(True)
            # Final objective after the last update (not counted as a step).
            with torch.no_grad():
                final_obj = float(objective(adv))
    finally:
        if was_training:
            agent.actor.train()

    delta = (adv.detach() - orig).squeeze(0).cpu().numpy()
    return {
        'adv': adv.detach().squeeze(0).cpu().numpy(),
        'obj_start': objectives[0] if objectives else float('nan'),
        'obj_end': final_obj,
        'obj_gain': final_obj - (objectives[0] if objectives else float('nan')),
        'grad_first': grad_norms[0] if grad_norms else float('nan'),
        'grad_last': grad_norms[-1] if grad_norms else float('nan'),
        'mean_abs_delta': float(np.abs(delta).mean()),
        'frac_at_boundary': float((np.abs(delta) >= 0.99 * epsilon).mean()),
    }


def decision_flips(agent, clean_state, adv_state, n_dest, k_paths, device):
    """Count per-destination argmax decisions changed by the perturbation."""
    with torch.no_grad():
        def decode(s):
            t = torch.tensor(np.asarray(s, dtype=np.float32), device=device).unsqueeze(0)
            out = agent.actor(t).squeeze(0).cpu().numpy()
            usable = n_dest * k_paths
            return out[:usable].reshape(n_dest, k_paths).argmax(axis=1)
        return int((decode(clean_state) != decode(adv_state)).sum()), n_dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='reward_fix_full_config.json')
    ap.add_argument('--variant', default='CC-Simple')
    ap.add_argument('--results-dir', default='data/results/reward_fix')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--epsilon', type=float, default=0.3)
    ap.add_argument('--warmup', type=int, default=24)
    ap.add_argument('--collect-steps', type=int, default=3)
    ap.add_argument('--max-samples', type=int, default=60)
    args = ap.parse_args()

    runner = StandaloneExperimentRunner(args.config, args.gpu, args.results_dir)
    cfg = runner.config

    vcfg = next((v for v in cfg['variants'] if v['name'] == args.variant), None)
    if vcfg is None:
        sys.exit(f"variant {args.variant} not in {args.config}")
    if vcfg.get('use_gnn', False):
        sys.exit("this diagnostic covers non-GNN variants only "
                 "(the GNN attack path needs the full multi-agent batch)")

    attack_eval = cfg.get('attack_eval', {})
    load = float(attack_eval.get('offered_load_factor', 2.0))
    hotspot = attack_eval.get('hotspot') or None

    maddpg, _, _ = runner._make_variant(vcfg)
    runner._load_variant_checkpoint(maddpg, vcfg['name'])
    env = runner._make_attack_env(hotspot)
    engine = env.engine

    n_dest = len(engine.topology.access_nodes)
    k_paths = max(1, maddpg.n_actions // n_dest)
    device = maddpg.agents[0].actor.device
    framework = runner.attack_framework
    framework.epsilon = args.epsilon
    framework.attack_type = 'packet_loss'

    print(f"variant={args.variant}  load={load}x  hotspot={'on' if hotspot else 'off'}  "
          f"eps={args.epsilon}  n_dest={n_dest}  K={k_paths}")

    samples = collect_observations(runner, maddpg, env, load,
                                   args.warmup, args.collect_steps)
    if len(samples) > args.max_samples:
        idx = np.linspace(0, len(samples) - 1, args.max_samples).astype(int)
        samples = [samples[i] for i in idx]
    print(f"collected {len(samples)} observations after {args.warmup} warm-up steps\n")

    eps = args.epsilon
    # (label, n_steps, step_alpha, freeze_weights, momentum, random_start)
    configs = [
        ('FGSM      n=1  a=eps',         1,  eps,           False, 0.0, False),
        ('PGD-shipped n=10 a=eps/n*2.5', 10, eps / 10 * 2.5, False, 0.0, False),
        ('PGD-full  n=10 a=eps',         10, eps,           False, 0.0, False),
        ('PGD-fine  n=20 a=eps/4',       20, eps / 4,       False, 0.0, False),
        # Congestion weights pinned to the clean observation: isolates "PGD cannot
        # help here" from "the objective moves under the attacker because it is
        # read from the perturbed state".
        ('PGD-frozenW n=10 a=eps/2',     10, eps / 2,       True,  0.0, False),
        # Momentum (MI-FGSM) is the standard remedy for the sign oscillation that
        # is costing every PGD config half its budget. Random start is the standard
        # PGD restart. Together these are a properly-tuned iterated attack.
        ('MI-FGSM  n=10 a=eps/4 mu=1',   10, eps / 4,       True,  1.0, False),
        ('MI-FGSM  n=20 a=eps/8 mu=1',   20, eps / 8,       True,  1.0, False),
        ('MI-FGSM+rand n=20 a=eps/8',    20, eps / 8,       True,  1.0, True),
    ]

    print(f"{'config':<30} {'obj gain':>9} {'grad1':>8} {'gradN':>8} "
          f"{'spend':>7} {'bound%':>7} {'flip%':>7}")
    print('-' * 82)

    for label, n_steps, alpha, freeze, mu, rand in configs:
        gains, g1, gn, spends, bounds = [], [], [], [], []
        flips = total = 0
        for agent_idx, obs in samples:
            agent = maddpg.agents[agent_idx]
            r = instrumented_pgd(framework, agent, engine, obs, agent_idx,
                                 eps, n_steps, alpha, freeze_weights=freeze,
                                 momentum=mu, random_start=rand)
            gains.append(r['obj_gain'])
            g1.append(r['grad_first'])
            gn.append(r['grad_last'])
            spends.append(r['mean_abs_delta'] / eps)
            bounds.append(r['frac_at_boundary'])
            c, t = decision_flips(agent, obs, r['adv'], n_dest, k_paths, device)
            flips += c
            total += t
        print(f"{label:<30} {np.mean(gains):>9.4f} {np.mean(g1):>8.4f} "
              f"{np.mean(gn):>8.4f} {np.mean(spends):>7.3f} "
              f"{np.mean(bounds) * 100:>6.1f}% {flips / max(1, total) * 100:>6.2f}%")

    print("\nspend  = mean |delta| / epsilon  (1.000 = full budget on every coordinate)")
    print("bound% = coordinates pinned to the +/-epsilon boundary")
    print("If PGD-shipped shows spend << 1.0 while FGSM shows 1.0, the shipped PGD")
    print("is under-spending its budget and the low flip rate is a tuning artefact.")


if __name__ == '__main__':
    main()
