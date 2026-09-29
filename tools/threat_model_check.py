#!/usr/bin/env python3
"""Check the threat-model feature mask on the production attack path.

Two checks, on real mid-episode observations of a frozen victim:

  1. LEGACY DIGEST. With no mask and no momentum, the adversarial states must be
     bit-identical to the code before the mask was added. Run once with
     --src <pre-patch src dir> and once with the current src; the digests of
     every (attack, n_steps) row must match.

  2. MASK INVARIANTS (current src only). With perturb_features = path_util or
     telemetry: features outside the mask are unchanged, masked features stay in
     [0,1], and no feature moves by more than epsilon. Also prints the clean
     value range of each feature group, to confirm they live in [0,1].

    python tools/threat_model_check.py                 # current src, both checks
    python tools/threat_model_check.py --src /tmp/old  # legacy digest only
"""
import argparse
import hashlib
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--src', default=os.path.join(os.path.dirname(__file__), '..', 'src'))
ap.add_argument('--config', default='reward_fix_full_config.json')
ap.add_argument('--variant', default='CC-Simple')
ap.add_argument('--results', default='data/results/reward_fix')
args = ap.parse_args()
sys.path.insert(0, args.src)
sys.path.insert(0, os.path.join(args.src, 'maddpg_clean'))

import torch  # noqa: E402
from standalone_experiment_runner import StandaloneExperimentRunner  # noqa: E402

EPS = 0.3
runner = StandaloneExperimentRunner(args.config, 0, args.results)
vcfg = next(v for v in runner.config['variants'] if v['name'] == args.variant)
maddpg, _, _ = runner._make_variant(vcfg)
runner._load_variant_checkpoint(maddpg, args.variant)
attack_eval = runner.config.get('attack_eval', {})
env = runner._make_attack_env(attack_eval.get('hotspot') or None)
engine = env.engine
engine.reset_with_load(offered_load_factor=float(attack_eval.get('offered_load_factor', 2.0)))
states = env.reset()
trainable = engine.trainable_host_indices
for _ in range(24):  # warm up so queues and path utilisations are non-trivial
    with torch.no_grad():
        acts = maddpg.choose_action([states[i] for i in trainable])
    states, _, _ = env.step(runner._build_full_actions(
        acts, engine.n_total_hosts, trainable, maddpg.n_actions))
clean = [np.asarray(states[i], dtype=np.float32) for i in trainable]
fw = runner.attack_framework
fw.epsilon = EPS


def attack(ti, atype, n_steps):
    fw.attack_type = atype
    fw.n_steps = n_steps
    fw.step_alpha = EPS if n_steps == 1 else EPS / n_steps * 2.5
    torch.manual_seed(1234 + ti)
    np.random.seed(1234 + ti)
    return np.asarray(fw.generate_adversarial_state(
        state=clean[ti], agent_network=maddpg.agents[ti], network_engine=engine,
        agent_index=ti), dtype=np.float32)


ROWS = [('packet_loss', 1), ('packet_loss', 10), ('logit_margin', 1),
        ('logit_congestion', 1), ('random', 1)]
print("== legacy digests (no mask, no momentum) ==")
for atype, n in ROWS:
    h = hashlib.sha256()
    for ti in range(len(trainable)):
        h.update(attack(ti, atype, n).round(6).tobytes())
    print(f"  {atype:<17} n_steps={n:<3d} {h.hexdigest()[:32]}")

if not hasattr(runner, '_observation_feature_groups'):
    sys.exit(0)  # pre-patch src: digests only

groups = runner._observation_feature_groups(engine)
X = np.stack(clean)
print(f"\n== feature groups (state_dims={engine.state_dims}) ==")
for g, idx in groups.items():
    print(f"  {g:<10} {len(idx):>3} features  [{idx[0]}..{idx[-1]}]  "
          f"clean range [{X[:, idx].min():.3f}, {X[:, idx].max():.3f}]")

print("\n== mask invariants ==")
ok = True
fw.momentum = 0.0
for g, idx in groups.items():
    fw.perturb_indices = idx
    outside = np.setdiff1d(np.arange(engine.state_dims), idx)
    for atype, n in ROWS:
        moved, worst_out, worst_eps, lo, hi = 0.0, 0.0, 0.0, 1.0, 0.0
        for ti in range(len(trainable)):
            adv = attack(ti, atype, n)
            d = adv - clean[ti]
            worst_out = max(worst_out, float(np.abs(d[outside]).max()))
            worst_eps = max(worst_eps, float(np.abs(d).max()))
            lo, hi = min(lo, float(adv[idx].min())), max(hi, float(adv[idx].max()))
            moved += float((np.abs(d[idx]) > 1e-7).mean()) / len(trainable)
        good = worst_out == 0.0 and worst_eps <= EPS + 1e-6 and lo >= 0.0 and hi <= 1.0
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {g:<10} {atype:<17} n={n:<3d} moved={moved:5.1%} "
              f"max|d_out|={worst_out:.2g} max|d|={worst_eps:.3f} range=[{lo:.3f},{hi:.3f}]")
fw.perturb_indices = None
print("\nALL MASK CHECKS PASS" if ok else "\nMASK CHECK FAILED")
