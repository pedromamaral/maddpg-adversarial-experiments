#!/usr/bin/env python3
"""Check the attack's threat model on real observations of a frozen victim.

For every attack type (FGSM, PGD-10, both logit objectives, the random control):
slots outside engine.path_util_slots are unchanged, the perturbed slots stay in
[0,1], and no slot moves by more than epsilon. Also prints a digest of the
adversarial states per attack, so a refactor of the attack path can be checked
for bit-identical output by running this before and after it (--src).

    python tools/threat_model_check.py [--src SRC_DIR] [--variant CC-Simple]
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
ROWS = [('packet_loss', 1), ('packet_loss', 10), ('logit_margin', 1),
        ('logit_congestion', 1), ('random', 1)]

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

slots = engine.path_util_slots
outside = np.setdiff1d(np.arange(engine.state_dims), slots)
X = np.stack(clean)
print(f"threat model: {len(slots)} of {engine.state_dims} slots [{slots[0]}..{slots[-1]}], "
      f"clean range [{X[:, slots].min():.3f}, {X[:, slots].max():.3f}]\n")

ok = True
for atype, n in ROWS:
    fw.attack_type, fw.n_steps = atype, n
    fw.step_alpha = EPS if n == 1 else EPS / n * 2.5
    h = hashlib.sha256()
    moved, d_out, d_max, lo, hi = 0.0, 0.0, 0.0, 1.0, 0.0
    for ti in range(len(trainable)):
        torch.manual_seed(1234 + ti)
        np.random.seed(1234 + ti)
        adv = np.asarray(fw.generate_adversarial_state(
            state=clean[ti], agent_network=maddpg.agents[ti], network_engine=engine,
            agent_index=ti), dtype=np.float32)
        h.update(adv.round(6).tobytes())
        d = adv - clean[ti]
        d_out = max(d_out, float(np.abs(d[outside]).max()))
        d_max = max(d_max, float(np.abs(d).max()))
        lo, hi = min(lo, float(adv[slots].min())), max(hi, float(adv[slots].max()))
        moved += float((np.abs(d[slots]) > 1e-7).mean()) / len(trainable)
    good = d_out == 0.0 and d_max <= EPS + 1e-6 and lo >= 0.0 and hi <= 1.0
    ok &= good
    print(f"{'OK ' if good else 'BAD'} {atype:<17} n={n:<3d} {h.hexdigest()[:24]}  "
          f"moved={moved:5.1%} max|d_out|={d_out:.2g} max|d|={d_max:.3f} "
          f"range=[{lo:.3f},{hi:.3f}]")
fw.n_steps, fw.step_alpha = 1, 0.0
print("\nALL CHECKS PASS" if ok else "\nCHECK FAILED")
