#!/usr/bin/env python3
"""Checksum adversarial states from the PRODUCTION attack path.

Run once with the patched improved_fgsm_attack.py and once with the pre-patch
version. The n_steps=1 (FGSM) digest MUST be identical across the two -- that is
the claim that every previously reported FGSM number still reproduces. The
n_steps=10 (PGD) digest is expected to differ: that path is what the patch fixes.
"""
import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'maddpg_clean'))

from standalone_experiment_runner import StandaloneExperimentRunner  # noqa: E402

CONFIG = 'reward_fix_full_config.json'
VARIANT = 'CC-Simple'
RESULTS = 'data/results/reward_fix'

runner = StandaloneExperimentRunner(CONFIG, 0, RESULTS)
vcfg = next(v for v in runner.config['variants'] if v['name'] == VARIANT)
maddpg, _, _ = runner._make_variant(vcfg)
runner._load_variant_checkpoint(maddpg, VARIANT)

attack_eval = runner.config.get('attack_eval', {})
env = runner._make_attack_env(attack_eval.get('hotspot') or None)
engine = env.engine
load = float(attack_eval.get('offered_load_factor', 2.0))

engine.reset_with_load(offered_load_factor=load)
states = env.reset()
trainable = engine.trainable_host_indices
n_total = engine.n_total_hosts

# Warm up so queues and k-path utilisations are non-trivial.
for _ in range(24):
    with torch.no_grad():
        acts = maddpg.choose_action([states[i] for i in trainable])
    states, _, _ = env.step(
        runner._build_full_actions(acts, n_total, trainable, maddpg.n_actions))

fw = runner.attack_framework
fw.epsilon = 0.3
fw.attack_type = 'packet_loss'

for n_steps in (1, 10):
    fw.n_steps = n_steps
    fw.step_alpha = 0.3 if n_steps == 1 else 0.3 / n_steps * 2.5
    h = hashlib.sha256()
    for ti, topo_idx in enumerate(trainable):
        # Deterministic seeding so any RNG inside the path cannot explain a diff.
        torch.manual_seed(1234 + ti)
        np.random.seed(1234 + ti)
        adv = fw.generate_adversarial_state(
            state=np.asarray(states[topo_idx], dtype=np.float32),
            agent_network=maddpg.agents[ti],
            network_engine=engine,
            agent_index=ti,
        )
        h.update(np.asarray(adv, dtype=np.float32).round(6).tobytes())
    print(f"n_steps={n_steps:<3d} digest={h.hexdigest()[:32]}")

fw.n_steps = 1
fw.step_alpha = 0.0
