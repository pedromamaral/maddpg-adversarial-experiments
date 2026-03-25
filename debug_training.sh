#!/bin/bash
# Quick debug test - run a single MADDPG variant with detailed error output

echo "🔍 Debug MADDPG Training"
echo "======================="

# Check if Docker image exists
if ! docker image inspect maddpg-adversarial:latest >/dev/null 2>&1; then
    echo "❌ Docker image not found. Run ./docker_setup.sh first."
    exit 1
fi

echo "🔧 Running debug test with detailed Python traceback..."

# Run with Python debugging enabled
docker run --rm --gpus all \
    -v $(pwd)/host_data:/workspace/data \
    -v $(pwd)/host_logs:/workspace/logs \
    maddpg-adversarial:latest \
    python -c "
import traceback
import sys
sys.path.append('/workspace/src/maddpg_clean')
sys.path.append('/workspace/src/attack_framework')

try:
    from standalone_experiment_runner import StandaloneExperimentRunner
    runner = StandaloneExperimentRunner('experiment_config.json', 0)
    
    # Test just one training step
    print('🧪 Testing MADDPG variant creation...')
    variant_config = {
        'name': 'CC-Simple',
        'critic_domain': 'central_critic',
        'neural_network': 'simple_q_network',
        'use_gnn': False,
        'actor_dims': 26,
        'critic_dims': 91,
        'n_agents': 65,
        'n_actions': 3
    }
    
    maddpg, network_engine, network_env = runner.create_maddpg_variant(variant_config)
    print('✅ MADDPG variant created successfully')
    
    print('🧪 Testing environment reset...')
    states = network_env.reset()
    print(f'✅ Environment reset. States shape: {len(states)}, First state shape: {len(states[0]) if states else 0}')
    
    print('🧪 Testing action selection...')
    actions = maddpg.choose_action(states)
    print(f'✅ Actions generated. Actions shape: {len(actions)}, First action shape: {len(actions[0]) if actions else 0}')
    
    print('🧪 Testing environment step...')
    next_states, rewards, info = network_env.step(actions)
    print(f'✅ Environment step completed. Rewards: {rewards[:3]}...')
    
    print('🎉 All basic operations successful!')
    
except Exception as e:
    print(f'❌ Error occurred: {e}')
    print('📋 Full traceback:')
    traceback.print_exc()
    sys.exit(1)
"

echo ""
echo "🎯 If this succeeds, the issue is in the training loop timing."
echo "🎯 If this fails, we'll see the exact error location."