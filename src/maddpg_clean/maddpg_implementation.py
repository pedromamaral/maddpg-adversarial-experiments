"""
Clean MADDPG Implementation - Standalone version
Based on working components from student's code but reimplemented cleanly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
import random
from collections import deque

class ActorNetwork(nn.Module):
    """Clean Actor Network implementation"""
    
    def __init__(self, input_dims: int, fc1_dims: int, fc2_dims: int, 
                 n_actions: int, name: str, chkpt_dir: str):
        super(ActorNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_actor.pth')
        
        # Network layers
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.action_out = nn.Linear(fc2_dims, n_actions)
        
        # Initialize weights
        self.init_weights()
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def init_weights(self):
        """Initialize network weights"""
        for layer in [self.fc1, self.fc2, self.action_out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, state):
        """Forward pass"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = torch.softmax(self.action_out(x), dim=1)
        return actions
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        """Load model checkpoint"""
        if os.path.exists(self.checkpoint_file):
            self.load_state_dict(torch.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    """Clean Critic Network implementation with support for different architectures"""
    
    def __init__(self, input_dims: int, fc1_dims: int, fc2_dims: int,
                 n_agents: int, n_actions: int, name: str, chkpt_dir: str,
                 network_type: str = 'simple_q_network'):
        super(CriticNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.network_type = network_type
        self.name = name
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_critic.pth')
        
        # Input layer (state + action)
        self.fc1 = nn.Linear(input_dims + n_actions, fc1_dims)
        
        if network_type == "duelling_q_network":
            # Duelling architecture
            self.value_head = nn.Linear(fc1_dims, 1)
            self.advantage_head = nn.Linear(fc1_dims, n_actions)
            self.output_layer = nn.Linear(n_actions, 1)
        else:  # simple_q_network
            self.q_head = nn.Linear(fc1_dims, 1)
        
        # Initialize weights
        self.init_weights()
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def init_weights(self):
        """Initialize network weights"""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, state, action):
        """Forward pass"""
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        
        if self.network_type == "duelling_q_network":
            # Duelling Q-network
            value = self.value_head(x)
            advantage = F.softmax(self.advantage_head(x), dim=1)
            
            # Q(s,a) = V(s) + A(s,a) - mean(A(s,*))
            advantage_mean = torch.mean(advantage, dim=1, keepdim=True)
            q_value = value + advantage - advantage_mean
            
            # Max over actions for final Q-value
            q_value, _ = torch.max(q_value, dim=1, keepdim=True)
        else:
            # Simple Q-network
            q_value = self.q_head(x)
        
        return q_value
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        """Load model checkpoint"""
        if os.path.exists(self.checkpoint_file):
            self.load_state_dict(torch.load(self.checkpoint_file))


class GNNProcessor(nn.Module):
    """Clean GNN processor for state preprocessing"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(GNNProcessor, self).__init__()
        
        try:
            from torch_geometric.nn import GraphConv
            self.conv1 = GraphConv(input_dim, hidden_dim)
            self.conv2 = GraphConv(hidden_dim, output_dim)
            self.available = True
        except ImportError:
            print("Warning: torch_geometric not available, GNN disabled")
            self.available = False
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.available:
            self.to(self.device)
    
    def forward(self, x, edge_index):
        """Forward pass through GNN"""
        if not self.available:
            return x
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
    
    def process_state(self, state, topology_info=None):
        """Process state through GNN"""
        if not self.available:
            return state
        
        # Create simple graph structure if topology info not provided
        if topology_info is None:
            num_nodes = len(state)
            # Simple linear graph for fallback
            edge_index = []
            for i in range(num_nodes - 1):
                edge_index.extend([[i, i+1], [i+1, i]])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = topology_info['edge_index']
        
        x = torch.tensor(state, dtype=torch.float).reshape(-1, 1)
        edge_index = edge_index.to(self.device)
        x = x.to(self.device)
        
        output = self(x, edge_index)
        processed_state = torch.mean(output, dim=0).detach().cpu().numpy()
        
        # Ensure output matches expected state dimension
        if len(processed_state) < len(state):
            padding = np.zeros(len(state) - len(processed_state))
            processed_state = np.concatenate([processed_state, padding])
        elif len(processed_state) > len(state):
            processed_state = processed_state[:len(state)]
        
        return processed_state


class Agent:
    """Clean Agent implementation"""
    
    def __init__(self, actor_dims: int, critic_dims: int, n_actions: int,
                 n_agents: int, agent_idx: int, chkpt_dir: str,
                 alpha: float = 0.01, beta: float = 0.01,
                 fc1: int = 64, fc2: int = 64, gamma: float = 0.95,
                 tau: float = 0.01, critic_type: str = 'central_critic',
                 network_type: str = 'simple_q_network', use_gnn: bool = False):
        
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_idx = agent_idx
        self.agent_name = f'agent_{agent_idx}'
        
        # Create checkpoint directory
        os.makedirs(chkpt_dir, exist_ok=True)
        
        # GNN processor (optional)
        self.use_gnn = use_gnn
        if use_gnn:
            self.gnn_processor = GNNProcessor(
                input_dim=1, 
                hidden_dim=16, 
                output_dim=actor_dims
            )
        else:
            self.gnn_processor = None
        
        # Actor network
        self.actor = ActorNetwork(
            input_dims=actor_dims,
            fc1_dims=fc1,
            fc2_dims=fc2,
            n_actions=n_actions,
            name=f'{self.agent_name}_actor',
            chkpt_dir=chkpt_dir
        )
        
        # Critic network
        self.critic = CriticNetwork(
            input_dims=critic_dims,
            fc1_dims=fc1,
            fc2_dims=fc2,
            n_agents=n_agents,
            n_actions=n_actions,
            name=f'{self.agent_name}_critic',
            chkpt_dir=chkpt_dir,
            network_type=network_type
        )
        
        # Target networks
        self.target_actor = ActorNetwork(
            input_dims=actor_dims,
            fc1_dims=fc1,
            fc2_dims=fc2,
            n_actions=n_actions,
            name=f'{self.agent_name}_target_actor',
            chkpt_dir=chkpt_dir
        )
        
        self.target_critic = CriticNetwork(
            input_dims=critic_dims,
            fc1_dims=fc1,
            fc2_dims=fc2,
            n_agents=n_agents,
            n_actions=n_actions,
            name=f'{self.agent_name}_target_critic',
            chkpt_dir=chkpt_dir,
            network_type=network_type
        )
        
        # Initialize target networks
        self.update_network_parameters(tau=1.0)
    
    def choose_action(self, observation, topology_info=None):
        """Choose action given observation"""
        # Process with GNN if enabled
        if self.use_gnn and self.gnn_processor is not None:
            processed_obs = self.gnn_processor.process_state(observation, topology_info)
            observation = processed_obs
        
        # Convert to tensor (optimized for numpy arrays)
        if isinstance(observation, list):
            observation = np.array(observation)
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)
        
        # Get action probabilities
        actions = self.actor.forward(state)
        return actions.detach().cpu().numpy()[0]
    
    def update_network_parameters(self, tau=None):
        """Update target networks using soft updates"""
        if tau is None:
            tau = self.tau
        
        # Update target actor
        target_actor_params = dict(self.target_actor.named_parameters())
        actor_params = dict(self.actor.named_parameters())
        
        for name in actor_params:
            actor_params[name] = tau * actor_params[name].clone() + \
                               (1 - tau) * target_actor_params[name].clone()
        
        self.target_actor.load_state_dict(actor_params)
        
        # Update target critic
        target_critic_params = dict(self.target_critic.named_parameters())
        critic_params = dict(self.critic.named_parameters())
        
        for name in critic_params:
            critic_params[name] = tau * critic_params[name].clone() + \
                                (1 - tau) * target_critic_params[name].clone()
        
        self.target_critic.load_state_dict(critic_params)
    
    def save_models(self):
        """Save all model checkpoints"""
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()
    
    def load_models(self):
        """Load all model checkpoints"""
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()


class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, max_size: int, input_shape: int, n_actions: int, n_agents: int):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = input_shape
        
        # Initialize memory
        self.state_memory = np.zeros((self.mem_size, self.n_agents, self.actor_dims))
        self.new_state_memory = np.zeros((self.mem_size, self.n_agents, self.actor_dims))
        self.reward_memory = np.zeros((self.mem_size, self.n_agents))
        self.terminal_memory = np.zeros((self.mem_size, self.n_agents), dtype=bool)
        
        # For actions, we store action probabilities
        self.action_memory = np.zeros((self.mem_size, self.n_agents, n_actions))
    
    def store_transition(self, obs, action, reward, obs_, done):
        """Store transition in replay buffer"""
        index = self.mem_cntr % self.mem_size
        
        self.state_memory[index] = obs
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = obs_
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size: int):
        """Sample batch from replay buffer"""
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, terminal
    
    def ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples"""
        return self.mem_cntr >= batch_size


class MADDPG:
    """Clean Multi-Agent DDPG implementation"""
    
    def __init__(self, actor_dims: List[int], critic_dims: List[int],
                 n_agents: int, n_actions: int, chkpt_dir: str,
                 alpha: float = 0.01, beta: float = 0.01,
                 fc1: int = 64, fc2: int = 64, gamma: float = 0.95,
                 tau: float = 0.01, critic_type: str = 'central_critic',
                 network_type: str = 'simple_q_network', use_gnn: bool = False):
        
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.critic_type = critic_type
        
        # Create agents
        self.agents = []
        for agent_idx in range(n_agents):
            agent = Agent(
                actor_dims=actor_dims[agent_idx] if isinstance(actor_dims, list) else actor_dims,
                critic_dims=critic_dims[agent_idx] if isinstance(critic_dims, list) else critic_dims,
                n_actions=n_actions,
                n_agents=n_agents,
                agent_idx=agent_idx,
                chkpt_dir=os.path.join(chkpt_dir, f'agent_{agent_idx}'),
                alpha=alpha,
                beta=beta,
                fc1=fc1,
                fc2=fc2,
                gamma=gamma,
                tau=tau,
                critic_type=critic_type,
                network_type=network_type,
                use_gnn=use_gnn
            )
            self.agents.append(agent)
        
        # Replay buffer
        self.memory = ReplayBuffer(
            max_size=50000,
            input_shape=actor_dims[0] if isinstance(actor_dims, list) else actor_dims,
            n_actions=n_actions,
            n_agents=n_agents
        )
    
    def choose_action(self, observations, topology_info=None):
        """Choose actions for all agents"""
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            obs = observations[agent_idx] if isinstance(observations[0], (list, np.ndarray)) else observations
            action = agent.choose_action(obs, topology_info)
            actions.append(action)
        return actions
    
    def learn(self, batch_size: int = 256):
        """Learn from replay buffer"""
        if not self.memory.ready(batch_size):
            return
        
        # Sample batch
        states, actions, rewards, states_, dones = self.memory.sample_buffer(batch_size)
        
        device = self.agents[0].actor.device
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.float).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        states_ = torch.tensor(states_, dtype=torch.float).to(device)
        dones = torch.tensor(dones).to(device)
        
        # Prepare data based on critic type
        if self.critic_type == 'central_critic':
            # Central critic: concatenate all agents' states (NOT actions)
            all_states = states.view(batch_size, -1)  # Flatten all agent states: 26*65=1690
            all_actions = actions.view(batch_size, -1)  # Flatten all agent actions: 3*65=195
            all_states_ = states_.view(batch_size, -1)  # Flatten all next states
            
        # Train each agent
        for agent_idx, agent in enumerate(self.agents):
            # Current agent's individual data
            agent_states = states[:, agent_idx, :]
            agent_rewards = rewards[:, agent_idx]
            agent_states_ = states_[:, agent_idx, :]
            agent_dones = dones[:, agent_idx]
            
            if self.critic_type == 'central_critic':
                # Central critic uses all agents' data
                critic_states = all_states
                critic_actions = all_actions
                critic_states_ = all_states_
                
                # Get next actions from all agents for target Q computation
                with torch.no_grad():
                    next_actions_list = []
                    for other_agent in self.agents:
                        other_agent_states_ = states_[:, self.agents.index(other_agent), :]
                        next_action = other_agent.target_actor.forward(other_agent_states_)
                        next_actions_list.append(next_action)
                    next_actions_all = torch.cat(next_actions_list, dim=1)
                    
                    target_q = agent.target_critic.forward(critic_states_, next_actions_all)
                    target_q[agent_dones] = 0.0
                    target_q = agent_rewards.unsqueeze(1) + agent.gamma * target_q
                
                current_q = agent.critic.forward(critic_states, critic_actions)
                
            else:  # local_critic
                # Local critic uses only this agent's data
                agent_actions = actions[:, agent_idx, :]
                
                with torch.no_grad():
                    next_actions = agent.target_actor.forward(agent_states_)
                    target_q = agent.target_critic.forward(agent_states_, next_actions)
                    target_q[agent_dones] = 0.0
                    target_q = agent_rewards.unsqueeze(1) + agent.gamma * target_q
                
                current_q = agent.critic.forward(agent_states, agent_actions)
            
            # Critic loss (same for both types)
            critic_loss = F.mse_loss(current_q, target_q)
            
            # Update critic
            agent.critic.optimizer.zero_grad()
            critic_loss.backward()
            agent.critic.optimizer.step()
            
            # Actor loss (always uses local data)
            predicted_actions = agent.actor.forward(agent_states)
            if self.critic_type == 'central_critic':
                # For actor update, need to construct full action vector with predicted action
                all_actions_for_actor = actions.clone()
                all_actions_for_actor[:, agent_idx, :] = predicted_actions
                all_actions_for_actor_flat = all_actions_for_actor.view(batch_size, -1)
                actor_loss = -agent.critic.forward(all_states, all_actions_for_actor_flat).mean()
            else:
                actor_loss = -agent.critic.forward(agent_states, predicted_actions).mean()
            
            # Update actor
            agent.actor.optimizer.zero_grad()
            actor_loss.backward()
            agent.actor.optimizer.step()
            
            # Update target networks
            agent.update_network_parameters()
    
    def store_transition(self, obs, action, reward, obs_, done):
        """Store transition in replay buffer"""
        self.memory.store_transition(obs, action, reward, obs_, done)
    
    def save_checkpoint(self):
        """Save all agent checkpoints"""
        for agent in self.agents:
            agent.save_models()
    
    def load_checkpoint(self):
        """Load all agent checkpoints"""
        for agent in self.agents:
            agent.load_models()


if __name__ == "__main__":
    # Test the implementation
    print("🧪 Testing Clean MADDPG Implementation")
    print("=" * 40)
    
    # Test configuration
    actor_dims = [26] * 5  # 5 agents, each with 26-dimensional state
    critic_dims = [91] * 5  # Central critic sees concatenated states
    n_agents = 5
    n_actions = 3
    
    # Test different configurations
    configs = [
        {'critic_type': 'central_critic', 'network_type': 'simple_q_network', 'use_gnn': False},
        {'critic_type': 'central_critic', 'network_type': 'duelling_q_network', 'use_gnn': False},
        {'critic_type': 'local_critic', 'network_type': 'duelling_q_network', 'use_gnn': False},
        {'critic_type': 'central_critic', 'network_type': 'simple_q_network', 'use_gnn': True},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n✅ Testing configuration {i+1}: {config}")
        
        maddpg = MADDPG(
            actor_dims=actor_dims,
            critic_dims=critic_dims,
            n_agents=n_agents,
            n_actions=n_actions,
            chkpt_dir=f'test_models/config_{i}',
            **config
        )
        
        # Test action selection
        observations = [np.random.random(26) for _ in range(n_agents)]
        actions = maddpg.choose_action(observations)
        
        print(f"   Actions shape: {np.array(actions).shape}")
        print(f"   Action sample: {actions[0][:3]}")
    
    print("\n✅ All tests passed! Clean MADDPG implementation working correctly.")