"""
Clean Network Environment - Standalone implementation
No external dependencies, complete routing simulation
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import random
import json
import os
import itertools

class NetworkTopology:
    """Clean network topology management"""
    
    def __init__(self, topology_type: str = "service_provider", n_nodes: int = 65):
        self.topology_type = topology_type
        self.n_nodes = n_nodes
        self.graph = self.create_topology()
        self.hosts = list(self.graph.nodes())
        self.path_cache = {}  # Cache for shortest paths to speed up routing decisions
        for src, dst in itertools.permutations(self.hosts, 2):
            try:
                self.path_cache[(src, dst)] = nx.shortest_path(self.graph, src, dst)
            except nx.NetworkXNoPath:
                self.path_cache[(src, dst)] = []
        
    def create_topology(self) -> nx.Graph:
        """Create network topology based on type"""
        
        if self.topology_type == "service_provider":
            return self.create_service_provider_topology()
        elif self.topology_type == "grid":
            return self.create_grid_topology()
        elif self.topology_type == "random":
            return self.create_random_topology()
        else:
            raise ValueError(f"Unknown topology type: {self.topology_type}")
    
    def create_service_provider_topology(self) -> nx.Graph:
        """Create service provider-like topology (similar to student's 65-node setup)"""
        
        # Create hierarchical topology
        G = nx.Graph()
        
        # Add nodes
        for i in range(self.n_nodes):
            G.add_node(f"H{i}")
        
        # Create hierarchical structure
        nodes = list(G.nodes())
        
        # Core tier (10% of nodes)
        core_size = max(3, int(0.1 * self.n_nodes))
        core_nodes = nodes[:core_size]
        
        # Connect core nodes in full mesh
        for i in range(len(core_nodes)):
            for j in range(i + 1, len(core_nodes)):
                G.add_edge(core_nodes[i], core_nodes[j], 
                          capacity=10.0, utilization=0.0)
        
        # Distribution tier (30% of nodes)
        dist_size = int(0.3 * self.n_nodes)
        dist_nodes = nodes[core_size:core_size + dist_size]
        
        # Connect distribution nodes to core
        for dist_node in dist_nodes:
            core_node = random.choice(core_nodes)
            G.add_edge(dist_node, core_node,
                      capacity=8.0, utilization=0.0)
            
            # Connect some distribution nodes to each other
            if random.random() < 0.3:
                other_dist = random.choice([n for n in dist_nodes if n != dist_node])
                if not G.has_edge(dist_node, other_dist):
                    G.add_edge(dist_node, other_dist,
                              capacity=5.0, utilization=0.0)
        
        # Access tier (remaining nodes)
        access_nodes = nodes[core_size + dist_size:]
        
        # Connect access nodes to distribution tier
        for access_node in access_nodes:
            dist_node = random.choice(dist_nodes)
            G.add_edge(access_node, dist_node,
                      capacity=2.0, utilization=0.0)
            
            # Some access nodes connect to core directly
            if random.random() < 0.1:
                core_node = random.choice(core_nodes)
                if not G.has_edge(access_node, core_node):
                    G.add_edge(access_node, core_node,
                              capacity=1.0, utilization=0.0)
        self.core_nodes = core_nodes
        self.dist_nodes = dist_nodes
        self.access_nodes = access_nodes
        return G
    
    def create_grid_topology(self) -> nx.Graph:
        """Create grid topology for testing"""
        side_length = int(np.sqrt(self.n_nodes))
        G = nx.grid_2d_graph(side_length, side_length)
        
        # Convert to string labels and add edge attributes
        mapping = {node: f"H{i}" for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        
        for u, v in G.edges():
            G[u][v]['capacity'] = 1.0
            G[u][v]['utilization'] = 0.0
        
        return G
    
    def create_random_topology(self) -> nx.Graph:
        """Create random topology"""
        G = nx.erdos_renyi_graph(self.n_nodes, 0.1)
        
        mapping = {node: f"H{node}" for node in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        
        for u, v in G.edges():
            G[u][v]['capacity'] = random.uniform(1.0, 5.0)
            G[u][v]['utilization'] = 0.0
        
        return G
    
    def get_neighbors(self, host: str) -> List[str]:
        """Get neighboring hosts"""
        return list(self.graph.neighbors(host))
    
    def get_edge_capacity(self, src: str, dst: str) -> float:
        """Get edge capacity between two nodes"""
        if self.graph.has_edge(src, dst):
            return self.graph[src][dst]['capacity']
        return 0.0
    
    def get_edge_utilization(self, src: str, dst: str) -> float:
        """Get current edge utilization"""
        if self.graph.has_edge(src, dst):
            return self.graph[src][dst]['utilization']
        return 0.0
    
    def update_edge_utilization(self, src: str, dst: str, utilization: float):
        """Update edge utilization"""
        if self.graph.has_edge(src, dst):
            self.graph[src][dst]['utilization'] = max(0.0, min(1.0, utilization))
    
    def get_available_bandwidth(self, src: str, dst: str) -> float:
        """Get available bandwidth (normalized)"""
        if self.graph.has_edge(src, dst):
            utilization = self.graph[src][dst]['utilization']
            return max(0.0, 1.0 - utilization)
        return 0.0


class FlowManager:
    """Manage network flows and routing decisions"""
    
    def __init__(self, topology: NetworkTopology):
        self.topology = topology
        self.active_flows = {}
        self.flow_counter = 0
        self.packet_stats = {
            'sent': 0,
            'delivered': 0,
            'dropped': 0
        }
    
    def generate_random_flows(self, n_flows: int = 10) -> Dict:
        """Generate realistic flows — access nodes are sources/sinks, core is transit only."""
        flows = {}
        hosts = self.topology.hosts
        
        # Use access nodes as sources (realistic), dist nodes occasionally
        if hasattr(self.topology, 'access_nodes') and len(self.topology.access_nodes) > 1:
            source_pool = self.topology.access_nodes  # 40 access nodes
            # 90% access→access, 10% access→distribution
            sink_pool = self.topology.access_nodes + self.topology.dist_nodes
        else:
            source_pool = self.topology.hosts  # fallback for non-SP topologies
            sink_pool = self.topology.hosts

        for _ in range(n_flows):
            src = random.choice(source_pool)
            dst = random.choice([h for h in sink_pool if h != src])
            
            flow_id = f"flow_{self.flow_counter}"
            self.flow_counter += 1
            
            flows[flow_id] = {
                'src': src,
                'dst': dst,
                'packets': random.randint(3, 10),
                'priority': random.choice(['high', 'medium', 'low']),
                'created_time': 0
            }
        
        return flows
    
    def route_packet(self, src: str, dst: str, chosen_next_hop: str) -> Dict:
        """Route a packet and return routing result"""
        result = {
            'success': False,
            'dropped': False,
            'path_used': [src],
            'utilization_impact': 0.0
        }
        
        # Check if next hop is valid neighbor
        neighbors = self.topology.get_neighbors(src)
        if chosen_next_hop not in neighbors:
            result['dropped'] = True
            result['reason'] = 'invalid_next_hop'
            return result
        
        # Check link capacity
        available_bw = self.topology.get_available_bandwidth(src, chosen_next_hop)
        packet_size = random.uniform(0.01, 0.05)  # Normalized packet size
        
        if available_bw < packet_size:
            result['dropped'] = True
            result['reason'] = 'congestion'
            return result
        
        # Route packet (simplified - use shortest path for remaining hops)
        try:
            path = self.topology.path_cache.get((chosen_next_hop, dst), [])
            if not path:
                raise nx.NetworkXNoPath
            full_path = [src] + path
            
            # Update link utilizations along path
            total_utilization = 0
            for i in range(len(full_path) - 1):
                u, v = full_path[i], full_path[i+1]
                current_util = self.topology.get_edge_utilization(u, v)
                new_util = current_util + packet_size
                self.topology.update_edge_utilization(u, v, new_util)
                total_utilization += packet_size
            
            result['success'] = True
            result['path_used'] = full_path
            result['utilization_impact'] = total_utilization
            
        except nx.NetworkXNoPath:
            result['dropped'] = True
            result['reason'] = 'no_path'
        
        return result


class NetworkEngine:
    """Clean Network Engine - Core simulation logic"""
    
    def __init__(self, topology_type: str = "service_provider", n_nodes: int = 65):
        self.topology = NetworkTopology(topology_type, n_nodes)
        self.flow_manager = FlowManager(self.topology)
        self.time_step = 0
        self.episode_stats = {
            'total_reward': 0.0,
            'packets_sent': 0,
            'packets_delivered': 0,
            'packets_dropped': 0,
            'total_delay': 0.0
        }
        
        # Generate initial flows
        self.current_flows = self.flow_manager.generate_random_flows(10)
    
    def get_all_hosts(self) -> List[str]:
        """Get all network hosts"""
        return self.topology.hosts
    
    def get_state(self, host: str, mode: int = 1) -> np.ndarray:
        """Get network state for a specific host"""
        neighbors = self.topology.get_neighbors(host)
        
        # Base state: [bandwidth_to_neighbors, flow_info, destinations]
        state = []
        
        # Bandwidth availability to neighbors (up to 4 neighbors, pad if less)
        bandwidth_states = []
        for i in range(4):  # Max 4 neighbors as in original
            if i < len(neighbors):
                neighbor = neighbors[i]
                available_bw = self.topology.get_available_bandwidth(host, neighbor)
                bandwidth_states.append(available_bw)
            else:
                bandwidth_states.append(0.0)
        
        state.extend(bandwidth_states)
        
        # Flow information for this host
        host_flows = [f for f in self.current_flows.values() 
                     if f['src'] == host or f['dst'] == host]
        
        flow_info = [
            len(host_flows),  # Number of flows
            len([f for f in host_flows if f['src'] == host]),  # Outgoing flows  
            len([f for f in host_flows if f['dst'] == host]),  # Incoming flows
        ]
        state.extend(flow_info)
        
        # Destination information (simplified)
        all_hosts = self.get_all_hosts()
        dest_indicators = []
        for i, potential_dest in enumerate(all_hosts[:15]):  # Top 15 destinations
            if potential_dest != host:
                # Check if this host has flows to this destination
                has_flow = any(f['dst'] == potential_dest for f in host_flows if f['src'] == host)
                dest_indicators.append(1.0 if has_flow else 0.0)
            else:
                dest_indicators.append(0.0)
        
        # Pad to 15 destinations
        while len(dest_indicators) < 15:
            dest_indicators.append(0.0)
        
        state.extend(dest_indicators[:15])
        
        # Global network metrics
        total_utilization = np.mean([
            self.topology.get_edge_utilization(u, v)
            for u, v in self.topology.graph.edges()
        ])
        
        state.extend([
            total_utilization,  # Global network utilization
            self.time_step / 1000.0,  # Normalized time step
            len(self.current_flows) / 100.0,  # Normalized flow count
        ])
        
        # Ensure state is exactly 26 dimensions (4+3+15+3+1=26)
        if len(state) < 26:
            state.extend([0.0] * (26 - len(state)))
        elif len(state) > 26:
            state = state[:26]
        
        return np.array(state, dtype=np.float32)
    
    def get_number_neighbors(self, host: str) -> int:
        """Get number of neighbors for a host"""
        return len(self.topology.get_neighbors(host))
    
    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], Dict]:
        """Execute one simulation step"""
        self.time_step += 1
        
        # Process actions for each host
        hosts = self.get_all_hosts()
        rewards = []
        next_states = []
        
        step_packets_sent = 0
        step_packets_delivered = 0
        step_packets_dropped = 0
        
        for i, host in enumerate(hosts):
            if i >= len(actions):
                # Default action if not provided
                action = np.array([1.0, 0.0, 0.0])  # Choose first neighbor
            else:
                action = actions[i]
            
            # Convert action to next hop selection
            neighbors = self.topology.get_neighbors(host)
            if len(neighbors) == 0:
                reward = 0.0
            else:
                # Choose neighbor based on action probabilities
                if len(action) >= len(neighbors):
                    neighbor_probs = action[:len(neighbors)]
                else:
                    neighbor_probs = np.pad(action, (0, len(neighbors) - len(action)))
                
                # Normalize probabilities
                if np.sum(neighbor_probs) > 0:
                    neighbor_probs = neighbor_probs / np.sum(neighbor_probs)
                else:
                    neighbor_probs = np.ones(len(neighbors)) / len(neighbors)
                
                chosen_neighbor_idx = int(np.argmax(neighbor_probs))
                chosen_neighbor = neighbors[chosen_neighbor_idx]
                
                # Route packets from this host
                host_reward = 0.0
                host_packets_delivered_this_host = 0
                host_flows = [f for f in self.current_flows.values() if f['src'] == host]
                
                for flow in host_flows:
                    step_packets_sent += flow['packets']
              
                    # Route each packet in the flow
                    for _ in range(flow['packets']):
                        result = self.flow_manager.route_packet(
                            host, flow['dst'], chosen_neighbor
                        )
                        if result['success']:
                            host_packets_delivered_this_host += 1
                            step_packets_delivered += 1
                        else:
                            step_packets_dropped += 1
                
                host_packets_attempted = sum(flow['packets'] for flow in host_flows)
                delivery_rate = host_packets_delivered_this_host / max(1, host_packets_attempted)
                # --- Congestion and balance signal from adjacent links ---
                neighbor_utils = [
                    self.topology.get_edge_utilization(host, nb)
                    for nb in neighbors
                ]
                max_util = max(neighbor_utils) if neighbor_utils else 0.0
                util_var  = float(np.var(neighbor_utils)) if len(neighbor_utils) > 1 else 0.0

                # --- Combined reward ---
                ALPHA, BETA, GAMMA = 1.0, 0.8, 0.4
                reward = ALPHA * delivery_rate - BETA * max_util - GAMMA * util_var
            
            rewards.append(reward)
            
            # Get next state
            next_state = self.get_state(host)
            next_states.append(next_state)
        
        # Update episode statistics
        self.episode_stats['packets_sent'] += step_packets_sent
        self.episode_stats['packets_delivered'] += step_packets_delivered
        self.episode_stats['packets_dropped'] += step_packets_dropped
        self.episode_stats['total_reward'] += sum(rewards)
        
        # Decay link utilizations (network recovery)
        for u, v in self.topology.graph.edges():
            current_util = self.topology.get_edge_utilization(u, v)
            new_util = current_util * 0.95  # 5% decay per step
            self.topology.update_edge_utilization(u, v, new_util)
        
        # Generate new flows occasionally
        if self.time_step % 20 == 0:
            new_flows = self.flow_manager.generate_random_flows(3)
            self.current_flows.update(new_flows)
            while len(self.current_flows) > 15:  # Limit total flows
                oldest = next(iter(self.current_flows))
                del self.current_flows[oldest]
        
        # Remove old flows
        if self.time_step % 50 == 0:
            flows_to_remove = random.sample(
                list(self.current_flows.keys()),
                min(20, len(self.current_flows) // 2)
            )
            for flow_id in flows_to_remove:
                del self.current_flows[flow_id]
        
        # Create info dict
        info = {
            'packets_sent': step_packets_sent,
            'packets_delivered': step_packets_delivered,
            'packets_dropped': step_packets_dropped,
            'packet_loss_rate': step_packets_dropped / max(1, step_packets_sent) * 100,
            'network_utilization': np.mean([
                self.topology.get_edge_utilization(u, v)
                for u, v in self.topology.graph.edges()
            ])
        }
        
        return next_states, rewards, info
    
    def reset(self):
        """Reset the network environment"""
        # Reset link utilizations
        for u, v in self.topology.graph.edges():
            self.topology.update_edge_utilization(u, v, 0.0)
        
        # Reset time and statistics
        self.time_step = 0
        self.episode_stats = {
            'total_reward': 0.0,
            'packets_sent': 0,
            'packets_delivered': 0,
            'packets_dropped': 0,
            'total_delay': 0.0
        }
        
        # Generate new flows
        self.current_flows = self.flow_manager.generate_random_flows(10)
    
    def get_episode_stats(self) -> Dict:
        """Get episode statistics"""
        stats = self.episode_stats.copy()
        if stats['packets_sent'] > 0:
            stats['packet_loss_rate'] = (stats['packets_dropped'] / stats['packets_sent']) * 100
            stats['delivery_rate'] = (stats['packets_delivered'] / stats['packets_sent']) * 100
        else:
            stats['packet_loss_rate'] = 0.0
            stats['delivery_rate'] = 0.0
        
        return stats
   
    def get_link_utilization_distribution(self) -> np.ndarray:
        """Return array of utilization values for all links (for distribution analysis)."""
        return np.array([
            self.topology.get_edge_utilization(u, v)
            for u, v in self.topology.graph.edges()
        ], dtype=np.float32)


class NetworkEnv:
    """Clean Network Environment wrapper"""
    
    def __init__(self, network_engine: NetworkEngine):
        self.engine = network_engine
    
    def reset(self):
        """Reset environment"""
        self.engine.reset()
        
        # Return initial states for all hosts
        initial_states = []
        for host in self.engine.get_all_hosts():
            state = self.engine.get_state(host)
            initial_states.append(state)
        
        return initial_states
    
    def step(self, actions: List[np.ndarray]):
        """Environment step"""
        return self.engine.step(actions)
    
    def get_stats(self) -> Dict:
        """Get environment statistics"""
        return self.engine.get_episode_stats()


if __name__ == "__main__":
    # Test the clean network environment
    print("🌐 Testing Clean Network Environment")
    print("=" * 40)
    
    # Create environment
    engine = NetworkEngine("service_provider", 10)  # Smaller for testing
    env = NetworkEnv(engine)
    
    print(f"✅ Created network with {len(engine.get_all_hosts())} hosts")
    
    # Test reset
    initial_states = env.reset()
    print(f"✅ Initial states shape: {np.array(initial_states).shape}")
    
    # Test a few steps
    for step in range(5):
        # Random actions for all hosts
        actions = []
        for host in engine.get_all_hosts():
            n_neighbors = engine.get_number_neighbors(host)
            if n_neighbors > 0:
                action = np.random.dirichlet(np.ones(3))  # 3 possible actions
            else:
                action = np.array([1.0, 0.0, 0.0])
            actions.append(action)
        
        next_states, rewards, info = env.step(actions)
        
        print(f"Step {step + 1}:")
        print(f"  Avg reward: {np.mean(rewards):.2f}")
        print(f"  Packet loss: {info['packet_loss_rate']:.1f}%")
        print(f"  Network util: {info['network_utilization']:.2f}")
    
    # Test statistics
    final_stats = env.get_stats()
    print(f"\n📊 Final episode stats:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Clean network environment working correctly!")