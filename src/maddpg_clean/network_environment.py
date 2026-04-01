"""
Network Environment — Hop-by-Hop MADDPG Routing
All 65 nodes are agents. Each step every node forwards one packet
from its local queue to a chosen neighbour (action = argmax over
neighbour softmax). The path_cache is used only by the OSPF
baseline evaluator; it plays no role during RL training.
"""

import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import random
import itertools

# ── Reward hyper-parameters ──────────────────────────────────────────────────
ALPHA = 1.0   # delivery weight
BETA  = 0.8   # max-utilisation penalty
GAMMA = 0.4   # utilisation variance penalty (load-balance incentive)

# ── Packet parameters ─────────────────────────────────────────────────────────
PACKET_SIZE    = 0.03   # normalised bandwidth cost per packet per hop
TTL_INIT       = 15     # max hops before drop (diameter ≈ 6, so 2× headroom)
UTIL_DECAY     = 0.95   # per-step link-utilisation decay
INIT_PACKETS   = 30     # packets injected at episode reset
INJECT_EVERY   = 20     # inject N_INJECT_BATCH new packets every N steps
N_INJECT_BATCH = 10


class NetworkTopology:
    """Hierarchical service-provider topology (65 nodes, 3 tiers)."""

    def __init__(self, topology_type: str = "service_provider",
                 n_nodes: int = 65, seed: int = 42):
        self.topology_type = topology_type
        self.n_nodes = n_nodes
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.graph = self._build()
        self.hosts = list(self.graph.nodes())

        # Shortest-path cache — used by OSPF baseline ONLY (not by RL agents).
        self.path_cache: Dict[Tuple, List] = {}
        for src, dst in itertools.permutations(self.hosts, 2):
            try:
                self.path_cache[(src, dst)] = nx.shortest_path(self.graph, src, dst)
            except nx.NetworkXNoPath:
                self.path_cache[(src, dst)] = []

    # ── topology builders ────────────────────────────────────────────────────

    def _build(self) -> nx.Graph:
        builders = {
            "service_provider": self._sp_topology,
            "grid":             self._grid_topology,
            "random":           self._random_topology,
        }
        if self.topology_type not in builders:
            raise ValueError(f"Unknown topology: {self.topology_type}")
        return builders[self.topology_type]()

    def _sp_topology(self) -> nx.Graph:
        G = nx.Graph()
        nodes = [f"H{i}" for i in range(self.n_nodes)]
        for n in nodes:
            G.add_node(n)

        core_size = max(3, int(0.10 * self.n_nodes))   #  6 core nodes
        dist_size = int(0.30 * self.n_nodes)            # 19 distribution nodes
        # remaining 40 = access nodes

        core_nodes   = nodes[:core_size]
        dist_nodes   = nodes[core_size:core_size + dist_size]
        access_nodes = nodes[core_size + dist_size:]

        # Core: full mesh — high capacity
        for i, j in itertools.combinations(core_nodes, 2):
            G.add_edge(i, j, capacity=10.0, utilization=0.0)

        # Distribution: connect to 1–2 core nodes
        for d in dist_nodes:
            G.add_edge(d, random.choice(core_nodes), capacity=8.0, utilization=0.0)
            if random.random() < 0.3:
                others = [n for n in dist_nodes if n != d]
                if others:
                    peer = random.choice(others)
                    if not G.has_edge(d, peer):
                        G.add_edge(d, peer, capacity=5.0, utilization=0.0)

        # Access: connect to 1 distribution node; 10% also direct to core
        for a in access_nodes:
            G.add_edge(a, random.choice(dist_nodes), capacity=2.0, utilization=0.0)
            if random.random() < 0.10:
                c = random.choice(core_nodes)
                if not G.has_edge(a, c):
                    G.add_edge(a, c, capacity=1.0, utilization=0.0)

        self.core_nodes   = core_nodes
        self.dist_nodes   = dist_nodes
        self.access_nodes = access_nodes
        return G

    def _grid_topology(self) -> nx.Graph:
        side = int(np.sqrt(self.n_nodes))
        G = nx.grid_2d_graph(side, side)
        mapping = {n: f"H{i}" for i, n in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        for u, v in G.edges():
            G[u][v].update(capacity=1.0, utilization=0.0)
        self.core_nodes = self.dist_nodes = []
        self.access_nodes = list(G.nodes())
        return G

    def _random_topology(self) -> nx.Graph:
        G = nx.erdos_renyi_graph(self.n_nodes, 0.1, seed=self._seed)
        mapping = {n: f"H{n}" for n in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        for u, v in G.edges():
            G[u][v].update(capacity=random.uniform(1.0, 5.0), utilization=0.0)
        self.core_nodes = self.dist_nodes = []
        self.access_nodes = list(G.nodes())
        return G

    # ── edge helpers ─────────────────────────────────────────────────────────

    def get_neighbors(self, node: str) -> List[str]:
        return list(self.graph.neighbors(node))

    def get_util(self, u: str, v: str) -> float:
        return self.graph[u][v]['utilization'] if self.graph.has_edge(u, v) else 0.0

    def set_util(self, u: str, v: str, val: float):
        if self.graph.has_edge(u, v):
            self.graph[u][v]['utilization'] = float(np.clip(val, 0.0, 1.0))

    def avail_bw(self, u: str, v: str) -> float:
        return max(0.0, 1.0 - self.get_util(u, v)) if self.graph.has_edge(u, v) else 0.0

    def reset_utils(self):
        for u, v in self.graph.edges():
            self.graph[u][v]['utilization'] = 0.0

    def decay_utils(self):
        for u, v in self.graph.edges():
            self.set_util(u, v, self.get_util(u, v) * UTIL_DECAY)


class NetworkEngine:
    """
    Hop-by-hop routing engine.

    At every timestep each of the 65 agents receives the queue of packets
    sitting at its node and forwards them through the chosen next-hop
    (argmax of the action vector over its neighbours).
    """

    def __init__(self, topology_type: str = "service_provider",
                 n_nodes: int = 65, seed: int = 42):
        self.topology = NetworkTopology(topology_type, n_nodes, seed)
        self.time_step = 0
        self.packet_queue: Dict[str, List[Dict]] = defaultdict(list)
        self._episode_stats = self._blank_stats()

        # Node-tier lookup — used by Paper-2 attack surface analysis
        self._tier: Dict[str, str] = {}
        for n in self.topology.core_nodes:   self._tier[n] = 'core'
        for n in self.topology.dist_nodes:   self._tier[n] = 'dist'
        for n in self.topology.access_nodes: self._tier[n] = 'access'

    # ── public API ───────────────────────────────────────────────────────────

    def get_all_hosts(self) -> List[str]:
        return self.topology.hosts

    def get_number_neighbors(self, host: str) -> int:
        return len(self.topology.get_neighbors(host))

    def get_tier(self, host: str) -> str:
        """Return 'core' | 'dist' | 'access' — used by attack-surface analysis."""
        return self._tier.get(host, 'access')

    def reset(self):
        self.topology.reset_utils()
        self.packet_queue = defaultdict(list)
        self.time_step = 0
        self._episode_stats = self._blank_stats()
        self._inject_packets(INIT_PACKETS)

    def get_state(self, host: str) -> np.ndarray:
        """
        26-dimensional observation for one agent:
          [0:4]   available bandwidth to neighbours (up to 4, zero-padded)
          [4]     normalised queue depth at this node
          [5]     destination diversity in queue
          [6]     normalised timestep
          [7:22]  one-hot: does queue contain a packet for each of
                  the first 15 hosts?
          [22]    mean adjacent link utilisation
          [23]    max adjacent link utilisation
          [24]    variance of adjacent link utilisation
          [25]    global mean link utilisation
        """
        nbrs  = self.topology.get_neighbors(host)
        state = []

        # 4 bandwidth slots
        for i in range(4):
            state.append(self.topology.avail_bw(host, nbrs[i]) if i < len(nbrs) else 0.0)

        # queue summary
        q = self.packet_queue[host]
        dsts_in_queue = {p['dst'] for p in q}
        state.append(min(len(q), 20) / 20.0)
        state.append(len(dsts_in_queue) / 65.0)
        state.append(self.time_step / 256.0)

        # destination indicator — 15 slots
        for h in self.topology.hosts[:15]:
            state.append(1.0 if h in dsts_in_queue else 0.0)
        while len(state) < 22:
            state.append(0.0)

        # link statistics — 4 slots
        utils = [self.topology.get_util(host, nb) for nb in nbrs]
        if utils:
            state.append(float(np.mean(utils)))
            state.append(float(np.max(utils)))
            state.append(float(np.var(utils)) if len(utils) > 1 else 0.0)
        else:
            state.extend([0.0, 0.0, 0.0])
        global_util = float(np.mean(
            [self.topology.get_util(u, v) for u, v in self.topology.graph.edges()]
        ))
        state.append(global_util)

        # guarantee exactly 26 dims
        state = state[:26]
        while len(state) < 26:
            state.append(0.0)
        return np.array(state, dtype=np.float32)

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], Dict]:
        """Execute one hop for every agent simultaneously."""
        self.time_step += 1
        hosts = self.topology.hosts
        next_queue: Dict[str, List[Dict]] = defaultdict(list)
        step_sent = step_delivered = step_dropped = 0
        rewards = []

        for i, host in enumerate(hosts):
            action = actions[i] if i < len(actions) else np.array([1.0, 0.0, 0.0])
            nbrs   = self.topology.get_neighbors(host)
            q      = self.packet_queue[host]

            if not nbrs or not q:
                rewards.append(0.0)
                continue

            # choose next-hop
            probs = np.array(action[:len(nbrs)], dtype=np.float64)
            if probs.sum() > 0:
                probs /= probs.sum()
            else:
                probs = np.ones(len(nbrs)) / len(nbrs)
            chosen = nbrs[int(np.argmax(probs))]

            # forward packets
            delivered = dropped = 0
            for pkt in q:
                step_sent += 1
                if self.topology.avail_bw(host, chosen) < PACKET_SIZE:
                    dropped += 1
                    step_dropped += 1
                    continue
                cur = self.topology.get_util(host, chosen)
                self.topology.set_util(host, chosen, cur + PACKET_SIZE)

                if chosen == pkt['dst']:
                    delivered += 1
                    step_delivered += 1
                elif pkt['ttl'] > 0:
                    next_queue[chosen].append({'dst': pkt['dst'], 'ttl': pkt['ttl'] - 1})
                else:
                    dropped += 1
                    step_dropped += 1   # TTL expired

            # per-agent reward
            total    = len(q)
            dr       = delivered / total
            utils    = [self.topology.get_util(host, nb) for nb in nbrs]
            mx_util  = max(utils)
            var_util = float(np.var(utils)) if len(utils) > 1 else 0.0
            rewards.append(ALPHA * dr - BETA * mx_util - GAMMA * var_util)

        # bookkeeping
        self.topology.decay_utils()
        self.packet_queue = next_queue
        if self.time_step % INJECT_EVERY == 0:
            self._inject_packets(N_INJECT_BATCH)

        self._episode_stats['packets_sent']      += step_sent
        self._episode_stats['packets_delivered'] += step_delivered
        self._episode_stats['packets_dropped']   += step_dropped
        self._episode_stats['total_reward']      += sum(rewards)

        next_states = [self.get_state(h) for h in hosts]
        info = {
            'packets_sent':      step_sent,
            'packets_delivered': step_delivered,
            'packets_dropped':   step_dropped,
            'packet_loss_rate':  step_dropped / max(1, step_sent) * 100,
            'network_utilization': float(np.mean(
                [self.topology.get_util(u, v) for u, v in self.topology.graph.edges()]
            )),
        }
        return next_states, rewards, info

    def get_episode_stats(self) -> Dict:
        s    = self._episode_stats.copy()
        sent = max(1, s['packets_sent'])
        s['packet_loss_rate'] = s['packets_dropped'] / sent * 100
        s['delivery_rate']    = s['packets_delivered'] / sent * 100
        return s

    def get_link_utilization_distribution(self) -> np.ndarray:
        return np.array(
            [self.topology.get_util(u, v) for u, v in self.topology.graph.edges()],
            dtype=np.float32
        )

    # ── OSPF / shortest-path baseline ────────────────────────────────────────

    def ospf_step(self) -> Dict:
        """
        One step under deterministic shortest-path routing.
        Used ONLY during Paper-1 evaluation to produce the OSPF baseline.
        RL agents play no role here.
        """
        next_queue: Dict[str, List[Dict]] = defaultdict(list)
        sent = delivered = dropped = 0

        for host, q in list(self.packet_queue.items()):
            for pkt in q:
                sent += 1
                path = self.topology.path_cache.get((host, pkt['dst']), [])
                if len(path) < 2:
                    dropped += 1
                    continue
                nxt = path[1]
                if self.topology.avail_bw(host, nxt) < PACKET_SIZE:
                    dropped += 1
                    continue
                cur = self.topology.get_util(host, nxt)
                self.topology.set_util(host, nxt, cur + PACKET_SIZE)
                if nxt == pkt['dst']:
                    delivered += 1
                elif pkt['ttl'] > 0:
                    next_queue[nxt].append({'dst': pkt['dst'], 'ttl': pkt['ttl'] - 1})
                else:
                    dropped += 1

        self.topology.decay_utils()
        self.packet_queue = next_queue
        if self.time_step % INJECT_EVERY == 0:
            self._inject_packets(N_INJECT_BATCH)
        self.time_step += 1
        return {
            'packets_sent':      sent,
            'packets_delivered': delivered,
            'packets_dropped':   dropped,
            'packet_loss_rate':  dropped / max(1, sent) * 100,
        }

    # ── internals ─────────────────────────────────────────────────────────────

    def _inject_packets(self, n: int):
        srcs = self.topology.access_nodes or self.topology.hosts
        dsts = (self.topology.access_nodes + self.topology.dist_nodes) or self.topology.hosts
        for _ in range(n):
            src = random.choice(srcs)
            dst = random.choice([h for h in dsts if h != src])
            self.packet_queue[src].append({'dst': dst, 'ttl': TTL_INIT})

    @staticmethod
    def _blank_stats() -> Dict:
        return {'total_reward': 0.0, 'packets_sent': 0,
                'packets_delivered': 0, 'packets_dropped': 0}


class NetworkEnv:
    """Thin wrapper — matches the interface used by StandaloneExperimentRunner."""

    def __init__(self, engine: NetworkEngine):
        self.engine = engine

    def reset(self) -> List[np.ndarray]:
        self.engine.reset()
        return [self.engine.get_state(h) for h in self.engine.get_all_hosts()]

    def step(self, actions: List[np.ndarray]):
        return self.engine.step(actions)

    def get_stats(self) -> Dict:
        return self.engine.get_episode_stats()