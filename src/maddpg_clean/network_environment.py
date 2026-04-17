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
BETA  = 2.0   # congestion penalty (only fires above CONGESTION_THRESHOLD)
GAMMA = 0.4   # utilisation variance penalty (load-balance incentive)
CONGESTION_THRESHOLD = 0.7  # utilisation below this is not penalised

# ── Packet parameters ─────────────────────────────────────────────────────────
PACKET_SIZE    = 0.03   # normalised bandwidth cost per packet per hop
TTL_INIT       = 15     # max hops before drop (diameter ≈ 6, so 2× headroom)
UTIL_DECAY     = 0.95   # per-step link-utilisation decay
INIT_PACKETS   = 30     # packets injected at episode reset
INJECT_EVERY   = 20     # inject N_INJECT_BATCH new packets every N steps
N_INJECT_BATCH = 10
DEST_BUCKETS = ("access", "dist", "core")
N_DEST_BUCKETS = len(DEST_BUCKETS)
K_PATHS = 3          # K shortest paths per access-destination for per-packet routing
SHAPING_WEIGHT = 0.4  # potential-based hop-progress reward shaping weight


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

        # K-shortest-paths cache — used by MADDPG per-destination routing.
        # Only (src, access_dst) pairs: core/dist nodes are not traffic endpoints.
        # Paths are deduplicated by first hop to maximise action diversity.
        self.kpath_cache: Dict[Tuple[str, str], List[List[str]]] = {}
        for src in self.hosts:
            for dst in self.access_nodes:
                if src != dst:
                    self.kpath_cache[(src, dst)] = self._build_distinct_kpaths(
                        self.graph, src, dst, K_PATHS
                    )

    @staticmethod
    def _build_distinct_kpaths(graph, src: str, dst: str, k: int) -> List[List[str]]:
        """Return up to k paths with distinct first hops, shortest first.

        Oversamples by 4× to find distinct first-hop alternatives; pads
        with the shortest path if fewer than k distinct first hops exist.
        """
        try:
            seen_fh: set = set()
            distinct: List[List[str]] = []
            for p in itertools.islice(
                nx.shortest_simple_paths(graph, src, dst), k * 4
            ):
                fh = p[1] if len(p) >= 2 else None
                if fh is not None and fh not in seen_fh:
                    seen_fh.add(fh)
                    distinct.append(p)
                    if len(distinct) == k:
                        break
            while distinct and len(distinct) < k:
                distinct.append(distinct[0])
            return distinct
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

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
        # Compute topology max degree — used to size the state/action vectors.
        self.max_degree = max(d for _, d in G.degree())
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
        self.max_degree = max(d for _, d in G.degree())
        return G

    def _random_topology(self) -> nx.Graph:
        G = nx.erdos_renyi_graph(self.n_nodes, 0.1, seed=self._seed)
        mapping = {n: f"H{n}" for n in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        for u, v in G.edges():
            G[u][v].update(capacity=random.uniform(1.0, 5.0), utilization=0.0)
        self.core_nodes = self.dist_nodes = []
        self.access_nodes = list(G.nodes())
        self.max_degree = max(d for _, d in G.degree())
        return G

    # ── path-cache helpers ───────────────────────────────────────────────────

    def refresh_path_cache(self):
        """Recompute shortest-path cache after topology changes (e.g. link failures).
        Called by NetworkEngine._inject_failures so the OSPF baseline re-routes
        correctly rather than following stale pre-failure paths.
        """
        self.path_cache.clear()
        for src, dst in itertools.permutations(self.hosts, 2):
            try:
                self.path_cache[(src, dst)] = nx.shortest_path(self.graph, src, dst)
            except nx.NetworkXNoPath:
                self.path_cache[(src, dst)] = []
        self.kpath_cache.clear()
        for src in self.hosts:
            for dst in self.access_nodes:
                if src != dst:
                    self.kpath_cache[(src, dst)] = self._build_distinct_kpaths(
                        self.graph, src, dst, K_PATHS
                    )

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
                 n_nodes: int = 65, seed: int = 42,
                 reward_config: Optional[Dict[str, float]] = None):
        self.topology_type = topology_type
        self.n_nodes = n_nodes
        self.topo_seed = seed
        self.topology = NetworkTopology(topology_type, n_nodes, seed)
        self.time_step = 0
        self.offered_load_factor = 1.0
        self.packet_queue: Dict[str, List[Dict]] = defaultdict(list)
        self._episode_stats = self._blank_stats()
        base_reward = {
            'delivery_weight': ALPHA,
            'max_util_penalty': BETA,
            'var_util_penalty': GAMMA,
            'drop_penalty': 0.6,
            'backlog_penalty': 0.05,
            'congestion_threshold': CONGESTION_THRESHOLD,
            'progress_shaping_weight': SHAPING_WEIGHT,
        }
        self.reward_cfg = {**base_reward, **(reward_config or {})}

        # Expose topology max degree so calling code can read actor_dims once.
        self.max_neighbors: int = self.topology.max_degree

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

    def get_adjacency_indices(self) -> List[List[int]]:
        """Return per-agent neighbor indices into the get_all_hosts() ordering."""
        hosts = self.topology.hosts
        host_idx = {h: idx for idx, h in enumerate(hosts)}
        return [
            [host_idx[nb] for nb in self.topology.get_neighbors(host) if nb in host_idx]
            for host in hosts
        ]

    def reset(self):
        self.topology.reset_utils()
        self.packet_queue = defaultdict(list)
        self.time_step = 0
        self.offered_load_factor = 1.0
        self._episode_stats = self._blank_stats()
        self._inject_packets(int(max(1, round(INIT_PACKETS * self.offered_load_factor))))

    def reset_with_load(self, offered_load_factor: float = 1.0):
        """Reset environment state with a configurable offered-load multiplier."""
        self.topology.reset_utils()
        self.packet_queue = defaultdict(list)
        self.time_step = 0
        self.offered_load_factor = float(max(0.1, offered_load_factor))
        self._episode_stats = self._blank_stats()
        self._inject_packets(int(max(1, round(INIT_PACKETS * self.offered_load_factor))))

    @property
    def state_dims(self) -> int:
        """Total state-vector size: max_neighbors + 50 fixed slots."""
        return self.max_neighbors + 50

    def get_state(self, host: str) -> np.ndarray:
        """
        (MAX_NEIGHBORS + 50)-dimensional observation for one agent.

        Slot layout:
          [0 : MAX_NEIGHBORS]           available bandwidth to each neighbour
                                        (zero-padded when node has fewer neighbours)
          [MAX_NEIGHBORS]               normalised queue depth
          [MAX_NEIGHBORS+1]             destination diversity in queue
          [MAX_NEIGHBORS+2]             normalised timestep
          [MAX_NEIGHBORS+3 :
           MAX_NEIGHBORS+43]            one-hot presence flag per access node (40 slots)
                                        1.0 if that access node appears in local queue
          [MAX_NEIGHBORS+43 :
           MAX_NEIGHBORS+46]            queue share by destination bucket (access, dist, core)
          [MAX_NEIGHBORS+46]            mean adjacent link utilisation
          [MAX_NEIGHBORS+47]            max adjacent link utilisation
          [MAX_NEIGHBORS+48]            variance of adjacent link utilisations
          [MAX_NEIGHBORS+49]            global mean link utilisation

          Total = max_neighbors + 50  (61 for service_provider/65-node/seed=42)
        """
        mn   = self.max_neighbors
        nbrs = self.topology.get_neighbors(host)
        state = []

        # MAX_NEIGHBORS bandwidth slots (zero-padded)
        for i in range(mn):
            state.append(self.topology.avail_bw(host, nbrs[i]) if i < len(nbrs) else 0.0)

        # queue summary — 3 slots
        q = self.packet_queue[host]
        dsts_in_queue = {p['dst'] for p in q}
        state.append(min(len(q), 20) / 20.0)
        state.append(len(dsts_in_queue) / 65.0)
        state.append(self.time_step / 256.0)

        # destination indicator — 40 slots (access nodes only; traffic endpoints in SP network)
        for h in self.topology.access_nodes:
            state.append(1.0 if h in dsts_in_queue else 0.0)

        # per-destination-bucket queue demand (access/dist/core)
        bucket_counts = {bucket: 0 for bucket in DEST_BUCKETS}
        for pkt in q:
            bucket_counts[self.get_tier(pkt['dst'])] += 1
        q_len = max(1, len(q))
        for bucket in DEST_BUCKETS:
            state.append(bucket_counts[bucket] / q_len)

        # link statistics — 4 slots
        utils = [self.topology.get_util(host, nb) for nb in nbrs]
        if utils:
            state.append(float(np.mean(utils)))
            state.append(float(np.max(utils)))
            state.append(float(np.var(utils)) if len(utils) > 1 else 0.0)
        else:
            state.extend([0.0, 0.0, 0.0])
        edge_utils = [self.topology.get_util(u, v) for u, v in self.topology.graph.edges()]
        global_util = float(np.mean(edge_utils)) if edge_utils else 0.0
        state.append(global_util)

        # guarantee exactly (mn + 50) dims
        total = mn + 50
        state = state[:total]
        while len(state) < total:
            state.append(0.0)
        return np.array(state, dtype=np.float32)

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], Dict]:
        """Execute one hop for every agent simultaneously."""
        self.time_step += 1
        hosts = self.topology.hosts
        next_queue: Dict[str, List[Dict]] = defaultdict(list)
        step_sent = step_delivered = step_dropped = 0
        rewards = []
        reward_components = {
            'delivery_term': 0.0,
            'max_util_term': 0.0,
            'var_util_term': 0.0,
            'drop_term': 0.0,
            'backlog_term': 0.0,
            'shaping_term': 0.0,
        }

        for i, host in enumerate(hosts):
            action = actions[i] if i < len(actions) else np.array([1.0, 0.0, 0.0])
            nbrs   = self.topology.get_neighbors(host)
            q      = self.packet_queue[host]

            if not nbrs or not q:
                rewards.append(0.0)
                continue

            # choose next-hop per destination (K-shortest-paths, per-packet)
            kpath_next_hops = self._select_kpath_next_hops(action, host, nbrs)

            # forward packets
            delivered = dropped = 0
            progress_sum = 0.0
            for pkt in q:
                pkt_hops = int(pkt.get('hops', 0))
                pkt_created = int(pkt.get('created_at', 0))
                step_sent += 1
                dst = pkt['dst']
                if dst in kpath_next_hops:
                    chosen = kpath_next_hops[dst]
                else:
                    # Non-access destination: fall back to shortest-path next hop
                    _sp = self.topology.path_cache.get((host, dst), [])
                    chosen = _sp[1] if len(_sp) >= 2 and _sp[1] in nbrs else nbrs[0]

                if self.topology.avail_bw(host, chosen) < PACKET_SIZE:
                    dropped += 1
                    step_dropped += 1
                    self._episode_stats['dropped_delay_samples'].append(
                        max(0, self.time_step - pkt_created)
                    )
                    self._episode_stats['dropped_hop_samples'].append(pkt_hops)
                    continue
                # Hop-progress shaping: reward movement toward destination
                d_before = len(self.topology.path_cache.get((host, dst), [])) - 1
                d_after = len(self.topology.path_cache.get((chosen, dst), [])) - 1
                if d_before > 0:
                    progress_sum += (d_before - d_after) / d_before
                cur = self.topology.get_util(host, chosen)
                self.topology.set_util(host, chosen, cur + PACKET_SIZE)

                if chosen == pkt['dst']:
                    delivered += 1
                    step_delivered += 1
                    final_hops = pkt_hops + 1
                    self._episode_stats['delivered_delay_samples'].append(
                        max(0, self.time_step - pkt_created)
                    )
                    self._episode_stats['delivered_hop_samples'].append(final_hops)
                elif pkt['ttl'] > 0:
                    next_queue[chosen].append({
                        'dst': pkt['dst'],
                        'ttl': pkt['ttl'] - 1,
                        'created_at': pkt_created,
                        'hops': pkt_hops + 1,
                    })
                else:
                    dropped += 1
                    step_dropped += 1   # TTL expired
                    self._episode_stats['dropped_delay_samples'].append(
                        max(0, self.time_step - pkt_created)
                    )
                    self._episode_stats['dropped_hop_samples'].append(pkt_hops + 1)

            # per-agent reward
            total    = len(q)
            dr       = delivered / total
            drop_ratio = dropped / total
            backlog_ratio = min(len(q), 20) / 20.0
            utils    = [self.topology.get_util(host, nb) for nb in nbrs]
            mx_util  = max(utils)
            var_util = float(np.var(utils)) if len(utils) > 1 else 0.0
            delivery_term = self.reward_cfg['delivery_weight'] * dr
            # Congestion penalty: only fires above threshold so agents are not
            # disincentivised from forwarding traffic at normal utilisation levels.
            threshold = float(self.reward_cfg.get('congestion_threshold', CONGESTION_THRESHOLD))
            max_util_term = self.reward_cfg['max_util_penalty'] * max(0.0, mx_util - threshold)
            var_util_term = self.reward_cfg['var_util_penalty'] * var_util
            drop_term = self.reward_cfg['drop_penalty'] * drop_ratio
            backlog_term = self.reward_cfg['backlog_penalty'] * backlog_ratio
            shaping_weight = float(self.reward_cfg.get('progress_shaping_weight', 0.0))
            shaping_term = shaping_weight * progress_sum / max(1, total)

            rewards.append(
                delivery_term - max_util_term - var_util_term - drop_term - backlog_term + shaping_term
            )
            reward_components['delivery_term'] += delivery_term
            reward_components['max_util_term'] += max_util_term
            reward_components['var_util_term'] += var_util_term
            reward_components['drop_term'] += drop_term
            reward_components['backlog_term'] += backlog_term
            reward_components['shaping_term'] += shaping_term

        # bookkeeping
        self.topology.decay_utils()
        self.packet_queue = next_queue
        if self.time_step % INJECT_EVERY == 0:
            self._inject_packets(int(max(1, round(N_INJECT_BATCH * self.offered_load_factor))))

        self._episode_stats['packets_sent']      += step_sent
        self._episode_stats['packets_delivered'] += step_delivered
        self._episode_stats['packets_dropped']   += step_dropped
        self._episode_stats['total_reward']      += sum(rewards)
        self._episode_stats['steps'] += 1
        for k, v in reward_components.items():
            self._episode_stats['reward_component_sums'][k] += v

        current_backlog = sum(len(q) for q in self.packet_queue.values())
        self._episode_stats['backlog_sum'] += current_backlog
        self._episode_stats['backlog_peak'] = max(
            self._episode_stats['backlog_peak'], current_backlog
        )

        util_values = [self.topology.get_util(u, v) for u, v in self.topology.graph.edges()]
        mean_util = float(np.mean(util_values)) if util_values else 0.0
        max_util = float(np.max(util_values)) if util_values else 0.0
        self._episode_stats['step_mean_utils'].append(mean_util)
        self._episode_stats['step_max_utils'].append(max_util)

        next_states = [self.get_state(h) for h in hosts]
        info = {
            'packets_sent':      step_sent,
            'packets_delivered': step_delivered,
            'packets_dropped':   step_dropped,
            'delivery_rate':     step_delivered / max(1, step_sent) * 100,
            'packet_loss_rate':  step_dropped / max(1, step_sent) * 100,
            'network_utilization': mean_util,
            'max_link_utilization': max_util,
            'backlog_packets': current_backlog,
            'reward_components': {
                'delivery_term': reward_components['delivery_term'],
                'max_util_term': reward_components['max_util_term'],
                'var_util_term': reward_components['var_util_term'],
                'drop_term': reward_components['drop_term'],
                'backlog_term': reward_components['backlog_term'],
                'shaping_term': reward_components['shaping_term'],
            },
        }
        return next_states, rewards, info

    def _select_kpath_next_hops(self, action: np.ndarray, host: str, nbrs: List[str]) -> Dict[str, str]:
        """Select next-hop per access destination using K-shortest-path action vector.

        Action layout: [dst0_path0, dst0_path1, dst0_path2, dst1_path0, ...]
        Length = 40 access_nodes × K_PATHS = 120

        Returns: dict mapping each access destination → chosen next-hop neighbor.
        """
        result: Dict[str, str] = {}
        if not nbrs:
            return result

        for i, dst in enumerate(self.topology.access_nodes):
            start = i * K_PATHS
            if len(action) >= start + K_PATHS:
                logits = action[start:start + K_PATHS]
            else:
                logits = np.ones(K_PATHS, dtype=np.float64)
            path_idx = int(np.argmax(logits))

            paths = self.topology.kpath_cache.get((host, dst), [])

            # Try chosen path first, then alternatives, then path_cache fallback
            chosen = None
            path_order = [path_idx] + [j for j in range(len(paths)) if j != path_idx]
            for pi in path_order:
                if pi < len(paths) and len(paths[pi]) >= 2:
                    nxt = paths[pi][1]
                    if nxt in nbrs:
                        chosen = nxt
                        break

            if chosen is None:
                sp = self.topology.path_cache.get((host, dst), [])
                chosen = sp[1] if len(sp) >= 2 and sp[1] in nbrs else nbrs[0]

            result[dst] = chosen
        return result

    def get_episode_stats(self) -> Dict:
        s = self._episode_stats
        sent = max(1, s['packets_sent'])
        injected = max(1, s['injected_count'])
        steps = max(1, s['steps'])

        delivered_delay = np.array(s['delivered_delay_samples'], dtype=np.float32)
        delivered_hops = np.array(s['delivered_hop_samples'], dtype=np.float32)
        step_mean_utils = np.array(s['step_mean_utils'], dtype=np.float32)
        step_max_utils = np.array(s['step_max_utils'], dtype=np.float32)
        per_agent_steps = max(1, steps * len(self.topology.hosts))

        return {
            'total_reward': s['total_reward'],
            'packets_sent': s['packets_sent'],
            'packets_injected': s['injected_count'],
            'packets_delivered': s['packets_delivered'],
            'packets_dropped': s['packets_dropped'],
            # hop_delivery_frac: fraction of *forwarding events* that are deliveries
            # (kept for transparency but not the primary paper metric)
            'hop_delivery_frac': s['packets_delivered'] / sent * 100,
            # end_to_end_pdr: fraction of *injected* packets successfully delivered
            'end_to_end_pdr': s['packets_delivered'] / injected * 100,
            # packet_loss_rate: fraction of forwarding events ending in a drop
            'packet_loss_rate': s['packets_dropped'] / sent * 100,
            'goodput_per_step': s['packets_delivered'] / steps,
            'backlog_end': sum(len(q) for q in self.packet_queue.values()),
            'backlog_avg': s['backlog_sum'] / steps,
            'backlog_peak': s['backlog_peak'],
            'delay_mean': float(np.mean(delivered_delay)) if delivered_delay.size else 0.0,
            'delay_p95': float(np.percentile(delivered_delay, 95)) if delivered_delay.size else 0.0,
            'hops_mean': float(np.mean(delivered_hops)) if delivered_hops.size else 0.0,
            'hops_p95': float(np.percentile(delivered_hops, 95)) if delivered_hops.size else 0.0,
            'util_mean': float(np.nanmean(step_mean_utils)) if step_mean_utils.size else 0.0,
            'util_p95': float(np.nanpercentile(step_mean_utils, 95)) if step_mean_utils.size else 0.0,
            'util_max': float(np.nanmax(step_max_utils)) if step_max_utils.size else 0.0,
            'overload_step_fraction': float(
                np.mean(step_max_utils >= 0.80)
            ) if step_max_utils.size else 0.0,
            'reward_components': {
                k: float(v / per_agent_steps)
                for k, v in s['reward_component_sums'].items()
            },
            'reward_config': self.reward_cfg,
        }

    def get_link_utilization_distribution(self) -> np.ndarray:
        edges = list(self.topology.graph.edges())
        if not edges:
            return np.array([0.0], dtype=np.float32)
        return np.array(
            [self.topology.get_util(u, v) for u, v in edges],
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
                pkt_hops = int(pkt.get('hops', 0))
                pkt_created = int(pkt.get('created_at', 0))
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
                    next_queue[nxt].append({
                        'dst': pkt['dst'],
                        'ttl': pkt['ttl'] - 1,
                        'created_at': pkt_created,
                        'hops': pkt_hops + 1,
                    })
                else:
                    dropped += 1

        self.topology.decay_utils()
        self.packet_queue = next_queue
        if self.time_step % INJECT_EVERY == 0:
            self._inject_packets(int(max(1, round(N_INJECT_BATCH * self.offered_load_factor))))
        self.time_step += 1
        return {
            'packets_sent':      sent,
            'packets_delivered': delivered,
            'packets_dropped':   dropped,
            'packet_loss_rate':  dropped / max(1, sent) * 100,
        }

    def ospf_queue_level_step(self) -> Dict:
        """
        Queue-level OSPF-like baseline.

        One next-hop is selected per host queue per step using shortest-path
        affinity to queued destinations. This baseline matches MADDPG's
        queue-level decision granularity while still using shortest-path priors.
        """
        next_queue: Dict[str, List[Dict]] = defaultdict(list)
        sent = delivered = dropped = 0

        for host, q in list(self.packet_queue.items()):
            nbrs = self.topology.get_neighbors(host)
            if not nbrs:
                dropped += len(q)
                sent += len(q)
                continue

            chosen = self._select_queue_level_next_hop(host, q, nbrs)
            for pkt in q:
                pkt_hops = int(pkt.get('hops', 0))
                pkt_created = int(pkt.get('created_at', 0))
                sent += 1
                if self.topology.avail_bw(host, chosen) < PACKET_SIZE:
                    dropped += 1
                    continue
                cur = self.topology.get_util(host, chosen)
                self.topology.set_util(host, chosen, cur + PACKET_SIZE)
                if chosen == pkt['dst']:
                    delivered += 1
                elif pkt['ttl'] > 0:
                    next_queue[chosen].append({
                        'dst': pkt['dst'],
                        'ttl': pkt['ttl'] - 1,
                        'created_at': pkt_created,
                        'hops': pkt_hops + 1,
                    })
                else:
                    dropped += 1

        self.topology.decay_utils()
        self.packet_queue = next_queue
        if self.time_step % INJECT_EVERY == 0:
            self._inject_packets(int(max(1, round(N_INJECT_BATCH * self.offered_load_factor))))
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
            self.packet_queue[src].append({
                'dst': dst,
                'ttl': TTL_INIT,
                'created_at': self.time_step,
                'hops': 0,
            })
        self._episode_stats['injected_count'] += n

    def _select_queue_level_next_hop(self, host: str, queue: List[Dict], nbrs: List[str]) -> str:
        """Pick one next-hop for the whole queue using shortest-path affinity."""
        if not queue or not nbrs:
            return nbrs[0] if nbrs else host

        scores = np.zeros(len(nbrs), dtype=np.float64)
        for pkt in queue:
            dst = pkt['dst']
            for i, nb in enumerate(nbrs):
                path = self.topology.path_cache.get((nb, dst), [])
                if not path:
                    continue
                # Prefer neighbors that keep packets closer to destination.
                scores[i] += 1.0 / max(1, len(path) - 1)

        best_idx = int(np.argmax(scores))
        return nbrs[best_idx]

    @staticmethod
    def _blank_stats() -> Dict:
        return {
            'total_reward': 0.0,
            'packets_sent': 0,
            'packets_injected': 0,
            'injected_count': 0,
            'packets_delivered': 0,
            'packets_dropped': 0,
            'steps': 0,
            'backlog_sum': 0,
            'backlog_peak': 0,
            'delivered_delay_samples': [],
            'delivered_hop_samples': [],
            'dropped_delay_samples': [],
            'dropped_hop_samples': [],
            'step_mean_utils': [],
            'step_max_utils': [],
            'reward_component_sums': {
                'delivery_term': 0.0,
                'max_util_term': 0.0,
                'var_util_term': 0.0,
                'drop_term': 0.0,
                'backlog_term': 0.0,
                'shaping_term': 0.0,
            },
        }


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