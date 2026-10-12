#!/usr/bin/env python3
"""Turn an Internet Topology Zoo graph into a provider-edge topology for the simulator.

The output is the edge-list format of NetworkEngine's 'service_provider_real'
topology ("node node capacity" per line; switches are named S*, endpoints
BS*, MECS*, CS*), loaded through topology.file in a config.

Construction, chosen to mirror the 86-node evaluation network:
  * the largest connected component, without self-loops, relabelled S1..Sn;
  * 14 provider-edge (PE) switches spread over the network by farthest-point
    selection on hop distance (deterministic, seeded with the highest-degree
    node), 7 hosting a site (BS<i> and MECS<i>) and 7 hosting a content server
    (CS<i>), alternating so that sites and servers interleave;
  * the Zoo carries no link capacities, so the core is provisioned for load:
    each switch-to-switch link gets a capacity proportional to its edge
    betweenness over PE-to-PE shortest paths, mapped linearly onto 300..1000
    and rounded to 50 (the range of the evaluation network); endpoint links get
    1000, as in the evaluation network.

    python tools/zoo_to_sp.py zoo/Deltacom.graphml configs/topologies/deltacom_sp.txt
"""
import sys

import networkx as nx

N_SITES = N_CS = 7


def main(src, dst):
    G = nx.Graph(nx.read_graphml(src))
    G.remove_edges_from(nx.selfloop_edges(G))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    order = sorted(G.nodes(), key=lambda n: (-G.degree(n), str(n)))
    G = nx.relabel_nodes(G, {n: f'S{i + 1}' for i, n in enumerate(order)})
    dist = dict(nx.all_pairs_shortest_path_length(G))

    # farthest-point PE selection among nodes of degree >= 2 (no stub routers)
    cand = [n for n in G if G.degree(n) >= 2]
    pes = [max(cand, key=lambda n: (G.degree(n), -int(n[1:])))]
    while len(pes) < N_SITES + N_CS:
        pes.append(max((n for n in cand if n not in pes),
                       key=lambda n: (min(dist[n][p] for p in pes), -int(n[1:]))))
    site_pes, cs_pes = pes[0::2], pes[1::2]

    bet = nx.edge_betweenness_centrality_subset(G, sources=pes, targets=pes, normalized=False)
    lo, hi = min(bet.values()), max(bet.values())
    cap = {e: 300 + 700 * (b - lo) / max(hi - lo, 1e-9) for e, b in bet.items()}

    lines = [f'# {src}: {G.number_of_nodes()} switches, {G.number_of_edges()} switch links; '
             f'site PEs {site_pes}, CS PEs {cs_pes}; core capacity by PE-to-PE edge betweenness']
    for i, pe in enumerate(site_pes, 1):
        lines += [f'BS{i} {pe} 1000', f'MECS{i} {pe} 1000']
    for i, pe in enumerate(cs_pes, 1):
        lines.append(f'CS{i} {pe} 1000')
    for (u, v), c in sorted(cap.items(), key=lambda kv: (int(kv[0][0][1:]), int(kv[0][1][1:]))):
        lines.append(f'{u} {v} {int(round(c / 50.0) * 50)}')
    with open(dst, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    degs = [d for _, d in G.degree()]
    print(f'{src}: {G.number_of_nodes()} switches, {G.number_of_edges()} links, max degree {max(degs)}, '
          f'PE hop distances min {min(dist[a][b] for a in pes for b in pes if a != b)}, '
          f'site PEs {site_pes}, CS PEs {cs_pes}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
