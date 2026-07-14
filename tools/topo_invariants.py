"""
Reviewer W1(a): show our evaluation topology is structurally representative of the
service-provider class, so the class-level claims are not instance artefacts.

Data: Internet Topology Zoo GraphML (mroughan/InternetTopologyZoo on GitHub,
graphml/ dir). Run in the maddpg-exp image (networkx present):
    docker run --rm -v $PWD/topo_check:/w -w /w \
        -e ZOO_DIR=zoo -e OUR_TOPO=our_topo.txt maddpg-exp:latest \
        python topo_invariants.py

Quantify the two structural invariants the Paper 1 results rest on, across a
sample of real service-provider topologies (Internet Topology Zoo) plus ours:

  (i)  path diversity   -- can K=3 hop/edge-disjoint alternates be found between
                           provider edges? (mean edge-disjoint paths; fraction of
                           core pairs with >=3)
  (ii) failure redundancy -- fraction of links that are bridges (single points of
                           failure); low => the redundancy the results exploit.

Zoo topologies carry no capacity, so absolute capacity-headroom is reported only
for ours; the class argument is carried by the structural invariants above, which
ARE encoded in the public data.
"""
import glob, os, random, statistics as st
import networkx as nx

random.seed(0)
ZOO_DIR = os.environ.get("ZOO_DIR", "zoo")
OUR_TOPO = os.environ.get("OUR_TOPO")  # path to our edge-list text


def largest_cc(G):
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def edge_disjoint_stats(G, sample=600):
    """Mean # edge-disjoint paths and fraction >=3, over pairs of degree>=2 nodes
    in the 2-core (excludes leaves, matching the provider-edge routing model)."""
    core = nx.k_core(G, k=2)
    if core.number_of_nodes() < 2:
        return float("nan"), float("nan"), 0
    nodes = list(core.nodes())
    pairs = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:]]
    if len(pairs) > sample:
        pairs = random.sample(pairs, sample)
    counts = []
    for u, v in pairs:
        try:
            counts.append(len(list(nx.edge_disjoint_paths(core, u, v))))
        except (nx.NetworkXNoPath, nx.NetworkXError):
            counts.append(0)
    frac3 = sum(c >= 3 for c in counts) / len(counts)
    return st.mean(counts), frac3, len(pairs)


def bridge_fraction(G):
    return len(list(nx.bridges(G))) / max(G.number_of_edges(), 1)


def analyse(name, G):
    G = largest_cc(nx.Graph(G))  # simple, connected
    degs = [d for _, d in G.degree()]
    mean_edp, frac3, npairs = edge_disjoint_stats(G)
    return {
        "name": name, "N": G.number_of_nodes(), "E": G.number_of_edges(),
        "meandeg": st.mean(degs), "frac_deg>=3": sum(d >= 3 for d in degs) / len(degs),
        "bridge_frac": bridge_fraction(G), "mean_edp": mean_edp,
        "frac_pairs_>=3": frac3, "npairs": npairs,
    }


def load_our_topology(path):
    G = nx.Graph()
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 2:
            G.add_edge(p[0], p[1])
    return G


rows = []
for fp in sorted(glob.glob(os.path.join(ZOO_DIR, "*.graphml"))):
    try:
        G = nx.read_graphml(fp)
        rows.append(analyse(os.path.splitext(os.path.basename(fp))[0], G))
    except Exception as e:
        print(f"  skip {fp}: {e}")

if OUR_TOPO and os.path.exists(OUR_TOPO):
    rows.append(analyse("Ours(SP-86)", load_our_topology(OUR_TOPO)))

hdr = f"{'topology':16}{'N':>5}{'E':>5}{'deg':>6}{'d>=3':>7}{'bridge':>8}{'meanEDP':>9}{'>=3paths':>10}"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['name']:16}{r['N']:5d}{r['E']:5d}{r['meandeg']:6.2f}"
          f"{r['frac_deg>=3']:7.2f}{r['bridge_frac']:8.2f}{r['mean_edp']:9.2f}{r['frac_pairs_>=3']:10.2f}")

zoo = [r for r in rows if r["name"] != "Ours(SP-86)"]
if zoo:
    print("\nZoo SP-class summary (mean over %d topologies):" % len(zoo))
    for k, lab in [("bridge_frac", "bridge fraction"), ("mean_edp", "mean edge-disjoint paths"),
                   ("frac_pairs_>=3", "fraction of core pairs with >=3 disjoint paths")]:
        vals = [r[k] for r in zoo]
        print(f"  {lab:48} {st.mean(vals):.2f}  (range {min(vals):.2f}-{max(vals):.2f})")
