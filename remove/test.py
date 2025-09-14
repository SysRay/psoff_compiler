# cfg2rvsdg.py
# Practical implementation of Section 4 (RESTRUCTURECFG + BUILDRVSDG)
# of "Perfect Reconstructability of Control Flow..." (user-supplied PDF).
#
# NOTE: This is a pragmatic implementation, not a formal mechanically
# verified transcription. It follows the loop/branch restructuring design
# from Section 4 (Tarjan SCC -> loop wrapping; head/branch/tail partition).
#
# Citations: Section 4 (loop & branch restructuring) and RVSDG building in the paper.
# :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}

from collections import defaultdict, deque, namedtuple
import itertools
import json
import copy
from typing import Dict, List, Set, Tuple, Optional

# --- Simple data structures -------------------------------------------------

# A CFG Node: id, region (start_pc, end_pc), successors (list of node ids), optional 'stmt' (string)


class CFGNode:
    def __init__(self, nid: str, start_pc: int, end_pc: int, stmt: Optional[str] = None):
        self.id = nid
        self.start_pc = start_pc
        self.end_pc = end_pc
        self.successors: List[str] = []
        self.predecessors: List[str] = []
        self.stmt = stmt or f"stmt_{nid}"

    def region(self):
        return (self.start_pc, self.end_pc)

    def __repr__(self):
        return f"CFGNode({self.id}, [{self.start_pc},{self.end_pc}], succ={self.successors})"

# Simple RVSDG node classes (hierarchical)


class RVNode:
    pass


class SimpleNode(RVNode):
    def __init__(self, nid, stmt, region):
        self.kind = "simple"
        self.id = nid
        self.stmt = stmt
        self.region = region  # (start_pc, end_pc)

    def to_dict(self):
        return {"kind": self.kind, "id": self.id, "stmt": self.stmt, "region": self.region}


class GammaNode(RVNode):
    def __init__(self, nid, predicate, regions: List['RVSDGRegion']):
        self.kind = "gamma"
        self.id = nid
        self.predicate = predicate  # the variable/pc that is predicate
        self.regions = regions  # list of RVSDGRegion (subgraphs)

    def to_dict(self):
        return {"kind": self.kind, "id": self.id, "predicate": self.predicate, "regions": [r.to_dict() for r in self.regions]}


class ThetaNode(RVNode):
    def __init__(self, nid, predicate, body: 'RVSDGRegion'):
        self.kind = "theta"
        self.id = nid
        self.predicate = predicate
        self.body = body

    def to_dict(self):
        return {"kind": self.kind, "id": self.id, "predicate": self.predicate, "body": self.body.to_dict()}

# Region is a small container for RVSDG contents (list of RVNodes)


class RVSDGRegion:
    def __init__(self, name):
        self.name = name
        self.nodes: List[RVNode] = []

    def add(self, node: RVNode):
        self.nodes.append(node)

    def to_dict(self):
        return {"region": self.name, "nodes": [n.to_dict() for n in self.nodes]}


# --- Graph utilities -------------------------------------------------------

def build_pred_succ(cfg_nodes: Dict[str, CFGNode]):
    for n in cfg_nodes.values():
        n.predecessors = []
    for n in cfg_nodes.values():
        for s in n.successors:
            if s in cfg_nodes:
                cfg_nodes[s].predecessors.append(n.id)


def tarjan_scc(cfg_nodes: Dict[str, CFGNode]) -> List[List[str]]:
    """
    Tarjan's algorithm for SCCs.
    Returns list of SCCs (each is list of node ids). Single-node SCCs for nodes with no self-loop.
    """
    index = {}
    lowlink = {}
    stack = []
    onstack = set()
    result = []
    current_index = 0

    def strongconnect(vid):
        nonlocal current_index
        index[vid] = current_index
        lowlink[vid] = current_index
        current_index += 1
        stack.append(vid)
        onstack.add(vid)

        for w in cfg_nodes[vid].successors:
            if w not in index:
                strongconnect(w)
                lowlink[vid] = min(lowlink[vid], lowlink[w])
            elif w in onstack:
                lowlink[vid] = min(lowlink[vid], index[w])

        if lowlink[vid] == index[vid]:
            scc = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                scc.append(w)
                if w == vid:
                    break
            result.append(scc)

    for v in cfg_nodes:
        if v not in index:
            strongconnect(v)
    return result


def reachable_from(entry: str, nodes: Dict[str, CFGNode], banned_edges: Set[Tuple[str, str]] = None) -> Set[str]:
    """BFS/DFS reachability from entry, ignoring any edges in banned_edges"""
    if banned_edges is None:
        banned_edges = set()
    seen = set()
    stack = [entry]
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        for s in nodes[v].successors:
            if (v, s) in banned_edges:
                continue
            if s not in seen and s in nodes:
                stack.append(s)
    return seen


def nodes_dominated_by_arc(entry: str, nodes: Dict[str, CFGNode], arc: Tuple[str, str]) -> Set[str]:
    """
    Approximate dominator set dominated by arc (u->v)
    Approach: remove arc (u->v) temporarily. Nodes that become unreachable from entry are dominated by that arc.
    This matches the paper's "dominator graph of arc a".
    """
    banned = {arc}
    reachable_with = reachable_from(entry, nodes, banned_edges=banned)
    # nodes dominated by arc are nodes that are reachable in the original graph but not when arc removed
    reachable_without = reachable_from(entry, nodes, banned_edges=set())
    dominated = set(reachable_without - reachable_with)
    return dominated

# --- High-level restructuring algorithms ----------------------------------


def restructure_loops(cfg_nodes: Dict[str, CFGNode], entry: str):
    """
    Loop restructuring: find SCCs with Tarjan, for each nontrivial SCC create a 'loop region' (Theta wrapper).
    We implement this by returning a 'supergraph' representation where each SCC is collapsed into a single meta-node
    and we produce a mapping scc_id -> member_nodes. We DO NOT mutate original nodes here; this function helps
    building the RVSDG recursively.
    Returns:
        scc_map: list of SCC lists (each list of node ids)
        membership: node_id -> scc_index (index into scc_map)
        meta_graph: adjacency between SCC indices (or single nodes)
    """
    sccs = tarjan_scc(cfg_nodes)
    # index SCCs, treat singletons as SCCs as well
    membership = {}
    for i, scc in enumerate(sccs):
        for v in scc:
            membership[v] = i

    meta_adj = defaultdict(set)
    for u in cfg_nodes:
        for v in cfg_nodes[u].successors:
            iu = membership[u]
            iv = membership.get(v, None)
            if iv is None:
                continue
            if iu != iv:
                meta_adj[iu].add(iv)

    return sccs, membership, meta_adj


def longest_linear_head(cfg_nodes: Dict[str, CFGNode], entry: str) -> List[str]:
    """
    Find longest linear chain (H) starting at entry where each node has exactly one successor
    and that successor has exactly one predecessor (linear chain property).
    We stop when hitting a branch (node with >1 successor) or a join (successor has multiple preds).
    """
    H = []
    cur = entry
    visited = set()
    while True:
        if cur in visited:
            break
        visited.add(cur)
        H.append(cur)
        succs = cfg_nodes[cur].successors
        if len(succs) != 1:
            break
        nxt = succs[0]
        # successor must have at most 1 predecessor to continue linearity
        if len(cfg_nodes[nxt].predecessors) > 1:
            break
        cur = nxt
    return H


def partition_head_branch_tail(cfg_nodes: Dict[str, CFGNode], entry: str):
    """
    Partition acyclic graph into H (head linear), Bj (branch dominator subgraphs for each fan-out of last head node), and T (tail).
    Implements the partitioning described in Section 4.2 using dominator-by-arc detection (reachability loss when removing arc).
    Returns a dict: { 'H': [...], 'B': [set(...), ...], 'T': set(...) , 'fanout_node': vB, 'fanout_arcs': [(vB, dst), ...] }
    """
    build_pred_succ(cfg_nodes)
    H = longest_linear_head(cfg_nodes, entry)
    last = H[-1]
    fanouts = cfg_nodes[last].successors
    # if last has no fanouts -> entire graph linear
    if not fanouts:
        return {"H": H, "B": [], "T": set(), "fanout_node": last, "fanout_arcs": []}

    # for each fanout arc last->dst compute dominated nodes
    B_prime = []
    fanout_arcs = []
    original_reachable = reachable_from(entry, cfg_nodes)
    for dst in fanouts:
        arc = (last, dst)
        dominated = nodes_dominated_by_arc(entry, cfg_nodes, arc)
        B_prime.append(set(dominated))
        fanout_arcs.append(arc)

    # The tail T is nodes not in H nor in any B_prime
    nodes_all = set(cfg_nodes.keys())
    union_Bprime = set().union(*B_prime) if B_prime else set()
    T = nodes_all - set(H) - union_Bprime

    # There is trimming step described in 4.2.1: if a continuation point in T has immediate predecessor in branch,
    # possibly pull predecessor into T unless all preds in branch. We implement a simple conservative trimming:
    # find continuation points = nodes in T with preds in branch subgraphs or in fanout arcs; if some continuation has
    # immediate predecessor in branch subgraph we pull that predecessor into T if not all preds are in branches.
    continuation_points = set()
    for n in T:
        for p in cfg_nodes[n].predecessors:
            if p in union_Bprime or p == last:
                continuation_points.add(n)
    # trimming: for each continuation point, if any immediate predecessor in branch subgraphs exists and NOT all preds are in branches,
    # then move that predecessor into T (i.e. remove it from branch subgraph). This mirrors the description in the paper.
    B = [set(b) for b in B_prime]
    for c in continuation_points:
        preds = cfg_nodes[c].predecessors
        # identify preds in branch subgraphs
        preds_in_branch = [p for p in preds if p in union_Bprime]
        for p in preds_in_branch:
            # find which B it belongs to
            b_idx = None
            for i, bset in enumerate(B):
                if p in bset:
                    b_idx = i
                    break
            if b_idx is None:
                continue
            # if not ALL preds are in branch subgraphs, pull this predecessor into T
            if not all((pp in union_Bprime) for pp in preds):
                B[b_idx].discard(p)
                T.add(p)
                union_Bprime.discard(p)

    # if some Bj becomes empty, we still keep a record of that (paper's algorithm keeps track)
    return {"H": H, "B": B, "T": T, "fanout_node": last, "fanout_arcs": fanout_arcs}

# --- RVSDG construction ---------------------------------------------------


_id_counter = itertools.count()


def fresh_id(prefix="n"):
    return f"{prefix}{next(_id_counter)}"


def build_rvsdg_from_region(node_ids: List[str], cfg_nodes_all: Dict[str, CFGNode], entry: str):
    """
    Build an RVSDGRegion from a list/set of node ids representing an acyclic subgraph or linear region.
    The routine will:
      - if the subgraph is single linear region with no branches -> create SimpleNodes for statements
      - else follow partitioning: head H, B_j, T -> create a GammaNode wrapping subregions
    This implements BUILDRVSDG* recursively for structured components (Section 4).
    """
    # Create local view of nodes
    nodes = {nid: copy.deepcopy(cfg_nodes_all[nid]) for nid in node_ids}
    build_pred_succ(nodes)

    # If nodes are empty, return empty region
    region = RVSDGRegion(name=f"region_{fresh_id('r')}")
    if not nodes:
        return region

    # If graph is linear (no branching points) just create simple nodes in topological order.
    branch_count = sum(1 for n in nodes.values() if len(n.successors) > 1)
    if branch_count == 0:
        # topological order via simple DFS (note: acyclic assumption here)
        topo = topo_sort_subgraph(nodes, entry)
        for vid in topo:
            nd = nodes[vid]
            sn = SimpleNode(nid=fresh_id("s"), stmt=nd.stmt,
                            region=nd.region())
            region.add(sn)
        return region

    # Otherwise apply partitioning into H, B, T
    part = partition_head_branch_tail(nodes, entry)
    H = part["H"]
    B_list = part["B"]
    T_set = part["T"]
    fanout_node = part["fanout_node"]

    # Build region: first add SimpleNodes for head
    for h in H:
        sn = SimpleNode(nid=fresh_id(
            "s"), stmt=nodes[h].stmt, region=nodes[h].region())
        region.add(sn)

    # Build gamma: each Bj becomes a subregion (convert set -> list entry)
    subregions = []
    for bset in B_list:
        if not bset:
            # empty branch -> represent as empty region (paper handles via p := k translations).
            subregions.append(RVSDGRegion(name=f"empty_{fresh_id('e')}"))
            continue
        # pick an entry for this branch subgraph: in the paper it's the target of the fanout arc
        # find the fanout arc whose dominated set includes these nodes
        # heuristic: choose smallest node id as entry if available
        entry_candidates = [n for n in bset if any(
            pred not in bset for pred in nodes[n].predecessors)]
        if entry_candidates:
            sub_entry = sorted(entry_candidates)[0]
        else:
            # fallback
            sub_entry = sorted(bset)[0]
        subreg = build_rvsdg_from_region(list(bset), cfg_nodes_all, sub_entry)
        subregions.append(subreg)

    # tail region
    tail_nodes = list(T_set)
    tail_entry = None
    if tail_nodes:
        # pick a tail entry: any node in tail with predecessor outside tail or head
        for t in tail_nodes:
            for p in cfg_nodes_all[t].predecessors:
                if p not in tail_nodes:
                    tail_entry = t
                    break
            if tail_entry:
                break
        if tail_entry is None:
            tail_entry = sorted(tail_nodes)[0]
        tail_region = build_rvsdg_from_region(
            tail_nodes, cfg_nodes_all, tail_entry)
    else:
        tail_region = RVSDGRegion(name=f"tail_empty_{fresh_id('t')}")

    # Predicate: use the fanout_node's stmt/predicate as gamma predicate (heuristic)
    predicate = f"pred_at_{fanout_node}"
    gamma = GammaNode(nid=fresh_id(
        "g"), predicate=predicate, regions=subregions)
    region.add(gamma)

    # then add tail region nodes (if any)
    # tail may include nodes that should come after the gamma
    if tail_region.nodes:
        # inline tail nodes into region
        for n in tail_region.nodes:
            region.add(n)
    return region


def topo_sort_subgraph(nodes: Dict[str, CFGNode], entry: str) -> List[str]:
    """
    Simple DFS-based topo order for acyclic subgraphs; falls back to reachable order for cycles.
    """
    visited = set()
    order = []

    def dfs(v):
        if v in visited:
            return
        visited.add(v)
        for w in nodes[v].successors:
            if w in nodes:
                dfs(w)
        order.append(v)
    if entry in nodes:
        dfs(entry)
    else:
        for v in nodes:
            if v not in visited:
                dfs(v)
    # produce reversed postorder so that defs come before uses roughly
    order.reverse()
    return order

# --- Public API ------------------------------------------------------------


def build_rvsdg_from_cfg(cfg_nodes: Dict[str, CFGNode], entry: str) -> RVSDGRegion:
    """
    Top-level builder: restructure loops (SCCs) and then either wrap SCC bodies into Theta nodes
    or build region directly for acyclic graphs.
    """
    build_pred_succ(cfg_nodes)
    sccs, membership, meta_adj = restructure_loops(cfg_nodes, entry)
    # We'll produce a top-level region that assembles theta nodes for any nontrivial SCCs and
    # constructs for the acyclic remaining graph pieces.

    # identify SCCs that are non-trivial (size > 1 or self-loop)
    top_region = RVSDGRegion(name="root")
    handled_nodes = set()

    # Map each SCC index to either a ThetaNode (if |SCC|>1 or self-loop) or to be processed inline
    for idx, scc in enumerate(sccs):
        if len(scc) > 1 or (len(scc) == 1 and scc[0] in cfg_nodes and scc[0] in cfg_nodes[scc[0]].successors):
            # treat as loop body L*. Create a ThetaNode with body built recursively (minus repetition arcs)
            # For simplicity we choose an arbitrary predicate name for loop (paper uses q/r constructs)
            body_entry = scc[0]
            body_region = build_rvsdg_from_region(
                list(scc), cfg_nodes, body_entry)
            theta = ThetaNode(nid=fresh_id(
                "th"), predicate=f"loop_pred_{idx}", body=body_region)
            top_region.add(theta)
            handled_nodes.update(scc)

    # Remaining nodes (not part of nontrivial SCCs) -> build region for those (acyclic)
    remaining = [n for n in cfg_nodes if n not in handled_nodes]
    if remaining:
        # choose entry among remaining (if original entry remained)
        rem_entry = entry if entry in remaining else remaining[0]
        rem_region = build_rvsdg_from_region(remaining, cfg_nodes, rem_entry)
        for node in rem_region.nodes:
            top_region.add(node)

    return top_region


# --- Example usage & demo -------------------------------------------------

def demo_small_cfg():
    # Build a small example CFG that includes a branch and a loop (toy)
    # Node ids are strings. Provide region pc ranges.
    G = {}

    def add(nid, start, end, stmt=None):
        G[nid] = CFGNode(nid, start, end, stmt=stmt)

    add("entry", 0, 0, "start")
    add("a", 1, 1, "x := x + 1")
    add("b", 2, 2, "if x>0 branch 0/1")
    add("c0", 3, 3, "y := y + 2")
    add("c1", 4, 4, "y := y - 2")
    add("d", 5, 5, "z := x+y")
    # a small loop SCC
    add("L1", 6, 6, "t := t-1")
    add("L2", 7, 7, "if t>0 branch 0/1")

    # connect edges
    G["entry"].successors = ["a"]
    G["a"].successors = ["b"]
    G["b"].successors = ["c0", "c1"]
    G["c0"].successors = ["d"]
    G["c1"].successors = ["d"]
    G["d"].successors = ["L1"]
    # loop edges
    G["L1"].successors = ["L2"]
    G["L2"].successors = ["L1", "exit"]  # repetition to L1, exit to exit
    add("exit", 8, 8, "return")
    G["exit"].successors = []

    # build preds
    build_pred_succ(G)

    print("CFG Nodes:")
    for n in G.values():
        print(n)

    r = build_rvsdg_from_cfg(G, "entry")
    # print out rvsdg structure
    print("\nResult RVSDG (json preview):")
    print(json.dumps(r.to_dict(), indent=2))


def demo_shared_region_cfg():
    G = {}

    def add(nid, start, end, stmt=None):
        G[nid] = CFGNode(nid, start, end, stmt=stmt)

    add("entry", 0, 0, "start")
    add("b", 1, 1, "if cond branch 0/1")
    add("x", 2, 2, "x := x+1")
    add("y", 3, 3, "y := y-1")
    add("z", 4, 4, "z := x+y")
    add("exit", 5, 5, "return")

    # edges
    G["entry"].successors = ["b"]
    G["b"].successors = ["x", "y"]
    G["x"].successors = ["z"]
    G["y"].successors = ["z"]
    G["z"].successors = ["exit"]
    G["exit"].successors = []

    build_pred_succ(G)

    print("CFG Nodes:")
    for n in G.values():
        print(n)

    r = build_rvsdg_from_cfg(G, "entry")
    print("\nResult RVSDG (json preview):")
    import json
    print(json.dumps(r.to_dict(), indent=2))


if __name__ == "__main__":
    #demo_small_cfg()
    demo_shared_region_cfg()
