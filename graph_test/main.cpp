// test.cpp
// High-efficiency, cache-local C++20 translation of the test.py RVSDG builder.
// - Uses integer node IDs (0..N-1) for cache locality
// - Adjacency lists are contiguous vectors
// - Avoids heavy dynamic allocations during graph traversals
// - Produces a JSON-like text output for the built RVSDG
//
// Compile: g++ -O3 -std=c++20 -march=native -flto -pipe test.cpp -o test

#include <cstdint>
#include <vector>
#include <string>
#include <functional>
#include <unordered_set>
#include <iostream>

using namespace std;

// -------------------- Basic types & utilities ------------------------------
using NodeId = int;
using EdgeKey = uint64_t;

inline EdgeKey packEdge(NodeId u, NodeId v) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(u)) << 32) |
        static_cast<uint64_t>(static_cast<uint32_t>(v));
}

// -------------------- CFG Node (compact) ----------------------------------
struct CFGNode {
    NodeId id;
    int start_pc;
    int end_pc;
    int stmt_idx;                // index into a shared string table
    vector<NodeId> succ;         // successors (contiguous vector)
    vector<NodeId> pred;         // predecessors
    CFGNode() = default;
    CFGNode(NodeId _id, int s, int e, int si) : id(_id), start_pc(s), end_pc(e), stmt_idx(si) {}
};

// -------------------- RVSDG lightweight representation --------------------
enum class RVKind { SIMPLE, GAMMA, THETA };

struct RVNode {
    RVKind kind;
    int id;                      // unique integer id for printing
    // SIMPLE:
    int stmt_idx = -1;
    pair<int, int> region = { -1,-1 };
    // GAMMA:
    int predicate_idx = -1;          // index into string table for predicate name
    vector<int> gamma_regions;       // indices into the global region store
    // THETA:
    int theta_body_region = -1;      // index into region store
    int theta_pred_idx = -1;
};

struct RVRegion {
    int id;
    string name;
    vector<int> node_indices;     // indices into global rv_nodes
};

// -------------------- Global containers for strings & constructed nodes ----
struct Universe {
    vector<string> strtab;                // statements, predicate names, region names
    vector<CFGNode> cfg;                  // nodes indexed by numeric NodeId
    vector<RVNode> rv_nodes;              // flattened RV nodes
    vector<RVRegion> rv_regions;          // regions
    int next_rvnode_id = 0;
    int next_region_id = 0;

    int intern_string(const string& s) {
        // Simple interning -- linear search (ok for small demos). If many strings use unordered_map.
        for (size_t i = 0; i < strtab.size(); ++i) if (strtab[i] == s) return (int)i;
        strtab.push_back(s);
        return (int)strtab.size() - 1;
    }

    int make_simple_node(int stmt_idx, pair<int, int> region) {
        RVNode n;
        n.kind = RVKind::SIMPLE;
        n.id = next_rvnode_id++;
        n.stmt_idx = stmt_idx;
        n.region = region;
        rv_nodes.push_back(move(n));
        return (int)rv_nodes.size() - 1;
    }
    int make_gamma_node(int pred_idx, vector<int> region_indices) {
        RVNode n;
        n.kind = RVKind::GAMMA;
        n.id = next_rvnode_id++;
        n.predicate_idx = pred_idx;
        n.gamma_regions = move(region_indices);
        rv_nodes.push_back(move(n));
        return (int)rv_nodes.size() - 1;
    }
    int make_theta_node(int pred_idx, int body_region_idx) {
        RVNode n;
        n.kind = RVKind::THETA;
        n.id = next_rvnode_id++;
        n.theta_pred_idx = pred_idx;
        n.theta_body_region = body_region_idx;
        rv_nodes.push_back(move(n));
        return (int)rv_nodes.size() - 1;
    }
    int make_region(const string& name) {
        RVRegion r;
        r.id = next_region_id++;
        r.name = name;
        rv_regions.push_back(move(r));
        return (int)rv_regions.size() - 1;
    }
};

// -------------------- Graph utilities (cache-friendly) ---------------------
void build_pred_succ(vector<CFGNode>& nodes) {
    for (auto& n : nodes) n.pred.clear();
    for (auto& n : nodes) {
        for (auto s : n.succ) {
            nodes[s].pred.push_back(n.id);
        }
    }
}

// Tarjan SCC (recursive) - uses contiguous vectors
vector<vector<NodeId>> tarjan_scc(const vector<CFGNode>& nodes) {
    int N = (int)nodes.size();
    vector<int> index(N, -1), low(N, 0);
    vector<char> onstack(N, 0);
    vector<NodeId> stack;
    stack.reserve(N);
    vector<vector<NodeId>> result;
    int cur = 0;

    function<void(NodeId)> strongconnect = [&](NodeId v) {
        index[v] = low[v] = cur++;
        stack.push_back(v);
        onstack[v] = 1;
        for (NodeId w : nodes[v].succ) {
            if (index[w] == -1) {
                strongconnect(w);
                low[v] = min(low[v], low[w]);
            }
            else if (onstack[w]) {
                low[v] = min(low[v], index[w]);
            }
        }
        if (low[v] == index[v]) {
            result.emplace_back();
            while (true) {
                NodeId w = stack.back();
                stack.pop_back();
                onstack[w] = 0;
                result.back().push_back(w);
                if (w == v) break;
            }
        }
        };

    for (int i = 0; i < N; ++i) if (index[i] == -1) strongconnect(i);
    return result;
}

// Reachability from entry, ignore banned edges
void reachable_from(NodeId entry, const vector<CFGNode>& nodes, vector<char>& seen,
    const unordered_set<EdgeKey>* banned_edges = nullptr) {
    int N = (int)nodes.size();
    fill(seen.begin(), seen.end(), 0);
    vector<NodeId> st;
    st.reserve(N);
    st.push_back(entry);
    while (!st.empty()) {
        NodeId v = st.back(); st.pop_back();
        if (seen[v]) continue;
        seen[v] = 1;
        for (NodeId s : nodes[v].succ) {
            EdgeKey ek = packEdge(v, s);
            if (banned_edges && banned_edges->find(ek) != banned_edges->end()) continue;
            if (!seen[s]) st.push_back(s);
        }
    }
}

// nodes dominated by arc (u->v): nodes that become unreachable when arc removed
vector<char> nodes_dominated_by_arc(NodeId entry, const vector<CFGNode>& nodes, pair<NodeId, NodeId> arc) {
    int N = (int)nodes.size();
    unordered_set<EdgeKey> banned;
    banned.reserve(4);
    banned.insert(packEdge(arc.first, arc.second));
    vector<char> with_banned(N), without_banned(N);
    reachable_from(entry, nodes, with_banned, &banned);
    reachable_from(entry, nodes, without_banned, nullptr);
    vector<char> dominated(N, 0);
    for (int i = 0; i < N; ++i) {
        if (without_banned[i] && !with_banned[i]) dominated[i] = 1;
    }
    return dominated;
}

// Topo order (simple DFS) for acyclic view (returns nodes in topo order)
vector<NodeId> topo_sort_subgraph(const vector<CFGNode>& nodes, NodeId entry, const vector<char>& in_subgraph) {
    int N = (int)nodes.size();
    vector<char> visited(N, 0);
    vector<NodeId> order;
    order.reserve(N);

    function<void(NodeId)> dfs = [&](NodeId v) {
        if (!in_subgraph[v] || visited[v]) return;
        visited[v] = 1;
        for (NodeId w : nodes[v].succ) {
            if (in_subgraph[w]) dfs(w);
        }
        order.push_back(v);
        };

    if (in_subgraph[entry]) dfs(entry);
    for (int i = 0; i < N; ++i) if (in_subgraph[i] && !visited[i]) dfs(i);
    reverse(order.begin(), order.end());
    return order;
}

// longest linear head starting from entry
vector<NodeId> longest_linear_head(const vector<CFGNode>& nodes, NodeId entry) {
    vector<NodeId> H;
    vector<char> seen(nodes.size(), 0);
    NodeId cur = entry;
    while (true) {
        if (cur < 0 || cur >= (NodeId)nodes.size()) break;
        if (seen[cur]) break;
        seen[cur] = 1;
        H.push_back(cur);
        if (nodes[cur].succ.size() != 1) break;
        NodeId nxt = nodes[cur].succ[0];
        if (nodes[nxt].pred.size() > 1) break;
        cur = nxt;
    }
    return H;
}

// Partition head/branch/tail
struct Partition {
    vector<NodeId> H;
    vector<vector<NodeId>> B; // each Bj as vector of nodes
    vector<char> T;           // boolean mask for tail nodes (size N)
    NodeId fanout_node;
    vector<pair<NodeId, NodeId>> fanout_arcs;
};

Partition partition_head_branch_tail(vector<CFGNode>& nodes, NodeId entry) {
    build_pred_succ(nodes);
    int N = (int)nodes.size();
    Partition P;
    P.T.assign(N, 0);

    P.H = longest_linear_head(nodes, entry);
    NodeId last = P.H.empty() ? entry : P.H.back();
    vector<NodeId> fanouts = nodes[last].succ;
    if (fanouts.empty()) {
        // everything is head or none
        return P;
    }
    // compute dominated sets for each fanout arc
    vector<vector<char>> dominated_masks;
    dominated_masks.reserve(fanouts.size());
    vector<char> union_B(N, 0);
    for (NodeId dst : fanouts) {
        P.fanout_arcs.emplace_back(last, dst);
        auto dom = nodes_dominated_by_arc(entry, nodes, { last, dst });
        dominated_masks.push_back(move(dom));
    }
    // collect Bj sets
    for (auto& mask : dominated_masks) {
        vector<NodeId> bj;
        for (int i = 0; i < N; ++i) if (mask[i]) {
            bj.push_back(i);
            union_B[i] = 1;
        }
        P.B.push_back(move(bj));
    }
    // tail nodes = all nodes - H - union_B
    vector<char> in_H(N, 0);
    for (auto v : P.H) in_H[v] = 1;
    for (int i = 0; i < N; ++i) {
        if (!in_H[i] && !union_B[i]) P.T[i] = 1;
    }
    // trimming: if a continuation point in T has predecessor in branch,
    // and not all preds are in branch -> move that pred to T (conservative)
    vector<int> continuation_points;
    for (int i = 0; i < N; ++i) if (P.T[i]) {
        for (auto p : nodes[i].pred) {
            if (union_B[p] || p == last) { continuation_points.push_back(i); break; }
        }
    }
    for (int c : continuation_points) {
        for (auto p : nodes[c].pred) {
            if (!union_B[p]) continue;
            bool all_preds_in_branch = true;
            for (auto pp : nodes[c].pred) {
                if (!union_B[pp]) { all_preds_in_branch = false; break; }
            }
            if (!all_preds_in_branch) {
                // remove p from whichever B it belongs to, and add to T
                for (auto& bvec : P.B) {
                    auto it = find(bvec.begin(), bvec.end(), p);
                    if (it != bvec.end()) {
                        bvec.erase(it);
                        P.T[p] = 1;
                        union_B[p] = 0;
                        break;
                    }
                }
            }
        }
    }
    P.fanout_node = last;
    return P;
}

// -------------------- RVSDG builder (recursive but operates on integer masks) -
struct Builder {
    Universe& U;

    Builder(Universe& un) : U(un) {}

    // Build a region from a list of node ids (subgraph) with entry id
    int build_region_from_nodes(const vector<NodeId>& node_list, NodeId entry) {
        int N = (int)U.cfg.size();
        // make a membership mask for quick checks
        vector<char> in_sub(N, 0);
        for (auto v : node_list) in_sub[v] = 1;

        // create a region object in Universe
        string rname = "region_" + to_string(U.next_region_id);
        int region_idx = U.make_region(rname);

        if (node_list.empty()) return region_idx;

        // count branch nodes inside the subgraph
        int branch_count = 0;
        for (auto v : node_list) {
            int succ_inside = 0;
            for (auto s : U.cfg[v].succ) if (in_sub[s]) ++succ_inside;
            if (succ_inside > 1) ++branch_count;
        }

        if (branch_count == 0) {
            // linear-ish: topo order
            auto topo = topo_sort_subgraph(U.cfg, entry, in_sub);
            for (auto vid : topo) {
                int stmt = U.cfg[vid].stmt_idx;
                pair<int, int> region = { U.cfg[vid].start_pc, U.cfg[vid].end_pc };
                int node_idx = U.make_simple_node(stmt, region);
                U.rv_regions[region_idx].node_indices.push_back(node_idx);
            }
            return region_idx;
        }

        // Otherwise partition into H/B/T
        // Build a temporary view nodes vector containing only the nodes in this subgraph,
        // but we operate on the full U.cfg and use in_sub mask for checks.
        // For partition function we need a nodes vector we can mutate preds for -> copy relevant nodes
        // For performance we create a small temporary nodes vector reindexed to 0..M-1
        vector<NodeId> map_to_global; map_to_global.reserve(node_list.size());
        unordered_map<NodeId, int> global_to_local; global_to_local.reserve(node_list.size() * 2);
        for (int i = 0; i < (int)node_list.size(); ++i) {
            map_to_global.push_back(node_list[i]);
            global_to_local[node_list[i]] = i;
        }
        // build a local nodes array
        struct LocalNode { NodeId gid; int s, e; int stmt; vector<int> succ; vector<int> pred; };
        vector<LocalNode> local;
        local.reserve(node_list.size());
        for (auto gid : map_to_global) {
            LocalNode ln;
            ln.gid = gid;
            ln.s = U.cfg[gid].start_pc;
            ln.e = U.cfg[gid].end_pc;
            ln.stmt = U.cfg[gid].stmt_idx;
            // successors only if inside subgraph
            for (auto s : U.cfg[gid].succ) if (in_sub[s]) ln.succ.push_back(global_to_local[s]);
            // predecessors will be built next
            local.push_back(move(ln));
        }
        // build preds
        for (int i = 0; i < (int)local.size(); ++i) {
            for (int s : local[i].succ) local[s].pred.push_back(i);
        }
        // Now adapt partition_head_branch_tail to local indices
        // We'll create a small vector<CFGNode>-like structure for the partition function:
        vector<CFGNode> local_cfg;
        local_cfg.reserve(local.size());
        for (int i = 0; i < (int)local.size(); ++i) {
            CFGNode cn(i, local[i].s, local[i].e, local[i].stmt);
            cn.succ.assign(local[i].succ.begin(), local[i].succ.end());
            cn.pred.assign(local[i].pred.begin(), local[i].pred.end());
            local_cfg.push_back(move(cn));
        }
        NodeId local_entry = global_to_local[entry];
        Partition part_local = partition_head_branch_tail(local_cfg, local_entry);

        // H: add simple nodes for head (converted back to global ids)
        for (auto local_h : part_local.H) {
            NodeId gid = map_to_global[local_h];
            int stmt = U.cfg[gid].stmt_idx;
            pair<int, int> region = { U.cfg[gid].start_pc, U.cfg[gid].end_pc };
            int node_idx = U.make_simple_node(stmt, region);
            U.rv_regions[region_idx].node_indices.push_back(node_idx);
        }

        // For each Bj, build subregion and collect its index
        vector<int> gamma_subregions;
        for (auto& bvec : part_local.B) {
            if (bvec.empty()) {
                // empty region
                int rr = U.make_region("empty_" + to_string(U.next_region_id));
                gamma_subregions.push_back(rr);
                continue;
            }
            // compute global ids of nodes in this branch
            vector<NodeId> branch_global;
            branch_global.reserve(bvec.size());
            for (auto local_id : bvec) branch_global.push_back(map_to_global[local_id]);
            // choose entry for branch: prefer nodes that have predecessor outside the branch
            NodeId chosen_entry = branch_global[0];
            for (auto gid : branch_global) {
                bool has_pred_outside = false;
                for (auto p : U.cfg[gid].pred) {
                    if (!in_sub[p]) { has_pred_outside = true; break; }
                    // or pred in_sub but not in this branch
                    bool in_this_branch = false;
                    for (auto gg : branch_global) if (gg == p) { in_this_branch = true; break; }
                    if (!in_this_branch) { has_pred_outside = true; break; }
                }
                if (has_pred_outside) { chosen_entry = gid; break; }
            }
            int subregion_idx = build_region_from_nodes(branch_global, chosen_entry);
            gamma_subregions.push_back(subregion_idx);
        }

        // Tail region (nodes in T)
        vector<NodeId> tail_global;
        for (int i = 0; i < (int)in_sub.size(); ++i) {
            if (!in_sub[i]) continue;
            if (part_local.T[global_to_local[i]]) {
                tail_global.push_back(i);
            }
        }
        int tail_region_idx = -1;
        if (!tail_global.empty()) {
            // choose tail entry: node with predecessor outside tail or head
            NodeId tail_entry = tail_global[0];
            unordered_set<NodeId> tailset(tail_global.begin(), tail_global.end());
            for (auto t : tail_global) {
                for (auto p : U.cfg[t].pred) {
                    if (!tailset.count(p)) { tail_entry = t; break; }
                }
            }
            tail_region_idx = build_region_from_nodes(tail_global, tail_entry);
        }
        else {
            tail_region_idx = U.make_region("tail_empty_" + to_string(U.next_region_id));
        }

        // Gamma node predicate: use fanout_node name heuristically
        string pred_name = "pred_at_" + to_string(map_to_global[part_local.fanout_node]);
        int pred_idx = U.intern_string(pred_name);
        int gamma_idx = U.make_gamma_node(pred_idx, gamma_subregions);
        U.rv_regions[region_idx].node_indices.push_back(gamma_idx);

        // inline tail nodes into region if tail has nodes
        if (tail_region_idx >= 0) {
            for (int ni : U.rv_regions[tail_region_idx].node_indices)
                U.rv_regions[region_idx].node_indices.push_back(ni);
        }
        return region_idx;
    }

    // Top-level build: find SCCs, wrap nontrivial SCCs as thetas, then build remaining region
    int build_from_cfg(NodeId entry) {
        build_pred_succ(U.cfg);
        auto sccs = tarjan_scc(U.cfg);
        int top_region_idx = U.make_region("root");
        vector<char> handled(U.cfg.size(), 0);

        // wrap nontrivial SCCs into thetas
        for (size_t idx = 0; idx < sccs.size(); ++idx) {
            auto& scc = sccs[idx];
            bool nontrivial = (scc.size() > 1);
            if (!nontrivial) {
                // check self-loop
                NodeId v = scc[0];
                for (auto s : U.cfg[v].succ) if (s == v) nontrivial = true;
            }
            if (nontrivial) {
                // build body region and create theta node
                vector<NodeId> body = scc;
                NodeId body_entry = body[0];
                int body_region = build_region_from_nodes(body, body_entry);
                string pred_name = "loop_pred_" + to_string(idx);
                int pred_idx = U.intern_string(pred_name);
                int theta_idx = U.make_theta_node(pred_idx, body_region);
                U.rv_regions[top_region_idx].node_indices.push_back(theta_idx);
                for (auto v : scc) handled[v] = 1;
            }
        }

        // remaining nodes -> build acyclic region
        vector<NodeId> remaining;
        remaining.reserve(U.cfg.size());
        for (auto& n : U.cfg) if (!handled[n.id]) remaining.push_back(n.id);
        if (!remaining.empty()) {
            NodeId rem_entry = (entry < (NodeId)U.cfg.size() && !handled[entry]) ? entry : remaining[0];
            int rem_region = build_region_from_nodes(remaining, rem_entry);
            // inline region nodes into top region
            for (int node_i : U.rv_regions[rem_region].node_indices)
                U.rv_regions[top_region_idx].node_indices.push_back(node_i);
        }
        return top_region_idx;
    }
};

// -------------------- JSON-like printer (simple, no external deps) ----------
string escape_json(const string& s) {
    string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\"') o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else o += c;
    }
    return o;
}

void print_rvsdg(const Universe& U, int top_region_idx, ostream& os) {
    os << "{\n  \"regions\": [\n";
    for (size_t ri = 0; ri < U.rv_regions.size(); ++ri) {
        const auto& R = U.rv_regions[ri];
        os << "    { \"id\": " << R.id << ", \"name\": \"" << escape_json(R.name) << "\", \"nodes\": [\n";
        for (size_t ni = 0; ni < R.node_indices.size(); ++ni) {
            const auto& N = U.rv_nodes[R.node_indices[ni]];
            os << "      { \"rv_id\": " << N.id << ", \"kind\": ";
            if (N.kind == RVKind::SIMPLE) {
                os << "\"simple\", \"stmt\": \"" << escape_json(U.strtab[N.stmt_idx]) << "\", \"region\": ["
                    << N.region.first << "," << N.region.second << "] }";
            }
            else if (N.kind == RVKind::GAMMA) {
                os << "\"gamma\", \"predicate\": \"" << escape_json(U.strtab[N.predicate_idx]) << "\", \"subregions\": [";
                for (size_t k = 0; k < N.gamma_regions.size(); ++k) {
                    os << N.gamma_regions[k];
                    if (k + 1 < N.gamma_regions.size()) os << ", ";
                }
                os << "] }";
            }
            else {
                os << "\"theta\", \"predicate\": \"" << escape_json(U.strtab[N.theta_pred_idx]) <<
                    "\", \"body_region\": " << N.theta_body_region << " }";
            }
            if (ni + 1 < R.node_indices.size()) os << ",";
            os << "\n";
        }
        os << "    ] }";
        if (ri + 1 < U.rv_regions.size()) os << ",";
        os << "\n";
    }
    os << "  ], \"top_region\": " << top_region_idx << "\n}\n";
}

// -------------------- Demo (mirrors provided Python demo) -------------------
int main() {
    Universe U;

    // helper to add nodes easily (returns NodeId)
    auto add_node = [&](int start, int end, const string& stmt) {
        int sid = U.intern_string(stmt);
        NodeId id = (NodeId)U.cfg.size();
        U.cfg.emplace_back(id, start, end, sid);
        return id;
        };

    // Build the 'shared region' demo from python
    NodeId entry = add_node(0, 0, "start");
    NodeId b = add_node(1, 1, "if cond branch 0/1");
    NodeId x = add_node(2, 2, "x := x+1");
    NodeId y = add_node(3, 3, "y := y-1");
    NodeId z = add_node(4, 4, "z := x+y");
    NodeId exitn = add_node(5, 5, "return");

    U.cfg[entry].succ = { b };
    U.cfg[b].succ = { x,y };
    U.cfg[x].succ = { z };
    U.cfg[y].succ = { z };
    U.cfg[z].succ = { exitn };
    U.cfg[exitn].succ = {};

    //build_pred_succ(U.cfg);

    Builder builder(U);
    int top = builder.build_from_cfg(entry);

    // Print RVSDG
    print_rvsdg(U, top, cout);

    return 0;
}
