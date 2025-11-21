// dominator_tree_sparse.hpp
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory_resource>
#include <optional>
#include <ranges>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace compiler::analysis {
template <typename T>
concept SparseGraphConcept = requires(T a, uint32_t idx) {
  { a.getSuccessors(idx) } -> std::ranges::range;
  { a.getPredecessors(idx) } -> std::ranges::range;
  { a.size() } -> std::convertible_to<uint32_t>;
};

template <SparseGraphConcept Graph>
class DominatorTreeSparse {
  public:
  using node_t  = uint32_t;
  using alloc_t = std::pmr::polymorphic_allocator<>;

  explicit DominatorTreeSparse(alloc_t alloc = {})
      : alloc_(alloc),
        node_to_dfs_(/*bucket count*/ 0, std::hash<node_t> {}, std::equal_to<node_t> {},
                     std::pmr::polymorphic_allocator<std::pair<const node_t, int>>(alloc_)) {}

  inline void calculate(const Graph& g, node_t exit) {
    if (build_) return;
    build_ = true;
    build(g, exit);
  }

  // Returns immediate dominator of `n` if exists; entry has no idom (returns std::nullopt).
  std::optional<node_t> get_idom(node_t n) const {
    auto it = node_to_dfs_.find(n);
    if (it == node_to_dfs_.end()) return std::nullopt;
    int dfn = it->second;
    if (dfn <= 0 || dfn > (int)idom_.size() - 1) return std::nullopt;
    int idom_dfn = idom_[dfn];
    if (idom_dfn <= 0) return std::nullopt;
    return vertex_[idom_dfn];
  }

  // True if a dominates b (conservative: checks via idoms; returns false if either node missing)
  bool dominates(node_t a, node_t b) const {
    auto ida = node_to_dfs_.find(a);
    auto idb = node_to_dfs_.find(b);
    if (ida == node_to_dfs_.end() || idb == node_to_dfs_.end()) return false;
    int da = ida->second;
    int db = idb->second;
    if (da == db) return true;
    // walk up idom chain from db to root, see if we hit da
    int cur = db;
    while (cur > 0 && cur != da)
      cur = idom_[cur];
    return cur == da;
  }

  private:
  bool    build_ = false;
  alloc_t alloc_;

  // map node id -> dfs number (1..N)
  std::unordered_map<node_t, int, std::hash<node_t>, std::equal_to<node_t>, std::pmr::polymorphic_allocator<std::pair<const node_t, int>>> node_to_dfs_;

  // arrays indexed by DFS number (0 is unused)
  std::pmr::vector<node_t>                vertex_; // vertex_[dfn] = node id
  std::pmr::vector<int>                   parent_; // parent_ in DFS tree (dfn)
  std::pmr::vector<int>                   semi_;
  std::pmr::vector<int>                   idom_;
  std::pmr::vector<int>                   ancestor_;
  std::pmr::vector<int>                   label_;
  std::pmr::vector<std::pmr::vector<int>> bucket_;

  // --- Lengauer-Tarjan helper functions (works on DFS numbers) ---
  void compress(int v) {
    int a = ancestor_[v];
    if (a == 0) return;
    if (ancestor_[a] != 0) {
      compress(a);
      if (semi_[label_[a]] < semi_[label_[v]]) label_[v] = label_[a];
      ancestor_[v] = ancestor_[a];
    }
  }

  int eval(int v) {
    if (ancestor_[v] == 0) return label_[v];
    compress(v);
    if (semi_[label_[ancestor_[v]]] >= semi_[label_[v]])
      return label_[v];
    else
      return label_[ancestor_[v]];
  }

  void link(int v, int w) {
    // make v parent of w in the forest
    ancestor_[w] = v;
    // label[w] already set
  }

  // Build dominator tree from graph g starting at entry
  void build(const Graph& g, node_t entry) {
    // Reserve small initial sizes (grow as needed)
    vertex_ = std::pmr::vector<node_t>(alloc_);
    vertex_.push_back(0); // index 0 unused

    parent_      = std::pmr::vector<int>(1, alloc_);
    parent_[0]   = 0;
    semi_        = std::pmr::vector<int>(1, alloc_);
    semi_[0]     = 0;
    idom_        = std::pmr::vector<int>(1, alloc_);
    idom_[0]     = 0;
    ancestor_    = std::pmr::vector<int>(1, alloc_);
    ancestor_[0] = 0;
    label_       = std::pmr::vector<int>(1, alloc_);
    label_[0]    = 0;
    bucket_      = std::pmr::vector<std::pmr::vector<int>>(alloc_);

    // DFS stack
    std::pmr::vector<node_t> stack(alloc_);
    stack.reserve(256);

    // iterative DFS to produce preorder numbering (dfn)
    int time = 0;
    // Use explicit stack of pairs: (node, iterator state) — but simpler: push node and mark visited on map
    stack.push_back(entry);
    // parent map for DFS by node id (sparse)
    std::pmr::unordered_map<node_t, node_t> node_parent(alloc_);

    // iterative DFS using manual stack of pairs (node, next-successor-iterator index)
    struct Frame {
      node_t      node;
      std::size_t next_idx;
    };

    std::pmr::vector<Frame> frames(alloc_);
    frames.reserve(256);
    frames.push_back({entry, 0});
    node_to_dfs_.reserve(1024);

    while (!frames.empty()) {
      Frame& fr = frames.back();
      node_t u  = fr.node;
      if (node_to_dfs_.find(u) == node_to_dfs_.end()) {
        // visit
        ++time;
        node_to_dfs_.emplace(u, time);
        vertex_.push_back(u);
        parent_.push_back(0);
        semi_.push_back(time);
        idom_.push_back(0);
        ancestor_.push_back(0);
        label_.push_back(time);
        bucket_.push_back(std::pmr::vector<int>(alloc_));
        // set parent if exists in node_parent
        auto npit = node_parent.find(u);
        if (npit != node_parent.end()) {
          node_t p      = npit->second;
          parent_[time] = node_to_dfs_.at(p);
        } else {
          parent_[time] = 0;
        }
      }

      // iterate successors incrementally
      bool pushed = false;
      auto succs  = g.getSuccessors(u);
      // We cannot index succs generically, so iterate and skip `fr.next_idx` successors using a local counter
      std::size_t idx = 0;
      for (auto s: succs) {
        if (idx < fr.next_idx) {
          ++idx;
          continue;
        }
        // We've got the next successor to handle
        // push successor frame (if not visited)
        if (node_to_dfs_.find(s) == node_to_dfs_.end()) {
          node_parent.emplace(s, u);
          frames.push_back({s, 0});
          ++fr.next_idx;
          pushed = true;
          break;
        } else {
          ++fr.next_idx;
          ++idx;
        }
      }
      if (!pushed) {
        // done with this node
        frames.pop_back();
      }
    }

    if (time == 0) return; // entry not present / empty graph

    int N = time;
    // ensure sizes exactly N+1
    // (already grown during DFS). Prepare preds_buf_

    std::pmr::vector<int> preds_buf(alloc_);
    preds_buf.reserve(64); // optional heuristic

    // Build predecessor lists on the fly when needed — the algorithm iterates predecessors of each vertex
    // For each vertex w in reverse DFS order:
    for (int w = N; w >= 2; --w) {
      preds_buf.clear();

      node_t n_w = vertex_[w];
      // gather predecessors in terms of DFS numbers
      for (auto pnode: g.getPredecessors(n_w)) {
        auto it = node_to_dfs_.find(pnode);
        if (it != node_to_dfs_.end()) preds_buf.push_back(it->second);
      }

      // compute semi
      int s      = parent_[w];
      int semi_w = s;
      for (int v: preds_buf) {
        int u = (v <= 0) ? v : eval(v);
        if (semi_[u] < semi_w) semi_w = semi_[u];
      }
      semi_[w] = semi_w;

      // add w to bucket[ vertex[semi[w]] ]
      bucket_[semi_w].push_back(w);
      link(parent_[w], w);

      // process bucket of parent
      int p = parent_[w];
      for (int v: bucket_[p]) {
        int u = eval(v);
        if (semi_[u] < semi_[v])
          idom_[v] = u;
        else
          idom_[v] = p;
      }
      bucket_[p].clear();
    }

    // final step
    for (int w = 2; w <= N; ++w) {
      if (idom_[w] != semi_[w]) idom_[w] = idom_[idom_[w]];
    }
    idom_[1] = 0; // entry has no idom
  }
};

template <SparseGraphConcept Graph>
class PostDominatorTreeSparse {
  public:
  using node_t  = uint32_t;
  using alloc_t = std::pmr::polymorphic_allocator<>;

  explicit PostDominatorTreeSparse(alloc_t alloc = {})
      : alloc_(alloc),
        node_to_dfs_(/*bucket count*/ 0, std::hash<node_t> {}, std::equal_to<node_t> {},
                     std::pmr::polymorphic_allocator<std::pair<const node_t, int>>(alloc_)) {}

  inline void calculate(const Graph& g, node_t exit) {
    if (build_) return;
    build_ = true;
    build(g, exit);
  }

  // Returns immediate post-dominator of `n` if exists; exit has no ipdom (returns std::nullopt).
  std::optional<node_t> get_ipdom(node_t n) const {
    auto it = node_to_dfs_.find(n);
    if (it == node_to_dfs_.end()) return std::nullopt;
    int dfn = it->second;
    if (dfn <= 0 || dfn > (int)idom_.size() - 1) return std::nullopt;
    int idom_dfn = idom_[dfn];
    if (idom_dfn <= 0) return std::nullopt;
    return vertex_[idom_dfn];
  }

  // True if a post-dominates b (conservative: checks via ipdoms; returns false if either node missing)
  bool postdominates(node_t a, node_t b) const {
    auto ida = node_to_dfs_.find(a);
    auto idb = node_to_dfs_.find(b);
    if (ida == node_to_dfs_.end() || idb == node_to_dfs_.end()) return false;
    int da = ida->second;
    int db = idb->second;
    if (da == db) return true;
    // walk up ipdom chain from db to root, see if we hit da
    int cur = db;
    while (cur > 0 && cur != da)
      cur = idom_[cur];
    return cur == da;
  }

  private:
  bool    build_ = false;
  alloc_t alloc_;

  // map node id -> dfs number (1..N)
  std::unordered_map<node_t, int, std::hash<node_t>, std::equal_to<node_t>, std::pmr::polymorphic_allocator<std::pair<const node_t, int>>> node_to_dfs_;

  // arrays indexed by DFS number (0 is unused)
  std::pmr::vector<node_t>                vertex_; // vertex_[dfn] = node id
  std::pmr::vector<int>                   parent_; // parent_ in DFS tree (dfn)
  std::pmr::vector<int>                   semi_;
  std::pmr::vector<int>                   idom_;
  std::pmr::vector<int>                   ancestor_;
  std::pmr::vector<int>                   label_;
  std::pmr::vector<std::pmr::vector<int>> bucket_;

  // --- Lengauer-Tarjan helper functions (works on DFS numbers) ---
  void compress(int v) {
    int a = ancestor_[v];
    if (a == 0) return;
    if (ancestor_[a] != 0) {
      compress(a);
      if (semi_[label_[a]] < semi_[label_[v]]) label_[v] = label_[a];
      ancestor_[v] = ancestor_[a];
    }
  }

  int eval(int v) {
    if (ancestor_[v] == 0) return label_[v];
    compress(v);
    if (semi_[label_[ancestor_[v]]] >= semi_[label_[v]])
      return label_[v];
    else
      return label_[ancestor_[v]];
  }

  void link(int v, int w) {
    // make v parent of w in the forest
    ancestor_[w] = v;
  }

  // Build post-dominator tree from REVERSED graph starting at exit
  void build(const Graph& g, node_t exit) {
    // Reserve small initial sizes (grow as needed)
    vertex_ = std::pmr::vector<node_t>(alloc_);
    vertex_.push_back(0); // index 0 unused

    parent_      = std::pmr::vector<int>(1, alloc_);
    parent_[0]   = 0;
    semi_        = std::pmr::vector<int>(1, alloc_);
    semi_[0]     = 0;
    idom_        = std::pmr::vector<int>(1, alloc_);
    idom_[0]     = 0;
    ancestor_    = std::pmr::vector<int>(1, alloc_);
    ancestor_[0] = 0;
    label_       = std::pmr::vector<int>(1, alloc_);
    label_[0]    = 0;
    bucket_      = std::pmr::vector<std::pmr::vector<int>>(alloc_);

    // DFS on REVERSED graph (using predecessors as successors)
    int                                     time = 0;
    std::pmr::unordered_map<node_t, node_t> node_parent(alloc_);

    struct Frame {
      node_t      node;
      std::size_t next_idx;
    };

    std::pmr::vector<Frame> frames(alloc_);
    frames.reserve(256);
    frames.push_back({exit, 0});
    node_to_dfs_.reserve(1024);

    while (!frames.empty()) {
      Frame& fr = frames.back();
      node_t u  = fr.node;
      if (node_to_dfs_.find(u) == node_to_dfs_.end()) {
        // visit
        ++time;
        node_to_dfs_.emplace(u, time);
        vertex_.push_back(u);
        parent_.push_back(0);
        semi_.push_back(time);
        idom_.push_back(0);
        ancestor_.push_back(0);
        label_.push_back(time);
        bucket_.push_back(std::pmr::vector<int>(alloc_));
        // set parent if exists in node_parent
        auto npit = node_parent.find(u);
        if (npit != node_parent.end()) {
          node_t p      = npit->second;
          parent_[time] = node_to_dfs_.at(p);
        } else {
          parent_[time] = 0;
        }
      }

      // Iterate PREDECESSORS (reversed edge direction for post-dominators)
      bool        pushed = false;
      auto        preds  = g.getPredecessors(u); // KEY CHANGE: use predecessors instead of successors
      std::size_t idx    = 0;
      for (auto s: preds) {
        if (idx < fr.next_idx) {
          ++idx;
          continue;
        }
        // push predecessor frame (if not visited)
        if (node_to_dfs_.find(s) == node_to_dfs_.end()) {
          node_parent.emplace(s, u);
          frames.push_back({s, 0});
          ++fr.next_idx;
          pushed = true;
          break;
        } else {
          ++fr.next_idx;
          ++idx;
        }
      }
      if (!pushed) {
        frames.pop_back();
      }
    }

    if (time == 0) return; // exit not present / empty graph

    int                   N = time;
    std::pmr::vector<int> preds_buf(alloc_);
    preds_buf.reserve(64);

    // Build predecessor lists in REVERSED graph (use successors as predecessors)
    for (int w = N; w >= 2; --w) {
      preds_buf.clear();

      node_t n_w = vertex_[w];
      // In reversed graph, predecessors are the original successors
      for (auto succ_node: g.getSuccessors(n_w)) { // KEY CHANGE: use successors
        auto it = node_to_dfs_.find(succ_node);
        if (it != node_to_dfs_.end()) preds_buf.push_back(it->second);
      }

      // compute semi
      int s      = parent_[w];
      int semi_w = s;
      for (int v: preds_buf) {
        int u = (v <= 0) ? v : eval(v);
        if (semi_[u] < semi_w) semi_w = semi_[u];
      }
      semi_[w] = semi_w;

      // add w to bucket[ vertex[semi[w]] ]
      bucket_[semi_w].push_back(w);
      link(parent_[w], w);

      // process bucket of parent
      int p = parent_[w];
      for (int v: bucket_[p]) {
        int u = eval(v);
        if (semi_[u] < semi_[v])
          idom_[v] = u;
        else
          idom_[v] = p;
      }
      bucket_[p].clear();
    }

    // final step
    for (int w = 2; w <= N; ++w) {
      if (idom_[w] != semi_[w]) idom_[w] = idom_[idom_[w]];
    }
    idom_[1] = 0; // exit has no ipdom
  }
};

struct BranchExitInfo {
  std::unordered_map<uint32_t, std::unordered_set<uint32_t>> exits;
  std::unordered_set<uint32_t>                               merges;
};

template <class Graph>
BranchExitInfo findBranchExitsAndMerges(const Graph& g, uint32_t branchNode, const std::vector<uint32_t>& fanouts) {
  using node_t = uint32_t;
  BranchExitInfo info;

  if (fanouts.empty()) return info;

  // Track which nodes are reachable by which branches
  std::unordered_map<node_t, std::unordered_set<int>> reachableBy;
  reachableBy.reserve(fanouts.size() * 10); // Heuristic: estimate graph size

  // First pass: find all merge points
  for (size_t j = 0; j < fanouts.size(); ++j) {
    node_t                     start = fanouts[j];
    std::unordered_set<node_t> visited;
    visited.reserve(20);
    std::vector<node_t> stack;
    stack.reserve(20);
    stack.push_back(start);

    while (!stack.empty()) {
      node_t v = stack.back();
      stack.pop_back();

      if (v == branchNode || visited.count(v)) continue;

      visited.insert(v);
      reachableBy[v].insert(j);

      // Detect merge and mark it
      if (reachableBy[v].size() > 1) {
        info.merges.insert(v);
        if (v != start) continue; // Stop unless it's our starting node
      }

      // Continue exploring
      for (auto succ: g.getSuccessors(v)) {
        if (!visited.count(succ)) {
          stack.push_back(succ);
        }
      }
    }
  }

  // Second pass: find exits (last nodes before merges)
  for (size_t j = 0; j < fanouts.size(); ++j) {
    node_t                     start = fanouts[j];
    std::unordered_set<node_t> visited;
    visited.reserve(20);
    std::vector<node_t> stack;
    stack.reserve(20);
    stack.push_back(start);

    while (!stack.empty()) {
      node_t v = stack.back();
      stack.pop_back();

      if (v == branchNode || visited.count(v)) continue;

      // Stop at merge points (unless it's our starting node)
      if (info.merges.count(v) && v != start) continue;

      visited.insert(v);

      auto successors = g.getSuccessors(v);

      // Check successors for merges and count them
      int mergeCount = 0;
      for (auto succ: successors) {
        if (info.merges.count(succ)) {
          mergeCount++;
        } else if (!visited.count(succ)) {
          stack.push_back(succ);
        }
      }

      // Add to exits if: has merge successors OR is a leaf
      if (mergeCount > 0 || successors.empty()) {
        info.exits[start].insert(v);
      }
    }
  }

  // Ensure every branch has an entry
  for (auto s: fanouts) {
    info.exits.try_emplace(s);
  }

  return info;
}
} // namespace compiler::analysis