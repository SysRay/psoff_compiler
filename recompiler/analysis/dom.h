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
#include <vector>

namespace compiler::analysis {

template <typename T>
concept DenseGraphConcept = requires(T g, uint32_t idx) {
  { g.getSuccessors(idx) } -> std::ranges::range;
  { g.getPredecessors(idx) } -> std::ranges::range;
  { g.getNodeCount() } -> std::convertible_to<uint32_t>;
  // IMPORTANT: node indices must be 0..N-1
};

template <DenseGraphConcept Graph>
class DominatorTreeDense {
  public:
  using node_t  = uint32_t;
  using alloc_t = std::pmr::polymorphic_allocator<>;

  explicit DominatorTreeDense(const Graph& g, node_t entry, alloc_t alloc = {}): alloc_(alloc) { build(g, entry); }

  std::optional<node_t> get_idom(node_t n) const {
    if (n >= node_to_dfs_.size()) return std::nullopt;
    int dfn = node_to_dfs_[n];
    if (dfn <= 0 || dfn >= (int)idom_.size()) return std::nullopt;
    int id = idom_[dfn];
    if (id <= 0) return std::nullopt;
    return vertex_[id];
  }

  bool dominates(node_t a, node_t b) const {
    if (a >= node_to_dfs_.size() || b >= node_to_dfs_.size()) return false;
    int da = node_to_dfs_[a];
    int db = node_to_dfs_[b];
    if (da <= 0 || db <= 0) return false;
    if (da == db) return true;
    int cur = db;
    while (cur > 0 && cur != da)
      cur = idom_[cur];
    return cur == da;
  }

  private:
  alloc_t alloc_;

  std::pmr::vector<int> node_to_dfs_; // size N, dfs number or 0 = not visited

  std::pmr::vector<node_t>                vertex_;
  std::pmr::vector<int>                   parent_, semi_, idom_, ancestor_, label_;
  std::pmr::vector<std::pmr::vector<int>> bucket_;

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
    return (semi_[label_[ancestor_[v]]] >= semi_[label_[v]]) ? label_[v] : label_[ancestor_[v]];
  }

  void link(int v, int w) { ancestor_[w] = v; }

  void build(const Graph& g, node_t entry) {
    uint32_t Nnodes = g.getNodeCount();
    node_to_dfs_    = std::pmr::vector<int>(Nnodes, 0, alloc_);

    vertex_.clear();
    vertex_.push_back(0);
    parent_   = std::pmr::vector<int>(1, alloc_);
    semi_     = std::pmr::vector<int>(1, alloc_);
    idom_     = std::pmr::vector<int>(1, alloc_);
    ancestor_ = std::pmr::vector<int>(1, alloc_);
    label_    = std::pmr::vector<int>(1, alloc_);
    bucket_   = std::pmr::vector<std::pmr::vector<int>>(alloc_);

    int time = 0;

    struct Frame {
      node_t u;
      size_t i;
    };

    std::pmr::vector<Frame> stk(alloc_);
    stk.reserve(128);
    stk.push_back({entry, 0});

    while (!stk.empty()) {
      auto&  fr = stk.back();
      node_t u  = fr.u;

      if (node_to_dfs_[u] == 0) {
        // visit
        ++time;
        node_to_dfs_[u] = time;
        vertex_.push_back(u);
        parent_.push_back(0);
        semi_.push_back(time);
        idom_.push_back(0);
        ancestor_.push_back(0);
        label_.push_back(time);
        bucket_.push_back(std::pmr::vector<int>(alloc_));

        if (time > 1) {
          node_t p      = stk[stk.size() - 2].u;
          parent_[time] = node_to_dfs_[p];
        }
      }

      auto   succs  = g.getSuccessors(u);
      bool   pushed = false;
      size_t idx    = 0;
      for (node_t s: succs) {
        if (idx++ < fr.i) continue;
        ++fr.i;
        if (node_to_dfs_[s] == 0) {
          stk.push_back({s, 0});
          pushed = true;
          break;
        }
      }
      if (!pushed) stk.pop_back();
    }

    int                   N = time;
    std::pmr::vector<int> preds(alloc_);
    preds.reserve(16);

    for (int w = N; w >= 2; --w) {
      preds.clear();
      node_t n_w = vertex_[w];
      for (auto p: g.getPredecessors(n_w)) {
        int d = node_to_dfs_[p];
        if (d) preds.push_back(d);
      }

      int s      = parent_[w];
      int semi_w = s;
      for (int v: preds) {
        int u = eval(v);
        if (semi_[u] < semi_w) semi_w = semi_[u];
      }
      semi_[w] = semi_w;

      bucket_[semi_w].push_back(w);
      link(parent_[w], w);

      int p = parent_[w];
      for (int v: bucket_[p]) {
        int u    = eval(v);
        idom_[v] = (semi_[u] < semi_[v]) ? u : p;
      }
      bucket_[p].clear();
    }

    for (int w = 2; w <= N; ++w)
      if (idom_[w] != semi_[w]) idom_[w] = idom_[idom_[w]];

    idom_[1] = 0;
  }
};

template <DenseGraphConcept Graph>
class PostDominatorTreeDense {
  public:
  using node_t  = uint32_t;
  using alloc_t = std::pmr::polymorphic_allocator<>;

  explicit PostDominatorTreeDense(const Graph& g, node_t exit, alloc_t alloc = {}): alloc_(alloc) { build(g, exit); }

  std::optional<node_t> get_ipdom(node_t n) const {
    if (n >= node_to_dfs_.size()) return std::nullopt;
    int dfn = node_to_dfs_[n];
    if (dfn <= 0 || dfn >= (int)idom_.size()) return std::nullopt;
    int id = idom_[dfn];
    if (id <= 0) return std::nullopt;
    return vertex_[id];
  }

  bool postdominates(node_t a, node_t b) const {
    if (a >= node_to_dfs_.size() || b >= node_to_dfs_.size()) return false;
    int da = node_to_dfs_[a];
    int db = node_to_dfs_[b];
    if (da <= 0 || db <= 0) return false;
    if (da == db) return true;
    int cur = db;
    while (cur > 0 && cur != da)
      cur = idom_[cur];
    return cur == da;
  }

  private:
  alloc_t alloc_;

  std::pmr::vector<int> node_to_dfs_;

  std::pmr::vector<node_t>                vertex_;
  std::pmr::vector<int>                   parent_, semi_, idom_, ancestor_, label_;
  std::pmr::vector<std::pmr::vector<int>> bucket_;

  void compress(int v) {
    int a = ancestor_[v];
    if (!a) return;
    if (ancestor_[a]) {
      compress(a);
      if (semi_[label_[a]] < semi_[label_[v]]) label_[v] = label_[a];
      ancestor_[v] = ancestor_[a];
    }
  }

  int eval(int v) {
    if (!ancestor_[v]) return label_[v];
    compress(v);
    return (semi_[label_[ancestor_[v]]] >= semi_[label_[v]]) ? label_[v] : label_[ancestor_[v]];
  }

  void link(int v, int w) { ancestor_[w] = v; }

  void build(const Graph& g, node_t exit) {
    uint32_t Nnodes = g.getNodeCount();
    node_to_dfs_    = std::pmr::vector<int>(Nnodes, 0, alloc_);

    vertex_.clear();
    vertex_.push_back(0);
    parent_   = std::pmr::vector<int>(1, alloc_);
    semi_     = std::pmr::vector<int>(1, alloc_);
    idom_     = std::pmr::vector<int>(1, alloc_);
    ancestor_ = std::pmr::vector<int>(1, alloc_);
    label_    = std::pmr::vector<int>(1, alloc_);
    bucket_   = std::pmr::vector<std::pmr::vector<int>>(alloc_);

    int time = 0;

    struct Frame {
      node_t u;
      size_t i;
    };

    std::pmr::vector<Frame> stk(alloc_);
    stk.push_back({exit, 0});

    while (!stk.empty()) {
      auto&  fr = stk.back();
      node_t u  = fr.u;
      if (node_to_dfs_[u] == 0) {
        ++time;
        node_to_dfs_[u] = time;
        vertex_.push_back(u);
        parent_.push_back(0);
        semi_.push_back(time);
        idom_.push_back(0);
        ancestor_.push_back(0);
        label_.push_back(time);
        bucket_.push_back(std::pmr::vector<int>(alloc_));
        if (time > 1) {
          node_t p      = stk[stk.size() - 2].u;
          parent_[time] = node_to_dfs_[p];
        }
      }

      auto   preds  = g.getPredecessors(u);
      bool   pushed = false;
      size_t idx    = 0;
      for (node_t s: preds) {
        if (idx++ < fr.i) continue;
        ++fr.i;
        if (node_to_dfs_[s] == 0) {
          stk.push_back({s, 0});
          pushed = true;
          break;
        }
      }
      if (!pushed) stk.pop_back();
    }

    int                   N = time;
    std::pmr::vector<int> preds(alloc_);
    preds.reserve(16);

    for (int w = N; w >= 2; --w) {
      preds.clear();
      node_t n_w = vertex_[w];
      for (auto succ: g.getSuccessors(n_w)) {
        int d = node_to_dfs_[succ];
        if (d) preds.push_back(d);
      }

      int s      = parent_[w];
      int semi_w = s;
      for (int v: preds) {
        int u = eval(v);
        if (semi_[u] < semi_w) semi_w = semi_[u];
      }
      semi_[w] = semi_w;
      bucket_[semi_w].push_back(w);
      link(parent_[w], w);
      int p = parent_[w];
      for (int v: bucket_[p]) {
        int u    = eval(v);
        idom_[v] = (semi_[u] < semi_[v]) ? u : p;
      }
      bucket_[p].clear();
    }

    for (int w = 2; w <= N; ++w)
      if (idom_[w] != semi_[w]) idom_[w] = idom_[idom_[w]];
    idom_[1] = 0;
  }
};

} // namespace compiler::analysis
