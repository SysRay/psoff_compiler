#pragma once
#include <algorithm>
#include <concepts>
#include <functional>
#include <iostream>
#include <memory_resource>
#include <set>
#include <span>
#include <vector>

namespace compiler::analysis {
using scc_node_t  = uint32_t;
using scc_nodes_t = std::pmr::set<scc_node_t>;

struct SCC {
  std::pmr::vector<scc_nodes_t> nodes;

  auto& get() const { return nodes; }

  SCC(std::pmr::polymorphic_allocator<> alloc): nodes(alloc) {}
};

template <typename T>
concept SCCBuilderConcept = requires(T a, uint32_t idx) {
  { a.size() } -> std::convertible_to<size_t>;
  { a.getSuccessors(idx) } -> std::ranges::range;
  { a.getPredecessors(idx) } -> std::ranges::range;
};

template <SCCBuilderConcept Graph>
class SCCBuilder {
  public:
  SCCBuilder(std::pmr::polymorphic_allocator<> alloc, Graph const& graph)
      : _allocator(alloc), _graph(graph), _state(graph.size(), TarjanState {}, alloc), _stack(alloc), _regionsOut(alloc) {
    _regionsOut.nodes.reserve(graph.size());
  }

  SCC calculate(scc_node_t from) {
    if (_state[from].index == -1) strongConnect(from);

    return std::move(_regionsOut);
  }

  private:
  struct TarjanState {
    int32_t index = -1, lowlink = -1;
    bool    onStack  : 1 = false;
    bool    selfLoop : 1 = false;
  };

  void strongConnect(int32_t v) {
    auto& s = _state[v];
    s.index = s.lowlink = _tarjanIndex++;
    _stack.push_back(v);
    s.onStack = true;

    for (auto w: _graph.getSuccessors(v)) {
      auto& t = _state[w];

      if (w == v) { // detect self-loop immediately
        t.selfLoop = true;
        continue;
      }

      if (t.index == -1) {
        strongConnect(w);
        s.lowlink = std::min(s.lowlink, t.lowlink);
      } else if (t.onStack) {
        s.lowlink = std::min(s.lowlink, t.index);
      }
    }

    if (s.lowlink == s.index) {
      scc_nodes_t region(_allocator);
      scc_node_t  w;

      bool isSelfLoop = false;
      do {
        w = _stack.back();
        _stack.pop_back();
        _state[w].onStack = false;
        region.insert(w);

        isSelfLoop |= _state[w].selfLoop;
      } while (w != v);

      if (region.size() > 1 || isSelfLoop) _regionsOut.nodes.push_back(std::move(region));
    }
  }

  std::pmr::polymorphic_allocator<> _allocator;
  Graph const&                      _graph;

  std::pmr::vector<TarjanState> _state;
  std::pmr::vector<scc_node_t>  _stack;
  int32_t                       _tarjanIndex = 0;
  SCC                           _regionsOut;
};

struct SCCMeta {
  std::pmr::set<scc_node_t> incoming; ///< preds outside → entry
  std::pmr::set<scc_node_t> outgoing; ///< exit → succs outside

  std::pmr::set<scc_node_t> preds; ///< nodes exits successors outside scc
  std::pmr::set<scc_node_t> succs; ///< nodes exits successors outside scc

  SCCMeta(std::pmr::polymorphic_allocator<> alloc): incoming(alloc), outgoing(alloc), preds(alloc), succs(alloc) {}
};

template <SCCBuilderConcept Graph>
SCCMeta const classifySCC(std::pmr::polymorphic_allocator<> alloc, Graph const& graph, scc_nodes_t const& nodes) {
  SCCMeta out(alloc);
  for (scc_node_t node: nodes) {
    // Check successors for exits and repetitions
    for (scc_node_t succ: graph.getSuccessors(node)) {
      if (nodes.contains(succ)) {
        // out.body.emplace(node); // Edge stays inside SCC
      } else { // Edge leaves SCC -> node is exit
        out.outgoing.insert(node);
        out.succs.insert(succ);
      }
    }

    // Check predecessors for entries
    for (auto pred: graph.getPredecessors(node)) {
      if (!nodes.contains(pred)) {
        // Incoming edge from outside SCC -> node is entry
        out.preds.insert(pred);
        out.incoming.insert(node);
      }
    }
  }
  return out;
}
} // namespace compiler::analysis
