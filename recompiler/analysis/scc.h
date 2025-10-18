#pragma once
#include <concepts>
#include <functional>
#include <iostream>
#include <memory_resource>
#include <set>
#include <vector>

namespace compiler::analysis {
using scc_node_t = uint32_t;
using scc_t      = std::pmr::vector<std::pmr::set<scc_node_t>>;

template <typename T>
concept RegionBuilderConcept = requires(T a, uint32_t idx) {
  { a.getNumRegions() } -> std::convertible_to<int32_t>;
  { a.getSuccessorsIdx(idx) } -> std::ranges::range;
};

template <RegionBuilderConcept Regions>
struct SCCBuilder {
  SCCBuilder(std::pmr::polymorphic_allocator<> allocator, Regions const& regions)
      : _allocator(allocator), _regions(regions), _state(regions.getNumRegions(), TarjanState {}, _allocator), _stack(_allocator), _sccs(_allocator) {}

  scc_t calculate() {
    for (int32_t i = 0; i < _regions.getNumRegions(); ++i) {
      if (_state[i].index == -1) { // If node 'i' hasn't been visited yet
        strongConnect(i);
      }
    }
    return std::move(_sccs);
  }

  private:
  void strongConnect(int32_t v) {
    auto& state   = _state[v];
    state.index   = _tarjanIndex;
    state.lowlink = _tarjanIndex;
    _tarjanIndex++;
    _stack.push_back(v);
    state.onStack = true;

    auto succs = _regions.getSuccessorsIdx(v);
    for (auto w: succs) {
      auto& item = _state[w];
      if (item.index == -1) {
        strongConnect(w);
        state.lowlink = std::min(state.lowlink, _state[w].lowlink);
      } else if (item.onStack) {
        state.lowlink = std::min(state.lowlink, item.index);
      }
    }

    if (state.lowlink == state.index) {
      std::pmr::set<scc_node_t> scc {_allocator};
      scc_node_t                w;
      do {
        w = _stack.back();
        _stack.pop_back();
        _state[w].onStack = false;
        scc.insert(w);
      } while (w != v);

      _sccs.push_back(std::move(scc));
    }
  }

  private:
  std::pmr::polymorphic_allocator<> _allocator;
  Regions const&                    _regions;

  struct TarjanState {
    int32_t index   = -1;
    int32_t lowlink = -1;
    bool    onStack = false;
  };

  std::pmr::vector<TarjanState> _state;
  std::pmr::vector<scc_node_t>  _stack;
  int32_t                       _tarjanIndex = 0;
  scc_t                         _sccs;
};

inline void dump(std::ostream& os, scc_t const& scc) {
  os << "\n Strongly Connected:\n";
  for (auto const& node: scc) {
    os << '{' << std::dec;
    for (auto id: node)
      os << id << ",";
    os << "}\n";
  }
}
} // namespace compiler::analysis