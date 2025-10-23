#pragma once

#include "logging.h"

#include <algorithm>
#include <concepts>
#include <format>
#include <functional>
#include <iostream>
#include <memory_resource>
#include <set>
#include <sstream>
#include <vector>

namespace compiler::analysis {
using scc_node_t = uint32_t;

struct SCCRegion {
  std::pmr::set<scc_node_t>                        nodes;
  std::pmr::set<scc_node_t>                        entries;
  std::pmr::set<scc_node_t>                        exits;
  std::pmr::set<std::pair<scc_node_t, scc_node_t>> repetitions;
  std::pmr::vector<SCCRegion>                      children;

  SCCRegion(std::pmr::polymorphic_allocator<> allocator): nodes(allocator), entries(allocator), exits(allocator), repetitions(allocator), children(allocator) {}
};

template <typename T>
concept RegionBuilderConcept = requires(T a, uint32_t idx) {
  { a.getNumRegions() } -> std::convertible_to<int32_t>;
  { a.getSuccessorsIdx(idx) } -> std::ranges::range;
};

template <RegionBuilderConcept Regions>
class SCCBuilder {
  public:
  SCCBuilder(std::pmr::polymorphic_allocator<> alloc, Regions const& regions, std::function<bool(uint32_t, uint32_t)> edgeFilter = {}, uint32_t depth = 0)
      : _allocator(alloc),
        _regions(regions),
        _edgeFilter(std::move(edgeFilter)),
        _state(regions.getNumRegions(), TarjanState {}, alloc),
        _stack(alloc),
        _regionsOut(alloc),
        _depth(depth) {}

  std::pmr::vector<SCCRegion> calculate() {
    LOG(eLOG_TYPE::DEBUG, "{}[Depth {}] Starting Tarjan on {} nodes", width(_depth), _depth, _regions.getNumRegions());
    for (int32_t i = 0; i < _regions.getNumRegions(); ++i)
      if (_state[i].index == -1) strongConnect(i);

    classifyArcs(_regionsOut);

    // remove non-loop singleton
    std::erase_if(_regionsOut, [](SCCRegion const& r) {
      if (r.nodes.size() > 1) return false;
      for (auto const& [u, v]: r.repetitions)
        if (u == v) return false;
      return true;
    });

    findNested(_regionsOut);
    return std::move(_regionsOut);
  }

  private:
  struct TarjanState {
    int32_t index = -1, lowlink = -1;
    bool    onStack = false;
  };

  void strongConnect(int32_t v) {
    auto& s = _state[v];
    s.index = s.lowlink = _tarjanIndex++;
    _stack.push_back(v);
    s.onStack = true;

    for (auto w: _regions.getSuccessorsIdx(v)) {
      if (_edgeFilter && !_edgeFilter(v, w)) continue;
      auto& t = _state[w];
      if (t.index == -1) {
        strongConnect(w);
        s.lowlink = std::min(s.lowlink, t.lowlink);
      } else if (t.onStack) {
        s.lowlink = std::min(s.lowlink, t.index);
      }
    }

    if (s.lowlink == s.index) {
      SCCRegion  region(_allocator);
      scc_node_t w;
      do {
        w = _stack.back();
        _stack.pop_back();
        _state[w].onStack = false;
        region.nodes.insert(w);
      } while (w != v);

      _regionsOut.push_back(std::move(region));
    }
  }

  void classifyArcs(std::pmr::vector<SCCRegion>& list) {
    for (auto& region: list) {
      std::pmr::set<scc_node_t> hasPredInside {_allocator};

      for (auto u: region.nodes) {
        for (auto v: _regions.getSuccessorsIdx(u)) {
          if (_edgeFilter && !_edgeFilter(u, v)) continue;

          if (region.nodes.contains(v)) {
            hasPredInside.insert(v);
            if (v == u) { // explicit self-loop
              region.repetitions.insert({u, v});
            } else if (v < u) {
              region.repetitions.insert({u, v});
            }
          } else {
            region.exits.insert(v);
          }
        }
      }

      for (auto n: region.nodes)
        if (!hasPredInside.contains(n)) region.entries.insert(n);

      if (region.entries.empty() && region.nodes.contains(0)) {
        region.entries.insert(0);
      }
    }
  }

  void findNested(std::pmr::vector<SCCRegion>& list) {
    for (auto& region: list) {
      if (region.nodes.size() < 2) continue;

      // Identify outer repetition arcs: target is entry
      std::pmr::set<std::pair<scc_node_t, scc_node_t>> outerReps {_allocator};
      for (auto const& [u, v]: region.repetitions)
        if (region.entries.contains(v)) outerReps.insert({u, v});

      if (outerReps.empty()) {
        LOG(eLOG_TYPE::DEBUG, [&]() {
          std::ostringstream oss;
          oss << width(_depth) << "[Depth " << _depth << "] Region: {{";

          for (auto n: region.nodes)
            oss << n << " ";

          oss << "}} has no outer reps, skipping recursion";
          return oss.str();
        }());

        continue;
      }

      LOG(eLOG_TYPE::DEBUG, [&]() {
        std::ostringstream oss;
        oss << width(_depth) << "[Depth " << _depth << "]  Recurse on region: {{";

        for (auto n: region.nodes) {
          oss << n << " ";
        }

        oss << "}}, outer reps:";
        for (auto const& [u, v]: outerReps) {
          oss << " (" << u << "→" << v << ")";
        }

        return oss.str();
      }());

      auto filter = [allowed = &region.nodes, outerReps = &outerReps](uint32_t u, uint32_t v) {
        if (!allowed->contains(u) || !allowed->contains(v)) return false;
        return !outerReps->contains({u, v});
      };

      SCCBuilder inner {_allocator, _regions, filter, _depth + 1};
      auto       nested = inner.calculate();

      for (auto& n: nested)
        if (n.nodes.size() > 1) region.children.push_back(std::move(n));
    }
  }

  std::pmr::polymorphic_allocator<> _allocator;
  Regions const&                    _regions;

  std::function<bool(uint32_t, uint32_t)> _edgeFilter;
  std::pmr::vector<TarjanState>           _state;
  std::pmr::vector<scc_node_t>            _stack;
  int32_t                                 _tarjanIndex = 0;
  std::pmr::vector<SCCRegion>             _regionsOut;
  uint32_t                                _depth;
};

// -----------------------------------------------------------------------------
// Dump helpers
// -----------------------------------------------------------------------------
inline void dumpRegion(std::ostream& os, SCCRegion const& r, uint32_t depth = 0) {
  auto const pad = width(depth);
  os << pad << "Nodes: {";
  for (auto n: r.nodes)
    os << n << ' ';
  os << "}\n" << pad << "Entries: {";
  for (auto e: r.entries)
    os << e << ' ';
  os << "} Exits: {";
  for (auto e: r.exits)
    os << e << ' ';
  os << "} Repetitions: {";
  for (auto [u, v]: r.repetitions)
    os << "(" << u << "," << v << ") ";
  os << "}\n";
  for (auto const& c: r.children) {
    os << pad << " Child:\n";
    dumpRegion(os, c, depth + 1);
  }
}

inline void dump(std::ostream& os, std::pmr::vector<SCCRegion> const& regs) {
  os << "\nHierarchical SCC Regions:\n";
  for (auto const& r: regs)
    dumpRegion(os, r, 1);
}

} // namespace compiler::analysis
