#pragma once
#include <algorithm>
#include <concepts>
#include <functional>
#include <iostream>
#include <memory_resource>
#include <set>
#include <vector>

namespace compiler::analysis {

using scc_node_t = uint32_t;

struct SCCRegion {
  std::pmr::set<scc_node_t>                        nodes;
  std::pmr::set<scc_node_t>                        entries;
  std::pmr::set<scc_node_t>                        exits;
  std::pmr::set<std::pair<scc_node_t, scc_node_t>> repetitions;
  std::pmr::vector<SCCRegion>                      children;
};

template <typename T>
concept RegionBuilderConcept = requires(T a, uint32_t idx) {
  { a.getNumRegions() } -> std::convertible_to<int32_t>;
  { a.getSuccessorsIdx(idx) } -> std::ranges::range;
};

template <RegionBuilderConcept Regions>
class SCCBuilder {
  public:
  SCCBuilder(std::pmr::polymorphic_allocator<> alloc, Regions const& regions, std::function<bool(uint32_t, uint32_t)> edgeFilter = {}, int depth = 0)
      : _allocator(alloc),
        _regions(regions),
        _edgeFilter(std::move(edgeFilter)),
        _state(regions.getNumRegions(), TarjanState {}, alloc),
        _stack(alloc),
        _regionsOut(alloc),
        _depth(depth) {}

  std::pmr::vector<SCCRegion> calculate() {
    printf("%*s[Depth %d] Starting Tarjan on %d nodes\n", _depth * 2, "", _depth, _regions.getNumRegions());
    for (int32_t i = 0; i < _regions.getNumRegions(); ++i)
      if (_state[i].index == -1) strongConnect(i);

    classifyArcs(_regionsOut);
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
      SCCRegion  region {std::pmr::set<scc_node_t> {_allocator}, std::pmr::set<scc_node_t> {_allocator}, std::pmr::set<scc_node_t> {_allocator},
                        std::pmr::set<std::pair<scc_node_t, scc_node_t>> {_allocator}, std::pmr::vector<SCCRegion> {_allocator}};
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
            if (v < u) region.repetitions.insert({u, v});
          } else
            region.exits.insert(v);
        }
      }
      for (auto n: region.nodes)
        if (!hasPredInside.contains(n)) region.entries.insert(n);

      // After computing entries via external predecessors
      if (region.entries.empty() && region.nodes.contains(0)) {
        region.entries.insert(0);
        printf("%*s[Depth %d] Treating node 0 as graph entry\n", _depth * 2, "", _depth);
      }

      printf("%*s[Depth %d] Region: {", _depth * 2, "", _depth);
      for (auto n: region.nodes)
        printf("%d ", n);
      printf("} entries:{");
      for (auto n: region.entries)
        printf("%d ", n);
      printf("} exits:{");
      for (auto n: region.exits)
        printf("%d ", n);
      printf("} reps:{");
      for (auto [u, v]: region.repetitions)
        printf("(%d,%d) ", u, v);
      printf("}\n");
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
        printf("%*s[Depth %d] Region {", _depth * 2, "", _depth);
        for (auto n: region.nodes)
          printf("%d ", n);
        printf("} has no outer reps, skipping recursion\n");
        continue;
      }

      printf("%*s[Depth %d] Recurse on region {", _depth * 2, "", _depth);
      for (auto n: region.nodes)
        printf("%d ", n);
      printf("}, outer reps:");
      for (auto [u, v]: outerReps)
        printf(" (%d→%d)", u, v);
      printf("\n");

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

  std::pmr::polymorphic_allocator<>       _allocator;
  Regions const&                          _regions;
  std::function<bool(uint32_t, uint32_t)> _edgeFilter;
  std::pmr::vector<TarjanState>           _state;
  std::pmr::vector<scc_node_t>            _stack;
  int32_t                                 _tarjanIndex = 0;
  std::pmr::vector<SCCRegion>             _regionsOut;
  int                                     _depth;
};

// -----------------------------------------------------------------------------
// Dump helpers
// -----------------------------------------------------------------------------
inline void dumpRegion(std::ostream& os, SCCRegion const& r, int depth = 0) {
  std::string pad(depth * 2, ' ');
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
