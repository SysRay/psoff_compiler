#pragma once

#include "ir/ir.h"
#include "types.h"

#include <cstdint>
#include <span>
#include <vector>

namespace compiler::ir {

class ControlFlow {
  CLASS_NO_COPY(ControlFlow);
  CLASS_NO_MOVE(ControlFlow);

  public:
  ControlFlow(std::pmr::polymorphic_allocator<> allocator, size_t expectedBlocks = 128): _successors(allocator), _predecessors(allocator) {
    _successors.reserve(expectedBlocks);
    _predecessors.reserve(expectedBlocks);
  }

  auto& accessSuccessors(nodeid_t id) { return _successors[id.value]; }

  auto& accessPredecessors(nodeid_t id) { return _predecessors[id.value]; }

  std::span<const nodeid_t> getSuccessors(nodeid_t id) const { return _successors[id.value]; }

  std::span<const nodeid_t> getPredecessors(nodeid_t id) const { return _predecessors[id.value]; }

  void addNode() {
    _successors.emplace_back();
    _predecessors.emplace_back();
  }

  auto size() const { return _successors.size(); }

  // ------------------------------------------------------------
  // Block edge manipulation
  // ------------------------------------------------------------

  void addEdge(nodeid_t from, nodeid_t to);
  void removeEdge(nodeid_t from, nodeid_t to);
  void redirectEdge(nodeid_t from, nodeid_t oldSucc, nodeid_t newSucc);
  void redirectEdgeReversed(nodeid_t oldPred, nodeid_t to, nodeid_t newPred);

  private:
  std::pmr::vector<std::pmr::vector<nodeid_t>> _successors;
  std::pmr::vector<std::pmr::vector<nodeid_t>> _predecessors;
};

} // namespace compiler::ir
