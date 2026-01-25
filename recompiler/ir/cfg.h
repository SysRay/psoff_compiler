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

  auto& accessSuccessors(blockid_t id) { return _successors[id.value]; }

  auto& accessPredecessors(blockid_t id) { return _predecessors[id.value]; }

  std::span<const blockid_t> getSuccessors(blockid_t id) const { return _successors[id.value]; }

  std::span<const blockid_t> getPredecessors(blockid_t id) const { return _predecessors[id.value]; }

  void addNode() {
    _successors.emplace_back();
    _predecessors.emplace_back();
  }

  auto size() const { return _successors.size(); }

  // ------------------------------------------------------------
  // Block edge manipulation
  // ------------------------------------------------------------

  void addEdge(blockid_t from, blockid_t to);
  void removeEdge(blockid_t from, blockid_t to);
  void redirectEdge(blockid_t from, blockid_t oldSucc, blockid_t newSucc);
  void redirectEdgeReversed(blockid_t oldPred, blockid_t to, blockid_t newPred);

  private:
  std::pmr::vector<std::pmr::vector<blockid_t>> _successors;
  std::pmr::vector<std::pmr::vector<blockid_t>> _predecessors;
};

} // namespace compiler::ir
