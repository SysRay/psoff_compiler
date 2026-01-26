#pragma once

#include "blocks.h"
#include "ir/ir.h"
#include "types.h"

#include <cstdint>
#include <span>
#include <vector>

namespace compiler::ir {

class ControlFlow {
  CLASS_NO_COPY(ControlFlow);

  inline void addNode() {
    _successors.emplace_back();
    _predecessors.emplace_back();
  }

  public:
  ControlFlow(std::pmr::polymorphic_allocator<> allocator, ir::rvsdg::IRBlocks& blocks): _successors(allocator), _predecessors(allocator), _blocks(blocks) {
    _successors.reserve(blocks.numBlocks());
    _predecessors.reserve(blocks.numBlocks());
  }

  auto& accessSuccessors(blockid_t id) { return _successors[id.value]; }

  auto& accessPredecessors(blockid_t id) { return _predecessors[id.value]; }

  std::span<const blockid_t> getSuccessors(blockid_t id) const { return _successors[id.value]; }

  std::span<const blockid_t> getPredecessors(blockid_t id) const { return _predecessors[id.value]; }

  auto size() const { return _successors.size(); }

  auto& getBlocks() { return _blocks; }

  auto& getBlocks() const { return _blocks; }

  // ------------------------------------------------------------
  // Block edge manipulation
  // ------------------------------------------------------------

  void addEdge(blockid_t from, blockid_t to);
  void removeEdge(blockid_t from, blockid_t to);
  void redirectEdge(blockid_t from, blockid_t oldSucc, blockid_t newSucc);
  void redirectEdgeReversed(blockid_t oldPred, blockid_t to, blockid_t newPred);

  // // Forward calls + create edges
  blockid_t inline createSimpleNode() {
    addNode();
    return _blocks.createSimpleNode();
  }

  blockid_t inline createGammaNode(uint8_t numBranches = 2) {
    addNode();
    return _blocks.createGammaNode(numBranches);
  }

  blockid_t inline createThetaNode() {
    addNode();
    return _blocks.createThetaNode();
  }

  blockid_t inline createLambdaNode() {
    addNode();
    return _blocks.createLambdaNode();
  }

  private:
  std::pmr::vector<std::pmr::vector<blockid_t>> _successors;
  std::pmr::vector<std::pmr::vector<blockid_t>> _predecessors;

  ir::rvsdg::IRBlocks& _blocks;
};

} // namespace compiler::ir
