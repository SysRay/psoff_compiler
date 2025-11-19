#pragma once

#include "ir/ir.h"
#include "rvsdg.h"

#include <cstdint>
#include <span>
#include <vector>

namespace compiler::cfg {

class ControlFlow {
  public:
  ControlFlow(std::pmr::polymorphic_allocator<> allocator, size_t expectedBlocks = 128)
      : _successors(allocator), _predecessors(allocator), _nodeBuilder(allocator, expectedBlocks), _instructions(allocator) {
    _successors.reserve(expectedBlocks);
    _predecessors.reserve(expectedBlocks);
  }

  auto& accessSuccessors(rvsdg::nodeid_t id) { return _successors[id.value]; }

  auto& accessPredecessors(rvsdg::nodeid_t id) { return _predecessors[id.value]; }

  std::span<const rvsdg::nodeid_t> getSuccessors(rvsdg::nodeid_t id) const { return _successors[id.value]; }

  std::span<const rvsdg::nodeid_t> getPredecessors(rvsdg::nodeid_t id) const { return _predecessors[id.value]; }

  rvsdg::nodeid_t inline createSimpleNode() {
    _successors.emplace_back();
    _predecessors.emplace_back();
    return _nodeBuilder.__createNode<rvsdg::SimpleNode>();
  }

  rvsdg::nodeid_t inline createGammaNode() {
    _successors.emplace_back();
    _predecessors.emplace_back();
    return _nodeBuilder.__createNode<rvsdg::GammaNode>();
  }

  rvsdg::nodeid_t inline createThetaNode() {
    _successors.emplace_back();
    _predecessors.emplace_back();
    return _nodeBuilder.__createNode<rvsdg::ThetaNode>(_nodeBuilder.createRegion());
  }

  rvsdg::nodeid_t inline createLambdaNode() {
    _successors.emplace_back();
    _predecessors.emplace_back();
    return _nodeBuilder.__createNode<rvsdg::LambdaNode>(_nodeBuilder.createRegion());
  }

  auto* nodes() { return &_nodeBuilder; }

  auto* nodes() const { return &_nodeBuilder; }

  auto* accessInstructions() { return &_instructions; }

  void swap(ir::InstructionManager&& inst) { std::swap(_instructions, inst); }

  // ------------------------------------------------------------
  // Block edge manipulation
  // ------------------------------------------------------------

  void addEdge(rvsdg::nodeid_t from, rvsdg::nodeid_t to);
  void removeEdge(rvsdg::nodeid_t from, rvsdg::nodeid_t to);
  void redirectEdge(rvsdg::nodeid_t from, rvsdg::nodeid_t oldSucc, rvsdg::nodeid_t newSucc);

  private:
  std::pmr::vector<std::pmr::vector<rvsdg::nodeid_t>> _successors;
  std::pmr::vector<std::pmr::vector<rvsdg::nodeid_t>> _predecessors;

  rvsdg::Builder         _nodeBuilder;
  ir::InstructionManager _instructions;
};

} // namespace compiler::cfg
