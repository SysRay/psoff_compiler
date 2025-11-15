#pragma once

#include "types.h"

#include <assert.h>
#include <cstdint>
#include <ostream>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

namespace compiler::ir::cfg {
class ControlFlow {
  static constexpr blocks::blockid_t START_ID {0};
  static constexpr blocks::blockid_t STOP_ID {1};

  public:
  ControlFlow(std::pmr::polymorphic_allocator<> allocator, size_t expectedBlocks = 128)
      : _allocator(allocator), _blocks(allocator), _successors(allocator), _predecessors(allocator) {
    _blocks.reserve(expectedBlocks);
    _successors.reserve(expectedBlocks);
    _predecessors.reserve(expectedBlocks);
  }

  size_t size() const { return _blocks.size(); }

  template <typename T, typename... Args>
  requires blocks::BlockConcept<T>
  blocks::blockid_t createNode(Args&&... args) {
    auto const id = blocks::blockid_t((uint32_t)_blocks.size());

    blocks::Base* block = _blocks.emplace_back(_allocator.allocate_object<T>(_allocator, std::forward<Args>(args)...));
    block->id           = id;

    _successors.emplace_back();
    _predecessors.emplace_back();

    return id;
  }

  auto& accessSuccessors(blocks::blockid_t id) { return _successors[id.value]; }

  auto& accessPredecessors(blocks::blockid_t id) { return _predecessors[id.value]; }

  std::span<const blocks::blockid_t> getSuccessors(blocks::blockid_t id) const { return _successors[id.value]; }

  std::span<const blocks::blockid_t> getPredecessors(blocks::blockid_t id) const { return _predecessors[id.value]; }

  blocks::blockid_t getStartId() const { return START_ID; }

  blocks::blockid_t getStopId() const { return STOP_ID; }

  const blocks::Base* getBlock(blocks::blockid_t id) const { return _blocks[id.value]; }

  blocks::Base* accessBlock(blocks::blockid_t id) { return _blocks[id.value]; }

  private:
  std::pmr::polymorphic_allocator<> _allocator;
  std::pmr::vector<blocks::Base*>   _blocks;

  std::pmr::vector<std::pmr::vector<blocks::blockid_t>> _successors;
  std::pmr::vector<std::pmr::vector<blocks::blockid_t>> _predecessors;
};
} // namespace compiler::ir::cfg