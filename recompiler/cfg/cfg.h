#pragma once

#include "types.h"

#include <cstdint>
#include <span>
#include <vector>

namespace compiler::cfg {

class ControlFlow {
  public:
  ControlFlow(std::pmr::polymorphic_allocator<> allocator, size_t expectedBlocks = 128)
      : _allocator(allocator), _blocks(allocator), _successors(allocator), _predecessors(allocator), _regions(allocator) {
    _blocks.reserve(expectedBlocks);
    _successors.reserve(expectedBlocks);
    _predecessors.reserve(expectedBlocks);
    _regions.reserve(64);

    createRegion(); // Create root
  }

  inline size_t blocksCount() const { return _blocks.size(); }

  inline size_t regionCount() const { return _regions.size(); }

  template <typename T, typename... Args>
  requires blocks::BlockConcept<T>
  blocks::blockid_t createNode(Args&&... args) {
    auto const id = blocks::blockid_t((uint32_t)_blocks.size());

    blocks::Base* block = _blocks.emplace_back(_allocator.new_object<T>(_allocator, std::forward<Args>(args)...));
    block->id           = id;

    _successors.emplace_back();
    _predecessors.emplace_back();

    return id;
  }

  blocks::blockid_t inline createBlock() { return createNode<blocks::BlockNode>(); }

  auto& accessSuccessors(blocks::blockid_t id) { return _successors[id.value]; }

  auto& accessPredecessors(blocks::blockid_t id) { return _predecessors[id.value]; }

  std::span<const blocks::blockid_t> getSuccessors(blocks::blockid_t id) const { return _successors[id.value]; }

  std::span<const blocks::blockid_t> getPredecessors(blocks::blockid_t id) const { return _predecessors[id.value]; }

  blocks::regionid_t createRegion() {
    auto rid = blocks::regionid_t((uint32_t)_regions.size());
    _regions.emplace_back(_allocator);
    _regions.back().id = rid;
    return rid;
  }

  blocks::regionid_t getRootRegionId() const { return blocks::regionid_t(0); }

  blocks::Region& accessRegion(blocks::regionid_t id) { return _regions[id.value]; }

  const blocks::Region& getRegion(blocks::regionid_t id) const { return _regions[id.value]; }

  const blocks::Base* getBlock(blocks::blockid_t id) const { return _blocks[id.value]; }

  blocks::Base* accessBlock(blocks::blockid_t id) { return _blocks[id.value]; }

  // ------------------------------------------------------------
  // Block edge manipulation
  // ------------------------------------------------------------

  void addEdge(blocks::blockid_t from, blocks::blockid_t to);
  void removeEdge(blocks::blockid_t from, blocks::blockid_t to);
  void replaceSuccessor(blocks::blockid_t from, blocks::blockid_t oldSucc, blocks::blockid_t newSucc);

  // ------------------------------------------------------------
  // Region membership manipulation
  // ------------------------------------------------------------

  bool regionContains(blocks::regionid_t rid, blocks::blockid_t bid) const;
  void moveBlockToRegion(blocks::blockid_t bid, blocks::regionid_t dest);

  void setRegionEntry(blocks::regionid_t rid, blocks::blockid_t bid) { _regions[rid.value].entry = bid; }

  void setRegionExit(blocks::regionid_t rid, blocks::blockid_t bid) { _regions[rid.value].exit = bid; }

  // ------------------------------------------------------------
  // Region hierarchy manipulation
  // ------------------------------------------------------------
  void addSubregion(blocks::regionid_t parent, blocks::regionid_t child);
  void removeSubregion(blocks::regionid_t parent, blocks::regionid_t child);

  // ------------------------------------------------------------
  // RegionNode helpers
  // ------------------------------------------------------------

  void addRegionToNode(blocks::blockid_t node, blocks::regionid_t rid);
  void addStructuredSuccessor(blocks::blockid_t node, blocks::blockid_t succ);

  //------------------------------------------------------------
  //  Block replacement / removal
  // ------------------------------------------------------------

  void replaceBlockInRegion(blocks::regionid_t rid, blocks::blockid_t oldB, blocks::blockid_t newB);
  void removeBlockFromRegion(blocks::regionid_t rid, blocks::blockid_t bid);

  private:
  std::pmr::polymorphic_allocator<> _allocator;

  std::pmr::vector<blocks::Base*>                       _blocks;
  std::pmr::vector<std::pmr::vector<blocks::blockid_t>> _successors;
  std::pmr::vector<std::pmr::vector<blocks::blockid_t>> _predecessors;

  std::pmr::vector<blocks::Region> _regions;
};

} // namespace compiler::cfg
