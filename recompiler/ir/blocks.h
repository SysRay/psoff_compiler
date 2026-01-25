#pragma once

#include "cfg.h"
#include "ir/ir.h"
#include "operations.h"
#include "types.h"

#include <assert.h>
#include <limits>
#include <memory_resource>
#include <span>
#include <variant>
#include <vector>

namespace compiler::ir::rvsdg {

struct Region {
  regionid_t id {};

  std::pmr::vector<uint32_t> arguments = {}; ///< arguments for this region
  std::pmr::vector<uint32_t> results   = {}; ///< results for this region

  std::pmr::vector<blockid_t> blocks = {}; ///< nodes belonging to this region

  Region(std::pmr::polymorphic_allocator<> alloc): arguments(alloc), results(alloc), blocks(alloc) {}
};

enum class eBlockType {
  Simple, ///< Linear, for operations
  Gamma,  ///< Decision point
  Theta,  ///< Tail controlled Loop
  Lambda, ///< Function
};

struct Base {
  blockid_t   id {};
  eBlockType  type;
  regionid_t parentRegion = {};

  std::pmr::vector<OutputOperandId_t> inputs  = {}; ///< inputs for this region
  std::pmr::vector<InputOperandId_t>  outputs = {}; ///< outputs indices for this region

  Base(eBlockType t): type(t) {}
};

struct SimpleBlock: Base {
  std::pmr::vector<InstructionId_t> instructions = {};

  SimpleBlock(std::pmr::polymorphic_allocator<> allocator): Base(eBlockType::Simple) {}
};

struct GammaBlock: Base {
  InputOperandId_t             predicate {};
  std::pmr::vector<regionid_t> branches = {}; ///< subregions

  GammaBlock(std::pmr::polymorphic_allocator<> alloc, uint8_t numBranches): Base(eBlockType::Gamma), branches(numBranches, alloc) {}
};

struct ThetaBlock: Base {
  regionid_t body; ///< loop body, first result is loop continuation predicate

  ThetaBlock(std::pmr::polymorphic_allocator<> alloc, regionid_t body): Base(eBlockType::Theta), body(body) {}
};

struct LambdaNode: Base {
  regionid_t body;

  LambdaNode(std::pmr::polymorphic_allocator<> alloc, regionid_t body): Base(eBlockType::Lambda), body(body) {}
};

template <typename T>
concept NodeConcept = std::derived_from<T, Base>;

class IRBlocks {
  CLASS_NO_COPY(IRBlocks);
  CLASS_NO_MOVE(IRBlocks);

  template <typename T, typename... Args>
  requires NodeConcept<T>
  T* __createNode(Args&&... args) {
    auto const id = blockid_t((uint32_t)_blocks.size());

    Base* block = _blocks.emplace_back(_blocks.get_allocator().new_object<T>(_blocks.get_allocator(), std::forward<Args>(args)...));
    block->id   = id;

    return (T*)block;
  }

  public:
  IRBlocks(std::pmr::polymorphic_allocator<> allocator, size_t expectedBlocks): _im(allocator), _cfg(allocator) {
    _blocks.reserve(expectedBlocks);
    _regions.reserve(64);
  }

  inline size_t blocksCount() const { return _blocks.size(); }

  inline size_t regionCount() const { return _regions.size(); }

  inline auto& getInstructions() { return _im; }

  inline auto const& getInstructions() const { return _im; }

  inline auto& getCfg() { return _cfg; }

  inline auto const& getCfg() const { return _cfg; }

  regionid_t createRegion() {
    auto rid = regionid_t((uint32_t)_regions.size());
    _regions.emplace_back(_regions.get_allocator());
    _regions.back().id = rid;
    return rid;
  }

  blockid_t inline createSimpleNode() {
    _cfg.addNode();
    return __createNode<SimpleBlock>()->id;
  }

  blockid_t inline createGammaNode(uint8_t numBranches = 2) {
    _cfg.addNode();
    auto node = __createNode<GammaBlock>(numBranches);

    for (auto& branch: node->branches)
      branch = createRegion();
    return node->id;
  }

  blockid_t inline createThetaNode() {
    _cfg.addNode();
    return __createNode<ThetaBlock>(createRegion())->id;
  }

  blockid_t inline createLambdaNode() {
    _cfg.addNode();
    return __createNode<LambdaNode>(createRegion())->id;
  }

  // ------------------------------------------------------------
  // Main function
  // ------------------------------------------------------------
  blockid_t getMainFunctionId() const { return _mainFunc; }

  LambdaNode const* getMainFunction() const { return getNode<LambdaNode>(_mainFunc); }

  LambdaNode* accessMainFunction() { return accessNode<LambdaNode>(_mainFunc); }

  void setMainFunction(blockid_t id) { _mainFunc = id; }

  // ------------------------------------------------------------
  // Region
  // ------------------------------------------------------------
  Region* accessRegion(regionid_t id) { return &_regions[id.value]; }

  const Region* getRegion(regionid_t id) const { return &_regions[id.value]; }

  const Base* getBase(blockid_t id) const { return _blocks[id.value]; }

  Base* accessBase(blockid_t id) { return _blocks[id.value]; }

  template <typename T>
  T const* getNode(blockid_t id) const {
    // todo check types
    return (T const*)getBase(id);
  }

  template <typename T>
  T* accessNode(blockid_t id) {
    // todo check types
    return (T*)accessBase(id);
  }

  template <typename Op, typename... Args>
  inline auto create(Args&&... args) {
    return Op::create(_im, std::forward<Args>(args)...);
  }

  // ------------------------------------------------------------
  // Region membership manipulation
  // ------------------------------------------------------------

  bool contains(regionid_t rid, blockid_t bid) const;
  void move(blockid_t bid, regionid_t dest);

  /**
   * @brief Insert Node src at pos of dst (becomes regionless)
   *
   * @param src
   * @param dst
   */
  bool insertToRegion(blockid_t src, blockid_t dst);

  //------------------------------------------------------------
  //  Block replacement / removal
  // ------------------------------------------------------------

  void replaceBlockInRegion(regionid_t rid, blockid_t oldB, blockid_t newB);
  void removeBlockFromRegion(regionid_t rid, blockid_t bid);

  private:
  std::pmr::vector<Base*>  _blocks;
  std::pmr::vector<Region> _regions;

  ir::IROperations _im;
  ControlFlow            _cfg;

  blockid_t _mainFunc {};
};
} // namespace compiler::ir::rvsdg
