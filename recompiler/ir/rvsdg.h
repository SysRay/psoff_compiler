#pragma once

#include "cfg.h"
#include "ir/ir.h"
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

  std::pmr::vector<nodeid_t> nodes = {}; ///< nodes belonging to this region

  Region(std::pmr::polymorphic_allocator<> alloc): arguments(alloc), results(alloc), nodes(alloc) {}
};

enum class eNodeType {
  SimpleNode, ///< Linear, for operations
  GammaNode,  ///< Decision point
  ThetaNode,  ///< Tail controlled Loop
  LambdaNode, ///< Function
};

struct Base {
  nodeid_t   id {};
  eNodeType  type;
  regionid_t parentRegion = {};

  std::pmr::vector<OutputOperandId_t> inputs  = {}; ///< inputs for this region
  std::pmr::vector<InputOperandId_t>  outputs = {}; ///< outputs indices for this region

  Base(eNodeType t): type(t) {}
};

struct SimpleNode: Base {
  std::pmr::vector<InstructionId_t> instructions = {};

  SimpleNode(std::pmr::polymorphic_allocator<> allocator): Base(eNodeType::SimpleNode) {}
};

struct GammaNode: Base {
  InputOperandId_t             predicate {};
  std::pmr::vector<regionid_t> branches = {}; ///< subregions

  GammaNode(std::pmr::polymorphic_allocator<> alloc, uint8_t numBranches): Base(eNodeType::GammaNode), branches(numBranches, alloc) {}
};

struct ThetaNode: Base {
  regionid_t body; ///< loop body, first result is loop continuation predicate

  ThetaNode(std::pmr::polymorphic_allocator<> alloc, regionid_t body): Base(eNodeType::ThetaNode), body(body) {}
};

struct LambdaNode: Base {
  regionid_t body;

  LambdaNode(std::pmr::polymorphic_allocator<> alloc, regionid_t body): Base(eNodeType::LambdaNode), body(body) {}
};

template <typename T>
concept NodeConcept = std::derived_from<T, Base>;

class Builder {
  CLASS_NO_COPY(Builder);
  CLASS_NO_MOVE(Builder);

  template <typename T, typename... Args>
  requires NodeConcept<T>
  T* __createNode(Args&&... args) {
    auto const id = nodeid_t((uint32_t)_blocks.size());

    Base* block = _blocks.emplace_back(_blocks.get_allocator().new_object<T>(_blocks.get_allocator(), std::forward<Args>(args)...));
    block->id   = id;

    return (T*)block;
  }

  public:
  Builder(std::pmr::polymorphic_allocator<> allocator, size_t expectedBlocks): _im(allocator), _cfg(allocator) {
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

  nodeid_t inline createSimpleNode() {
    _cfg.addNode();
    return __createNode<SimpleNode>()->id;
  }

  nodeid_t inline createGammaNode(uint8_t numBranches = 2) {
    _cfg.addNode();
    auto node = __createNode<GammaNode>(numBranches);

    for (auto& branch: node->branches)
      branch = createRegion();
    return node->id;
  }

  nodeid_t inline createThetaNode() {
    _cfg.addNode();
    return __createNode<ThetaNode>(createRegion())->id;
  }

  nodeid_t inline createLambdaNode() {
    _cfg.addNode();
    return __createNode<LambdaNode>(createRegion())->id;
  }

  // ------------------------------------------------------------
  // Main function
  // ------------------------------------------------------------
  nodeid_t getMainFunctionId() const { return _mainFunc; }

  LambdaNode const* getMainFunction() const { return getNode<LambdaNode>(_mainFunc); }

  LambdaNode* accessMainFunction() { return accessNode<LambdaNode>(_mainFunc); }

  void setMainFunction(nodeid_t id) { _mainFunc = id; }

  // ------------------------------------------------------------
  // Region
  // ------------------------------------------------------------
  Region* accessRegion(regionid_t id) { return &_regions[id.value]; }

  const Region* getRegion(regionid_t id) const { return &_regions[id.value]; }

  const Base* getNodeBase(nodeid_t id) const { return _blocks[id.value]; }

  Base* accessNodeBase(nodeid_t id) { return _blocks[id.value]; }

  template <typename T>
  T const* getNode(nodeid_t id) const {
    // todo check types
    return (T const*)getNodeBase(id);
  }

  template <typename T>
  T* accessNode(nodeid_t id) {
    // todo check types
    return (T*)accessNodeBase(id);
  }

  template <typename Op, typename... Args>
  inline auto create(Args&&... args) {
    return Op::create(_im, std::forward<Args>(args)...);
  }

  // ------------------------------------------------------------
  // Region membership manipulation
  // ------------------------------------------------------------

  bool regionContains(regionid_t rid, nodeid_t bid) const;
  void moveNodeToRegion(nodeid_t bid, regionid_t dest);

  /**
   * @brief Insert Node src at pos of dst (becomes regionless)
   *
   * @param src
   * @param dst
   */
  bool insertNodeToRegion(nodeid_t src, nodeid_t dst);

  //------------------------------------------------------------
  //  Block replacement / removal
  // ------------------------------------------------------------

  void replaceBlockInRegion(regionid_t rid, nodeid_t oldB, nodeid_t newB);
  void removeBlockFromRegion(regionid_t rid, nodeid_t bid);

  private:
  std::pmr::vector<Base*>  _blocks;
  std::pmr::vector<Region> _regions;

  ir::InstructionManager _im;
  ControlFlow            _cfg;

  nodeid_t _mainFunc {};
};
} // namespace compiler::ir::rvsdg
