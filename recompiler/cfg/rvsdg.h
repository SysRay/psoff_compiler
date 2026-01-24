#pragma once

#include "ir/ir.h"

#include <assert.h>
#include <limits>
#include <memory_resource>
#include <variant>
#include <vector>

namespace compiler::cfg::rvsdg {

struct nodeid_t {
  using underlying_t = uint32_t;

  static inline constexpr nodeid_t NO_VALUE() { return nodeid_t(std::numeric_limits<underlying_t>::max()); };

  underlying_t value = NO_VALUE().value;

  constexpr nodeid_t() = default;

  constexpr explicit nodeid_t(underlying_t v): value(v) {}

  constexpr operator underlying_t() const { return value; }

  constexpr bool operator==(nodeid_t const&) const = default;

  constexpr bool isValid() const { return value != NO_VALUE().value; }
};

struct edge_t {
  nodeid_t from;
  nodeid_t to;

  constexpr operator std::pair<nodeid_t, nodeid_t>() const { return {from, to}; }

  constexpr bool operator==(const edge_t&) const = default;

  edge_t(nodeid_t from, nodeid_t to): from(from), to(to) {}

  edge_t(nodeid_t::underlying_t from, nodeid_t::underlying_t to): from(nodeid_t(from)), to(nodeid_t(to)) {}
};

struct regionid_t {
  using underlying_t = uint32_t;

  underlying_t value = std::numeric_limits<underlying_t>::max();

  constexpr regionid_t() = default;

  constexpr explicit regionid_t(underlying_t v): value(v) {}

  constexpr operator underlying_t() const { return value; }

  bool isValid() const { return value != std::numeric_limits<underlying_t>::max(); }
};

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
  std::pmr::vector<regionid_t> branches = {}; ///< subregions

  GammaNode(std::pmr::polymorphic_allocator<> alloc): Base(eNodeType::GammaNode), branches(alloc) {}
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
  public:
  Builder(std::pmr::polymorphic_allocator<> allocator, size_t expectedBlocks) {
    _blocks.reserve(expectedBlocks);
    _regions.reserve(64);
  }

  inline size_t blocksCount() const { return _blocks.size(); }

  inline size_t regionCount() const { return _regions.size(); }

  regionid_t createRegion() {
    auto rid = regionid_t((uint32_t)_regions.size());
    _regions.emplace_back(_regions.get_allocator());
    _regions.back().id = rid;
    return rid;
  }

  template <typename T, typename... Args>
  requires NodeConcept<T>
  nodeid_t __createNode(Args&&... args) {
    auto const id = nodeid_t((uint32_t)_blocks.size());

    Base* block = _blocks.emplace_back(_blocks.get_allocator().new_object<T>(_blocks.get_allocator(), std::forward<Args>(args)...));
    block->id   = id;

    return id;
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

  nodeid_t _mainFunc {};
};
} // namespace compiler::cfg::rvsdg
