#pragma once

#include "fixed_containers/fixed_vector.hpp"
#include "ir/cfg/types.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory_resource>
#include <optional>
#include <span>
#include <vector>

namespace compiler {
class Builder;

namespace ir {
struct InstCore;
}
} // namespace compiler

namespace compiler::frontend::analysis {
using regionid_t = uint32_t;

enum class CFGNodeType : uint8_t { Simple, Gamma, Theta };
using astnodeid_t = int16_t;

struct CFGNode {
  CFGNodeType type;
  uint32_t    instrStart;
  uint32_t    instrEnd;

  // Children array:
  // Gamma: [thenBranch, elseBranch]
  // Theta: [body, nullptr]
  // Simple: [nullptr, nullptr]
  std::array<astnodeid_t, 2> bodies {-1, -1};

  char auxPredicate {'\0'}; // todo add with ssa, or use operand?

  // Factory functions
  static CFGNode makeSimple(uint32_t start, uint32_t end) { return CFGNode {CFGNodeType::Simple, start, end, {-1, -1}, '\0'}; }

  static CFGNode makeGamma(uint32_t start, uint32_t end, astnodeid_t thenNode, astnodeid_t elseNode, char pred = '\0') {
    return CFGNode {CFGNodeType::Gamma, start, end, {thenNode, elseNode}, pred};
  }

  static CFGNode makeTheta(uint32_t start, uint32_t end, astnodeid_t bodyNode) { return CFGNode {CFGNodeType::Theta, start, end, {bodyNode, -1}, '\0'}; }
};

class RegionBuilder {
  friend class StructuredCFGBuilder;

  public:
  static constexpr regionid_t NO_REGION = -1;

  RegionBuilder(uint32_t N, std::pmr::memory_resource* pool): _regions {pool} {
    _regions.reserve(128);
    _regions.emplace_back(Region(0, N));
  }

  void addJump(regionid_t from, regionid_t to);
  void addReturn(regionid_t from);
  void addCondJump(regionid_t from, regionid_t to);

  // todo use small vector
  std::vector<regionid_t> getSuccessors(regionid_t start) const;
  std::vector<regionid_t> getPredecessors(regionid_t start) const;

  fixed_containers::FixedVector<int32_t, 2> getSuccessorsIdx(uint32_t region_idx) const;

  /**
   * @brief Get the Region given a index
   *
   * @param index
   * @return std::pair<uint32_t, uint32_t> start, end
   */
  std::pair<regionid_t, uint32_t> findRegion(uint32_t index) const;
  std::pair<uint32_t, uint32_t>   getRegion(uint32_t index) const;

  auto getNumRegions() const { return _regions.size(); }

  void dump(std::ostream& os, void* region) const;

  void for_each(auto cb) const {
    for (auto const& region: _regions) {
      cb(region.start, region.end, (void*)&region);
    }
  }

  protected:
  struct Region {
    uint32_t start = 0;
    uint32_t end   = 0;

    uint32_t trueSucc = NO_SUCC;

    bool hasJump = false;

    static constexpr uint32_t NO_SUCC = UINT32_MAX;

    Region(uint32_t s, uint32_t e): start(s), end(e) {}

    // Region() = default;

    inline bool hasTrueSucc() const { return trueSucc != NO_SUCC; }

    inline bool hasFalseSucc() const { return !hasJump; }
  };

  std::pmr::vector<Region> _regions;

  using regionsit_t = decltype(_regions)::iterator;

  regionid_t  getRegionIndex(uint32_t pos) const;
  regionsit_t splitRegion(uint32_t pos);
};

enum class SimpleNodeKind { Basic, Sequence, Branch, Loop };

template <template <class, class> class Container>
struct SimpleNode {
  SimpleNodeKind kind;
  uint32_t       instrStart = 0;
  uint32_t       instrEnd   = 0;

  using NodeContainer = Container<SimpleNode, std::pmr::polymorphic_allocator<SimpleNode>>;
  NodeContainer children;     ///< children for Sequence or Loop body
  NodeContainer alternatives; ///< for Branch: alternatives are nodes (each alternative is a subtree)

  SimpleNode(auto pool): kind(SimpleNodeKind::Basic), children(pool), alternatives(pool) {}
};

using SimpleNode_t = SimpleNode<std::vector>;
SimpleNode_t transformStructuredCFG(std::pmr::memory_resource* allocPool, std::pmr::memory_resource* tempPool, RegionBuilder& regions);

void dump(std::ostream& os, SimpleNode_t const* node);
void dump(std::ostream& os, SimpleNode_t const* node, ir::InstCore const* instructions);

ir::cfg::ControlFlow transformCFG(std::pmr::memory_resource* allocPool, std::pmr::memory_resource* tempPool, RegionBuilder& regions);

} // namespace compiler::frontend::analysis