#pragma once

#include "fixed_containers/fixed_vector.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <memory_resource>
#include <optional>
#include <span>
#include <vector>

namespace compiler {
class Builder;
}

namespace compiler::ir {
struct InstCore;
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

  RegionBuilder(uint32_t N, auto& pool): _regions {&pool} {
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

  std::vector<Region, std::pmr::polymorphic_allocator<Region>> _regions;

  using regionsit_t = decltype(_regions)::iterator;

  regionid_t  getRegionIndex(uint32_t pos) const;
  regionsit_t splitRegion(uint32_t pos);
};

struct SimpleNode {
  enum class Kind { Basic, Sequence, Branch, Loop };
  Kind       kind;
  regionid_t rid        = RegionBuilder::NO_REGION; // meaningful for Basic
  uint32_t   instrStart = 0;
  uint32_t   instrEnd   = 0;

  std::optional<char> auxPredicate;

  // children for Sequence or Loop body
  std::pmr::vector<SimpleNode> children;

  // for Branch: alternatives are nodes (each alternative is a subtree)
  std::pmr::vector<SimpleNode> alternatives;

  SimpleNode(std::pmr::monotonic_buffer_resource& pool): kind(Kind::Basic), children(&pool), alternatives(&pool) {}
};

SimpleNode transformStructuredCFG(std::pmr::monotonic_buffer_resource& alloc_pool, std::pmr::monotonic_buffer_resource& temp_pool, RegionBuilder& regions);

void dump(std::ostream& os, const SimpleNode* node);

} // namespace compiler::ir