#pragma once
#include <cstdint>
#include <memory_resource>
#include <string_view>
#include <vector>

namespace compiler::ir {

class RegionBuilder {
  public:
  RegionBuilder(uint32_t N, auto& pool): regions {&pool} {
    regions.reserve(128);
    regions.emplace_back(0, N);
  }

  void addJump(uint32_t from, uint32_t to);
  void addReturn(uint32_t from);
  void addCondJump(uint32_t from, uint32_t to);

  std::vector<uint32_t> getSuccessors(uint32_t start) const;
  std::vector<uint32_t> getPredecessors(uint32_t start) const;

  void dump(std::ostream& os) const;

  private:
  struct Region {
    uint32_t start = 0;
    uint32_t end   = 0;

    uint32_t trueSucc  = NO_SUCC;
    uint32_t falseSucc = NO_SUCC;

    bool hasJump = false;

    static constexpr uint32_t NO_SUCC = UINT32_MAX;

    Region(uint32_t s, uint32_t e): start(s), end(e) {}

    Region() = default;

    inline bool hasTrueSucc() const { return trueSucc != NO_SUCC; }

    inline bool hasFalseSucc() const { return falseSucc != NO_SUCC; }
  };

  std::vector<Region, std::pmr::polymorphic_allocator<Region>> regions;

  size_t   getRegionIndex(uint32_t pos) const;
  uint32_t splitRegionAt(uint32_t pos);

  std::pair<uint32_t, uint32_t> splitRegionAround(uint32_t pos);
};
} // namespace compiler::ir