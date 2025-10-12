#pragma once

#include <cstdint>
#include <functional>
#include <memory_resource>
#include <span>
#include <vector>

namespace compiler::ir {
struct InstCore;
using regionid_t = uint32_t;

class RegionBuilder {
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

  /**
   * @brief Get the Region for index
   *
   * @param index
   * @return std::pair<uint32_t, uint32_t> start, end
   */
  std::pair<regionid_t, uint32_t> findRegion(uint32_t index) const;

  // std::pair<regionid_t, uint32_t> getRegion(regionid_t start) const;

  auto getNumRegions() const { return _regions.size(); }

  void dump(std::ostream& os, void* region) const;

  void for_each(auto cb) const {
    for (auto const& region: _regions) {
      cb(region.start, region.end, (void*)&region);
    }
  }

  private:
  struct Region {
    uint32_t start = 0;
    uint32_t end   = 0;

    uint32_t trueSucc  = NO_SUCC;
    uint32_t falseSucc = NO_SUCC;

    bool hasJump = false;

    static constexpr uint32_t NO_SUCC = UINT32_MAX;

    Region(uint32_t s, uint32_t e): start(s), end(e) {}

    // Region() = default;

    inline bool hasTrueSucc() const { return trueSucc != NO_SUCC; }

    inline bool hasFalseSucc() const { return falseSucc != NO_SUCC; }
  };

  std::vector<Region, std::pmr::polymorphic_allocator<Region>> _regions;

  using regionsit_t = decltype(_regions)::iterator;

  regionid_t getRegionIndex(uint32_t pos) const;

  regionsit_t splitRegion(uint32_t pos);
};
} // namespace compiler::ir