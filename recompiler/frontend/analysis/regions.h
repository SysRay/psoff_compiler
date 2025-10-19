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

namespace compiler::frontend::analysis {
using regionid_t = uint32_t;

// todo handle dead regions
class RegionBuilder {
  public:
  static constexpr regionid_t NO_REGION = -1;

  RegionBuilder(uint32_t N, std::pmr::polymorphic_allocator<> allocator): _regions {allocator} {
    _regions.reserve(128);
    _regions.emplace_back(Region(0, N));
  }

  void addJump(regionid_t from, regionid_t to);
  void addReturn(regionid_t from);
  void addCondJump(regionid_t from, regionid_t to);

  template <typename V>
  requires std::invocable<V, regionid_t>
  void visitSuccessors(regionid_t start, V&& visitor) const {
    size_t const idx = getRegionIndex(start);
    const auto&  r   = _regions[idx];

    if (r.hasTrueSucc()) visitor(r.trueSucc);
    if (r.hasFalseSucc()) { // Fallthrough
      if (idx + 1 < _regions.size() && _regions[idx + 1].start == r.end) visitor(_regions[idx + 1].start);
    }
  }

  template <typename V>
  requires std::invocable<V, regionid_t>
  void visitPredecessors(regionid_t start, V&& visitor) const {
    for (const auto& r: _regions) {
      visitSuccessors(r.start, [&visitor, start](regionid_t succ) {
        if (succ == start) visitor(succ);
      });
    }
  }

  // todo change to visitor ?
  fixed_containers::FixedVector<int32_t, 2> getSuccessorsIdx(uint32_t region_idx) const;

  /**
   * @brief Get the Region given a index
   *
   * @param index
   * @return std::pair<uint32_t, uint32_t> start, end
   */
  std::pair<regionid_t, uint32_t> findRegion(uint32_t index) const;
  std::pair<uint32_t, uint32_t>   getRegion(uint32_t index) const;

  auto const& getRegions() const { return _regions; }

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
} // namespace compiler::frontend::analysis