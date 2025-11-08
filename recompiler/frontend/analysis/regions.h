#pragma once

#include "fixed_containers/fixed_vector.hpp"
#include "include/checkpoint_resource_fwd.h"
#include "ir/cfg/types.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory_resource>
#include <optional>
#include <span>
#include <vector>

namespace compiler::frontend::analysis {
using region_t = uint32_t;

struct regionid_t {
  region_t value         = std::numeric_limits<region_t>::max();
  constexpr regionid_t() = default;

  constexpr explicit regionid_t(uint32_t v): value(v) {}

  constexpr operator uint32_t() const { return value; }

  constexpr bool operator==(regionid_t const&) const = default;

  constexpr bool isValid() const { return value != UINT32_MAX; }
};

inline constexpr regionid_t NO_REGION = regionid_t {UINT32_MAX};

class RegionBuilder {

  public:
  RegionBuilder(region_t N, std::pmr::polymorphic_allocator<> allocator): _regions {allocator} {
    _regions.reserve(128);
    _regions.emplace_back(Region(0, N));
  }

  void addJump(region_t from, region_t to);
  void addReturn(region_t from);
  void addCondJump(region_t from, region_t to);

  template <typename V>
  requires std::invocable<V, region_t>
  void visitSuccessors(region_t from, V&& visitor) const {
    size_t const idx = getRegionIndex(from);
    const auto&  r   = _regions[idx];

    if (r.hasFalseSucc()) { // Fallthrough
      if (idx + 1 < _regions.size() && _regions[idx + 1].start == r.end) visitor(_regions[idx + 1].start);
    }
    if (r.hasTrueSucc()) visitor(r.trueSucc);
  }

  template <typename V>
  requires std::invocable<V, region_t>
  void visitPredecessors(region_t from, V&& visitor) const {
    for (uint32_t n = 0; n < _regions.size(); ++n) {
      const auto& r = _regions[n];
      if (r.hasFalseSucc()) {
        if (1 + n < _regions.size() && _regions[1 + n].start == from) visitor(r.start);
      }
      if (r.hasTrueSucc() && r.trueSucc == from) visitor(r.start);
    }
  }

  fixed_containers::FixedVector<regionid_t, 2> getSuccessorsIdx(regionid_t id) const;

  /**
   * @brief
   *
   * @param index
   * @return std::pair<uint32_t, uint32_t> start, end
   */
  std::pair<region_t, region_t> findRegion(region_t from) const;

  /**
   * @brief
   *
   * @param index
   * @return std::pair<uint32_t, uint32_t> start, end
   */
  std::pair<region_t, region_t> getRegion(regionid_t id) const;
  regionid_t                    getRegionIndex(region_t from) const;

  auto const& getRegions() const { return _regions; }

  auto getNumRegions() const { return _regions.size(); }

  void dump(std::ostream& os, void* region) const;

  inline void dumpAll(std::ostream& os) const {
    os << "Regions dump:\n";
    for (auto const& region: _regions) {
      dump(os, (void*)&region);
    }
  }

  void for_each(auto cb) const {
    for (auto const& region: _regions) {
      cb(region.start, region.end, (void*)&region);
    }
  }

  protected:
  struct Region {
    region_t start = 0;
    region_t end   = 0;

    region_t trueSucc = NO_SUCC;

    bool hasJump = false;

    static constexpr region_t NO_SUCC = UINT32_MAX;

    Region(region_t s, region_t e): start(s), end(e) {}

    // Region() = default;

    inline bool hasTrueSucc() const { return trueSucc != NO_SUCC; }

    inline bool hasFalseSucc() const { return !hasJump; }
  };

  std::pmr::vector<Region> _regions;

  using regionsit_t = decltype(_regions)::iterator;
  regionsit_t splitRegion(region_t from);
};

struct StartRegion {
  regionid_t id;
};

struct StopRegion {
  regionid_t id;
};

struct BasicRegion {
  regionid_t id;
  region_t   begin;
  region_t   end;
};

struct CondRegion {
  regionid_t id;
  regionid_t startTrue, startFalse; //< Subgraph start nodes per branch
  regionid_t mergeId;               ///< Subgraph branch merge node
};

struct LoopRegion {
  regionid_t id;
  regionid_t startId; ///< Subgraph start node
  regionid_t exitId;  ///< Subgraph Loop exit node
  regionid_t contId;  ///< Subgraph Loop continue node (break back to start )
};

using RegionNode = std::variant<StartRegion, StopRegion, BasicRegion, CondRegion, LoopRegion>;

/**
 * @brief Create graph from regions.
 *
 */
class RegionGraph {
  static constexpr regionid_t START_ID {0};
  static constexpr regionid_t STOP_ID {1};

  std::pmr::vector<RegionNode>                   nodes;
  std::pmr::vector<std::pmr::vector<regionid_t>> succ;
  std::pmr::vector<std::pmr::vector<regionid_t>> pred;

  public:
  RegionGraph(std::pmr::polymorphic_allocator<> allocator, const RegionBuilder& rb);

  size_t getNodeCount() const { return nodes.size(); }

  auto& accessSuccessors(regionid_t idx) { return succ[idx.value]; }

  auto& accessPredecessors(regionid_t idx) { return pred[idx.value]; }

  std::span<const regionid_t> getSuccessors(regionid_t idx) const { return succ[idx.value]; }

  std::span<const regionid_t> getPredecessors(regionid_t idx) const { return pred[idx.value]; }

  regionid_t getStartId() const { return START_ID; }

  regionid_t getStopId() const { return STOP_ID; }

  const RegionNode& getNode(regionid_t id) const { return nodes[id.value]; }

  RegionNode& getNode(regionid_t id) { return nodes[id.value]; }

  template <typename T>
  regionid_t createNode() {
    auto const id = regionid_t((uint32_t)nodes.size());

    auto& item = nodes.emplace_back(T());
    std::visit([id](auto& region) { region.id = id; }, item);

    succ.emplace_back();
    pred.emplace_back();

    return id;
  }
};

void structurizeRegions(util::checkpoint_resource& checkpoint_resource, analysis::RegionGraph& regionGraph);

void dump(std::ostream& os, const compiler::frontend::analysis::RegionGraph& g);
} // namespace compiler::frontend::analysis