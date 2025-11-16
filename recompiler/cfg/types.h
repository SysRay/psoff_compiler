#pragma once

#include "ir/ir.h"

#include <assert.h>
#include <limits>
#include <memory_resource>
#include <variant>
#include <vector>

namespace compiler::cfg {

namespace blocks {

struct blockid_t {
  using underlying_t = uint32_t;

  static inline constexpr blockid_t NO_VALUE() { return blockid_t(std::numeric_limits<underlying_t>::max()); };

  underlying_t value = NO_VALUE().value;

  constexpr blockid_t() = default;

  constexpr explicit blockid_t(underlying_t v): value(v) {}

  constexpr operator underlying_t() const { return value; }

  constexpr bool operator==(blockid_t const&) const = default;

  constexpr bool isValid() const { return value != NO_VALUE().value; }
};

struct edge_t {
  blockid_t from;
  blockid_t to;

  constexpr operator std::pair<blockid_t, blockid_t>() const { return {from, to}; }

  constexpr bool operator==(const edge_t&) const = default;

  edge_t(blockid_t from, blockid_t to): from(from), to(to) {}

  edge_t(blockid_t::underlying_t from, blockid_t::underlying_t to): from(blockid_t(from)), to(blockid_t(to)) {}
};

enum class eBlockType { Block, RegionNode };

struct Base {
  blockid_t  id {};
  eBlockType type;

  Base(eBlockType t): type(t) {}
};

struct BlockNode: Base {
  uint32_t opbegin {};
  uint32_t opend {};

  BlockNode(std::pmr::polymorphic_allocator<> allocator): Base(eBlockType::Block) {}
};

struct regionid_t {
  using underlying_t = uint32_t;

  underlying_t value = std::numeric_limits<underlying_t>::max();

  constexpr regionid_t() = default;

  constexpr explicit regionid_t(underlying_t v): value(v) {}

  constexpr operator underlying_t() const { return value; }

  bool isValid() const { return value != std::numeric_limits<underlying_t>::max(); }
};

/* Region: a nested subgraph of blocks */
struct Region {
  regionid_t id {};
  blockid_t  entry {}; ///< entry block of the region
  blockid_t  exit {};  ///< (optional) structured exit

  std::pmr::vector<blockid_t>  blocks;     ///< blocks belonging to this region
  std::pmr::vector<regionid_t> subregions; ///< nested regions inside this region

  Region(std::pmr::polymorphic_allocator<> alloc): blocks(alloc), subregions(alloc) {}
};

/* RegionNode: a block that controls entering regions */
struct RegionNode: Base {
  std::pmr::vector<regionid_t> regions;                ///< subregions executed from this node
  std::pmr::vector<blockid_t>  successorsAfterRegions; ///< successors after region exit (merge point)

  RegionNode(std::pmr::polymorphic_allocator<> alloc): Base(eBlockType::RegionNode), regions(alloc), successorsAfterRegions(alloc) {}
};

template <typename T>
concept BlockConcept = std::derived_from<T, blocks::Base>;

} // namespace blocks
} // namespace compiler::cfg
