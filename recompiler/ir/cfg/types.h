#pragma once

#include "../ir.h"

#include <assert.h>
#include <memory_resource>
#include <variant>
#include <vector>

namespace compiler::ir::cfg {

namespace blocks {
struct blockid_t {
  using underlying_t = uint32_t;

  static inline constexpr blockid_t NO_VALUE() { return blockid_t(std::numeric_limits<blockid_t::underlying_t>::max()); };

  underlying_t value = NO_VALUE();

  constexpr blockid_t() = default;

  constexpr explicit blockid_t(underlying_t v): value(v) {}

  constexpr operator underlying_t() const { return value; }

  constexpr bool operator==(blockid_t const&) const = default;

  constexpr bool isValid() const { return value != NO_VALUE().value; }
};

struct block_edge_t {
  blockid_t from;
  blockid_t to;

  constexpr operator std::pair<blockid_t, blockid_t>() const { return {from, to}; }

  constexpr bool operator==(const block_edge_t&) const = default;

  block_edge_t(blockid_t from, blockid_t to): from(from), to(to) {}
};

enum class eBlockType { Start, Stop, Basic, Loop, Cond };

struct Base {
  blockid_t  id = {};
  eBlockType type;

  Base(eBlockType type): type(type) {}
};

struct StartBlock: public Base {
  blockid_t parent = {};

  StartBlock(std::pmr::polymorphic_allocator<> allocator): Base(eBlockType::Start) {}
};

struct StopBlock: public Base {
  blockid_t parent = {};

  StopBlock(std::pmr::polymorphic_allocator<> allocator): Base(eBlockType::Stop) {}
};

struct BasicBlock: public Base {
  uint32_t opbegin = {};
  uint32_t opend   = {};

  BasicBlock(std::pmr::polymorphic_allocator<> allocator): Base(eBlockType::Basic) {}
};

struct CondBlock: public Base {
  blockid_t    mergeId   = {}; ///< Subgraph branch merge node
  ir::InstCore predicate = {};

  std::pmr::vector<blockid_t> branches; //< Subgraph start nodes per branch

  CondBlock(std::pmr::polymorphic_allocator<> allocator, uint8_t numBranches = 2): Base(eBlockType::Cond), branches(numBranches, allocator) {}
};

struct LoopBlock: public Base {
  blockid_t id       = {};
  blockid_t headerId = {}; ///< Subgraph start node
  blockid_t exitId   = {}; ///< Subgraph Loop exit node
  blockid_t contId   = {}; ///< Subgraph Loop continue node (break back to start )

  LoopBlock(std::pmr::polymorphic_allocator<> allocator): Base(eBlockType::Loop) {}
};

template <typename T>
concept BlockConcept = std::derived_from<T, blocks::Base>;
} // namespace blocks

} // namespace compiler::ir::cfg