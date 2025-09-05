#pragma once
#include <bit>
#include <cstdint>
#include <memory_resource>
#include <span>
#include <vector>

using Id                = uint32_t;
constexpr Id kInvalidId = ~Id(0);

// -------------------------
// Data-Oriented IRSoA: Organize by algorithm access patterns
// -------------------------
class IRSoA {
  public:
  static constexpr uint32_t kMaxOps = 4;

  private:
  // DATA LAYOUT: Organize by how algorithms access the data

  // Most common access pattern: iterate instructions, check opcode + operands
  // Pack these tightly for maximum cache hits during traversal
  struct InstCore {
    uint16_t opcode;
    uint16_t flags;
    uint32_t ops[kMaxOps]; // operands packed together
  };

  static_assert(sizeof(InstCore) == 20); // 3.2 per cache line

  // Less frequent access: def-use info, only touched during SSA construction/analysis
  struct InstrDefUse {
    uint32_t def_vreg;
    uint32_t orig_def;
    Id       next, prev; // intrusive list pointers
  };

  static_assert(sizeof(InstrDefUse) == 16); // 4 per cache line

  // Block structure: head/tail accessed together during block iteration
  struct BlockCore {
    Id       head, tail;
    uint32_t edge_start; // index into edges array
    uint16_t edge_count;
    uint16_t _pad;
  };

  static_assert(sizeof(BlockCore) == 16); // 4 per cache line

  // THE DATA TREE - organized by access frequency and patterns
  std::vector<InstCore>   instr_core_;   // Hot path data
  std::vector<InstrDefUse> instr_defuse_; // Cold analysis data
  std::vector<BlockCore>   blocks_;       // Block structure
  std::vector<Id>          edges_;        // Flat successor list

  Id instr_count_;
  Id block_count_;

  public:
  IRSoA(): instr_count_(0), block_count_(0) {}

  // CORE OPERATIONS: Functions that work on the data tree

  [[nodiscard]] Id instr_count() const noexcept { return instr_count_; }

  [[nodiscard]] Id block_count() const noexcept { return block_count_; }

  // Primary instruction access - single cache line for hot data
  [[nodiscard]] const InstCore& instr_core(Id id) const noexcept { return instr_core_[id]; }

  [[nodiscard]] InstCore& instr_core(Id id) noexcept { return instr_core_[id]; }

  // Def-use access - separate when doing SSA analysis
  [[nodiscard]] const InstrDefUse& instr_defuse(Id id) const noexcept { return instr_defuse_[id]; }

  [[nodiscard]] InstrDefUse& instr_defuse(Id id) noexcept { return instr_defuse_[id]; }

  // Block access
  [[nodiscard]] const BlockCore& block(Id id) const noexcept { return blocks_[id]; }

  [[nodiscard]] BlockCore& block(Id id) noexcept { return blocks_[id]; }

  [[nodiscard]] std::span<const Id> block_edges(Id block_id) const noexcept {
    const auto& b = blocks_[block_id];
    return {edges_.data() + b.edge_start, b.edge_count};
  }

  // ALGORITHM-ORIENTED FUNCTIONS: Work on data tree efficiently

  // Hot path: instruction iteration (most common operation)
  template <typename Func>
  void for_each_instr_core(Func&& func) const {
    for (Id i = 0; i < instr_count_; ++i) {
      func(i, instr_core_[i]);
    }
  }

  // Cache-friendly instruction filtering by opcode
  template <typename Func>
  void for_matching_opcodes(uint16_t target_opcode, Func&& func) const {
    // Vectorizable loop over packed opcode data
    for (Id i = 0; i < instr_count_; ++i) {
      if (instr_core_[i].opcode == target_opcode) {
        func(i, instr_core_[i]);
      }
    }
  }

  // Block traversal - sequential access pattern
  template <typename Func>
  void for_each_block(Func&& func) const {
    for (Id i = 0; i < block_count_; ++i) {
      func(i, blocks_[i]);
    }
  }

  // Instruction traversal within block - follows linked list
  template <typename Func>
  void for_instrs_in_block(Id block_id, Func&& func) {
    Id instr = blocks_[block_id].head;
    while (instr != kInvalidId) {
      func(instr, instr_core_[instr]);
      instr = instr_defuse_[instr].next;
    }
  }

  // Bulk operations on contiguous data
  void batch_set_flags(std::span<const Id> ids, uint16_t flag_mask) noexcept {
    for (Id id: ids) {
      instr_core_[id].flags |= flag_mask;
    }
  }

  void batch_clear_flags(std::span<const Id> ids, uint16_t flag_mask) noexcept {
    const uint16_t clear_mask = ~flag_mask;
    for (Id id: ids) {
      instr_core_[id].flags &= clear_mask;
    }
  }

  // SIMD-friendly operand scanning
  std::vector<Id> find_uses_of(uint32_t vreg) const {
    std::vector<Id> uses;
    uses.reserve(16); // typical case

    // Hot loop over packed operand data
    for (Id i = 0; i < instr_count_; ++i) {
      const auto& core = instr_core_[i];
      for (uint32_t j = 0; j < kMaxOps; ++j) {
        if (core.ops[j] == vreg) {
          uses.push_back(i);
          break; // found in this instruction
        }
      }
    }
    return uses;
  }

  // DATA TREE CONSTRUCTION

  void reserve(std::size_t instr_capacity, std::size_t block_capacity, std::size_t edge_capacity) {
    instr_core_.reserve(instr_capacity);
    instr_defuse_.reserve(instr_capacity);
    blocks_.reserve(block_capacity);
    edges_.reserve(edge_capacity);
  }

  [[nodiscard]] Id add_instruction(uint16_t opcode, uint16_t flags = 0) {
    Id id = instr_count_++;

    instr_core_.emplace_back();
    instr_defuse_.emplace_back();

    auto& core  = instr_core_[id];
    core.opcode = opcode;
    core.flags  = flags;
    for (uint32_t i = 0; i < kMaxOps; ++i) {
      core.ops[i] = kInvalidId;
    }

    auto& defuse    = instr_defuse_[id];
    defuse.def_vreg = kInvalidId;
    defuse.orig_def = kInvalidId;
    defuse.next     = kInvalidId;
    defuse.prev     = kInvalidId;

    return id;
  }

  [[nodiscard]] Id add_block() {
    Id id = block_count_++;

    blocks_.emplace_back();
    auto& block      = blocks_[id];
    block.head       = kInvalidId;
    block.tail       = kInvalidId;
    block.edge_start = static_cast<uint32_t>(edges_.size());
    block.edge_count = 0;

    return id;
  }

  void add_block_edge(Id block_id, Id successor) {
    auto& block = blocks_[block_id];

    // First edge for this block
    if (block.edge_count == 0) {
      block.edge_start = static_cast<uint32_t>(edges_.size());
    }

    edges_.push_back(successor);
    block.edge_count++;
  }

  // Link instruction into block's intrusive list
  void append_to_block(Id block_id, Id instr_id) {
    auto& block        = blocks_[block_id];
    auto& instr_defuse = instr_defuse_[instr_id];

    if (block.head == kInvalidId) {
      // First instruction in block
      block.head = block.tail = instr_id;
      instr_defuse.next = instr_defuse.prev = kInvalidId;
    } else {
      // Append to end
      instr_defuse_[block.tail].next = instr_id;
      instr_defuse.prev              = block.tail;
      instr_defuse.next              = kInvalidId;
      block.tail                     = instr_id;
    }
  }

  // INTROSPECTION
  [[nodiscard]] std::size_t memory_usage() const noexcept {
    return instr_core_.capacity() * sizeof(InstCore) + instr_defuse_.capacity() * sizeof(InstrDefUse) + blocks_.capacity() * sizeof(BlockCore) +
           edges_.capacity() * sizeof(Id);
  }

  // Debug: verify data layout assumptions
  static void verify_layout() {
    static_assert(alignof(InstCore) <= 8);
    static_assert(alignof(InstrDefUse) <= 8);
    static_assert(alignof(BlockCore) <= 8);
    static_assert(sizeof(InstCore) == 20);
    static_assert(sizeof(InstrDefUse) == 16);
    static_assert(sizeof(BlockCore) == 16);
  }
};