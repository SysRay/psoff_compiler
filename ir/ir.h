#pragma once

#include "types.h"

namespace compiler::ir {

struct Operand {
  struct {
    OperandKind_t  kind: config::kOperandKindBits   = 0;
    OperandFlags_t flags: config::kOperandFlagsBits = 0;
  };

  OperandType type = OperandType::i32();
};

struct InstrCore {
  InstructionKind_t  kind;
  InstructionFlags_t flags;

  struct {
    uint8_t numDst : 4;
    uint8_t numSrc : 4;
  };

  Operand operands[config::kMaxOps];

  // // Flags
  constexpr inline bool hasSideEffects() {
    return (flags & kHasSideEffects) != 0;
  }

  constexpr inline bool writesExec() {
    return (flags & kWritesEXEC) != 0;
  }

  constexpr inline bool isVirtual() {
    return (flags & kVirtual) != 0;
  }

  constexpr bool isBarrier() {
    return (flags & kBarrier) != 0;
  }
};

static_assert(sizeof(InstrCore) <= 64); ///< 1 cache line
static_assert(config::kMaxOps <= 15);   ///< 4 bits
} // namespace compiler::ir