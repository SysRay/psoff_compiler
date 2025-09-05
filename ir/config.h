#pragma once
#include <cstdint>

namespace compiler {
using OperandKind_t = uint16_t;

using OperandIndex_t = uint16_t;
using OperandType_t  = uint16_t;
using OperandFlags_t = uint8_t;

using InstructionId_t    = uint32_t;
using InstructionKind_t  = uint16_t;
using InstructionFlags_t = uint8_t;
using InstructionGroup_t = uint8_t;

} // namespace compiler

namespace compiler::config {
static constexpr uint32_t kMaxOps = 5;

constexpr uint32_t kOperandKindBits  = 9;
constexpr uint32_t kOperandFlagsBits = 7;
} // namespace compiler::config