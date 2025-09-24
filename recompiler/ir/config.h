#pragma once
#include <stdint.h>

namespace compiler {
using OperandKind_t = uint16_t;

using OperandIndex_t = uint16_t;
using OperandType_t  = uint16_t;
using OperandFlags_t = uint8_t;

using InstructionId_t       = uint32_t;
using InstructionKind_t     = uint16_t;
using InstructionUserData_t = uint16_t;
using InstructionFlags_t    = uint8_t;
using InstructionGroup_t    = uint8_t;

} // namespace compiler

namespace compiler::config {
static constexpr uint32_t kMaxOps    = 5;
static constexpr uint32_t kMaxSrcOps = 3;
static constexpr uint32_t kMaxDstOps = 2;
} // namespace compiler::config