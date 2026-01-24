#pragma once

#include "arith/instructions.h"
#include "core/instructions.h"

namespace compiler::ir::dialect {

inline constexpr const ir::InstCore& getInfo(eDialect dialect, InstructionKind_t instr) {
  switch (dialect) {
    case eDialect::kCore: return core::getInfo((core::eInstKind)instr);
    case eDialect::kArith: return arith::getInfo((arith::eInstKind)instr);
  }
}

inline constexpr const std::string_view getInstrKindStr(eDialect dialect, InstructionKind_t instr) {
  switch (dialect) {
    case eDialect::kCore: return core::getInstrKindStr((core::eInstKind)instr);
    case eDialect::kArith: return arith::getInstrKindStr((arith::eInstKind)instr);
  }
}

} // namespace compiler::ir::dialect