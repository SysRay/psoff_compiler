#pragma once

#include "frontend/ir_types.h"
#include "frontend/parser.h"
#include "ir/dialects/types.h"

namespace compiler::frontend::translate {
InstructionKind_t handleSop1(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleSop2(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleSop2(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleSopc(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleSopk(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleSopp(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleSmrd(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode);

InstructionKind_t handleVop1(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode, bool extended);
InstructionKind_t handleVop2(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode, bool extended);
InstructionKind_t handleVop3(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleVopc(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode, bool extended);
InstructionKind_t handleVintrp(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode);

InstructionKind_t handleExp(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleMubuf(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleMtbuf(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleMimg(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleDs(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode);

inline compiler::ir::dialect::OpSrc createSrc(eOperandKind kind = eOperandKind::Unset()) {
  return compiler::ir::dialect::OpSrc(getOperandKind(kind), 0);
}

inline compiler::ir::dialect::OpDst createDst(eOperandKind kind = eOperandKind::Unset()) {
  return compiler::ir::dialect::OpDst(getOperandKind(kind), 0);
}

inline compiler::ir::dialect::OpSrc createSrc(SsaId_t ssa) {
  return compiler::ir::dialect::OpSrc(ssa, 0);
}
} // namespace compiler::frontend::translate