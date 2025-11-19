#pragma once

#include "frontend/parser.h"

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
} // namespace compiler::frontend::translate