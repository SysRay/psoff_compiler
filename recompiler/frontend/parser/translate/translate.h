#pragma once

#include "../parser.h"

namespace compiler {
class Builder;
}

namespace compiler::frontend::translate {
InstructionKind_t handleSop1(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleSop2(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleSop2(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleSopc(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleSopk(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleSopp(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleSmrd(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);

InstructionKind_t handleVop1(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode, bool extended);
InstructionKind_t handleVop2(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode, bool extended);
InstructionKind_t handleVop3(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleVopc(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode, bool extended);
InstructionKind_t handleVintrp(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);

InstructionKind_t handleExp(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleMubuf(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleMtbuf(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleMimg(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
InstructionKind_t handleDs(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
} // namespace compiler::frontend::translate