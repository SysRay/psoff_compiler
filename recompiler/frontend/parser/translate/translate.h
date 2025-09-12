#pragma once

#include "../parser.h"

namespace compiler {
class Builder;
}

namespace compiler::frontend::translate {
bool handleSop1(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
bool handleSop2(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
bool handleSop2(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
bool handleSopc(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
bool handleSopk(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
bool handleSopp(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
bool handleSmrd(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);

bool handleVop1(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode, bool extended);
bool handleVop2(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode, bool extended);
bool handleVop3(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
bool handleVopc(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode, bool extended);
bool handleVintrp(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);

bool handleExp(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
bool handleMubuf(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
bool handleMtbuf(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
bool handleMimg(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
bool handleDs(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
bool handleCustom(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
} // namespace compiler::frontend::translate