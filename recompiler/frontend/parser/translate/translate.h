#pragma once

#include "../parser.h"

namespace compiler {
class Builder;
}

namespace compiler::frontend::translate {
ir::InstCore handleSop1(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleSop2(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleSop2(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleSopc(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleSopk(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleSopp(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleSmrd(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);

ir::InstCore handleVop1(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode, bool extended);
ir::InstCore handleVop2(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode, bool extended);
ir::InstCore handleVop3(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleVopc(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode, bool extended);
ir::InstCore handleVintrp(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);

ir::InstCore handleExp(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleMubuf(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleMtbuf(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleMimg(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleDs(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleCustom(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode);
} // namespace compiler::frontend::translate