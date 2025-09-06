#pragma once

#include "../parser.h"

namespace compiler::frontend {
struct ShaderInput;
}

namespace compiler::frontend::translate {
ir::InstCore handleSop1(ShaderInput const& ctx, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleSop2(ShaderInput const& ctx, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleSop2(ShaderInput const& ctx, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleSopc(ShaderInput const& ctx, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleSopk(ShaderInput const& ctx, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleSopp(ShaderInput const& ctx, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleSmrd(ShaderInput const& ctx, parser::pc_t pc, parser::code_p_t* pCode);

ir::InstCore handleVop1(ShaderInput const& ctx, parser::pc_t pc, parser::code_p_t* pCode, bool extended);
ir::InstCore handleVop2(ShaderInput const& ctx, parser::pc_t pc, parser::code_p_t* pCode, bool extended);
ir::InstCore handleVop3(ShaderInput const& ctx, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleVopc(ShaderInput const& ctx, parser::pc_t pc, parser::code_p_t* pCode, bool extended);
ir::InstCore handleVintrp(ShaderInput const& ctx, parser::pc_t pc, parser::code_p_t* pCode);

ir::InstCore handleExp(ShaderInput const& ctx, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleMubuf(ShaderInput const& ctx, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleMtbuf(ShaderInput const& ctx, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleMimg(ShaderInput const& ctx, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleDs(ShaderInput const& ctx, parser::pc_t pc, parser::code_p_t* pCode);
ir::InstCore handleCustom(ShaderInput const& ctx, parser::pc_t pc, parser::code_p_t* pCode);
} // namespace compiler::frontend::translate