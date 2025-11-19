#include "../debug_strings.h"
#include "../instruction_builder.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "frontend/ir_types.h"
#include "translate.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
InstructionKind_t handleVintrp(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = VINTRP(**pCode);
  auto const op   = (parser::eOpcode)(OPcodeStart_VINTRP + inst.template get<VINTRP::Field::OP>());

  auto const vdst    = OpDst(eOperandKind((eOperandKind_t)inst.template get<VINTRP::Field::VDST>()));
  auto       src0    = OpSrc(eOperandKind((eOperandKind_t)inst.template get<VINTRP::Field::VSRC>()));
  auto const channel = (uint8_t)inst.template get<VINTRP::Field::ATTRCHAN>();
  auto const attr    = (uint8_t)inst.template get<VINTRP::Field::ATTR>();

  create::IRBuilder ir(ctx.instructions);
  create::IRBuilder vir(ctx.instructions, true);
  if (src0.kind.isLiteral()) {
    *pCode += 1;
    src0 = OpSrc(ir.literalOp(**pCode));
  }
  *pCode += 1;
  return conv(op);
}
} // namespace compiler::frontend::translate