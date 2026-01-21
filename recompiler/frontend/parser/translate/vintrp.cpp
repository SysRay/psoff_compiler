#include "../debug_strings.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "frontend/ir_types.h"
#include "ir/dialects/arith/builder.h"
#include "ir/dialects/core/builder.h"
#include "translate.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
InstructionKind_t handleVintrp(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = VINTRP(**pCode);
  auto const op   = (parser::eOpcode)(OPcodeStart_VINTRP + inst.template get<VINTRP::Field::OP>());

  auto const vdst    = createDst(eOperandKind((eOperandKind_t)inst.template get<VINTRP::Field::VDST>()));
  auto       src0    = createSrc(eOperandKind((eOperandKind_t)inst.template get<VINTRP::Field::VSRC>()));
  auto const channel = (uint8_t)inst.template get<VINTRP::Field::ATTRCHAN>();
  auto const attr    = (uint8_t)inst.template get<VINTRP::Field::ATTR>();

  using namespace ir::dialect;
  if (eOperandKind(src0.kind).isLiteral()) {
    *pCode += 1;
    src0 = createSrc(ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_u64 = **pCode}, ir::OperandType::i32()));
  }
  *pCode += 1;
  return conv(op);
}
} // namespace compiler::frontend::translate