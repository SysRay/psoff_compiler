#include "../../frontend.h"
#include "../debug_strings.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "instruction_builder.h"
#include "translate.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
bool handleVintrp(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = VINTRP(**pCode);
  auto const op   = (parser::eOpcode)(OPcodeStart_VINTRP + inst.template get<VINTRP::Field::OP>());

  auto const vdst    = OpDst(eOperandKind::create((OperandKind_t)inst.template get<VINTRP::Field::VDST>()));
  auto const src0    = OpSrc(eOperandKind::create((OperandKind_t)inst.template get<VINTRP::Field::VSRC>()));
  auto const channel = (uint8_t)inst.template get<VINTRP::Field::ATTRCHAN>();
  auto const attr    = (uint8_t)inst.template get<VINTRP::Field::ATTR>();

  if (src0.kind.isLiteral()) {
    *pCode += 1;
    builder.createInstruction(create::literalOp(**pCode));
  }
  *pCode += 1;
  return true;
}
} // namespace compiler::frontend::translate