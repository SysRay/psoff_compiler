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
  auto const op   = (parser::eOpcode)inst.template get<VINTRP::Field::OP>();

  auto const vdst    = (eOperandKind)inst.template get<VINTRP::Field::VDST>();
  auto const src0    = (eOperandKind)inst.template get<VINTRP::Field::VSRC>();
  auto const channel = (uint8_t)inst.template get<VINTRP::Field::ATTRCHAN>();
  auto const attr    = (uint8_t)inst.template get<VINTRP::Field::ATTR>();

  if (src0 == eOperandKind::Literal) {
    *pCode += 1;
    builder.createInstruction(createLiteral(**pCode));
  }
  *pCode += 1;
  return true;
}
} // namespace compiler::frontend::translate