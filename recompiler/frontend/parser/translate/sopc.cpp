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
bool handleSopc(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = SOPC(**pCode);
  auto const op   = (parser::eOpcode)inst.template get<SOPC::Field::OP>();

  auto const src0 = (eOperandKind)inst.template get<SOPC::Field::SSRC0>();
  auto const src1 = (eOperandKind)inst.template get<SOPC::Field::SSRC1>();

  if (src0 == eOperandKind::Literal || src1 == eOperandKind::Literal) {
    *pCode += 1;
    builder.createInstruction(createLiteral(**pCode));
  }

  *pCode += 1;
  return true;
}
} // namespace compiler::frontend::translate