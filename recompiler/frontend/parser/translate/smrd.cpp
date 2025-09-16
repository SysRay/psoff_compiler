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
bool handleSmrd(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  auto       inst = SMRD(**pCode);
  auto const op   = (parser::eOpcode)inst.template get<SMRD::Field::OP>();

  auto const sdst      = (eOperandKind)inst.template get<SMRD::Field::SDST>();
  auto const sBase     = (eOperandKind)inst.template get<SMRD::Field::SBASE>();
  auto const offsetImm = inst.template get<SMRD::Field::OFFSET>();
  auto const sOffset   = (eOperandKind)offsetImm; // either imm or op
  auto const isImm     = (bool)inst.template get<SMRD::Field::IMM>();

  if (!isImm && sOffset == eOperandKind::Literal) {
    *pCode += 1;
    builder.createInstruction(create::literalOp(**pCode));
  }

  *pCode += 1;
  return {};
}
} // namespace compiler::frontend::translate