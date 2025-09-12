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
bool handleVop3(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  // todo sdst or not
  auto       inst = VOP3(getU64(*pCode));
  auto const op   = (parser::eOpcode)inst.template get<VOP3::Field::OP>();

  auto const vdst = (eOperandKind)inst.template get<VOP3::Field::VDST>();
  auto const src0 = (eOperandKind)inst.template get<VOP3::Field::SRC0>();
  auto const src1 = (eOperandKind)inst.template get<VOP3::Field::SRC1>();
  auto const src2 = (eOperandKind)inst.template get<VOP3::Field::SRC2>();

  *pCode += 2;
  return true;
}
} // namespace compiler::frontend::translate