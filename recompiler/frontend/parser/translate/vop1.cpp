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
bool handleVop1(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode, bool extended) {
  using namespace parser;

  parser::eOpcode op;
  eOperandKind    vdst0;
  eOperandKind    src0;

  if (extended) { // todo sdst or not
    auto inst = VOP3(getU64(*pCode));
    op        = (parser::eOpcode)inst.template get<VOP3::Field::OP>();

    vdst0 = (eOperandKind)inst.template get<VOP3::Field::VDST>();
    src0  = (eOperandKind)inst.template get<VOP3::Field::SRC0>();
    *pCode += 1;
  } else {
    auto inst = VOP1(**pCode);
    op        = (parser::eOpcode)inst.template get<VOP1::Field::OP>();

    vdst0 = (eOperandKind)inst.template get<VOP1::Field::VDST>();
    src0  = (eOperandKind)inst.template get<VOP1::Field::SRC0>();
    if (src0 == eOperandKind::Literal) {
      *pCode += 1;
      builder.createInstruction(create::literalOp(**pCode));
    }
  }

  *pCode += 1;
  return true;
}
} // namespace compiler::frontend::translate