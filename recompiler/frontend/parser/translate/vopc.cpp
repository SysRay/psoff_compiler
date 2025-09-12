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
bool handleVopc(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode, bool extended) {
  using namespace parser;

  parser::eOpcode op;
  eOperandKind    vdst0;
  eOperandKind    src0, src1;

  if (extended) { // todo sdst or not
    auto inst = VOP3(getU64(*pCode));
    op        = (parser::eOpcode)inst.template get<VOP3::Field::OP>();

    vdst0 = (eOperandKind)inst.template get<VOP3::Field::VDST>();
    src0  = (eOperandKind)inst.template get<VOP3::Field::SRC0>();
    src1  = (eOperandKind)inst.template get<VOP3::Field::SRC1>();
    *pCode += 1;
  } else {
    auto inst = VOPC(**pCode);
    op        = (parser::eOpcode)inst.template get<VOPC::Field::OP>();
    src0      = (eOperandKind)inst.template get<VOPC::Field::SRC0>();
    src1      = (eOperandKind)inst.template get<VOPC::Field::VSRC1>();

    if (src0 == eOperandKind::Literal || src1 == eOperandKind::Literal) {
      *pCode += 1;
      builder.createInstruction(createLiteral(**pCode));
    }
  }

  *pCode += 1;
  return true;
}
} // namespace compiler::frontend::translate