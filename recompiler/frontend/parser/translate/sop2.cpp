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
bool handleSop2(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = SOP2(**pCode);
  auto const op   = (parser::eOpcode)inst.template get<SOP2::Field::OP>();

  auto const sdst = (eOperandKind)inst.template get<SOP2::Field::SDST>();
  auto const src0 = (eOperandKind)inst.template get<SOP2::Field::SSRC0>();
  auto const src1 = (eOperandKind)inst.template get<SOP2::Field::SSRC1>();

  if (src0 == eOperandKind::Literal || src1 == eOperandKind::Literal) {
    *pCode += 1;
    builder.createInstruction(create::literalOp(**pCode));
  }

  *pCode += 1;

  switch (op) {
    case eOpcode::S_ADD_U32: {

    } break;
    case eOpcode::S_SUB_U32: {

    } break;
    case eOpcode::S_ADD_I32: {

    } break;
    case eOpcode::S_SUB_I32: {

    } break;
    case eOpcode::S_ADDC_U32: {

    } break;
    case eOpcode::S_SUBB_U32: {

    } break;
    case eOpcode::S_MIN_I32: {

    } break;
    case eOpcode::S_MIN_U32: {

    } break;
    case eOpcode::S_MAX_I32: {

    } break;
    case eOpcode::S_MAX_U32: {

    } break;
    case eOpcode::S_CSELECT_B32: {

    } break;
    case eOpcode::S_CSELECT_B64: {

    } break;
    case eOpcode::S_AND_B32: {

    } break;
    case eOpcode::S_AND_B64: {

    } break;
    case eOpcode::S_OR_B32: {

    } break;
    case eOpcode::S_OR_B64: {

    } break;
    case eOpcode::S_XOR_B32: {

    } break;
    case eOpcode::S_XOR_B64: {

    } break;
    case eOpcode::S_ANDN2_B32: {

    } break;
    case eOpcode::S_ANDN2_B64: {

    } break;
    case eOpcode::S_ORN2_B32: {

    } break;
    case eOpcode::S_ORN2_B64: {

    } break;
    case eOpcode::S_NAND_B32: {

    } break;
    case eOpcode::S_NAND_B64: {

    } break;
    case eOpcode::S_NOR_B32: {

    } break;
    case eOpcode::S_NOR_B64: {

    } break;
    case eOpcode::S_XNOR_B32: {

    } break;
    case eOpcode::S_XNOR_B64: {

    } break;
    case eOpcode::S_LSHL_B32: {

    } break;
    case eOpcode::S_LSHL_B64: {

    } break;
    case eOpcode::S_LSHR_B32: {

    } break;
    case eOpcode::S_LSHR_B64: {

    } break;
    case eOpcode::S_ASHR_I32: {

    } break;
    case eOpcode::S_ASHR_I64: {

    } break;
    case eOpcode::S_BFM_B32: {

    } break;
    case eOpcode::S_BFM_B64: {

    } break;
    case eOpcode::S_MUL_I32: {

    } break;
    case eOpcode::S_BFE_U32: {

    } break;
    case eOpcode::S_BFE_I32: {
    } break;
    case eOpcode::S_BFE_U64: {
    } break;
    case eOpcode::S_BFE_I64: {
    } break;
    case eOpcode::S_CBRANCH_G_FORK: {
    } break;
    case eOpcode::S_ABSDIFF_I32: {
    } break;
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }

  return true;
}
} // namespace compiler::frontend::translate