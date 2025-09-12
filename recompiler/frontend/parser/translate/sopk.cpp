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
bool handleSopk(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = SOPK(**pCode);
  auto const op   = (eOpcode)inst.template get<SOPK::Field::OP>();

  auto const sdst  = (eOperandKind)inst.template get<SOPK::Field::SDST>();
  auto const imm16 = (int16_t)inst.template get<SOPK::Field::SIMM16>();

  *pCode += 1;

  switch (op) {
    case eOpcode::S_MOVK_I32: {

    } break;
    case eOpcode::S_CMOVK_I32: {

    } break;
    case eOpcode::S_CMPK_EQ_I32: {

    } break;
    case eOpcode::S_CMPK_LG_I32: {

    } break;
    case eOpcode::S_CMPK_GT_I32: {

    } break;
    case eOpcode::S_CMPK_GE_I32: {

    } break;
    case eOpcode::S_CMPK_LT_I32: {

    } break;
    case eOpcode::S_CMPK_LE_I32: {

    } break;
    case eOpcode::S_CMPK_EQ_U32: {

    } break;
    case eOpcode::S_CMPK_LG_U32: {

    } break;
    case eOpcode::S_CMPK_GT_U32: {

    } break;
    case eOpcode::S_CMPK_GE_U32: {

    } break;
    case eOpcode::S_CMPK_LT_U32: {

    } break;
    case eOpcode::S_CMPK_LE_U32: {

    } break;
    case eOpcode::S_ADDK_I32: {

    } break;
    case eOpcode::S_MULK_I32: {

    } break;
    case eOpcode::S_CBRANCH_I_FORK: { // handled in branch
    } break;
    case eOpcode::S_GETREG_B32: {
    } break;
    case eOpcode::S_SETREG_B32: {
    } break;
    case eOpcode::S_GETREG_REGRD_B32: {
    } break;
    case eOpcode::S_SETREG_IMM32_B32: {
      builder.createInstruction(createLiteral(**pCode));
      *pCode += 1;
    } break;
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }
  return true;
}
} // namespace compiler::frontend::translate