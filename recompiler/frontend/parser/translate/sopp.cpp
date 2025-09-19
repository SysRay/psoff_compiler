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
bool handleSopp(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = SOPP(**pCode);
  auto const op   = (parser::eOpcode)inst.template get<SOPP::Field::OP>();

  auto const offset = (int16_t)inst.template get<SOPP::Field::SIMM16>();

  *pCode += 1;

  switch (op) {
    case eOpcode::S_NOP: break; // ignore
    case eOpcode::S_ENDPGM: {
      builder.createInstruction(ir::getInfo(ir::eInstKind::ReturnOp));
    } break;
    case eOpcode::S_BRANCH: {
      builder.createInstruction(create::jumpAbsOp(4 + pc + 4 * (int64_t)offset));
    } break;
    case eOpcode::S_CBRANCH_SCC0: {
      builder.createInstruction(create::constantOp(eOperandKind::CustomTemp0Lo, 4 + pc + 4 * (int64_t)offset, ir::OperandType::i64()));
      builder.createInstruction(create::cjumpAbsOp(eOperandKind::Scc, true, eOperandKind::CustomTemp0Lo));
    } break;
    case eOpcode::S_CBRANCH_SCC1: {
      builder.createInstruction(create::constantOp(eOperandKind::CustomTemp0Lo, 4 + pc + 4 * (int64_t)offset, ir::OperandType::i64()));
      builder.createInstruction(create::cjumpAbsOp(eOperandKind::Scc, false, eOperandKind::CustomTemp0Lo));
    } break;
    case eOpcode::S_CBRANCH_VCCZ: {
      builder.createInstruction(create::constantOp(eOperandKind::CustomTemp0Lo, 4 + pc + 4 * (int64_t)offset, ir::OperandType::i64()));
      builder.createInstruction(create::cjumpAbsOp(eOperandKind::VccZ, false, eOperandKind::CustomTemp0Lo));
    } break;
    case eOpcode::S_CBRANCH_VCCNZ: {
      builder.createInstruction(create::constantOp(eOperandKind::CustomTemp0Lo, 4 + pc + 4 * (int64_t)offset, ir::OperandType::i64()));
      builder.createInstruction(create::cjumpAbsOp(eOperandKind::VccZ, true, eOperandKind::CustomTemp0Lo));
    } break;
    case eOpcode::S_CBRANCH_EXECZ: {
      builder.createInstruction(create::constantOp(eOperandKind::CustomTemp0Lo, 4 + pc + 4 * (int64_t)offset, ir::OperandType::i64()));
      builder.createInstruction(create::cjumpAbsOp(eOperandKind::ExecZ, false, eOperandKind::CustomTemp0Lo));
    } break;
    case eOpcode::S_CBRANCH_EXECNZ: {
      builder.createInstruction(create::constantOp(eOperandKind::CustomTemp0Lo, 4 + pc + 4 * (int64_t)offset, ir::OperandType::i64()));
      builder.createInstruction(create::cjumpAbsOp(eOperandKind::ExecZ, true, eOperandKind::CustomTemp0Lo));
    } break;
    case eOpcode::S_BARRIER: {
      builder.createInstruction(ir::getInfo(ir::eInstKind::BarrierOp));
    } break;
    // case eOpcode::S_SETKILL: {} break; // Does not exist
    case eOpcode::S_WAITCNT:
      break; // ignore
    // case eOpcode::S_SETHALT: {} break; // Does not exist
    case eOpcode::S_SLEEP: break; // ignore
    case eOpcode::S_SETPRIO: {
    } break;
    case eOpcode::S_SENDMSG: {
    } break;
    // case eOpcode::S_SENDMSGHALT: {} break; // Does not exist
    // case eOpcode::S_TRAP: {} break; // Does not exist
    case eOpcode::S_ICACHE_INV: {
    } break;
    case eOpcode::S_INCPERFLEVEL: {
    } break;
    case eOpcode::S_DECPERFLEVEL: {
    } break;
    case eOpcode::S_TTRACEDATA: {
    } break;
    // case eOpcode::S_CBRANCH_CDBGSYS: {} break; // Does not exist
    // case eOpcode::S_CBRANCH_CDBGUSER: {} break; // Does not exist
    // case eOpcode::S_CBRANCH_CDBGSYS_OR_USER: {} break; // Does not exist
    // case eOpcode::S_CBRANCH_CDBGSYS_AND_USER: {} break;// Does not exist
    default: throw std::runtime_error(std::format("missing inst {}", debug::getDebug(op))); break;
  }
  return true;
}
} // namespace compiler::frontend::translate