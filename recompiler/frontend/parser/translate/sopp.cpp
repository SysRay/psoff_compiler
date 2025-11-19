#include "../debug_strings.h"
#include "../instruction_builder.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "frontend/ir_types.h"
#include "translate.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
InstructionKind_t handleSopp(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = SOPP(**pCode);
  auto const op   = (parser::eOpcode)(OPcodeStart_SOPP + inst.template get<SOPP::Field::OP>());

  auto const offset = (int16_t)inst.template get<SOPP::Field::SIMM16>();

  create::IRBuilder ir(ctx.instructions);
  *pCode += 1;

  switch (op) {
    case eOpcode::S_NOP: break; // ignore
    case eOpcode::S_ENDPGM: {
      ir.returnOp();
    } break;
    case eOpcode::S_BRANCH: {
      auto const target = OpSrc(ctx.instructions.createConstant(ir::ConstantValue {.value_u64 = 4 + pc + 4 * (int64_t)offset}));
      ir.jumpAbsOp(target);
    } break;
    case eOpcode::S_CBRANCH_SCC0: {
      auto const target = OpSrc(ctx.instructions.createConstant(ir::ConstantValue {.value_u64 = 4 + pc + 4 * (int64_t)offset}));
      ir.cjumpAbsOp(OpSrc(eOperandKind::SCC()), true, target);
    } break;
    case eOpcode::S_CBRANCH_SCC1: {
      auto const target = OpSrc(ctx.instructions.createConstant(ir::ConstantValue {.value_u64 = 4 + pc + 4 * (int64_t)offset}));
      ir.cjumpAbsOp(OpSrc(eOperandKind::SCC()), false, target);
    } break;
    case eOpcode::S_CBRANCH_VCCZ: {
      auto const target = OpSrc(ctx.instructions.createConstant(ir::ConstantValue {.value_u64 = 4 + pc + 4 * (int64_t)offset}));
      ir.cjumpAbsOp(OpSrc(eOperandKind::VCC(), true, false), false, target);
    } break;
    case eOpcode::S_CBRANCH_VCCNZ: {
      auto const target = OpSrc(ctx.instructions.createConstant(ir::ConstantValue {.value_u64 = 4 + pc + 4 * (int64_t)offset}));
      ir.cjumpAbsOp(OpSrc(eOperandKind::VCC()), true, target);
    } break;
    case eOpcode::S_CBRANCH_EXECZ: {
      auto const target = OpSrc(ctx.instructions.createConstant(ir::ConstantValue {.value_u64 = 4 + pc + 4 * (int64_t)offset}));
      ir.cjumpAbsOp(OpSrc(eOperandKind::EXEC(), true, false), false, target);
    } break;
    case eOpcode::S_CBRANCH_EXECNZ: {
      auto const target = OpSrc(ctx.instructions.createConstant(ir::ConstantValue {.value_u64 = 4 + pc + 4 * (int64_t)offset}));
      ir.cjumpAbsOp(OpSrc(eOperandKind::EXEC()), true, target);
    } break;
    case eOpcode::S_BARRIER: {
      ir.barrierOp();
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
  return conv(op);
}
} // namespace compiler::frontend::translate