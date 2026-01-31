#include "../debug_strings.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "frontend/ir_types.h"
#include "ir/dialects/arith/builder.h"
#include "ir/dialects/core/builder.h"
#include "translate.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
InstructionKind_t handleSopp(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = SOPP(**pCode);
  auto const op   = (parser::eOpcode)(OPcodeStart_SOPP + inst.template get<SOPP::Field::OP>());

  auto const offset = (int16_t)inst.template get<SOPP::Field::SIMM16>();

  *pCode += 1;

  using namespace ir::dialect;
  switch (op) {
    case eOpcode::S_NOP: break; // ignore
    case eOpcode::S_ENDPGM: {
      ctx.create<core::ReturnOp>();
    } break;
    case eOpcode::S_BRANCH: {
      auto const target =
          ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = (int64_t)(4 + pc) + 4 * (int64_t)offset}, ir::OperandType::i64());
      ctx.create<core::JumpAbsOp>(target);
    } break;
    case eOpcode::S_CBRANCH_SCC0: {
      auto const target =
          ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = (int64_t)(4 + pc) + 4 * (int64_t)offset}, ir::OperandType::i64());

      auto pred = ctx.create<arith::NotOp>(createDst(), createSrc(eOperandKind::SCC()), ir::OperandType::i1());
      ctx.create<core::CjumpAbsOp>(pred, target);
    } break;
    case eOpcode::S_CBRANCH_SCC1: {
      auto const target =
          ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = (int64_t)(4 + pc) + 4 * (int64_t)offset}, ir::OperandType::i64());
      ctx.create<core::CjumpAbsOp>(createSrc(eOperandKind::SCC()), target);
    } break;
    case eOpcode::S_CBRANCH_VCCZ: {
      auto const target =
          ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = (int64_t)(4 + pc) + 4 * (int64_t)offset}, ir::OperandType::i64());

      auto pred = ctx.create<arith::NotOp>(createDst(), createSrc(eOperandKind::VCC()), ir::OperandType::i1());
      ctx.create<core::CjumpAbsOp>(pred, target);
    } break;
    case eOpcode::S_CBRANCH_VCCNZ: {
      auto const target =
          ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = (int64_t)(4 + pc) + 4 * (int64_t)offset}, ir::OperandType::i64());
      ctx.create<core::CjumpAbsOp>(createSrc(eOperandKind::VCC()), target);
    } break;
    case eOpcode::S_CBRANCH_EXECZ: {
      auto const target =
          ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = (int64_t)(4 + pc) + 4 * (int64_t)offset}, ir::OperandType::i64());

      auto pred = ctx.create<arith::NotOp>(createDst(), createSrc(eOperandKind::EXEC()), ir::OperandType::i1());
      ctx.create<core::CjumpAbsOp>(pred, target);
    } break;
    case eOpcode::S_CBRANCH_EXECNZ: {
      auto const target =
          ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_i64 = (int64_t)(4 + pc) + 4 * (int64_t)offset}, ir::OperandType::i64());
      ctx.create<core::CjumpAbsOp>(createSrc(eOperandKind::EXEC()), target);
    } break;
    case eOpcode::S_BARRIER: {
      ctx.create<core::BarrierOp>();
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