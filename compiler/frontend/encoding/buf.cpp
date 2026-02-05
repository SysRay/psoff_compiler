#include "../gfx/encoding_types.h"
#include "../gfx/operand_types.h"
#include "../parser.h"
#include "opcodes_table.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend {

uint8_t Parser::handleSmrd(CodeBlock& cb, pc_t pc, uint32_t const* pCode) {
  auto       inst = SMRD(*pCode);
  auto const op   = (eOpcode)(OPcodeStart_SMRD + inst.template get<SMRD::Field::OP>());

  auto const sdst      = eOperandKind((eOperandKind_t)inst.template get<SMRD::Field::SDST>());
  auto const sBase     = eOperandKind((eOperandKind_t)inst.template get<SMRD::Field::SBASE>());
  auto const offsetImm = inst.template get<SMRD::Field::OFFSET>();
  auto       sOffset   = eOperandKind((eOperandKind_t)offsetImm); // either imm or op
  auto const isImm     = (bool)inst.template get<SMRD::Field::IMM>();

  uint8_t size = sizeof(uint32_t);
  if (!isImm && sOffset.isLiteral()) {
    size = sizeof(uint64_t);
    // src0 = createSrc(ctx.create<core::ConstantOp>(createDst(), ir::ConstantValue {.value_u64 = **pCode}, ir::OperandType::i32()));
  }

  return size;
}

uint8_t Parser::handleMubuf(CodeBlock& cb, pc_t pc, uint32_t const* pCode) {
  auto       inst = MUBUF(getU64(pCode));
  auto const op   = (eOpcode)(OPcodeStart_MUBUF + inst.template get<MUBUF::Field::OP>());

  return sizeof(uint64_t);
}

uint8_t Parser::handleMtbuf(CodeBlock& cb, pc_t pc, uint32_t const* pCode) {
  auto       inst = MTBUF(getU64(pCode));
  auto const op   = (eOpcode)(OPcodeStart_MTBUF + inst.template get<MTBUF::Field::OP>());

  return sizeof(uint64_t);
}
} // namespace compiler::frontend