#include "../gfx/encoding_types.h"
#include "../parser.h"
#include "opcodes_table.h"

#include <bitset>
#include <format>
#include <stdexcept>

namespace compiler::frontend {

uint8_t Parser::handleExp(CodeBlock& cb, pc_t pc, uint32_t const* pCode) {
  auto inst = EXP(getU64(pCode));

  std::bitset<4> const enable     = inst.template get<EXP::Field::EN>();
  uint8_t const        target     = inst.template get<EXP::Field::TGT>();
  bool const           compressed = inst.template get<EXP::Field::COMPR>();
  bool const           isLast     = inst.template get<EXP::Field::DONE>();

  auto src0 = eOperandKind::VGPR(inst.template get<EXP::Field::VSRC0>());
  auto src1 = eOperandKind::VGPR(inst.template get<EXP::Field::VSRC1>());
  auto src2 = eOperandKind::VGPR(inst.template get<EXP::Field::VSRC2>());
  auto src3 = eOperandKind::VGPR(inst.template get<EXP::Field::VSRC2>());

  if (target >= 0x0 && target <= 0x7) { // Attachments
    bool const useExecMask = inst.template get<EXP::Field::VM>();
    if (useExecMask) {
      // ctx.create<core::DiscardOp>(createSrc(eOperandKind::EXEC()));
    }

  } else if (target == 0x8 && enable.any()) { // Write to depth

  } else if (target >= 0xc && target <= 0xf) { // Vertex output
    auto const pos = target - 0xc;

  } else if (target >= 0x20 && target <= 0x3F) { //  output params
    auto const pos = target - 0x20;
  }

  return sizeof(uint64_t);
}

} // namespace compiler::frontend