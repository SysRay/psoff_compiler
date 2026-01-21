#include "../debug_strings.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "ir/dialects/arith/builder.h"
#include "ir/dialects/core/builder.h"
#include "translate.h"

#include <bitset>
#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
InstructionKind_t handleExp(parser::Context& ctx, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto inst = EXP(getU64(*pCode));

  std::bitset<4> const enable     = inst.template get<EXP::Field::EN>();
  uint8_t const        target     = inst.template get<EXP::Field::TGT>();
  bool const           compressed = inst.template get<EXP::Field::COMPR>();
  bool const           isLast     = inst.template get<EXP::Field::DONE>();

  using namespace compiler::ir::dialect;
  auto src0 = createSrc(eOperandKind::VGPR(inst.template get<EXP::Field::VSRC0>()));
  auto src1 = createSrc(eOperandKind::VGPR(inst.template get<EXP::Field::VSRC1>()));
  auto src2 = createSrc(eOperandKind::VGPR(inst.template get<EXP::Field::VSRC2>()));
  auto src3 = createSrc(eOperandKind::VGPR(inst.template get<EXP::Field::VSRC2>()));

  *pCode += 2;

  if (target >= 0x0 && target <= 0x7) { // Attachments
    bool const useExecMask = inst.template get<EXP::Field::VM>();
    if (useExecMask) {
      ctx.create<core::DiscardOp>(createSrc(eOperandKind::EXEC()));
    }

  } else if (target == 0x8 && enable.any()) { // Write to depth

  } else if (target >= 0xc && target <= 0xf) { // Vertex output
    auto const pos = target - 0xc;

  } else if (target >= 0x20 && target <= 0x3F) { //  output params
    auto const pos = target - 0x20;
  }
  return conv(eOpcode::EXP);
}
} // namespace compiler::frontend::translate