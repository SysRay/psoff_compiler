#include "frontend/ir_types.h"
#include "../debug_strings.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "../instruction_builder.h"
#include "translate.h"

#include <bitset>
#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
InstructionKind_t handleExp(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto inst = EXP(getU64(*pCode));

  std::bitset<4> const enable     = inst.template get<EXP::Field::EN>();
  uint8_t const        target     = inst.template get<EXP::Field::TGT>();
  bool const           compressed = inst.template get<EXP::Field::COMPR>();
  bool const           isLast     = inst.template get<EXP::Field::DONE>();

  auto src0 = OpSrc(eOperandKind::VGPR(inst.template get<EXP::Field::VSRC0>()));
  auto src1 = OpSrc(eOperandKind::VGPR(inst.template get<EXP::Field::VSRC1>()));
  auto src2 = OpSrc(eOperandKind::VGPR(inst.template get<EXP::Field::VSRC2>()));
  auto src3 = OpSrc(eOperandKind::VGPR(inst.template get<EXP::Field::VSRC2>()));

  *pCode += 2;

  if (target >= 0x0 && target <= 0x7) { // Attachments
    bool const useExecMask = inst.template get<EXP::Field::VM>();
    if (useExecMask) {
      builder.createInstruction(create::discardOp(OpSrc(eOperandKind::EXEC(), true, false)));
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