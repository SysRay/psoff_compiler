#include "instruction_builder.h"

// #include "ir/types.h"

namespace compiler::frontend::translate {
ir::InstCore createLiteral(uint32_t value) {
  auto inst              = ir::getInfo(ir::eInstKind::ConstantOp);
  inst.srcConstant.value = value;
  inst.srcConstant.type  = ir::OperandType::i32();
  return inst;
}
} // namespace compiler::frontend::translate