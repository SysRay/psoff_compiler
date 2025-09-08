#include "builder.h"
#include "ir/instructions.h"

namespace compiler::frontend {
uint64_t getAddr(uint64_t addr) {
  // todo move to user defined
  return addr;
  // return Platform::System::mem_convert();
}
} // namespace compiler::frontend

int main() {
  compiler::Builder builder(100);
  using namespace compiler::ir;

  auto& add = builder.createInstruction(getInfo(eInstKind::AddF32Op));

  return 0;
}
