#include "builder.h"
#include "ir/instructions.h"

int main() {
  compiler::Builder builder(100);
  using namespace compiler::ir;

  auto& add = builder.createInstruction(getInfo(eInstKind::AddF32Op));

  return 0;
}
