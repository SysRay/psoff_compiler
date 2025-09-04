#include "builder.h"

int main() {
  compiler::Builder builder(100);

  auto& add = builder.createInstruction(5, 0, 1, 2);

  return 0;
}
