#include "instructions.h"

namespace compiler::ir::debug {

std::string_view getInstrKindStr(eInstKind code) {
#define X(name, ...)        #name,
#define X_NO_OPS(name, ...) #name,
#define X_NO_SRC(name, ...) #name,
#define X_NO_DST(name, ...) #name,
  static constexpr std::string_view kDebugTable[] = {INSTRUCTION_LIST};
#undef X
#undef X_NO_OPS
#undef X_NO_SRC
  return kDebugTable[(std::underlying_type<eInstKind>::type)code];
}
} // namespace compiler::ir::debug