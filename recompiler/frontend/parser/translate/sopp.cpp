#include "../../frontend.h"
#include "../debug_strings.h"
#include "../opcodes_table.h"
#include "builder.h"
#include "encodings.h"
#include "instruction_builder.h"
#include "translate.h"

#include <format>
#include <stdexcept>

namespace compiler::frontend::translate {
bool handleSopp(Builder& builder, parser::pc_t pc, parser::code_p_t* pCode) {
  using namespace parser;

  auto       inst = SOPP(**pCode);
  auto const op   = (parser::eOpcode)inst.template get<SOPP::Field::OP>();

  auto const offset = (int16_t)inst.template get<SOPP::Field::SIMM16>();

  *pCode += 1;
  return true;
}
} // namespace compiler::frontend::translate