#include "debug_strings.h"

#include "frontend/debug_strings.h"
#include "instructions.h"

#include <assert.h>
#include <bit>
#include <format>
#include <ostream>

namespace compiler::ir::debug {
static std::string_view isVirtual(InstCore const& op) {
  if (op.flags.is_set(eInstructionFlags::kVirtual)) {
    return "v_";
  }
  return "s_";
}

static void printIMM(std::ostream& os, InstConstant const& value) {
  assert(!type.is_vector());
  assert(!type.is_array());

  switch (value.type.base()) {
    case OperandType::eBase::i1: {
      os << std::bit_cast<bool>((bool)value.value_u64);
    } break;
    case OperandType::eBase::i8:
    case OperandType::eBase::i16:
    case OperandType::eBase::i32:
    case OperandType::eBase::i64: {
      if (value.type.is_signed()) {
        if (value.value_i64 > 10000)
          os << "0x" << std::hex << value.value_i64;
        else
          os << value.value_i64;
      } else {
        if (value.value_u64 > 10000)
          os << "0x" << std::hex << value.value_u64;
        else
          os << value.value_u64;
      }
    } break;
    case OperandType::eBase::f32: {
      os << (float)value.value_f64;
    } break;
    case OperandType::eBase::f64: {
      os << value.value_f64;
    } break;
  }
}

static void getDst(std::ostream& os, InstCore const& op) {
  for (uint8_t n = 0; n < op.numDst; ++n) {
    os << " ";
    frontend::debug::printOperandDst(os, op.dstOperands[n]);
  }
}

static void getSrc(std::ostream& os, InstCore const& op) {
  if (op.group == eInstructionGroup::kConstant) {
    printIMM(os, op.srcConstant);
    return;
  }

  for (uint8_t n = 0; n < op.numSrc; ++n) {
    os << " ";
    frontend::debug::printOperandSrc(os, op.srcOperands[n]);
  }
}

static void printTypes(std::ostream& os, InstCore const& op) {
  os << "\t(";
  for (uint8_t n = 0; n < op.numDst; ++n) {
    frontend::debug::printType(os, op.dstOperands[n].type);
    if (n < op.numDst - 1) os << ", ";
  }

  for (uint8_t n = 0; n < op.numSrc; ++n) {
    os << ", ";
    frontend::debug::printType(os, op.srcOperands[n].type);
  }
  os << ")";
}

static void getDebug_generic(std::ostream& os, InstCore const& op) {
  os << isVirtual(op) << getInstrKindStr((eInstKind)op.kind) << " ";
  if (op.numDst > 0 || op.numSrc > 0) {
    getDst(os, op);
    os << ",";
    getSrc(os, op);

    printTypes(os, op);
  }
  os << std::endl;
}

void getDebug(std::ostream& os, InstCore const& op) {
  switch (op.kind) {

    default: return getDebug_generic(os, op);
  }
}
} // namespace compiler::ir::debug