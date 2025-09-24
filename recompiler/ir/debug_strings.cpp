#include "debug_strings.h"

#include "frontend/debug_strings.h"
#include "instructions.h"

#include <assert.h>
#include <bit>
#include <format>
#include <ostream>

namespace compiler::ir::debug {
static std::string_view isVirtual(InstCore const& op) {
  if (op.isVirtual()) {
    return "v_";
  }
  return "s_";
}

static void printIMM(std::ostream& os, uint64_t value, OperandType type) {
  assert(!type.is_vector());
  assert(!type.is_array());

  switch (type.base()) {
    case OperandType::eBase::i1: {
      os << std::bit_cast<bool>((bool)value);
    } break;
    case OperandType::eBase::i8:
    case OperandType::eBase::i16:
    case OperandType::eBase::i32:
    case OperandType::eBase::i64: {
      if (type.is_signed()) {
        if ((int64_t)value > 10000)
          os << "0x" << std::hex << (int64_t)value;
        else
          os << (int64_t)value;
      } else {
        if (value > 10000)
          os << "0x" << std::hex << value;
        else
          os << value;
      }
    } break;
    case OperandType::eBase::f32: {
      os << std::bit_cast<float>((uint32_t)value);
    } break;
    case OperandType::eBase::f64: {
      os << std::bit_cast<double>(value);
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
    printIMM(os, op.srcConstant.value, op.srcConstant.type);
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