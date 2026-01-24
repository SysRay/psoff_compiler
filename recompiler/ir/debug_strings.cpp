#include "debug_strings.h"

#include "cfg/cfg.h"
#include "dialects/dialects.h"
#include "frontend/debug_strings.h"

#include <assert.h>
#include <bit>
#include <format>
#include <ostream>
#include <unordered_set>

namespace compiler::ir::debug {
static std::string_view isVirtual(InstCore const& op) {
  if (op.flags.is_set(eInstructionFlags::kVirtual)) {
    return "v_";
  }
  return "s_";
}

static void printIMM(std::ostream& os, ConstantValue const& value, OperandType type) {
  assert(!type.is_vector());
  assert(!type.is_array());

  switch (type.base()) {
    case OperandType::eBase::i1: {
      os << std::bit_cast<bool>((bool)value.value_u64);
    } break;
    case OperandType::eBase::i8:
    case OperandType::eBase::i16:
    case OperandType::eBase::i32:
    case OperandType::eBase::i64: {
      if (type.is_signed()) {
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

static void getDst(std::ostream& os, InstructionManager const& im, InstCore const& op) {
  for (uint8_t n = 0; n < op.numDst; ++n) {
    auto const& item = im.getOperand(op.getOutputId(n));
    os << "%" << item.ssa.ssaValue;

    if (item.hasKind()) {
      os << "($";
      frontend::debug::printOperandDst(os, item);
      os << "), ";
    }
    if (n < op.numDst - 1) os << ", ";
  }
}

static void getSrc(std::ostream& os, InstructionManager const& im, InstCore const& op) {
  for (uint8_t n = 0; n < op.numSrc; ++n) {
    os << " $";
    frontend::debug::printOperandSrc(os, im.getOperand(op.getInputId(n)));
  }
}

static void printTypes(std::ostream& os, InstructionManager const& im, InstCore const& op) {
  if (op.numSrc > 0) {
    os << "   (";
    for (uint8_t n = 0; n < op.numSrc; ++n) {
      frontend::debug::printType(os, im.getOperand(op.getInputId(n)).type);
      if (n < op.numSrc - 1) os << ", ";
    }
    os << ")";
  }

  if (op.numDst > 0) {
    os << " -> ";
    for (uint8_t n = 0; n < op.numDst; ++n) {
      frontend::debug::printType(os, im.getOperand(op.getOutputId(n)).type);
      if (n < op.numDst - 1) os << ", ";
    }
  }
}

static void getDebug_generic(std::ostream& os, InstructionManager const& im, InstCore const& op) {
  if (op.numDst > 0) {
    getDst(os, im, op);
    os << " = ";
  }

  os << isVirtual(op) << dialect::getInstrKindStr(op.dialect, op.kind);

  if (op.numSrc > 0) {
    getSrc(os, im, op);
  } else if (op.isConstant()) {
    os << " #";
    printIMM(os, im.getConstant(op.getConstantId()), im.getOperand(op.getOutputId(0)).type);
  }

  printTypes(os, im, op);
  os << std::endl;
}

void getDebug(std::ostream& os, InstructionManager const& im, InstCore const& op) {
  switch (op.kind) {

    default: return getDebug_generic(os, im, op);
  }
}
} // namespace compiler::ir::debug