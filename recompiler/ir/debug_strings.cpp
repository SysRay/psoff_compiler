#include "debug_strings.h"

#include "blocks.h"
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

static void getDst(std::ostream& os, IROperations const& im, InstCore const& op) {
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

static void getSrc(std::ostream& os, IROperations const& im, InstCore const& op) {
  for (uint8_t n = 0; n < op.numSrc; ++n) {
    os << " $";
    frontend::debug::printOperandSrc(os, im.getOperand(op.getInputId(n)));
  }
}

static void printTypes(std::ostream& os, IROperations const& im, InstCore const& op) {
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

static void getDebug_generic(std::ostream& os, IROperations const& im, InstCore const& op) {
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

void getDebug(std::ostream& os, IROperations const& im, InstCore const& op) {
  switch (op.kind) {

    default: return getDebug_generic(os, im, op);
  }
}

void dumpBlock(std::ostream& os, const rvsdg::IRBlocks& builder, blockid_t bid, const std::string& indent) {
  const auto* B = builder.getBase(bid);

  // Block header: ^bbX:
  os << indent << "^bb" << B->id.value << ":\n";

  // Print successors
  auto succs = builder.getCfg().getSuccessors(bid);
  if (!succs.empty()) {
    os << indent << "  successors:";
    for (auto s: succs)
      os << " ^bb" << s.value;
    os << "\n";
  }
}

static void dumpNode(std::ostream& os, const rvsdg::IRBlocks& builder, blockid_t bid, const std::string& indent);

void dumpRegion(std::ostream& os, const rvsdg::IRBlocks& builder, regionid_t rid, const std::string& indent) {
  auto R = builder.getRegion(rid);

  os << indent << "region @" << R->id.value << ":\n";

  // // Entry block comment
  // if (R->entry.isValid()) os << indent << "  // entry: ^bb" << R->entry.value << "\n";

  // Dump blocks in region order
  for (auto bid: R->blocks)
    dumpNode(os, builder, bid, indent + "  ");

  // // Exit block comment
  // if (R->exit.isValid()) os << indent << "  // exit: ^bb" << R->exit.value << "\n";

  os << indent << "}\n";
}

void dumpNode(std::ostream& os, const rvsdg::IRBlocks& builder, blockid_t bid, const std::string& indent) {
  const auto* B = builder.getBase(bid);

  os << indent << "^bb" << B->id;

  using namespace rvsdg;
  switch (B->type) {
    case eBlockType::Simple: {
      auto node = builder.getNode<SimpleBlock>(B->id);
      os << " Simple {\n";

      auto& im = builder.getInstructions();
      for (auto id: node->instructions) {
        os << indent << "  ";
        ir::debug::getDebug(os, im, im.getInstr(id));
      }
    } break;
    case eBlockType::Gamma: {
      auto node = builder.getNode<GammaBlock>(B->id);
      os << " Gamma ";

      os << "{\n";
      for (uint32_t n = 0; n < node->branches.size(); ++n) {
        dumpRegion(os, builder, node->branches[n], indent + "  ");
      }
    } break;
    case eBlockType::Theta: {
      auto node = builder.getNode<ThetaBlock>(B->id);
      os << " Theta {\n";
      dumpRegion(os, builder, node->body, indent + "  ");
    } break;
    case eBlockType::Lambda: {
      auto node = builder.getNode<LambdaNode>(B->id);
      os << " Lambda {\n";
      dumpRegion(os, builder, node->body, indent + "  ");
    } break;
  }

  if (!B->outputs.empty()) {
    os << indent << "  Outputs: ";
    for (uint8_t n = 0; n < B->outputs.size(); ++n) {
      auto const& item = builder.getInstructions().getOperand(B->outputs[n]);
      if (item.isSSA()) {
        os << "%" << item.ssaId;
      } else {
        os << "unset";
      }
      if (n < B->outputs.size() - 1) os << ", ";
    }

    os << "   (";
    for (uint8_t n = 0; n < B->outputs.size(); ++n) {
      auto const& item = builder.getInstructions().getOperand(B->outputs[n]);
      frontend::debug::printType(os, item.type);
      if (n < B->outputs.size() - 1) os << ", ";
    }
    os << ")";
  }
  os << std::endl;

  auto succs = builder.getCfg().getSuccessors(bid);
  if (!succs.empty()) {
    os << indent << "  successors:";
    for (auto s: succs)
      os << " ^bb" << s.value;
    os << std::endl;
  }
  os << indent << "}\n";
}

void dumpCFG(std::ostream& os, const rvsdg::IRBlocks& builder) {
  os << "builder {\n";

  if (builder.getMainFunctionId().isValid()) {
    dumpNode(os, builder, builder.getMainFunctionId(), "  ");
  }

  os << "}\n";
}
} // namespace compiler::ir::debug