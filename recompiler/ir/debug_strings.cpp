#include "debug_strings.h"

#include "cfg/cfg.h"
#include "frontend/debug_strings.h"
#include "instructions.h"

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

static void printIMM(std::ostream& os, InstConstant const& value) {
  assert(!value.type.is_vector());
  assert(!value.type.is_array());

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

static void inline printIndent(std::ostream& os, int indent) {
  os << std::string(indent, ' ');
}

static void dumpSuccPred(std::ostream& os, const compiler::ir::cfg::ControlFlow& g, ir::cfg::blocks::blockid_t id, int indent) {
  printIndent(os, indent);
  os << "succ: ";
  auto succ = g.getSuccessors(id);
  if (succ.empty())
    os << "(none)";
  else
    for (auto s: succ)
      os << s.value << " ";
  os << "\n";

  printIndent(os, indent);
  os << "pred: ";
  auto pred = g.getPredecessors(id);
  if (pred.empty())
    os << "(none)";
  else
    for (auto p: pred)
      os << p.value << " ";
  os << "\n";
}

static void dumpNodeHeader(std::ostream& os, const ir::cfg::blocks::Base* base) {
  switch (base->type) {
    case cfg::blocks::eBlockType::Start: {
      os << "StartRegion{id=" << base->id.value << "}";
    } break;
    case cfg::blocks::eBlockType::Stop: {
      os << "StopRegion{id=" << base->id.value << "}";
    } break;
    case cfg::blocks::eBlockType::Basic: {
      auto block = (cfg::blocks::BasicBlock*)base;
      os << "BasicRegion{id=" << block->id.value << ", begin=" << block->opbegin << ", end=" << block->opend << "}";
    } break;
    case cfg::blocks::eBlockType::Loop: {
      auto block = (cfg::blocks::LoopBlock*)base;
      os << "LoopRegion{id=" << block->id.value << ", start=" << block->headerId.value << ", exit=" << block->exitId.value
         << ", continue=" << block->contId.value << "}";
    } break;
    case cfg::blocks::eBlockType::Cond: {
      auto block = (cfg::blocks::CondBlock*)base;
      os << "CondRegion{id=" << block->id.value << ", merge=" << block->mergeId.value << ", branches: ";

      for (uint32_t n = 0; n < block->branches.size(); ++n) {
        os << "B_" << n << "=" << block->branches[n].value << ", ";
      }
      os << "}";
    } break;
  }
}

static void dumpSubgraph(std::ostream& os, const compiler::ir::cfg::ControlFlow& g, ir::cfg::blocks::blockid_t id, ir::cfg::blocks::blockid_t stop,
                         std::unordered_set<uint32_t>& visited, int indent) {
  if (visited.contains(id.value) || id == ir::cfg::blocks::blockid_t()) return;

  visited.insert(id.value);
  const auto base = g.getBlock(id);

  // Node header
  printIndent(os, indent);
  dumpNodeHeader(os, base);
  os << "\n";

  // Print succ/pred
  dumpSuccPred(os, g, id, indent + 1);

  if (id.value == stop.value) return;

  // Structural recursion only when needed
  switch (base->type) {
    case cfg::blocks::eBlockType::Start: {
    } break;
    case cfg::blocks::eBlockType::Stop: {
    } break;
    case cfg::blocks::eBlockType::Basic: {
    } break;
    case cfg::blocks::eBlockType::Loop: {
      auto block = (cfg::blocks::LoopBlock*)base;
      os << std::string(indent + 1, ' ') << "{\n";
      dumpSubgraph(os, g, block->headerId, block->exitId, visited, indent + 2);
      os << std::string(indent + 1, ' ') << "}\n";
    } break;
    case cfg::blocks::eBlockType::Cond: {
      auto block = (cfg::blocks::CondBlock*)base;
      for (uint32_t n = 0; n < block->branches.size(); ++n) {
        os << std::string(indent + 1, ' ') << "B_" << n << ":{\n";
        dumpSubgraph(os, g, block->branches[n], block->mergeId, visited, indent + 2);
        os << std::string(indent + 1, ' ') << "}\n";
      }
    } break;
  }

  // Continue linear chain only if it's linear
  for (auto succ: g.getSuccessors(id)) {
    dumpSubgraph(os, g, succ, stop, visited, indent);
  }
}

void dump(std::ostream& os, const compiler::ir::cfg::ControlFlow& g) {
  os << "RegionGraph Structure:\n";
  std::unordered_set<uint32_t> visited;
  dumpSubgraph(os, g, g.getStartId(), g.getStopId(), visited, 0);
}
} // namespace compiler::ir::debug