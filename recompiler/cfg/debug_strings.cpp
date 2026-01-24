#include "debug_strings.h"

#include "frontend/debug_strings.h"
#include "ir/debug_strings.h"

#include <ostream>

namespace compiler::cfg {
void dumpBlock(std::ostream& os, const ControlFlow& cfg, rvsdg::nodeid_t bid, const std::string& indent) {
  const auto* B = cfg.nodes()->getNodeBase(bid);

  // Block header: ^bbX:
  os << indent << "^bb" << B->id.value << ":\n";

  // Print successors
  auto succs = cfg.getSuccessors(bid);
  if (!succs.empty()) {
    os << indent << "  successors:";
    for (auto s: succs)
      os << " ^bb" << s.value;
    os << "\n";
  }
}

static void dumpNode(std::ostream& os, const ControlFlow& cfg, rvsdg::nodeid_t bid, const std::string& indent);

void dumpRegion(std::ostream& os, const ControlFlow& cfg, rvsdg::regionid_t rid, const std::string& indent) {
  auto R = cfg.nodes()->getRegion(rid);

  os << indent << "region @" << R->id.value << ":\n";

  // // Entry block comment
  // if (R->entry.isValid()) os << indent << "  // entry: ^bb" << R->entry.value << "\n";

  // Dump blocks in region order
  for (auto bid: R->nodes)
    dumpNode(os, cfg, bid, indent + "  ");

  // // Exit block comment
  // if (R->exit.isValid()) os << indent << "  // exit: ^bb" << R->exit.value << "\n";

  os << indent << "}\n";
}

void dumpNode(std::ostream& os, const ControlFlow& cfg, rvsdg::nodeid_t bid, const std::string& indent) {
  const auto* B = cfg.nodes()->getNodeBase(bid);

  os << indent << "^bb" << B->id;

  switch (B->type) {
    case rvsdg::eNodeType::SimpleNode: {
      auto node = cfg.nodes()->getNode<rvsdg::SimpleNode>(B->id);
      os << " Simple {\n";

      auto im = cfg.getInstructions();
      for (auto id: node->instructions) {
        os << indent << "  ";
        ir::debug::getDebug(os, *im, im->getInstr(id));
      }
    } break;
    case rvsdg::eNodeType::GammaNode: {
      auto node = cfg.nodes()->getNode<rvsdg::GammaNode>(B->id);
      os << " Gamma {\n";
      for (uint32_t n = 0; n < node->branches.size(); ++n) {
        dumpRegion(os, cfg, node->branches[n], indent + "  ");
      }
    } break;
    case rvsdg::eNodeType::ThetaNode: {
      auto node = cfg.nodes()->getNode<rvsdg::ThetaNode>(B->id);
      os << " Theta {\n";
      dumpRegion(os, cfg, node->body, indent + "  ");
    } break;
    case rvsdg::eNodeType::LambdaNode: {
      auto node = cfg.nodes()->getNode<rvsdg::LambdaNode>(B->id);
      os << " Lambda {\n";
      dumpRegion(os, cfg, node->body, indent + "  ");
    } break;
  }

  if (!B->outputs.empty()) {
    os << indent << "  Outputs: ";
    for (uint8_t n = 0; n < B->outputs.size(); ++n) {
      auto const& item = cfg.getInstructions()->getOperand(B->outputs[n]);
      if (item.isSSA()) {
        os << "%" << item.ssaId;
      } else {
        os << "unset";
      }
      if (n < B->outputs.size() - 1) os << ", ";
    }

    os << "   (";
    for (uint8_t n = 0; n < B->outputs.size(); ++n) {
      auto const& item = cfg.getInstructions()->getOperand(B->outputs[n]);
      frontend::debug::printType(os, item.type);
      if (n < B->outputs.size() - 1) os << ", ";
    }
    os << ")";
  }
  os << std::endl;

  auto succs = cfg.getSuccessors(bid);
  if (!succs.empty()) {
    os << indent << "  successors:";
    for (auto s: succs)
      os << " ^bb" << s.value;
    os << std::endl;
  }
  os << indent << "}\n";
}

void dumpCFG(std::ostream& os, const ControlFlow& cfg) {
  os << "cfg {\n";

  if (cfg.nodes()->getMainFunctionId().isValid()) {
    dumpNode(os, cfg, cfg.nodes()->getMainFunctionId(), "  ");
  }

  os << "}\n";
}
} // namespace compiler::cfg
