#include "debug_strings.h"

#include <ostream>

namespace compiler::cfg {
void dumpBlock(std::ostream& os, const ControlFlow& cfg, blocks::blockid_t bid, const std::string& indent) {
  const auto* B = cfg.getBlock(bid);

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

void dumpRegion(std::ostream& os, const ControlFlow& cfg, blocks::regionid_t rid, const std::string& indent) {
  const auto& R = cfg.getRegion(rid);

  os << indent << "region @" << R.id.value << " {\n";

  // Entry block comment
  if (R.entry.isValid()) os << indent << "  // entry: ^bb" << R.entry.value << "\n";

  // Dump blocks in region order
  for (auto bid: R.blocks)
    dumpBlock(os, cfg, bid, indent + "  ");

  // Dump nested regions
  for (auto srid: R.subregions)
    dumpRegion(os, cfg, srid, indent + "  ");

  // Exit block comment
  if (R.exit.isValid()) os << indent << "  // exit: ^bb" << R.exit.value << "\n";

  os << indent << "}\n";
}

void dumpCFG(std::ostream& os, const ControlFlow& cfg) {
  os << "cfg {\n";

  // Dump all regions
  for (size_t i = 0; i < cfg.regionCount(); ++i) {
    blocks::regionid_t rid((uint32_t)i);
    dumpRegion(os, cfg, rid, "  ");
  }

  // Dump any blocks not in regions (should not happen normally)
  os << "  // blocks without region:\n";
  for (size_t i = 0; i < cfg.blocksCount(); ++i) {
    blocks::blockid_t bid((uint32_t)i);

    bool inRegion = false;
    for (size_t r = 0; r < cfg.regionCount(); ++r) {
      auto& R = cfg.getRegion(blocks::regionid_t((uint32_t)r));
      for (auto b: R.blocks)
        if (b == bid) {
          inRegion = true;
          break;
        }
      if (inRegion) break;
    }

    if (!inRegion) dumpBlock(os, cfg, bid, "  ");
  }

  os << "}\n";
}
} // namespace compiler::cfg
