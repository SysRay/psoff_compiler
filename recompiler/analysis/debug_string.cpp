#include "debug_strings.h"
#include "scc.h"

namespace compiler::analysis::debug {

void dump(std::ostream& os, SCC const& sccs) {
  for (auto const& nodes: sccs.get()) {
    os << "Nodes: {";
    for (auto n: nodes)
      os << n << ' ';
    os << "}\n";
  }
}

void dump(std::ostream& os, SCCMeta const& meta) {
  os << "Preds: {";
  for (auto u: meta.preds)
    os << u << ",";
  os << "} Entries: {";
  for (auto e: meta.incoming)
    os << e << ' ';
  os << "} Exits: {";
  for (auto e: meta.outgoing)
    os << e << ' ';
  os << "} Succs: {";
  for (auto u: meta.succs)
    os << u << ",";
  os << "}\n";
}
} // namespace compiler::analysis::debug
