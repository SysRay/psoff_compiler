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
  os << "Entries: {";
  for (auto e: meta.entries)
    os << e << ' ';
  os << "} Exits: {";
  for (auto e: meta.exits)
    os << e << ' ';
  os << "} Repetitions: {";
  for (auto u: meta.body)
    os << u << ",";
  os << "}\n";
}
} // namespace compiler::analysis::debug
