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

void dump(std::ostream& os, SCCEdges const& meta) {
  os << "Entries: {";
  for (auto [from,to]: meta.entryEdges)
    os << from << "->" << to  << ' ';
  os << "} Exits: {";
  for (auto [from,to]: meta.exitEdges)
    os << from << "->" << to  << ' ';
  os << "} continue: {";
  for (auto [from,to]: meta.backEdges)
    os << from << "->" << to  << ' ';
  os << "}\n";
}
} // namespace compiler::analysis::debug
