#include <ostream>

namespace compiler::analysis {

struct SCC;
struct SCCEdges;

namespace debug {

void dump(std::ostream& os, SCC const& sccs);
void dump(std::ostream& os, SCCEdges const& meta);
} // namespace debug
} // namespace compiler::analysis