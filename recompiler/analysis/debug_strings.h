#include <ostream>

namespace compiler::analysis {

struct SCC;
struct SCCMeta;

namespace debug {

void dump(std::ostream& os, SCC const& sccs);
void dump(std::ostream& os, SCCMeta const& meta);
} // namespace debug
} // namespace compiler::analysis