#pragma once

#include "ir.h"

#include <string>

namespace compiler::ir {
class IROperations;

namespace rvsdg {
class IRBlocks;
}
} // namespace compiler::ir

namespace compiler::ir::debug {
// todo needs IROperations
void getDebug(std::ostream& os, IROperations const& im, InstCore const& op);

void dumpBlock(std::ostream& os, const rvsdg::IRBlocks& cfg, nodeid_t bid, const std::string& indent = "");

void dumpRegion(std::ostream& os, const rvsdg::IRBlocks& cfg, regionid_t rid, const std::string& indent = "");

void dumpCFG(std::ostream& os, const rvsdg::IRBlocks& cfg);
} // namespace compiler::ir::debug