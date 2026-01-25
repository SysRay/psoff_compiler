#pragma once

#include "ir.h"

#include <string>

namespace compiler::ir::rvsdg {
class Builder;

}

namespace compiler::ir::debug {
// todo needs instructionmanager
void getDebug(std::ostream& os, InstructionManager const& im, InstCore const& op);

void dumpBlock(std::ostream& os, const rvsdg::Builder& cfg, nodeid_t bid, const std::string& indent = "");

void dumpRegion(std::ostream& os, const rvsdg::Builder& cfg, regionid_t rid, const std::string& indent = "");

void dumpCFG(std::ostream& os, const rvsdg::Builder& cfg);
} // namespace compiler::ir::debug