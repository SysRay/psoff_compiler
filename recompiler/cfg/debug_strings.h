#pragma once
#include "cfg.h"

#include <ostream>

namespace compiler::cfg {
void dumpBlock(std::ostream& os, const ControlFlow& cfg, rvsdg::nodeid_t bid, const std::string& indent = "");

void dumpRegion(std::ostream& os, const ControlFlow& cfg, rvsdg::regionid_t rid, const std::string& indent = "");

void dumpCFG(std::ostream& os, const ControlFlow& cfg);
} // namespace compiler::cfg
