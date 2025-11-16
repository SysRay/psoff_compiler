#pragma once
#include "cfg.h"

namespace compiler::cfg {
void dumpBlock(std::ostream& os, const ControlFlow& cfg, blocks::blockid_t bid, const std::string& indent = "");

void dumpRegion(std::ostream& os, const ControlFlow& cfg, blocks::regionid_t rid, const std::string& indent = "");

void dumpCFG(std::ostream& os, const ControlFlow& cfg);
} // namespace compiler::cfg
