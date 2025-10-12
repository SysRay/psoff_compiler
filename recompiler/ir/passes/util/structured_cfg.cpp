#include "structured_cfg.h"

#include <algorithm>
#include <iostream>
#include <string>

namespace compiler::ir {
void StructuredCFGBuilder::identifySCCs(regionid_t entry) {
  _sccs.clear();
  _tarjanState.clear();
  _tarjanIndex = 0;

  strongConnect(entry);
}

void StructuredCFGBuilder::strongConnect(regionid_t v) {
  auto& state   = _tarjanState[v];
  state.index   = _tarjanIndex;
  state.lowlink = _tarjanIndex;
  _tarjanIndex++;
  _tarjanStack.push(v);
  state.onStack = true;

  auto succs = _regions.getSuccessors(v);
  for (auto w: succs) {
    if (_tarjanState[w].index == -1) {
      strongConnect(w);
      state.lowlink = std::min(state.lowlink, _tarjanState[w].lowlink);
    } else if (_tarjanState[w].onStack) {
      state.lowlink = std::min(state.lowlink, _tarjanState[w].index);
    }
  }

  if (state.lowlink == state.index) {
    std::set<regionid_t> scc;
    regionid_t           w;
    do {
      w = _tarjanStack.top();
      _tarjanStack.pop();
      _tarjanState[w].onStack = false;
      scc.insert(w);
    } while (w != v);

    if (scc.size() > 1) {
      _sccs.push_back(std::move(scc));
    }
  }
}

bool StructuredCFGBuilder::isInSCC(regionid_t rid) const {
  for (const auto& scc: _sccs) {
    if (scc.count(rid) > 0) return true;
  }
  return false;
}

std::set<regionid_t> StructuredCFGBuilder::findLoopBody(const std::set<regionid_t>& scc, regionid_t entry) {

  std::set<regionid_t> body = scc;

  // Remove exit edges from consideration
  std::set<regionid_t> toRemove;
  for (auto rid: body) {
    auto succs           = _regions.getSuccessors(rid);
    bool hasExternalSucc = false;
    for (auto succ: succs) {
      if (body.count(succ) == 0) {
        hasExternalSucc = true;
        break;
      }
    }
    // Keep all nodes for now
  }

  return body;
}

std::unique_ptr<ASTNode> StructuredCFGBuilder::restructureLoop(const std::set<regionid_t>& scc, regionid_t entry) {

  // Mark as processed first to avoid infinite recursion
  for (auto rid: scc) {
    _processed.insert(rid);
  }

  // Build loop body recursively, handling branches
  auto bodyAST = restructureLoopBody(entry, scc);

  auto loopNode          = std::make_unique<ASTNode>(ASTNodeType::Loop, bodyAST->instrStart, bodyAST->instrEnd);
  loopNode->body         = std::move(bodyAST);
  loopNode->auxPredicate = 'q';

  return loopNode;
}

std::unique_ptr<ASTNode> StructuredCFGBuilder::restructureLoopBody(regionid_t entry, const std::set<regionid_t>& scc) {

    std::vector<std::unique_ptr<ASTNode>> bodyNodes;
    std::set<regionid_t>                  visited;

    // DFS traversal inside the loop SCC starting from 'entry'
    std::function<void(regionid_t)> traverse = [&](regionid_t current) {
        if (visited.count(current) > 0 || scc.count(current) == 0) {
            return;
        }

        visited.insert(current);

        auto succs = _regions.getSuccessors(current);

        // Filter successors to only those in SCC (ignore loop exit)
        std::vector<regionid_t> sccSuccs;
        for (auto succ : succs) {
            if (scc.count(succ) > 0) {
                sccSuccs.push_back(succ);
            }
        }

        if (sccSuccs.empty()) {
            // Leaf node (no successors inside SCC)
            bodyNodes.push_back(createBasicNode(current));
        }
        else if (sccSuccs.size() == 1) {
            // Linear flow: add the basic node and continue
            bodyNodes.push_back(createBasicNode(current));
            traverse(sccSuccs[0]);
        }
        else {
            // Branch inside loop: conservative safe handling
            //  -> create a Branch node but make each alternative the immediate successor (basic node)
            // This is conservative but prevents the collector from accidentally including the whole loop
            // body in one alternative and avoids recursion into the same SCC.
            auto branchNode = std::make_unique<ASTNode>(ASTNodeType::Branch, 0, 0);
            auto [start, end] = _regions.findRegion(current);
            branchNode->instrStart = start;
            branchNode->instrEnd = end;
            branchNode->auxPredicate = 'p';

            // Add the header basic node before the branch
            bodyNodes.push_back(createBasicNode(current));

            for (auto succ : sccSuccs) {
                // If succ is already visited, skip to avoid duplication / loops
                if (visited.count(succ) > 0) continue;

                // Mark succ visited so outer traversal won't re-visit it
                visited.insert(succ);

                // Create a simple alternative composed of the successor basic node only.
                // This matches expected AST for simple branches inside loops like your example.
                branchNode->alternatives.push_back(createBasicNode(succ));
            }

            if (!branchNode->alternatives.empty()) {
                bodyNodes.push_back(std::move(branchNode));
            }
        }
        };

    traverse(entry);

    return createSequence(std::move(bodyNodes));
}

std::unique_ptr<ASTNode> StructuredCFGBuilder::restructureAcyclic(regionid_t entry, const std::set<regionid_t>& region) {

  if (region.empty()) {
    return std::make_unique<ASTNode>(ASTNodeType::Basic, 0, 0);
  }

  if (_processed.count(entry) > 0) {
    return createBasicNode(entry);
  }

  // Check if entry is part of a loop
  for (const auto& scc: _sccs) {
    if (scc.count(entry) > 0 && _processed.count(entry) == 0) {
      auto loopNode = restructureLoop(scc, entry);

      // After loop, continue with remaining regions
      std::set<regionid_t> remaining = region;
      for (auto rid: scc) {
        remaining.erase(rid);
      }

      if (remaining.empty()) {
        return loopNode;
      }

      // Find next region after loop
      regionid_t next = RegionBuilder::NO_REGION;
      for (auto rid: scc) {
        auto succs = _regions.getSuccessors(rid);
        for (auto succ: succs) {
          if (remaining.count(succ) > 0) {
            next = succ;
            break;
          }
        }
        if (next != RegionBuilder::NO_REGION) break;
      }

      if (next != RegionBuilder::NO_REGION) {
        std::vector<std::unique_ptr<ASTNode>> nodes;
        nodes.push_back(std::move(loopNode));
        nodes.push_back(restructureAcyclic(next, remaining));
        return createSequence(std::move(nodes));
      }

      return loopNode;
    }
  }

  if (isLinear(entry, region)) {
    std::vector<std::unique_ptr<ASTNode>> nodes;

    regionid_t           current = entry;
    std::set<regionid_t> visited;

    while (current != RegionBuilder::NO_REGION && region.count(current) > 0 && visited.count(current) == 0) {

      visited.insert(current);
      nodes.push_back(createBasicNode(current));

      auto succs = _regions.getSuccessors(current);
      if (succs.empty() || succs.size() > 1) break;

      current = succs[0];

      // Don't follow back to already visited nodes
      if (visited.count(current) > 0) break;

      auto preds = _regions.getPredecessors(current);
      if (preds.size() > 1) break;
    }

    return createSequence(std::move(nodes));
  }

  // Handle branching
  auto succs = _regions.getSuccessors(entry);

  if (succs.empty()) {
    return createBasicNode(entry);
  }

  if (succs.size() == 1) {
    auto next = succs[0];
    if (region.count(next) > 0 && _processed.count(next) == 0) {
      std::vector<std::unique_ptr<ASTNode>> nodes;
      nodes.push_back(createBasicNode(entry));

      std::set<regionid_t> remaining = region;
      remaining.erase(entry);
      nodes.push_back(restructureAcyclic(next, remaining));

      return createSequence(std::move(nodes));
    }
    return createBasicNode(entry);
  }

  // Multiple successors - create branch
  auto branchNode = std::make_unique<ASTNode>(ASTNodeType::Branch, 0, 0);

  auto [start, end]        = _regions.findRegion(entry);
  branchNode->instrStart   = start;
  branchNode->instrEnd     = end;
  branchNode->auxPredicate = 'p';

  // Add entry as head
  std::vector<std::unique_ptr<ASTNode>> sequence;
  sequence.push_back(createBasicNode(entry));

  // Process each alternative
  for (auto succ: succs) {
    if (region.count(succ) > 0 && _processed.count(succ) == 0) {
      std::set<regionid_t> branchRegion;

      // Simple: just include the immediate successor
      branchRegion.insert(succ);

      auto alt = restructureAcyclic(succ, branchRegion);
      branchNode->alternatives.push_back(std::move(alt));
    }
  }

  if (!branchNode->alternatives.empty()) {
    sequence.push_back(std::move(branchNode));
    return createSequence(std::move(sequence));
  }

  return createBasicNode(entry);
}

std::unique_ptr<ASTNode> StructuredCFGBuilder::build() {
  _processed.clear();

  identifySCCs(0);

  std::set<regionid_t> allRegions;
  _regions.for_each([&](uint32_t start, uint32_t, void*) {
    auto [rid, _] = _regions.findRegion(start);
    allRegions.insert(rid);
  });

  return restructureAcyclic(0, allRegions);
}

std::unique_ptr<ASTNode> StructuredCFGBuilder::createBasicNode(regionid_t rid) {
  auto [start, end] = _regions.findRegion(rid);
  return std::make_unique<ASTNode>(ASTNodeType::Basic, start, end);
}

std::unique_ptr<ASTNode> StructuredCFGBuilder::createSequence(std::vector<std::unique_ptr<ASTNode>> nodes) {

  if (nodes.empty()) {
    return std::make_unique<ASTNode>(ASTNodeType::Basic, 0, 0);
  }

  if (nodes.size() == 1) {
    return std::move(nodes[0]);
  }

  auto seq      = std::make_unique<ASTNode>(ASTNodeType::Sequence, nodes.front()->instrStart, nodes.back()->instrEnd);
  seq->children = std::move(nodes);

  return seq;
}

bool StructuredCFGBuilder::isLinear(regionid_t entry, const std::set<regionid_t>& region) {

  regionid_t           current = entry;
  std::set<regionid_t> visited;

  while (current != RegionBuilder::NO_REGION && region.count(current) > 0) {

    if (visited.count(current) > 0) return false;
    visited.insert(current);

    auto succs = _regions.getSuccessors(current);
    if (succs.size() > 1) return false;
    if (succs.empty()) break;

    auto next = succs[0];
    if (region.count(next) == 0) break;

    auto preds = _regions.getPredecessors(next);
    if (preds.size() > 1) return false;

    current = next;
  }

  return true;
}

static std::string indent(int level) {
  return std::string(level * 2, ' ');
}

static std::string nodeTypeName(ASTNodeType type) {
  switch (type) {
    case ASTNodeType::Basic: return "Basic";
    case ASTNodeType::Branch: return "Branch";
    case ASTNodeType::Loop: return "Loop";
    case ASTNodeType::Sequence: return "Sequence";
    default: return "Unknown";
  }
}

void StructuredCFGBuilder::dumpAST(std::ostream& os, const ASTNode* node, int depth) const {
  if (!node) return;

  std::string ind       = indent(depth);
  std::string connector = depth > 0 ? "-- " : "";

  os << ind << connector << nodeTypeName(node->type) << " [" << node->instrStart << ", " << node->instrEnd << ")";

  uint32_t instrCount = node->instrEnd - node->instrStart;
  if (instrCount > 0) {
    os << " (" << instrCount << " instr)";
  }

  if (node->auxPredicate) {
    os << " {pred: " << *node->auxPredicate << "}";
  }

  os << "\n";

  switch (node->type) {
    case ASTNodeType::Branch:
      for (size_t i = 0; i < node->alternatives.size(); ++i) {
        os << indent(depth + 1) << "- Alt[" << i << "]:\n";
        dumpAST(os, node->alternatives[i].get(), depth + 2);
      }
      break;

    case ASTNodeType::Loop:
      if (node->body) {
        os << indent(depth + 1) << "- Body:\n";
        dumpAST(os, node->body.get(), depth + 2);
      }
      break;

    case ASTNodeType::Sequence:
      for (size_t i = 0; i < node->children.size(); ++i) {
        os << indent(depth + 1) << "- Step[" << i << "]:\n";
        dumpAST(os, node->children[i].get(), depth + 2);
      }
      break;

    case ASTNodeType::Basic: break;
  }
}

} // namespace compiler::ir
