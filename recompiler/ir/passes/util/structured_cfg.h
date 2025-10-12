#pragma once

#include "region_graph.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <set>
#include <stack>
#include <unordered_map>
#include <unordered_set>

namespace compiler::ir {

enum class ASTNodeType { Basic, Branch, Loop, Sequence };

struct ASTNode {
  ASTNodeType type;
  uint32_t    instrStart;
  uint32_t    instrEnd;

  std::unique_ptr<ASTNode>              thenBranch;
  std::unique_ptr<ASTNode>              elseBranch;
  std::vector<std::unique_ptr<ASTNode>> alternatives;
  std::unique_ptr<ASTNode>              body;
  std::vector<std::unique_ptr<ASTNode>> children;
  std::optional<char>                   auxPredicate;

  ASTNode(ASTNodeType t, uint32_t start, uint32_t end): type(t), instrStart(start), instrEnd(end) {}
};

class StructuredCFGBuilder {
  public:
  explicit StructuredCFGBuilder(const RegionBuilder& regions): _regions(regions) {}

  std::unique_ptr<ASTNode> build();

  void dumpAST(std::ostream& os, const ASTNode* node, int indent = 0) const;

  private:
  const RegionBuilder& _regions;

  struct TarjanState {
    int  index   = -1;
    int  lowlink = -1;
    bool onStack = false;
  };

  std::unordered_map<regionid_t, TarjanState> _tarjanState;
  std::stack<regionid_t>                      _tarjanStack;
  int                                         _tarjanIndex = 0;
  std::vector<std::set<regionid_t>>           _sccs;
  std::unordered_set<regionid_t>              _processed;

  void identifySCCs(regionid_t entry);
  void strongConnect(regionid_t v);

  std::unique_ptr<ASTNode> restructure(regionid_t entry, const std::set<regionid_t>& region);

  std::unique_ptr<ASTNode> restructureLoop(const std::set<regionid_t>& scc, regionid_t entry);
  std::unique_ptr<ASTNode> restructureLoopBody(regionid_t entry, const std::set<regionid_t>& scc);
  std::unique_ptr<ASTNode> restructureAcyclic(regionid_t entry, const std::set<regionid_t>& region);

  bool                 isInSCC(regionid_t rid) const;
  std::set<regionid_t> findLoopBody(const std::set<regionid_t>& scc, regionid_t entry);

  std::unique_ptr<ASTNode> createBasicNode(regionid_t rid);
  std::unique_ptr<ASTNode> createSequence(std::vector<std::unique_ptr<ASTNode>> nodes);

  bool isLinear(regionid_t entry, const std::set<regionid_t>& region);
};

} // namespace compiler::ir