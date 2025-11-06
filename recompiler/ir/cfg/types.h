#pragma once

#include <assert.h>
#include <cstdint>
#include <ostream>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

namespace compiler::ir::cfg {

enum class NodeType : uint8_t { Block = 0, Cond, Loop };
constexpr uint8_t NodeTypeSize = 3;

struct NodeId {
  union {
    struct {
      uint16_t index : 14;
      uint16_t type  : 2;
    };

    uint16_t raw;
  };

  constexpr NodeId(uint16_t index, NodeType type): index(index), type((uint16_t)type) {}

  constexpr NodeId(uint16_t raw): raw(raw) {}

  bool operator==(NodeId rhs) const { return index == rhs.index && type == rhs.type; }
};

constexpr NodeId InvalidNode(0x3FF, NodeType::Block);

struct NodeBase {
  NodeId prev = InvalidNode;
  NodeId next = InvalidNode;

  NodeType type;

  NodeBase(NodeType t): type(t) {}
};

struct NodeBlock: public NodeBase {
  NodeBlock(): NodeBase(NodeType::Block) {}
};

struct NodeCond: public NodeBase {

  NodeId ifBranchFront = InvalidNode, ifBranchBack = InvalidNode;
  NodeId elseBranchFront = InvalidNode, elseBranchBack = InvalidNode;

  NodeCond(): NodeBase(NodeType::Cond) {}
};

struct NodeLoop: public NodeBase {
  NodeId header    = InvalidNode; ///< entry node
  NodeId bodyFront = InvalidNode, bodyBack = InvalidNode;

  NodeLoop(): NodeBase(NodeType::Loop) {}
};

class ControlFlow {
  public:
  ControlFlow(std::pmr::polymorphic_allocator<> allocator): _allocator(allocator), _nodesBlock(allocator), _nodesCond(allocator), _nodesLoop(allocator) {}

  template <typename T, typename... Args>
  requires(std::is_same_v<T, NodeBlock> || std::is_same_v<T, NodeCond> || std::is_same_v<T, NodeLoop>)
  NodeId createNode(Args&&... args) {
    if constexpr (std::is_same_v<T, NodeBlock>) {
      auto& ref = _nodesBlock.emplace_back(std::forward<Args>(args)...);
      return NodeId(_nodesBlock.size() - 1, NodeType::Block);
    } else if constexpr (std::is_same_v<T, NodeCond>) {
      auto& ref = _nodesCond.emplace_back(std::forward<Args>(args)...);
      return NodeId(_nodesCond.size() - 1, NodeType::Cond);
    } else if constexpr (std::is_same_v<T, NodeLoop>) {
      auto& ref = _nodesLoop.emplace_back(std::forward<Args>(args)...);
      return NodeId(_nodesLoop.size() - 1, NodeType::Loop);
    }
  }

  auto getType(NodeId id) const { return static_cast<NodeType>(id.type); }

  NodeBase& getBase(NodeId id) {
    switch (static_cast<NodeType>(id.type)) {
      case NodeType::Block: return _nodesBlock[id.index];
      case NodeType::Cond: return _nodesCond[id.index];
      case NodeType::Loop: return _nodesLoop[id.index];
    }
  }

  template <typename T>
  requires(std::is_same_v<T, NodeBlock> || std::is_same_v<T, NodeCond> || std::is_same_v<T, NodeLoop>)
  auto& getNode(NodeId id) {
    if constexpr (std::is_same_v<T, NodeBlock>) {
      assert(getType(id) == NodeType::Block);
    } else if constexpr (std::is_same_v<T, NodeCond>) {
      assert(getType(id) == NodeType::Cond);
    } else if constexpr (std::is_same_v<T, NodeLoop>) {
      assert(getType(id) == NodeType::Loop);
    }
    return (T&)getBase(id);
  }

  inline void connect(NodeId parent, NodeId child) {
    auto& parentBase = getBase(parent);
    auto& childBase  = getBase(child);

    parentBase.next = child;
    childBase.prev  = parent;
  }

  void dump(std::ostream& os) {
    auto cur = NodeId(0, NodeType::Block);
    os << "{\n";
    do {
      auto& base = getBase(cur);

      switch (base.type) {
        case NodeType::Block: {
          auto& node = getNode<NodeBlock>(cur);
          os << "Block:" << cur.index;
        } break;
        case NodeType::Cond: {
          auto& node = getNode<NodeCond>(cur);
          os << "if(" << ") {\n";

          os << "   }\n";

          if (node.elseBranchFront != InvalidNode) {
            os << "else(" << ") {\n";

            os << "   }\n";
          }
        } break;
        case NodeType::Loop: {
          auto& node = getNode<NodeCond>(cur);
          os << "while(" << ") {\n";

          os << "   }\n";
        } break;
      }

      cur = base.next;
    } while (cur != InvalidNode);
    os << "}\n";
  }

  private:
  std::pmr::polymorphic_allocator<> _allocator;

  std::pmr::vector<NodeBlock> _nodesBlock;
  std::pmr::vector<NodeCond>  _nodesCond;
  std::pmr::vector<NodeLoop>  _nodesLoop;
};
} // namespace compiler::ir::cfg