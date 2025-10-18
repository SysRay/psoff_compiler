#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

namespace compiler::ir::cfg {

enum class NodeType : uint8_t { Block, Cond, Loop };
constexpr uint8_t NodeTypeSize = 3;

struct NodeId {
  uint16_t index : 14;
  uint16_t type  : 2;

  constexpr NodeId(uint16_t index, NodeType type): index(index), type((uint16_t)type) {}
};

constexpr NodeId InvalidNode(0x3FF, NodeType::Block);

struct NodeBase {
  NodeId prev = InvalidNode;
  NodeId next = InvalidNode;

  NodeBase(NodeType t) {}
};

struct Node {
  NodeBase base {NodeType::Block};
};

struct NodeCond {
  NodeBase          base {NodeType::Cond};
  char              predicate = '0'; // todo
  std::span<NodeId> thenBranch;
  std::span<NodeId> elseBranch;
  NodeId            merge = InvalidNode;
};

struct NodeLoop {
  NodeBase base {NodeType::Loop};

  NodeId            header = InvalidNode;
  std::span<NodeId> bodyBranch;
  NodeId            merge = InvalidNode;
};

class ControlFlow {
  public:
  ControlFlow(std::pmr::polymorphic_allocator<> allocator): _nodesBlock(allocator), _nodesCond(allocator), _nodesLoop(allocator) {}

  template <typename T, typename... Args>
  requires(std::is_same_v<T, Node> || std::is_same_v<T, NodeCond> || std::is_same_v<T, NodeLoop>)
  NodeId createNode(Args&&... args) {
    if constexpr (std::is_same_v<T, Node>) {
      auto& ref = _nodesBlock.emplace_back(std::forward<Args>(args)...);
      return NodeId(_nodesBlock.size() - 1, NodeType::Block);
    } else if constexpr (std::is_same_v<T, NodeCond>) {
      auto& ref     = _nodesCond.emplace_back(std::forward<Args>(args)...);
      ref.base.type = NodeType::Cond;
      return NodeId(_nodesBlock.size() - 1, NodeType::Cond);
    } else if constexpr (std::is_same_v<T, NodeLoop>) {
      auto& ref     = _nodesLoop.emplace_back(std::forward<Args>(args)...);
      ref.base.type = NodeType::Loop;
      return NodeId(_nodesBlock.size() - 1, NodeType::Loop);
    }
  }

  NodeBase& baseOf(NodeId id) {
    switch (static_cast<NodeType>(id.type)) {
      case NodeType::Block: return _nodesBlock[id.index].base;
      case NodeType::Cond: return _nodesCond[id.index].base;
      case NodeType::Loop: return _nodesLoop[id.index].base;
    }
  }

  private:
  std::pmr::vector<Node>     _nodesBlock;
  std::pmr::vector<NodeCond> _nodesCond;
  std::pmr::vector<NodeLoop> _nodesLoop;
};

inline void connect(ControlFlow& arena, NodeId& parent, NodeId& child) {
  auto& parentBase = arena.baseOf(parent);
  auto& childBase  = arena.baseOf(child);

  parentBase.next = child;
  childBase.prev  = parent;
}
} // namespace compiler::ir::cfg