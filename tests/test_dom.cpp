#include "analysis/debug_strings.h"
#include "analysis/dom.h"
#include "analysis/dom_sparse.h"

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory_resource>
#include <ranges>
#include <type_traits>
#include <vector>

// Simple test graph structure (same as example)
struct TestGraph {
  std::vector<std::vector<uint32_t>> succ, pred;

  TestGraph(size_t n): succ(n), pred(n) {}

  void addEdge(uint32_t u, uint32_t v) {
    succ[u].push_back(v);
    pred[v].push_back(u);
  }

  auto getSuccessors(uint32_t idx) const { return std::views::all(succ[idx]); }

  auto getPredecessors(uint32_t idx) const { return std::views::all(pred[idx]); }

  auto size() const { return succ.size(); }
};

template <typename Graph>
struct SparseImpl {
  using DomTree                         = compiler::analysis::DominatorTreeSparse<Graph>;
  static constexpr bool needs_allocator = true;
};

template <typename Graph>
struct DenseImpl {
  using DomTree                         = compiler::analysis::DominatorTreeDense<Graph>;
  static constexpr bool needs_allocator = false;
};

template <typename ImplWrapper>
class DominatorTreeTest: public ::testing::Test {
  protected:
  static constexpr bool needs_allocator = ImplWrapper::needs_allocator;

  auto createDomTree(TestGraph& g, size_t entry) {
    if constexpr (needs_allocator) {
      auto dom = typename ImplWrapper::DomTree(&pool);
      dom.calculate(g, entry);
      return dom;
    } else {
      auto dom = typename ImplWrapper::DomTree();
      dom.calculate(g, entry);
      return dom;
    }
  }

  std::pmr::monotonic_buffer_resource pool;
};

using Implementations = ::testing::Types<SparseImpl<TestGraph>, DenseImpl<TestGraph>>;
TYPED_TEST_SUITE(DominatorTreeTest, Implementations);

TYPED_TEST(DominatorTreeTest, SimpleLinearGraph) {
  TestGraph g(4);
  g.addEdge(0, 1);
  g.addEdge(1, 2);
  g.addEdge(2, 3);

  auto dom = this->createDomTree(g, 0);

  EXPECT_EQ(dom.get_idom(0), std::nullopt); // entry has no idom
  EXPECT_EQ(dom.get_idom(1), 0u);
  EXPECT_EQ(dom.get_idom(2), 1u);
  EXPECT_EQ(dom.get_idom(3), 2u);
}

TYPED_TEST(DominatorTreeTest, DiamondGraph) {
  //   0
  //  / \
  // 1   2
  // |   |
  // 3   |
  //  \ /
  //   4
  //   |
  //   5

  TestGraph g(6);
  g.addEdge(0, 1);
  g.addEdge(0, 2);
  g.addEdge(1, 3);
  g.addEdge(3, 4);
  g.addEdge(2, 4);
  g.addEdge(4, 5);

  auto dom = this->createDomTree(g, 0);

  EXPECT_EQ(dom.get_idom(0), std::nullopt);
  EXPECT_EQ(dom.get_idom(1), 0u);
  EXPECT_EQ(dom.get_idom(2), 0u);
  EXPECT_EQ(dom.get_idom(3), 1u);
  EXPECT_EQ(dom.get_idom(4), 0u); // both 1 and 2 dominated by 0
  EXPECT_EQ(dom.get_idom(5), 4u);
}

TYPED_TEST(DominatorTreeTest, BranchGraph) {
  //   0
  //  / \
  // 1   3
  //  \
  //   3

  TestGraph g(4);
  g.addEdge(0, 1);
  g.addEdge(0, 3);
  g.addEdge(1, 3);

  auto dom = this->createDomTree(g, 0);

  EXPECT_EQ(dom.get_idom(0), std::nullopt);
  EXPECT_EQ(dom.get_idom(1), 0u);
  EXPECT_EQ(dom.get_idom(3), 0u); // both 1 and 2 dominated by 0
}

// // POST DOMINATOR

template <typename Graph>
struct PostSparseImpl {
  using DomTree                         = compiler::analysis::PostDominatorTreeSparse<Graph>;
  static constexpr bool needs_allocator = true;
};

template <typename Graph>
struct PostDenseImpl {
  using DomTree                         = compiler::analysis::PostDominatorTreeDense<Graph>;
  static constexpr bool needs_allocator = false;
};

template <typename ImplWrapper>
class PostDominatorTreeTest: public ::testing::Test {
  protected:
  static constexpr bool needs_allocator = ImplWrapper::needs_allocator;

  auto createDomTree(TestGraph& g, size_t entry) {
    if constexpr (needs_allocator) {
      auto dom = typename ImplWrapper::DomTree({&pool});
      dom.calculate(g, entry);
      return dom;
    } else {
      auto dom = typename ImplWrapper::DomTree();
      dom.calculate(g, entry);
      return dom;
    }
  }

  std::pmr::monotonic_buffer_resource pool;
};

using PostImplementations = ::testing::Types<PostSparseImpl<TestGraph>, PostDenseImpl<TestGraph>>;
TYPED_TEST_SUITE(PostDominatorTreeTest, PostImplementations);

TYPED_TEST(PostDominatorTreeTest, DiamondGraph) {
  //   0
  //  / \
  // 1   2
  //  \ /
  //   3

  TestGraph g(4);
  g.addEdge(0, 1);
  g.addEdge(0, 2);
  g.addEdge(1, 3);
  g.addEdge(2, 3);

  auto dom = this->createDomTree(g, 3);

  EXPECT_EQ(dom.get_ipdom(0), 3);
  EXPECT_EQ(dom.get_ipdom(1), 3u);
  EXPECT_EQ(dom.get_ipdom(2), 3u);
  EXPECT_EQ(dom.get_ipdom(3), std::nullopt);
}

TYPED_TEST(PostDominatorTreeTest, BranchWithMultipleExitsIntoTail) {
  //      0 (branch)
  //     / \
  //    1   5
  //    |   |
  //    2   6
  //   / \
  //  3   4
  //  |   |
  //  6   5
  //      |
  //      6

  TestGraph g(7);
  g.addEdge(0, 1);
  g.addEdge(0, 5);
  g.addEdge(1, 2);
  g.addEdge(2, 3);
  g.addEdge(2, 4);
  g.addEdge(3, 6);
  g.addEdge(4, 5);
  g.addEdge(5, 6);

  auto dom = this->createDomTree(g, 6);

  EXPECT_EQ(dom.get_ipdom(0), 6);
  EXPECT_EQ(dom.get_ipdom(1), 2u);
  EXPECT_EQ(dom.get_ipdom(2), 6u);
  EXPECT_EQ(dom.get_ipdom(3), 6u);
}