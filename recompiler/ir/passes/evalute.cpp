#include "builder.h"
#include "frontend/frontend.h"
#include "passes.h"
#include "util/region_graph.h"

#include <set>
#include <span>
#include <stack>
#include <unordered_map>
#include <vector>

namespace compiler::ir::passes {

class Evaluate {
  public:
  Evaluate(Builder& builder, RegionBuilder& regions)
      : _checkpoint(&builder.getTempBuffer()), _builder(builder), _regions(regions), _visited(&_checkpoint), _cache(&_checkpoint) {}

  ~Evaluate() = default;

  std::optional<InstConstant> check(uint32_t index, Operand const& reg) {
    auto const [start, end] = _regions.findRegion(index);
    return findInstructionIterative(reg, index, start);
  }

  private:
  using CacheKey = std::pair<uint32_t, regionid_t>;

  struct CacheHash {
    size_t operator()(CacheKey const& k) const noexcept { return (static_cast<size_t>(k.first) << 32) ^ k.second; }
  };

  std::pmr::monotonic_buffer_resource _checkpoint;
  Builder&                            _builder;
  RegionBuilder&                      _regions;

  std::pmr::set<regionid_t>                                                 _visited;
  std::pmr::unordered_map<CacheKey, std::optional<InstConstant>, CacheHash> _cache;

  std::optional<InstConstant> evaluate(ir::InstCore const& instr, std::span<InstConstant> inputs) {
    // TODO: implement actual instruction evaluation
    return inputs[0];
  }

  std::optional<InstConstant> checkIterative(uint32_t startIndex, regionid_t region) {
    struct Frame {
      uint32_t                                     index;
      regionid_t                                   region;
      uint8_t                                      nextSrc;
      std::array<InstConstant, config::kMaxSrcOps> inputs;
    };

    auto const& instructions = _builder.getInstructions();

    std::stack<Frame, std::vector<Frame>> stack;
    stack.push({startIndex, region, 0, {}});

    while (!stack.empty()) {
      auto& frame                             = stack.top();
      auto [index, curRegion, srcIdx, inputs] = frame;

      CacheKey key {index, curRegion};
      if (auto it = _cache.find(key); it != _cache.end()) {
        auto cached = it->second;
        stack.pop();
        if (!stack.empty() && cached) {
          auto& parent                    = stack.top();
          parent.inputs[parent.nextSrc++] = *cached;
        }
        continue;
      }

      auto const& instr = instructions[index];

      // Constant — directly return
      if (instr.isConstant()) {
        _cache[key] = instr.srcConstant;
        stack.pop();
        if (!stack.empty()) {
          auto& parent                    = stack.top();
          parent.inputs[parent.nextSrc++] = instr.srcConstant;
        }
        continue;
      }

      // Still have sources to resolve
      if (frame.nextSrc < instr.numSrc) {
        auto const& op       = instr.srcOperands[frame.nextSrc];
        auto        srcConst = findInstructionIterative(op, index, curRegion);
        if (!srcConst) {
          _cache[key] = std::nullopt;
          stack.pop();
          continue;
        }
        frame.inputs[frame.nextSrc++] = *srcConst;
        continue;
      }

      // All inputs ready, evaluate
      auto result = evaluate(instr, std::span(frame.inputs.data(), instr.numSrc));
      _cache[key] = result;
      stack.pop();

      // Pass result to parent
      if (!stack.empty() && result) {
        auto& parent                    = stack.top();
        parent.inputs[parent.nextSrc++] = *result;
      }
    }

    if (auto it = _cache.find({startIndex, region}); it != _cache.end()) return it->second;
    return std::nullopt;
  }

  std::optional<InstConstant> findInstructionIterative(Operand const& reg, uint32_t index, regionid_t region) {
    auto const& instructions = _builder.getInstructions();
    auto        regBase      = frontend::eOperandKind::import(reg.kind).base();

    for (int64_t i = index - 1; i >= static_cast<int64_t>(region); --i) {
      auto const& instr = instructions[i];
      for (uint8_t d = 0; d < instr.numDst; ++d) {
        auto dstBase = frontend::eOperandKind::import(instr.dstOperands[d].kind).base();
        if (dstBase == regBase) {
          return checkIterative(static_cast<uint32_t>(i), region);
        }
      }
    }

    // TODO: check predecessors
    return std::nullopt;
  }
};

std::optional<InstConstant> evaluate(Builder& builder, RegionBuilder& regions, uint32_t index, Operand const& reg) {
  return Evaluate(builder, regions).check(index, reg);
}
} // namespace compiler::ir::passes