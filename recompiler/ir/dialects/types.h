#pragma once

#include "ir/ir.h"

#include <string_view>
#include <tuple>

namespace compiler::ir::dialect {
namespace internal {
struct InstDef {
  InstCore                     core;
  std::array<OutputOperand, 5> dstOperands; // adjust if needed
  std::array<InputOperand, 5>  srcOperands; // adjust if needed
  std::string_view             name;
};

namespace o {
constexpr OutputOperand i1 {.type = OperandType::i1()};
constexpr OutputOperand i32 {.type = OperandType::i32()};
constexpr OutputOperand i64 {.type = OperandType::i64()};
constexpr OutputOperand f32 {.type = OperandType::f32()};
constexpr OutputOperand f64 {.type = OperandType::f64()};
// ...
} // namespace o

namespace i {
constexpr InputOperand i1 {.type = OperandType::i1()};
constexpr InputOperand i32 {.type = OperandType::i32()};
constexpr InputOperand i64 {.type = OperandType::i64()};
constexpr InputOperand f32 {.type = OperandType::f32()};
constexpr InputOperand f64 {.type = OperandType::f64()};
} // namespace i

// ...
constexpr auto makeInstDef(eDialect dialect, InstructionKind_t kind, util::Flags<eInstructionFlags> flags, auto&& dstOps, auto&& srcOps,
                           std::string_view name) {
  // Helper to convert container to std::array
  auto toArray = []<typename Container>(const auto& container) -> Container {
    Container result {};
    size_t    i = 0;
    for (const auto& item: container) {
      result[i++] = item;
    }
    return result;
  };

  return InstDef {.core =
                      InstCore {
                          .kind    = kind,
                          .dialect = dialect,
                          .flags   = flags,
                          .numDst  = (uint8_t)dstOps.size(),
                          .numSrc  = (uint8_t)srcOps.size(),
                      },
                  .dstOperands = toArray.template operator()<decltype(InstDef::dstOperands)>(dstOps),
                  .srcOperands = toArray.template operator()<decltype(InstDef::srcOperands)>(srcOps),
                  .name        = name};
}

} // namespace internal

class IRResult {
  public:
  IRResult(ir::InstructionManager& ir, InstructionId_t id): _id(id), _ir(ir) {}

  operator SsaId_t() const { return _ir.getDef(_id, 0); }

  operator InstructionId_t() const { return _id; }

  private:
  ir::InstructionManager& _ir;
  InstructionId_t         _id;
};
} // namespace compiler::ir::dialect