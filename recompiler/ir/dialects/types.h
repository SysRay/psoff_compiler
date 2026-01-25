#pragma once

#include "ir/operations.h"
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

struct OpSrc {
  OperandKind_t  kind {-1};
  OperandFlags_t flags {0};
  SsaId_t        ssa = {};

  constexpr explicit OpSrc() {}

  constexpr explicit OpSrc(OperandKind_t kind, OperandFlags_t flags = 0): kind(kind), flags(flags) {}

  constexpr explicit OpSrc(SsaId_t op, OperandFlags_t flags = 0): flags(flags), ssa(op) {}

  constexpr OpSrc& operator=(OpSrc const& other) = default;
};

struct OpDst {
  OperandKind_t  kind {-1};
  OperandFlags_t flags {0};

  constexpr explicit OpDst() {}

  constexpr explicit OpDst(OperandKind_t kind, OperandFlags_t flags = 0): kind(kind), flags(flags) {}

  constexpr OpDst& operator=(OpDst const& other) = default;
};

class IRResult {
  public:
  IRResult(ir::IROperations& ir, InstructionId_t id): _id(id), _ir(ir) {}

  operator SsaId_t() const { return _ir.getDef(_id, 0); }

  operator InstructionId_t() const { return _id; }

  operator OpSrc() const { return OpSrc(_ir.getDef(_id, 0)); }

  private:
  ir::IROperations& _ir;
  InstructionId_t         _id;
};
} // namespace compiler::ir::dialect