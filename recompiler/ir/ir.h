#pragma once

#include "include/flags.h"
#include "types.h"

#include <array>
#include <deque>
#include <type_traits>

namespace compiler::ir {

struct Operand {
  OperandId_t    id    = 0;
  OperandKind_t  kind  = 0;
  OperandFlags_t flags = 0;

  ConstantId_t constantId = {};

  OperandType type = OperandType::i32();
};

struct alignas(16) InstCore {
  InstructionKind_t                  kind  = -1;
  eInstructionGroup                  group = eInstructionGroup::kUnknown;
  util::Flags<ir::eInstructionFlags> flags;

  InstructionUserData_t userData = 0;

  uint8_t numDst;
  uint8_t numSrc;

  OperandId_t dstStartId; ///< index into operand table
  OperandId_t srcStartId; ///< index into operand table

  inline bool isValid() const { return group != eInstructionGroup::kUnknown; }

  inline bool isConstant() const { return flags.is_set(ir::eInstructionFlags::kConstant); }
};

static_assert(sizeof(InstCore) <= 16); ///< cache lines
static_assert(config::kMaxOps <= 15);  ///< only 4 bits

// // Handle enum bits to underlying conversion
template <typename Enum>
struct Flags {
  static_assert(std::is_enum_v<Enum>, "Flags<T> requires an enum type");

  using underlying_t = std::underlying_type_t<Enum>;
  underlying_t value {};

  constexpr Flags(): value(0) {}

  constexpr Flags(Enum flag): value(static_cast<underlying_t>(flag)) {}

  constexpr Flags(underlying_t raw): value(raw) {}

  constexpr operator underlying_t() const { return value; }

  constexpr bool has(Enum flag) const { return (value & static_cast<underlying_t>(flag)) != 0; }
};

template <typename Enum>
constexpr Flags<Enum> operator|(Enum lhs, Enum rhs) {
  using U = std::underlying_type_t<Enum>;
  return Flags<Enum>(static_cast<U>(lhs) | static_cast<U>(rhs));
}

template <typename Enum>
constexpr Flags<Enum> operator|(Flags<Enum> lhs, Enum rhs) {
  return Flags<Enum>(lhs.value | static_cast<std::underlying_type_t<Enum>>(rhs));
}

template <typename Enum>
constexpr Flags<Enum> operator|(Enum lhs, Flags<Enum> rhs) {
  return Flags<Enum>(static_cast<std::underlying_type_t<Enum>>(lhs) | rhs.value);
}

struct ConstantValue {
  union {
    int64_t  value_i64;
    uint64_t value_u64;
    double   value_f64;
  };
};

class InstructionManager {
  public:
  InstructionManager(std::pmr::polymorphic_allocator<> allocator, size_t expectedInstructions = 256)
      : _instructions(allocator), _operands(allocator), _constants(allocator) {}

  InstructionId_t createInstruction(InstCore const& instr, bool isVirtual = false);

  inline InstCore& accessInstr(InstructionId_t id) { return _instructions[id]; }

  inline const InstCore& getInstr(InstructionId_t id) const { return _instructions[id]; }

  inline Operand& getDst(InstructionId_t id, uint32_t index) {
    const InstCore& inst = _instructions[id];
    return _operands[inst.dstStartId + index];
  }

  inline const Operand& getDst(InstructionId_t id, uint32_t index) const {
    const InstCore& inst = _instructions[id];
    return _operands[inst.dstStartId + index];
  }

  inline Operand& getSrc(InstructionId_t id, uint32_t index) {
    const InstCore& inst = _instructions[id];
    return _operands[inst.srcStartId + index];
  }

  inline const Operand& getSrc(InstructionId_t id, uint32_t index) const {
    const InstCore& inst = _instructions[id];
    return _operands[inst.srcStartId + index];
  }

  // void setDstValueId(InstructionId_t id, uint32_t idx, ValueId_t v) { getDst(id, idx).valueId = v; }

  // void setSrcValueId(InstructionId_t id, uint32_t idx, ValueId_t v) { getSrc(id, idx).valueId = v; }

  inline void setDstOperand(InstructionId_t id, uint32_t idx, const Operand& op) { getDst(id, idx) = op; }

  inline void setSrcOperand(InstructionId_t id, uint32_t idx, const Operand& op) { getSrc(id, idx) = op; }

  inline Operand& getOperand(OperandId_t id) { return _operands[id]; }

  inline const Operand& getOperand(OperandId_t id) const { return _operands[id]; }

  inline OperandId_t operandCount() const { return _operands.size(); }

  inline auto instructionCount() const { return _instructions.size(); }

  inline ConstantId_t createConstant(const ConstantValue& c) {
    // todo make unique
    _constants.push_back(c);
    return ConstantId_t(_constants.size() - 1);
  }

  auto& access() { return _instructions; };

  auto& get() const { return _instructions; };

  private:
  std::pmr::deque<InstCore>      _instructions; // todo boost::stable_vector?
  std::pmr::deque<Operand>       _operands;
  std::pmr::deque<ConstantValue> _constants;
};

} // namespace compiler::ir