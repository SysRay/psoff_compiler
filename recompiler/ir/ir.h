#pragma once

#include "include/flags.h"
#include "types.h"
#include "utility/util.h"

#include <array>
#include <deque>
#include <memory_resource>
#include <type_traits>
#include <vector>

namespace compiler::ir {

struct InputOperand {
  OperandKind_t  kind  = -1;
  OperandFlags_t flags = 0;

  SsaId_t ssaId = {};

  OperandType type = OperandType::i32();

  bool isSSA() const { return ssaId.isValid(); }
};

struct OutputOperand {
  OperandKind_t  kind  = -1;
  OperandFlags_t flags = 0;

  struct {
    SsaId_t  ssaValue    = {};
    uint32_t resultIndex = 0;
  } ssa = {};

  OperandType type = OperandType::i32();

  inline bool hasKind() const { return kind >= 0; }
};

struct alignas(16) InstCore {
  InstructionKind_t kind    = -1;
  eDialect          dialect = eDialect::kCore;

  util::Flags<ir::eInstructionFlags> flags;

  InstructionUserData_t userData = 0;

  uint8_t numDst;
  uint8_t numSrc;

  OutputOperandId_t dstStartId; ///< index into operand table

  union {
    InputOperandId_t srcStartId; ///< index into operand table

    ConstantId_t constantId;
  };

  inline bool isValid() const { return kind != -1; }

  inline bool isConstant() const { return flags.is_set(eInstructionFlags::kConstant); }

  inline auto getSrcStart() const { return srcStartId; }

  inline auto getConstantId() const { return constantId; }

  inline auto getOutputId(uint8_t n) const { return OutputOperandId_t(dstStartId + n); }

  inline auto getInputId(uint8_t n) const { return InputOperandId_t(srcStartId + n); }
};

static_assert(sizeof(InstCore) <= 16); ///< cache lines

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
  CLASS_NO_COPY(InstructionManager);
  CLASS_NO_MOVE(InstructionManager);

  public:
  InstructionManager(std::pmr::polymorphic_allocator<> allocator, size_t expectedInstructions = 256)
      : _instructions(allocator), _outputs(allocator), _inputs(allocator), _constants(allocator), _ssa(allocator) {
    //_instructions.reserve(expectedInstructions);
  }

  InstructionId_t createInstruction(InstCore const& instr, bool isVirtual = false);
  SsaId_t         createSSA(OutputOperandId_t def);

  InputOperandId_t  createInput(ir::OperandType type);
  OutputOperandId_t createOutput(ir::OperandType type);

  inline InstCore& accessInstr(InstructionId_t id) { return _instructions[id]; }

  inline const InstCore& getInstr(InstructionId_t id) const { return _instructions[id]; }

  inline OutputOperandId_t getDst(InstructionId_t id, uint32_t index) const {
    const InstCore& inst = _instructions[id];
    return OutputOperandId_t(inst.dstStartId + index);
  }

  inline InputOperandId_t getSrc(InstructionId_t id, uint32_t index) const {
    const InstCore& inst = _instructions[id];
    return InputOperandId_t(inst.srcStartId + index);
  }

  inline SsaId_t getDef(OutputOperandId_t id) const { return getOperand(id).ssa.ssaValue; }

  inline SsaId_t getDef(InstructionId_t id, uint32_t index) const { return getOperand(getDst(id, index)).ssa.ssaValue; }

  // void setDstValueId(InstructionId_t id, uint32_t idx, ValueId_t v) { getDst(id, idx).valueId = v; }

  // void setSrcValueId(InstructionId_t id, uint32_t idx, ValueId_t v) { getSrc(id, idx).valueId = v; }

  inline void setDstOperand(InstructionId_t id, uint32_t idx, const OutputOperand& op) { getOperand(getDst(id, idx)) = op; }

  inline void setSrcOperand(InstructionId_t id, uint32_t idx, const InputOperand& op) { getOperand(getSrc(id, idx)) = op; }

  inline OutputOperand& getOperand(OutputOperandId_t id) { return _outputs[id]; }

  inline OutputOperand& getOperand(SsaId_t id) { return _outputs[_ssa[id].def]; }

  inline InputOperand& getOperand(InputOperandId_t id) { return _inputs[id]; }

  inline OutputOperand const& getOperand(OutputOperandId_t id) const { return _outputs[id]; }

  inline OutputOperand const& getOperand(SsaId_t id) const { return _outputs[_ssa[id].def]; }

  inline InputOperand const& getOperand(InputOperandId_t id) const { return _inputs[id]; }

  inline auto outputsSize() const { return _outputs.size(); }

  inline auto inputsSize() const { return _inputs.size(); }

  inline auto instructionCount() const { return _instructions.size(); }

  inline ConstantId_t createConstant(const ConstantValue& c) {
    // todo make unique
    _constants.push_back(c);
    return ConstantId_t(_constants.size() - 1);
  }

  inline auto const& getConstant(ConstantId_t id) const { return _constants[id]; }

  auto& access() { return _instructions; };

  auto& get() const { return _instructions; };

  void connect(InputOperandId_t sink, SsaId_t source) {
    auto& op = getOperand(sink);
    op.ssaId = source;

    _ssa[source].uses.push_back(sink);
  }

  private:
  struct SsaValueInfo {
    OutputOperandId_t                  def {};  ///< which output defines this value
    std::pmr::vector<InputOperandId_t> uses {}; ///< all input operands that use it
  };

  std::pmr::deque<InstCore>      _instructions; // todo boost::stable_vector?
  std::pmr::deque<OutputOperand> _outputs;
  std::pmr::deque<InputOperand>  _inputs;
  std::pmr::deque<ConstantValue> _constants;
  std::pmr::deque<SsaValueInfo>  _ssa;
};

} // namespace compiler::ir