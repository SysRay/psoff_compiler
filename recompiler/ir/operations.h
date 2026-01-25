#pragma once

#include "ir.h"

#include <array>
#include <deque>
#include <memory_resource>
#include <type_traits>
#include <vector>

namespace compiler::ir {

class IROperations {
  CLASS_NO_COPY(IROperations);
  CLASS_NO_MOVE(IROperations);

  public:
  IROperations(std::pmr::polymorphic_allocator<> allocator, size_t expectedInstructions = 256)
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