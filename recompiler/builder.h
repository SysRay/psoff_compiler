#pragma once

#include "frontend/shader_input.h"
#include "include/flags.h"
#include "ir/ir.h"

#include <array>
#include <memory_resource>

namespace compiler {
constexpr auto operator""_MB(size_t x) -> size_t {
  return 1024L * 1024L * x;
}

constexpr size_t MEMORY_SIZE = 1_MB;

using ShaderDump_t = std::vector<uint8_t>;

enum class ShaderBuildFlags : uint16_t {
  ISDUMP = (1 << 0),
  ISNEO  = (1 << 1),
};

struct HostMapping {
  uint64_t pc      = 0;
  uint64_t host    = 0;
  uint32_t size_dw = 0;
};

uint64_t getAddr(uint64_t addr); ///< User implementation

class Builder {
  public:
  Builder(size_t numInstructions = 0, util::Flags<ShaderBuildFlags> const& flags = {})
      : _buffer(std::make_unique_for_overwrite<uint8_t[]>(MEMORY_SIZE)), _debugFlags(flags) {
    if (numInstructions != 0) {
      numInstructions = 2048;
    }
    _instructions.reserve(numInstructions);
  }

  bool createShader(frontend::ShaderStage stage, uint32_t id, frontend::ShaderHeader const* header, uint32_t const* gpuRegs);
  bool createShader(ShaderDump_t const&);
  bool createDump(frontend::ShaderHeader const* header, uint32_t const* gpuRegs) const;

  // // Create instructions
  auto& createInstruction(ir::InstCore&& instr) { return _instructions.emplace_back(std::move(instr)); }

  auto& createVirtualInst(ir::InstCore&& instr) {
    instr.flags |= ir::Flags<ir::eInstructionFlags>(ir::eInstructionFlags::kVirtual);
    return _instructions.emplace_back(std::move(instr));
  }

  // auto& createInstruction(ir::InstCore& instr) { return _instructions.emplace_back(instr); }

  // // Getter
  std::string_view getName() const { return _name; }

  inline auto const& getShaderInput() const { return _shaderInput; }

  HostMapping* getHostMapping(uint64_t pc);

  // // Getter for flags
  template <ShaderBuildFlags item>
  constexpr bool is_set() const {
    return _debugFlags.is_set(item);
  }

  void print() const;

  private:
  bool processBinary();

  private:
  std::unique_ptr<uint8_t[]>          _buffer;
  std::pmr::monotonic_buffer_resource _pool {_buffer.get(), MEMORY_SIZE};

  std::pmr::vector<ir::InstCore> _instructions {&_pool};

  util::Flags<ShaderBuildFlags> _debugFlags = {};
  frontend::ShaderInput         _shaderInput;

  std::array<HostMapping, 2> _hostMapping {};

  char _name[32] = {"dump"};
};

} // namespace compiler