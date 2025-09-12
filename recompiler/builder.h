#pragma once

#include "frontend/shader_input.h"
#include "include/flags.h"
#include "ir/ir.h"

#include <memory_resource>

namespace compiler {
constexpr auto operator""_MB(size_t x) -> size_t {
  return 1024L * 1024L * x;
}

constexpr size_t MEMORY_SIZE = 1_MB;

using ShaderDump_t = std::vector<uint8_t>;

enum class ShaderDebugFlags : uint16_t {
  ISDUMP = (1 << 0),
};

class Builder {
  public:
  Builder(size_t numInstructions = 0, util::Flags<ShaderDebugFlags> const& flags = {})
      : _buffer(std::make_unique_for_overwrite<uint8_t[]>(MEMORY_SIZE)), _debugFlags(flags) {
    if (numInstructions != 0) {
      numInstructions = 2048;
    }
    _instructions.reserve(numInstructions);
  }

  auto& createInstruction(ir::InstCore const& instr) { return _instructions.emplace_back(instr); }

  bool createShader(frontend::ShaderStage stage, uint32_t id, frontend::ShaderHeader const* header, uint32_t const* gpuRegs);
  bool createShader(ShaderDump_t const&);

  bool createDump(frontend::ShaderHeader const* header, uint32_t const* gpuRegs) const;

  std::string_view getName() const { return _name; }

  private:
  std::unique_ptr<uint8_t[]>          _buffer;
  std::pmr::monotonic_buffer_resource _pool {_buffer.get(), MEMORY_SIZE};

  std::pmr::vector<ir::InstCore> _instructions {&_pool};

  util::Flags<ShaderDebugFlags> _debugFlags = {};
  frontend::ShaderInput         _shaderInput;

  char _name[32] = {'\0'};
};

} // namespace compiler