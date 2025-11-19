#pragma once
#include "frontend/parser/shader_input.h"
#include "include/checkpoint_resource.h"
#include "include/flags.h"

#include <array>
#include <memory_resource>

constexpr auto operator""_MB(size_t x) -> size_t {
  return 1024L * 1024L * x;
}

namespace compiler {

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
  Builder(util::Flags<ShaderBuildFlags> const& flags = {}): _bufferTemp(std::make_unique_for_overwrite<uint8_t[]>(MEMORY_SIZE)), _debugFlags(flags) {}

  bool createShader(frontend::ShaderStage stage, uint32_t id, frontend::ShaderHeader const* header, uint32_t const* gpuRegs);
  bool createShader(ShaderDump_t const&);
  bool createDump(frontend::ShaderHeader const* header, uint32_t const* gpuRegs) const;

  // // Getter
  std::string_view getName() const { return _name; }

  inline auto const& getShaderInput() const { return _shaderInput; }

  HostMapping* getHostMapping(uint64_t pc);

  auto* getTempBuffer() { return &_poolTemp; }

  auto* getBuffer() { return &_pool; }

  // // Getter for flags
  template <ShaderBuildFlags item>
  constexpr bool is_set() const {
    return _debugFlags.is_set(item);
  }

  void print() const;

  private:
  bool processBinary();

  private:
  std::pmr::monotonic_buffer_resource _pool {MEMORY_SIZE};

  std::unique_ptr<uint8_t[]> _bufferTemp;
  util::checkpoint_resource  _poolTemp {_bufferTemp.get(), MEMORY_SIZE};

  util::Flags<ShaderBuildFlags> _debugFlags = {};
  frontend::ShaderInput         _shaderInput;

  std::array<HostMapping, 2> _hostMapping {};

  char _name[32] = {"dump"};
};

} // namespace compiler