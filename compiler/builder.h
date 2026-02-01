#pragma once
#include "frontend/shader_input.h"
#include "frontend/shader_types.h"
#include "util/flags.h"

#include <array>
#include <string_view>
#include <vector>

// mlir
#include <mlir/IR/MLIRContext.h>

constexpr auto operator""_MB(unsigned long long x) -> size_t {
  return 1024L * 1024L * x;
}

namespace compiler {

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
  Builder(util::Flags<ShaderBuildFlags> const& flags = {});

  bool createShader(frontend::ShaderStage stage, uint32_t id, frontend::ShaderHeader const* header, uint32_t const* gpuRegs);
  bool createShader(ShaderDump_t const&);
  bool createDump(frontend::ShaderHeader const* header, uint32_t const* gpuRegs) const;

  // // Getter
  std::string_view getName() const { return _name.data(); }

  HostMapping* getHostMapping(uint64_t pc);

  auto getContext() { return &_mlirCtx; }

  // // Getter for flags
  template <ShaderBuildFlags item>
  constexpr bool is_set() const {
    return _debugFlags.is_set(item);
  }

  void print() const;

  private:
  bool processBinary();

  private:
  util::Flags<ShaderBuildFlags> _debugFlags = {};
  std::array<HostMapping, 2>    _hostMapping {};
  frontend::ShaderInput         _shaderInput;

  mlir::MLIRContext _mlirCtx;

  std::array<char, 32> _name = {"dump"};
};

} // namespace compiler