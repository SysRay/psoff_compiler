#pragma once
#include "compiler_types.h"
#include "frontend/shader_input.h"
#include "frontend/shader_types.h"
#include "operand_types.h"
#include "util/flags.h"

#include <array>
#include <string_view>
#include <vector>

// mlir
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

namespace compiler {

struct HostMapping {
  uint64_t pc      = std::numeric_limits<uint64_t>::max();
  uint64_t host    = 0;
  uint32_t size_dw = 0;
};

class CompilerCtx {
  public:
  CompilerCtx(util::Flags<ShaderBuildFlags> const& flags = {});

  std::string_view getName() const { return _name.data(); }

  void setName(std::string_view name) {
    auto const pos = name.copy(_name.data(), sizeof(_name) - 1);
    _name[pos]     = '\0';
  }

  HostMapping* getHostMapping(uint64_t pc);

  auto& getHostMapping() const { return _hostMapping; }

  void setHostMapping(uint64_t pc, uint32_t const* vaddr, uint32_t size_dw = 0);

  auto getContext() { return &_mlirCtx; }

  auto getModule() { return &_mlirModule; }

  auto& getShaderInput() { return _shaderInput; }

  auto& getShaderInput() const { return _shaderInput; }

  template <ShaderBuildFlags item>
  constexpr bool is_set() const {
    return _debugFlags.is_set(item);
  }

  bool processBinary();

  auto& types() const { return _types; }

  private:
  protected:
  util::Flags<ShaderBuildFlags> _debugFlags = {};

  std::array<HostMapping, 4> _hostMapping {};
  frontend::ShaderInput      _shaderInput;
  mlir::MLIRContext          _mlirCtx;
  mlir::ModuleOp             _mlirModule;

  OperandTypeCache _types;

  std::array<char, 32> _name = {"main"};
};

} // namespace compiler