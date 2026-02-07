#pragma once
#include "compiler_types.h"
#include "frontend/shader_input.h"
#include "frontend/shader_types.h"
#include "util/flags.h"

namespace compiler {

bool createShader(frontend::ShaderStage stage, uint32_t id, frontend::ShaderHeader const* header, uint32_t const* gpuRegs,
                  util::Flags<ShaderBuildFlags> const& flags = {});
bool createShader(ShaderDump_t const&, util::Flags<ShaderBuildFlags> const& flags = {});

} // namespace compiler