#pragma once
#include "shader_types.h"

#include <array>
#include <string>
#include <variant>

namespace compiler::frontend {

uint64_t            getShaderBase(ShaderStage stage, uint32_t const* gpuRegs);
ShaderHeader const* getShaderHeader(const uint32_t* code);
uint32_t const*     getShaderUserData(ShaderStage, uint32_t const* gpuRegs);

struct ShaderInput;

// // Shader stage data
struct ShaderVertexData {
  uint32_t stepRate0     = 0; ///< instance rate
  uint32_t stepRate1     = 0; ///< instance rate
  uint8_t  numComponents = 0; ///< For vgpr defaults
  void     init(ShaderInput& ctx, ShaderHeader const* header, uint32_t const* gpuRegs);
};

struct ShaderVertexExportData {
  void init(ShaderInput& ctx, ShaderHeader const* header, uint32_t const* gpuRegs);
};

struct ShaderVertexLocalData {
  void init(ShaderInput& ctx, ShaderHeader const* header, uint32_t const* gpuRegs);
};

struct ShaderFragmentData {
  void init(ShaderInput& ctx, ShaderHeader const* header, uint32_t const* gpuRegs);
};

struct ShaderCopyData {
  void init(ShaderInput& ctx, ShaderHeader const* header, uint32_t const* gpuRegs);
};

struct ShaderComputeData {
  void init(ShaderInput& ctx, ShaderHeader const* header, uint32_t const* gpuRegs);
};

struct ShaderGeometryData {
  void init(ShaderInput& ctx, ShaderHeader const* header, uint32_t const* gpuRegs);
};

struct ShaderTessCntrlData {
  void init(ShaderInput& ctx, ShaderHeader const* header, uint32_t const* gpuRegs);
};

struct ShaderTessEvalData {
  void init(ShaderInput& ctx, ShaderHeader const* header, uint32_t const* gpuRegs);
};

using ShaderStageData_t = std::variant<ShaderComputeData, ShaderVertexData, ShaderVertexExportData, ShaderVertexLocalData, ShaderFragmentData,
                                       ShaderGeometryData, ShaderCopyData, ShaderTessCntrlData, ShaderTessEvalData>;

// // -

struct ShaderInput {
  ShaderStage stage;

  uint16_t totalSGPR;
  uint16_t totalVGPR;

  uint16_t                                   userSGPRSize;
  std::array<uint32_t, SPEC_TOTAL_USER_SGPR> userSGPR;

  ShaderStageData_t stageData;

  ShaderLogicalStage getLogicalStage() const;
};
} // namespace compiler::frontend