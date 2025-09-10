#include "shader_input.h"

#include "include/si_ci_vi_merged_offset.h"
#include "parser/registers.h"

namespace compiler::frontend {
extern uint64_t getAddr(uint64_t addr);

uint64_t getShaderBase(ShaderStage stage, uint32_t const* gpuRegs) {
  using namespace Pal::Gfx6::Chip;
  switch (stage) {
    case ShaderStage::Compute: return getAddr(((uint64_t)gpuRegs[mmCOMPUTE_PGM_HI] << 40u) | ((uint64_t)gpuRegs[mmCOMPUTE_PGM_LO] << 8u));
    case ShaderStage::Vertex: return getAddr(((uint64_t)gpuRegs[mmSPI_SHADER_PGM_HI_VS] << 40u) | ((uint64_t)gpuRegs[mmSPI_SHADER_PGM_LO_VS] << 8u));
    case ShaderStage::VertexExport: return getAddr(((uint64_t)gpuRegs[mmSPI_SHADER_PGM_HI_ES] << 40u) | ((uint64_t)gpuRegs[mmSPI_SHADER_PGM_LO_ES] << 8u));
    case ShaderStage::VertexLocal: return getAddr(((uint64_t)gpuRegs[mmSPI_SHADER_PGM_HI_LS] << 40u) | ((uint64_t)gpuRegs[mmSPI_SHADER_PGM_LO_LS] << 8u));
    case ShaderStage::Fragment: return getAddr(((uint64_t)gpuRegs[mmSPI_SHADER_PGM_HI_PS] << 40u) | ((uint64_t)gpuRegs[mmSPI_SHADER_PGM_LO_PS] << 8u));
    case ShaderStage::Geometry: return getAddr(((uint64_t)gpuRegs[mmSPI_SHADER_PGM_HI_GS] << 40u) | ((uint64_t)gpuRegs[mmSPI_SHADER_PGM_LO_GS] << 8u));
    case ShaderStage::Copy: return getAddr(((uint64_t)gpuRegs[mmSPI_SHADER_PGM_HI_VS] << 40u) | ((uint64_t)gpuRegs[mmSPI_SHADER_PGM_LO_VS] << 8u));
    case ShaderStage::TessellationCtrl: return getAddr(((uint64_t)gpuRegs[mmSPI_SHADER_PGM_HI_HS] << 40u) | ((uint64_t)gpuRegs[mmSPI_SHADER_PGM_LO_HS] << 8u));
    case ShaderStage::TessellationEval: return getAddr(((uint64_t)gpuRegs[mmSPI_SHADER_PGM_HI_VS] << 40u) | ((uint64_t)gpuRegs[mmSPI_SHADER_PGM_LO_VS] << 8u));
  }
}

ShaderLogicalStage ShaderInput::getLogicalStage() const {
  switch (stage) {
    case ShaderStage::Compute: return ShaderLogicalStage::Compute;
    case ShaderStage::Vertex: return ShaderLogicalStage::Vertex;
    case ShaderStage::VertexExport: return ShaderLogicalStage::Vertex;
    case ShaderStage::VertexLocal: return ShaderLogicalStage::Vertex;
    case ShaderStage::Fragment: return ShaderLogicalStage::Fragment;
    case ShaderStage::Geometry: return ShaderLogicalStage::Geometry;
    case ShaderStage::Copy: return ShaderLogicalStage::Geometry;
    case ShaderStage::TessellationCtrl: return ShaderLogicalStage::Tesselation;
    case ShaderStage::TessellationEval: return ShaderLogicalStage::Vertex;
  }
}

uint32_t const* getShaderUserData(ShaderStage stage, uint32_t const* gpuRegs) {
  // todo get correct offset for copy and tess
  using namespace Pal::Gfx6::Chip;
  switch (stage) {
    case ShaderStage::Compute: return &gpuRegs[mmCOMPUTE_USER_DATA_0];
    case ShaderStage::Vertex: return &gpuRegs[mmSPI_SHADER_USER_DATA_VS_0];
    case ShaderStage::VertexExport: return &gpuRegs[mmSPI_SHADER_USER_DATA_ES_0];
    case ShaderStage::VertexLocal: return &gpuRegs[mmSPI_SHADER_USER_DATA_LS_0];
    case ShaderStage::Fragment: return &gpuRegs[mmSPI_SHADER_USER_DATA_PS_0];
    case ShaderStage::Geometry: return &gpuRegs[mmSPI_SHADER_USER_DATA_GS_0];
    case ShaderStage::Copy: return &gpuRegs[mmSPI_SHADER_USER_DATA_VS_0];
    case ShaderStage::TessellationCtrl: return &gpuRegs[mmSPI_SHADER_USER_DATA_HS_0];
    case ShaderStage::TessellationEval: return &gpuRegs[mmSPI_SHADER_USER_DATA_HS_0];
  }
}

template <typename T, uint32_t offset>
uint16_t getTotalVgprs(uint32_t const* gpuRegs) {
  using namespace parser;
  auto const reg = T(gpuRegs[offset]);
  return 4 * (1 + reg.template get<T::Field::VGPRS>());
}

template <typename T, uint32_t offset>
uint16_t getTotalSgprs(uint32_t const* gpuRegs) {
  using namespace parser;
  auto const reg = T(gpuRegs[offset]);
  return 6 + 8 * reg.template get<T::Field::SGPRS>();
}

template <typename T, uint32_t offset>
uint16_t getTotalUserSgprs(uint32_t const* gpuRegs) {
  using namespace parser;
  auto const reg = T(gpuRegs[offset]);
  return reg.template get<T::Field::USER_SGPR>();
}

ShaderHeader const* getShaderHeader(const uint32_t* code) {
  if (code[0] == 0xBEEB03FF) return reinterpret_cast<const ShaderHeader*>(code + 2 * static_cast<size_t>(code[1] + 1));

  // search signature 0x4f, 0x72, 0x62, 0x53, 0x68, 0x64, 0x72
  auto base = code;

  while (true) {
    if (*base == 0x5362724f && (0xFFFFFF & base[1]) == 0x00726468) break;
    ++base;
  }
  return reinterpret_cast<const ShaderHeader*>(base);
}

void ShaderVertexData::init(ShaderInput& ctx, ShaderHeader const* header, uint32_t const* gpuRegs) {
  using namespace Pal::Gfx6::Chip;

  ctx.totalSGPR = getTotalSgprs<parser::SPI_SHADER_PGM_RSRC1_VS, mmSPI_SHADER_PGM_RSRC1_VS>(gpuRegs);
  ctx.totalVGPR = getTotalVgprs<parser::SPI_SHADER_PGM_RSRC1_VS, mmSPI_SHADER_PGM_RSRC1_VS>(gpuRegs);

  ctx.userSGPRSize     = getTotalUserSgprs<parser::SPI_SHADER_PGM_RSRC2_VS, mmSPI_SHADER_PGM_RSRC2_VS>(gpuRegs);
  auto const* userData = &gpuRegs[mmSPI_SHADER_USER_DATA_VS_0];
  for (uint8_t n = 0; n < ctx.userSGPRSize; ++n) {
    ctx.userSGPR[n] = userData[n];
  }

  auto const reg = parser::SPI_SHADER_PGM_RSRC1_VS(gpuRegs[mmSPI_SHADER_PGM_RSRC1_VS]);
  numComponents  = reg.template get<parser::SPI_SHADER_PGM_RSRC1_VS::Field::VGPR_COMP_CNT>();

  stepRate0 = gpuRegs[mmVGT_INSTANCE_STEP_RATE_0];
  stepRate1 = gpuRegs[mmVGT_INSTANCE_STEP_RATE_1];
}

void ShaderVertexExportData::init(ShaderInput& ctx, ShaderHeader const* header, uint32_t const* gpuRegs) {
  using namespace Pal::Gfx6::Chip;

  ctx.totalSGPR = getTotalSgprs<parser::SPI_SHADER_PGM_RSRC1_ES, mmSPI_SHADER_PGM_RSRC1_ES>(gpuRegs);
  ctx.totalVGPR = getTotalVgprs<parser::SPI_SHADER_PGM_RSRC1_ES, mmSPI_SHADER_PGM_RSRC1_ES>(gpuRegs);

  ctx.userSGPRSize     = getTotalUserSgprs<parser::SPI_SHADER_PGM_RSRC2_ES, mmSPI_SHADER_PGM_RSRC2_ES>(gpuRegs);
  auto const* userData = &gpuRegs[mmSPI_SHADER_USER_DATA_ES_0];
  for (uint8_t n = 0; n < ctx.userSGPRSize; ++n)
    ctx.userSGPR[n] = userData[n];
}

void ShaderVertexLocalData::init(ShaderInput& ctx, ShaderHeader const* header, uint32_t const* gpuRegs) {
  using namespace Pal::Gfx6::Chip;

  ctx.totalSGPR = getTotalSgprs<parser::SPI_SHADER_PGM_RSRC1_LS, mmSPI_SHADER_PGM_RSRC1_LS>(gpuRegs);
  ctx.totalVGPR = getTotalVgprs<parser::SPI_SHADER_PGM_RSRC1_LS, mmSPI_SHADER_PGM_RSRC1_LS>(gpuRegs);

  ctx.userSGPRSize     = getTotalUserSgprs<parser::SPI_SHADER_PGM_RSRC2_LS, mmSPI_SHADER_PGM_RSRC2_LS>(gpuRegs);
  auto const* userData = &gpuRegs[mmSPI_SHADER_USER_DATA_LS_0];
  for (uint8_t n = 0; n < ctx.userSGPRSize; ++n)
    ctx.userSGPR[n] = userData[n];
}

void ShaderFragmentData::init(ShaderInput& ctx, ShaderHeader const* header, uint32_t const* gpuRegs) {
  using namespace Pal::Gfx6::Chip;

  ctx.totalSGPR = getTotalSgprs<parser::SPI_SHADER_PGM_RSRC1_PS, mmSPI_SHADER_PGM_RSRC1_PS>(gpuRegs);
  ctx.totalVGPR = getTotalVgprs<parser::SPI_SHADER_PGM_RSRC1_PS, mmSPI_SHADER_PGM_RSRC1_PS>(gpuRegs);

  ctx.userSGPRSize     = getTotalUserSgprs<parser::SPI_SHADER_PGM_RSRC2_PS, mmSPI_SHADER_PGM_RSRC2_PS>(gpuRegs);
  auto const* userData = &gpuRegs[mmSPI_SHADER_USER_DATA_PS_0];
  for (uint8_t n = 0; n < ctx.userSGPRSize; ++n)
    ctx.userSGPR[n] = userData[n];
}

void ShaderComputeData::init(ShaderInput& ctx, ShaderHeader const* header, uint32_t const* gpuRegs) {
  using namespace Pal::Gfx6::Chip;

  ctx.totalSGPR = getTotalSgprs<parser::COMPUTE_PGM_RSRC1, mmCOMPUTE_PGM_RSRC1>(gpuRegs);
  ctx.totalVGPR = getTotalVgprs<parser::COMPUTE_PGM_RSRC1, mmCOMPUTE_PGM_RSRC1>(gpuRegs);

  ctx.userSGPRSize     = getTotalUserSgprs<parser::COMPUTE_PGM_RSRC2, mmCOMPUTE_PGM_RSRC2>(gpuRegs);
  auto const* userData = &gpuRegs[mmCOMPUTE_USER_DATA_0];
  for (uint8_t n = 0; n < ctx.userSGPRSize; ++n)
    ctx.userSGPR[n] = userData[n];
}

void ShaderGeometryData::init(ShaderInput& ctx, ShaderHeader const* header, uint32_t const* gpuRegs) {
  using namespace Pal::Gfx6::Chip;

  ctx.totalSGPR = getTotalSgprs<parser::SPI_SHADER_PGM_RSRC1_GS, mmSPI_SHADER_PGM_RSRC1_GS>(gpuRegs);
  ctx.totalVGPR = getTotalVgprs<parser::SPI_SHADER_PGM_RSRC1_GS, mmSPI_SHADER_PGM_RSRC1_GS>(gpuRegs);

  ctx.userSGPRSize     = getTotalUserSgprs<parser::SPI_SHADER_PGM_RSRC2_GS, mmSPI_SHADER_PGM_RSRC2_GS>(gpuRegs);
  auto const* userData = &gpuRegs[mmSPI_SHADER_USER_DATA_GS_0];
  for (uint8_t n = 0; n < ctx.userSGPRSize; ++n)
    ctx.userSGPR[n] = userData[n];
}

void ShaderCopyData::init(ShaderInput& ctx, ShaderHeader const* header, uint32_t const* gpuRegs) {
  using namespace Pal::Gfx6::Chip;

  ctx.totalSGPR = getTotalSgprs<parser::SPI_SHADER_PGM_RSRC1_VS, mmSPI_SHADER_PGM_RSRC1_VS>(gpuRegs);
  ctx.totalVGPR = getTotalVgprs<parser::SPI_SHADER_PGM_RSRC1_VS, mmSPI_SHADER_PGM_RSRC1_VS>(gpuRegs);

  ctx.userSGPRSize     = getTotalUserSgprs<parser::SPI_SHADER_PGM_RSRC2_VS, mmSPI_SHADER_PGM_RSRC2_VS>(gpuRegs);
  auto const* userData = &gpuRegs[mmSPI_SHADER_USER_DATA_VS_0];
  for (uint8_t n = 0; n < ctx.userSGPRSize; ++n)
    ctx.userSGPR[n] = userData[n];
}

void ShaderTessCntrlData::init(ShaderInput& ctx, ShaderHeader const* header, uint32_t const* gpuRegs) {
  using namespace Pal::Gfx6::Chip;

  ctx.totalSGPR = getTotalSgprs<parser::SPI_SHADER_PGM_RSRC1_HS, mmSPI_SHADER_PGM_RSRC1_HS>(gpuRegs);
  ctx.totalVGPR = getTotalVgprs<parser::SPI_SHADER_PGM_RSRC1_HS, mmSPI_SHADER_PGM_RSRC1_HS>(gpuRegs);

  ctx.userSGPRSize     = getTotalUserSgprs<parser::SPI_SHADER_PGM_RSRC2_HS, mmSPI_SHADER_PGM_RSRC2_HS>(gpuRegs);
  auto const* userData = &gpuRegs[mmSPI_SHADER_USER_DATA_HS_0];
  for (uint8_t n = 0; n < ctx.userSGPRSize; ++n)
    ctx.userSGPR[n] = userData[n];
}

void ShaderTessEvalData::init(ShaderInput& ctx, ShaderHeader const* header, uint32_t const* gpuRegs) {
  using namespace Pal::Gfx6::Chip;

  ctx.totalSGPR = getTotalSgprs<parser::SPI_SHADER_PGM_RSRC1_VS, mmSPI_SHADER_PGM_RSRC1_VS>(gpuRegs);
  ctx.totalVGPR = getTotalVgprs<parser::SPI_SHADER_PGM_RSRC1_VS, mmSPI_SHADER_PGM_RSRC1_VS>(gpuRegs);

  ctx.userSGPRSize     = getTotalUserSgprs<parser::SPI_SHADER_PGM_RSRC2_VS, mmSPI_SHADER_PGM_RSRC2_VS>(gpuRegs);
  auto const* userData = &gpuRegs[mmSPI_SHADER_USER_DATA_HS_0];
  for (uint8_t n = 0; n < ctx.userSGPRSize; ++n)
    ctx.userSGPR[n] = userData[n];
}

} // namespace compiler::frontend
