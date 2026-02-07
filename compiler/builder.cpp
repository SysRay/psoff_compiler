#include "builder.h"

#include "alpaca/alpaca.h"
#include "compiler_ctx.h"
#include "logging.h"

#include <filesystem>
#include <format>

namespace compiler {

struct DumpData {
  frontend::ShaderInput shaderInput;

  struct InstData {
    uint64_t              pc {};
    std::vector<uint32_t> instructions {};
  };

  std::array<InstData, 2> shaders;
};

static std::string_view getFileTpye(frontend::ShaderStage stage) {
  using namespace frontend;
  switch (stage) {
    case ShaderStage::Compute: return "cs";
    case ShaderStage::Vertex: return "vv";
    case ShaderStage::VertexExport: return "ve";
    case ShaderStage::VertexLocal: return "vl";
    case ShaderStage::Fragment: return "fs";
    case ShaderStage::Geometry: return "gs";
    case ShaderStage::Copy: return "cp";
    case ShaderStage::TessellationCtrl: return "hs";
    case ShaderStage::TessellationEval: return "te";
  }
}

static bool createDump(CompilerCtx& ctx, frontend::ShaderHeader const* header, uint32_t const* gpuRegs) {
  // // Create dump data
  DumpData data {.shaderInput = ctx.getShaderInput()};

  // Collect memory
  auto const& hostMapping = ctx.getHostMapping();
  for (uint8_t n = 0; n < hostMapping.size(); ++n) {
    auto const& itemHost = hostMapping[n];
    if (itemHost.host == 0) continue;

    auto& item = data.shaders[n];
    item.pc    = itemHost.pc;
    item.instructions.resize(itemHost.size_dw);
    std::memcpy(item.instructions.data(), (void*)itemHost.host, sizeof(uint32_t) * item.instructions.size());
  }
  // -

  std::filesystem::create_directory("shader_dumps");

  uint16_t             numBytes = 0;
  std::vector<uint8_t> binaryData;

  constexpr auto OPTIONS = alpaca::options::fixed_length_encoding;
  numBytes               = alpaca::serialize<OPTIONS>(data, binaryData);

  auto const path = std::filesystem::path("shader_dumps") / (std::string(ctx.getName()) + ".bin");
  try {
    std::ofstream file(path, std::ios::binary);
    file.write((char const*)binaryData.data(), binaryData.size());
    file.flush();
  } catch (...) {
    LOG(eLOG_TYPE::ERROR, "Couldn't dump shader {}", path.string());
    return false;
  }
  LOG(eLOG_TYPE::INFO, "dumped shader at {}", path.string());
  return true;
}

bool createShader(frontend::ShaderStage stage, uint32_t id, frontend::ShaderHeader const* header, uint32_t const* gpuRegs,
                  util::Flags<ShaderBuildFlags> buildFlags) {

  CompilerCtx compilerCtx(buildFlags);

  if (buildFlags.is_set(ShaderBuildFlags::ISDEBUG) || buildFlags.is_set(ShaderBuildFlags::WITHDUMP)) {
    compilerCtx.setName(std::format("{}_{:#x}_{}", getFileTpye(stage), header->hash0, id));
  }

  // Get shader data
  using namespace frontend;

  auto& shaderInput = compilerCtx.getShaderInput();

  shaderInput.stage = stage;
#define __INIT(name)                                                                                                                                           \
  {                                                                                                                                                            \
    name obj;                                                                                                                                                  \
    obj.init(shaderInput, header, gpuRegs);                                                                                                                    \
    shaderInput.stageData = obj;                                                                                                                               \
  }

  switch (stage) {
    case ShaderStage::Compute: __INIT(ShaderComputeData) break;
    case ShaderStage::Vertex: __INIT(ShaderVertexData) break;
    case ShaderStage::VertexExport: __INIT(ShaderVertexExportData) break;
    case ShaderStage::VertexLocal: __INIT(ShaderVertexLocalData) break;
    case ShaderStage::Fragment: __INIT(ShaderFragmentData) break;
    case ShaderStage::Geometry: __INIT(ShaderGeometryData) break;
    case ShaderStage::Copy: __INIT(ShaderCopyData) break;
    case ShaderStage::TessellationCtrl: __INIT(ShaderTessCntrlData) break;
    case ShaderStage::TessellationEval: __INIT(ShaderTessEvalData) break;
  }
#undef __INIT

  uint64_t const base = getShaderBase(shaderInput.stage, gpuRegs);

  // Register mapping
  compilerCtx.setHostMapping(0, (uint32_t const*)base, header->length / sizeof(uint32_t));
  auto result = compilerCtx.processBinary();

  if (buildFlags.is_set(ShaderBuildFlags::WITHDUMP)) {
  }
  return result;
}

bool createShader(ShaderDump_t const& dump, util::Flags<ShaderBuildFlags> buildFlags) {
  buildFlags.set(ShaderBuildFlags::ISDUMP);

  size_t         start   = 0;
  size_t         end     = dump.size();
  constexpr auto OPTIONS = alpaca::options::fixed_length_encoding;

  DumpData data;

  std::error_code ec;
  alpaca::deserialize<OPTIONS>(data, dump, start, end, ec);
  if (ec) {
    std::cerr << "Deserialization failed: " << ec.message() << "\n";
    return false;
  }

  CompilerCtx compilerCtx(buildFlags);

  auto& shaderInput = compilerCtx.getShaderInput();
  shaderInput       = data.shaderInput;

  // Register mapping
  for (uint8_t n = 0; n < data.shaders.size(); ++n) {
    auto const& item = data.shaders[n];

    compilerCtx.setHostMapping(item.pc, item.instructions.data(), item.instructions.size());
  }

  return compilerCtx.processBinary();
}
} // namespace compiler