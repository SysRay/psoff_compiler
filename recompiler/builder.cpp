#include "builder.h"

#include "alpaca/alpaca.h"

#include <filesystem>

namespace compiler {
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

bool Builder::createShader(frontend::ShaderStage stage, uint32_t id, frontend::ShaderHeader const* header, uint32_t const* gpuRegs) {
  { // Create name
    size_t const len = std::format("{}_{:#x}_{}", getFileTpye(stage), header->hash0, id).copy(_name, sizeof(_name) - 1);
    _name[len]       = '\0';
  }

  return false;
}

bool Builder::createShader(ShaderDump_t const& dump) {
  _debugFlags.set(ShaderDebugFlags::ISDUMP);
  return false;
}

struct DumpData {
  frontend::ShaderInput shaderInput;
  std::vector<uint32_t> instructions;
  std::vector<uint32_t> fetchInstructions;
};

bool Builder::createDump(frontend::ShaderHeader const* header, uint32_t const* gpuRegs) const {
  // // Create dump data
  DumpData data {.shaderInput = _shaderInput};

  // Collect memory
  {
    uint64_t const base = getShaderBase(_shaderInput.stage, gpuRegs);
    data.instructions.resize(header->length / sizeof(uint32_t));
    std::memcpy(data.instructions.data(), (uint32_t const*)base, header->length);
  }
  // todo fetch shader
  // -

  std::filesystem::create_directory("shader_dumps");

  uint16_t             numBytes = 0;
  std::vector<uint8_t> binaryData;

  constexpr auto OPTIONS = alpaca::options::fixed_length_encoding;
  numBytes               = alpaca::serialize<OPTIONS>(data, binaryData);

  auto const path = std::filesystem::path("shader_dumps") / (std::string(_name) + ".bin");
  try {
    std::ofstream file(path, std::ios::binary);
    file.write((char const*)binaryData.data(), binaryData.size());
    file.flush();
  } catch (...) {
    printf("Couldn't dump shader %S\n", path.c_str());
    return false;
  }
  printf("dumped shader at %S\n", path.c_str());
  return true;
}

} // namespace compiler