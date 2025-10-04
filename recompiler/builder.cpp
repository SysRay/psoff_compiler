#include "builder.h"

#include "alpaca/alpaca.h"
#include "frontend/parser/parser.h"
#include "ir/debug_strings.h"
#include "ir/passes/passes.h"

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

void Builder::print() const {
  for (auto const& inst: _instructions) {
    ir::debug::getDebug(std::cout, inst);
  }
}

bool Builder::createShader(frontend::ShaderStage stage, uint32_t id, frontend::ShaderHeader const* header, uint32_t const* gpuRegs) {
  { // Create name
    size_t const len = std::format("{}_{:#x}_{}", getFileTpye(stage), header->hash0, id).copy(_name, sizeof(_name) - 1);
    _name[len]       = '\0';
  }

  // Get shader data
  using namespace frontend;

  _shaderInput.stage = stage;
#define __INIT(name)                                                                                                                                           \
  {                                                                                                                                                            \
    name obj;                                                                                                                                                  \
    obj.init(_shaderInput, header, gpuRegs);                                                                                                                   \
    _shaderInput.stageData = obj;                                                                                                                              \
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

  uint64_t const base     = getShaderBase(_shaderInput.stage, gpuRegs);
  auto const     baseHost = getAddr(base);

  // Register mapping
  _hostMapping[0].pc      = base;
  _hostMapping[0].host    = baseHost;
  _hostMapping[0].size_dw = header->length / sizeof(uint32_t);

  return processBinary();
}

struct DumpData {
  frontend::ShaderInput shaderInput;

  struct InstData {
    uint64_t              pc {};
    std::vector<uint32_t> instructions {};
  };

  std::array<InstData, 2> shaders;
};

bool Builder::createShader(ShaderDump_t const& dump) {
  _debugFlags.set(ShaderBuildFlags::ISDUMP);

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

  _shaderInput = data.shaderInput;

  // Register mapping
  for (uint8_t n = 0; n < data.shaders.size(); ++n) {
    auto const& item = data.shaders[n];

    auto& itemHost = _hostMapping[n];
    itemHost.pc    = item.pc;

    if (item.instructions.empty()) continue;
    itemHost.host    = (uint64_t)item.instructions.data();
    itemHost.size_dw = item.instructions.size();
  }

  return processBinary();
}

bool Builder::createDump(frontend::ShaderHeader const* header, uint32_t const* gpuRegs) const {
  // // Create dump data
  DumpData data {.shaderInput = _shaderInput};

  // Collect memory
  for (uint8_t n = 0; n < _hostMapping.size(); ++n) {
    auto const& itemHost = _hostMapping[n];
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

HostMapping* Builder::getHostMapping(uint64_t pc) {
  { // Search existing
    auto it = std::find_if(_hostMapping.begin(), _hostMapping.end(), [pc](auto const& item) { return item.pc == pc; });
    if (it != _hostMapping.end()) {
      return &*it;
    }
  }
  { // Search free
    auto it = std::find_if(_hostMapping.begin(), _hostMapping.end(), [](auto const& item) { return item.pc == 0; });
    if (it != _hostMapping.end()) {
      it->pc   = pc;
      it->host = getAddr(pc);
      return &*it;
    }
  }
  return nullptr;
}

bool Builder::processBinary() {
  auto const      pcStart = _hostMapping[0].pc;
  uint32_t const* pCode   = (uint32_t const*)_hostMapping[0].host;
  auto const      size    = _hostMapping[0].size_dw;
  if (pCode == nullptr) return false;

  {
    // parse instructions
    ir::passes::pcmapping_t pcMapping {&_poolTemp}; // map pc to instructions for resolving jmp
    pcMapping.reserve(_hostMapping[0].size_dw);

    auto curCode = pCode;
    try {
      while (curCode < (pCode + size)) {
        auto const pc = pcStart + (frontend::parser::pc_t)curCode - (frontend::parser::pc_t)pCode;
        pcMapping.push_back({pc, _instructions.size()});
        frontend::parser::parseInstruction(*this, pc, &curCode);
      }
    } catch (std::runtime_error const& ex) {
      printf("%s error:%s", _name, ex.what());
      return {};
    }

    // create code regions
    if (!ir::passes::createRegions(*this, pcMapping)) {
      printf("Couldn't create regions");
      return false;
    }

    _poolTemp.release();
  }
  return true;
}
} // namespace compiler