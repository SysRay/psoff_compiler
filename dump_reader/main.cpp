#include "builder.h"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace compiler {
uint64_t getAddr(uint64_t addr) {
  printf("Error: getAddr called!");
  exit(1);
  return 0;
}
} // namespace compiler

int main(int argc, char* argv[]) {
  // // Read Dump
  std::filesystem::path fp = argv[1];
  if (!std::filesystem::exists(fp)) {
    printf("Missing file: %S\n", fp.c_str());
    return 1;
  }

  auto const fsize = std::filesystem::file_size(fp);

  compiler::ShaderDump_t data(fsize);

  std::ifstream file;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  try {
    std::cout << "Opening " << fp << std::endl;
    file.open(fp, std::ios::binary);
  } catch (std::ifstream::failure e) {
    printf("Exception openFile %s\n", e.what());
    return 2;
  }

  file.read((char*)data.data(), fsize);

  // // Build Shader
  compiler::Builder builder(1000);
  builder.createShader(data);
  builder.print();

  return 0;
}
