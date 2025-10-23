#include "logging.h"

#include <iostream>

namespace compiler {
static constexpr std::string_view TYPES[] = {"DEBUG| ", "INFO| ", "ERROR| "};

void __LOG_IMPL(eLOG_TYPE type, std::string_view formatted_message) {
  std::cout << TYPES[(uint8_t)type] << formatted_message << "\n";
}
} // namespace compiler