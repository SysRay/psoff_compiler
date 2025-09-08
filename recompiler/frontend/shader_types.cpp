#include "shader_types.h"

namespace compiler::frontend {
eSwizzle getSwizzle(Swizzle_t swizzle, uint8_t index) {
  auto const           item  = (swizzle >> (3u * index)) & 0b111; // 3 bits per item
  constexpr std::array table = {eSwizzle::Zero, eSwizzle::One, (eSwizzle)0, (eSwizzle)0, eSwizzle::R, eSwizzle::G, eSwizzle::B, eSwizzle::A};
  return table[item];
}
} // namespace compiler::frontend