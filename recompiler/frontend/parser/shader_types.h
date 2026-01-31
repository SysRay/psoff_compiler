#pragma once
#include "include/common.h"

#include <array>
#include <stdint.h>

namespace compiler::frontend {
constexpr uint8_t SPEC_SLOTS_PER_CHUNK             = 32;
constexpr uint8_t SPEC_TOTAL_USER_SGPR             = 16;
constexpr uint8_t SPEC_NUM_INTERNAL_GLOBAL_BUFFERS = 0x0C;
constexpr uint8_t SPEC_SRT_MAX_SIZE                = 8;
constexpr size_t  SPEC_FETCHSHADER_MAX_SIZE_DW     = 1000; ///< Limit fetsh shader parsing

enum class ShaderStage : uint8_t { Compute, Vertex, VertexExport, VertexLocal, Fragment, Geometry, Copy, TessellationCtrl, TessellationEval };

enum class ShaderLogicalStage : uint8_t { Compute, Vertex, Fragment, Geometry, Tesselation };

enum class FpRoundMode : uint8_t {
  NearestEven = 0,
  PlusInf     = 1,
  MinInf      = 2,
  ToZero      = 3,
};

enum class FpDenormMode : uint8_t {
  InOutFlush      = 0,
  InAllowOutFlush = 1,
  InFlushOutAllow = 2,
  InOutAllow      = 3,
};

PACK(struct ShaderHeader {
  uint8_t  signature[7];
  uint8_t  version;
  uint32_t pssl_or_cg  : 1;
  uint32_t cached      : 1;
  uint32_t type        : 4;
  uint32_t source_type : 2;
  uint32_t length      : 24; ///< in bytes
  uint8_t  chunk_usage_base_offset_dw;
  uint8_t  num_input_usage_slots;
  uint8_t  is_srt                 : 1;
  uint8_t  is_srt_used_info_valid : 1;
  uint8_t  is_extended_usage_info : 1;
  uint8_t  reserved2              : 5;
  uint8_t  reserved3;
  uint32_t hash0;
  uint32_t hash1;
  uint32_t crc32;
});

PACK(struct ResourceVBuffer {
  uint64_t base          : 44; // [43:0]    base byte address
  uint64_t mtype_L1s     : 2;  // [45:44]   mtype for scalar L1
  uint64_t mtype_L2      : 2;  // [47:46]   mtype for L2
  uint64_t stride        : 14; // [61:48]   stride in bytes (0–16383)
  uint64_t cache_swizzle : 1;  // [62]      swizzle TC L1 banks
  uint64_t swizzle_en    : 1;  // [63]      swizzle enable

  uint64_t num_records : 32; // [95:64]   record count (in units of stride)

  uint64_t dst_sel_xyzw : 12;

  uint64_t nfmt : 3; // [110:108] numeric format

  uint64_t dfmt         : 4; // [114:111] data format
  uint64_t element_size : 2; // [116:115] element size (2/4/8/16 bytes)
  uint64_t index_stride : 2; // [118:117] index stride (8/16/32/64)
  uint64_t addtid_en    : 1; // [119]     add thread ID
  uint64_t reserved1    : 1; // [120]     reserved
  uint64_t hash_en      : 1; // [121]     enable address hashing
  uint64_t reserved2    : 1; // [122]     reserved
  uint64_t mtype        : 3; // [125:123] mtype for L1
  uint64_t type         : 2; // [127:126] type (must be 0 for buffer)
});

PACK(struct ResourceTBuffer {
  uint64_t baseaddr256  : 38; // [37:0]
  uint64_t mtype_L2     : 2;  // [39:38]
  uint64_t min_lod      : 12; // [51:40] (fixed-point 4.8)
  uint64_t dfmt         : 6;  // [57:52]
  uint64_t nfmt         : 4;  // [61:58]
  uint64_t mtype_L1_lsb : 2;  // [63:62]

  uint64_t width      : 14; // [77:64]
  uint64_t height     : 14; // [91:78]
  uint64_t perf_mod   : 3;  // [94:92]
  uint64_t interlaced : 1;  // [95]

  uint64_t dst_sel_xyzw : 12; // [98:107]

  uint64_t base_level   : 4; // [111:108]
  uint64_t last_level   : 4; // [115:112]
  uint64_t tiling_idx   : 5; // [120:116]
  uint64_t pow2pad      : 1; // [121]
  uint64_t mtype_L1_msb : 1; // [122]
  uint64_t reserved1    : 1; // [123]
  uint64_t type         : 4; // [127:124]

  // --- High 128 bits ---
  uint64_t depth           : 13; // [140:128]
  uint64_t pitch           : 14; // [154:141]
  uint64_t reserved2       : 5;  // [159:155]
  uint64_t base_array      : 13; // [172:160]
  uint64_t last_array      : 13; // [185:173]
  uint64_t reserved3       : 6;  // [191:186]
  uint64_t min_lod_warn    : 12; // [203:192] (fixed-point 4.8)
  uint64_t counter_bank_id : 8;  // [211:204]
  uint64_t LOD_hdw_cnt_en  : 1;  // [212]
  uint64_t reserved4       : 43; // [255:213]
});

PACK(struct ResourceSampler {
  // Bits 0–31
  uint64_t clamp_x            : 3; // [2:0]
  uint64_t clamp_y            : 3; // [5:3]
  uint64_t clamp_z            : 3; // [8:6]
  uint64_t max_aniso_ratio    : 3; // [11:9]
  uint64_t depth_cmp_func     : 3; // [14:12]
  uint64_t force_unorm_coords : 1; // [15]
  uint64_t aniso_thresh       : 3; // [18:16]
  uint64_t mc_coord_trunc     : 1; // [19]
  uint64_t force_degamma      : 1; // [20]
  uint64_t aniso_bias         : 6; // [26:21]
  uint64_t trunc_coord        : 1; // [27]
  uint64_t disable_cube_wrap  : 1; // [28]
  uint64_t filter_mode        : 2; // [30:29]
  uint64_t reserved0          : 1; // [31]

  // Bits 32–63
  uint64_t min_lod  : 12; // [43:32]  - fixed point 4.8
  uint64_t max_lod  : 12; // [55:44]  - fixed point 4.8
  uint64_t perf_mip : 4;  // [59:56]
  uint64_t perf_z   : 4;  // [63:60]

  // Bits 64–95
  uint64_t lod_bias      : 14; // [77:64] - signed fixed point 5.8
  uint64_t lod_bias_sec  : 6;  // [83:78] - signed fixed point 1.4
  uint64_t xy_mag_filter : 2;  // [85:84]
  uint64_t xy_min_filter : 2;  // [87:86]
  uint64_t z_filter      : 2;  // [89:88]
  uint64_t mip_filter    : 2;  // [91:90]
  uint64_t reserved1     : 2;  // [93:92]

  // Bits 96–127
  uint64_t border_color_ptr  : 12; // [107:96]
  uint64_t reserved2         : 18; // [125:108]
  uint64_t border_color_type : 2;  // [127:126]
});

static_assert(sizeof(ResourceVBuffer) == 4 * sizeof(uint32_t));
static_assert(sizeof(ResourceTBuffer) == 8 * sizeof(uint32_t));
static_assert(sizeof(ResourceSampler) == 4 * sizeof(uint32_t));

enum class eSwizzle { Zero, One, R, G, B, A };
using Swizzle_t = uint16_t; ///< dst_sel_xyzw

eSwizzle getSwizzle(Swizzle_t swizzle, uint8_t index);
} // namespace compiler::frontend