#pragma once
#include "include/bitfield.h"

#include <stdint.h>

namespace compiler::frontend::translate {
constexpr inline uint64_t getU64(uint32_t const* pCode) {
  return ((uint64_t)pCode[1] << 32) | pCode[0];
}

#define VOP1_FIELDS(X)                                                                                                                                         \
  X(SRC0, 0, 9)                                                                                                                                                \
  X(OP, 9, 8)                                                                                                                                                  \
  X(VDST, 17, 8)
DEFINE_BITFIELD_STRUCT(VOP1, uint32_t, VOP1_FIELDS)

#define VOP2_FIELDS(X)                                                                                                                                         \
  X(SRC0, 0, 9)                                                                                                                                                \
  X(VSRC1, 9, 8)                                                                                                                                               \
  X(VDST, 17, 8)                                                                                                                                               \
  X(OP, 25, 6)
DEFINE_BITFIELD_STRUCT(VOP2, uint32_t, VOP2_FIELDS)

#define VOPC_FIELDS(X)                                                                                                                                         \
  X(SRC0, 0, 9)                                                                                                                                                \
  X(VSRC1, 9, 8)                                                                                                                                               \
  X(OP, 17, 8)
DEFINE_BITFIELD_STRUCT(VOPC, uint32_t, VOPC_FIELDS)

#define VOP3_FIELDS(X)                                                                                                                                         \
  X(VDST, 0, 8)                                                                                                                                                \
  X(ABS, 8, 3)                                                                                                                                                 \
  X(CLAMP, 11, 1)                                                                                                                                              \
  X(OP, 17, 9)                                                                                                                                                 \
  X(SRC0, 32, 9)                                                                                                                                               \
  X(SRC1, 41, 9)                                                                                                                                               \
  X(SRC2, 50, 9)                                                                                                                                               \
  X(OMOD, 59, 2)                                                                                                                                               \
  X(NEG, 61, 3)
DEFINE_BITFIELD_STRUCT(VOP3, uint64_t, VOP3_FIELDS)

#define VOP3_SDST_FIELDS(X)                                                                                                                                    \
  X(VDST, 0, 8)                                                                                                                                                \
  X(SDST, 8, 7)                                                                                                                                                \
  X(OP, 17, 9)                                                                                                                                                 \
  X(SRC0, 32, 9)                                                                                                                                               \
  X(SRC1, 41, 9)                                                                                                                                               \
  X(SRC2, 50, 9)                                                                                                                                               \
  X(OMOD, 59, 2)                                                                                                                                               \
  X(NEG, 61, 3)
DEFINE_BITFIELD_STRUCT(VOP3_SDST, uint64_t, VOP3_SDST_FIELDS)

#define VINTRP_FIELDS(X)                                                                                                                                       \
  X(VSRC, 0, 8)                                                                                                                                                \
  X(ATTRCHAN, 8, 2)                                                                                                                                            \
  X(ATTR, 10, 6)                                                                                                                                               \
  X(OP, 16, 2)                                                                                                                                                 \
  X(VDST, 18, 8)
DEFINE_BITFIELD_STRUCT(VINTRP, uint32_t, VINTRP_FIELDS)

#define SOP1_FIELDS(X)                                                                                                                                         \
  X(SSRC0, 0, 8)                                                                                                                                               \
  X(OP, 8, 8)                                                                                                                                                  \
  X(SDST, 16, 7)
DEFINE_BITFIELD_STRUCT(SOP1, uint32_t, SOP1_FIELDS)

#define SOP2_FIELDS(X)                                                                                                                                         \
  X(SSRC0, 0, 8)                                                                                                                                               \
  X(SSRC1, 8, 8)                                                                                                                                               \
  X(SDST, 16, 7)                                                                                                                                               \
  X(OP, 23, 7)
DEFINE_BITFIELD_STRUCT(SOP2, uint32_t, SOP2_FIELDS)

#define SOPC_FIELDS(X)                                                                                                                                         \
  X(SSRC0, 0, 8)                                                                                                                                               \
  X(SSRC1, 8, 8)                                                                                                                                               \
  X(OP, 16, 7)
DEFINE_BITFIELD_STRUCT(SOPC, uint32_t, SOPC_FIELDS)

#define SOPK_FIELDS(X)                                                                                                                                         \
  X(SIMM16, 0, 16)                                                                                                                                             \
  X(SDST, 16, 7)                                                                                                                                               \
  X(OP, 23, 5)
DEFINE_BITFIELD_STRUCT(SOPK, uint32_t, SOPK_FIELDS)

#define SOPP_FIELDS(X)                                                                                                                                         \
  X(SIMM16, 0, 16)                                                                                                                                             \
  X(OP, 16, 7)
DEFINE_BITFIELD_STRUCT(SOPP, uint32_t, SOPP_FIELDS)

#define EXP_FIELDS(X)                                                                                                                                          \
  X(EN, 0, 4)                                                                                                                                                  \
  X(TGT, 4, 6)                                                                                                                                                 \
  X(COMPR, 10, 1)                                                                                                                                              \
  X(DONE, 11, 1)                                                                                                                                               \
  X(VM, 12, 1)                                                                                                                                                 \
  X(VSRC0, 32, 8)                                                                                                                                              \
  X(VSRC1, 40, 8)                                                                                                                                              \
  X(VSRC2, 48, 8)                                                                                                                                              \
  X(VSRC3, 56, 8)
DEFINE_BITFIELD_STRUCT(EXP, uint64_t, EXP_FIELDS)

#define DS_FIELDS(X)                                                                                                                                           \
  X(OFFSET0, 0, 8)                                                                                                                                             \
  X(OFFSET1, 8, 8)                                                                                                                                             \
  X(GDS, 17, 1)                                                                                                                                                \
  X(OP, 18, 8)                                                                                                                                                 \
  X(ADDR, 32, 8)                                                                                                                                               \
  X(DATA0, 40, 8)                                                                                                                                              \
  X(DATA1, 48, 8)                                                                                                                                              \
  X(VDST, 56, 8)
DEFINE_BITFIELD_STRUCT(DS, uint64_t, DS_FIELDS)

#define MIMG_FIELDS(X)                                                                                                                                         \
  X(DMASK, 8, 4)                                                                                                                                               \
  X(UNORM, 12, 1)                                                                                                                                              \
  X(GLC, 13, 1)                                                                                                                                                \
  X(DA, 14, 1)                                                                                                                                                 \
  X(R128, 15, 1)                                                                                                                                               \
  X(TFE, 16, 1)                                                                                                                                                \
  X(LWE, 17, 1)                                                                                                                                                \
  X(OP, 18, 7)                                                                                                                                                 \
  X(SLC, 25, 1)                                                                                                                                                \
  X(VADDR, 32, 8)                                                                                                                                              \
  X(VDATA, 40, 8)                                                                                                                                              \
  X(SRSRC, 48, 5)                                                                                                                                              \
  X(SSAMP, 53, 5)
DEFINE_BITFIELD_STRUCT(MIMG, uint64_t, MIMG_FIELDS)

#define MTBUF_FIELDS(X)                                                                                                                                        \
  X(OFFSET, 0, 12)                                                                                                                                             \
  X(OFFEN, 12, 1)                                                                                                                                              \
  X(IDXEN, 13, 1)                                                                                                                                              \
  X(GLC, 14, 1)                                                                                                                                                \
  X(OP, 16, 3)                                                                                                                                                 \
  X(DFMT, 19, 4)                                                                                                                                               \
  X(NFMT, 23, 3)                                                                                                                                               \
  X(VADDR, 32, 8)                                                                                                                                              \
  X(VDATA, 40, 8)                                                                                                                                              \
  X(SRSRC, 48, 5)                                                                                                                                              \
  X(SLC, 54, 1)                                                                                                                                                \
  X(TFE, 55, 1)                                                                                                                                                \
  X(SOFFSET, 56, 8)
DEFINE_BITFIELD_STRUCT(MTBUF, uint64_t, MTBUF_FIELDS)

#define MUBUF_FIELDS(X)                                                                                                                                        \
  X(OFFSET, 0, 12)                                                                                                                                             \
  X(OFFEN, 12, 1)                                                                                                                                              \
  X(IDXEN, 13, 1)                                                                                                                                              \
  X(GLC, 14, 1)                                                                                                                                                \
  X(LDS, 16, 1)                                                                                                                                                \
  X(OP, 18, 7)                                                                                                                                                 \
  X(VADDR, 32, 8)                                                                                                                                              \
  X(VDATA, 40, 8)                                                                                                                                              \
  X(SRSRC, 48, 5)                                                                                                                                              \
  X(SLC, 54, 1)                                                                                                                                                \
  X(TFE, 55, 1)                                                                                                                                                \
  X(SOFFSET, 56, 8)
DEFINE_BITFIELD_STRUCT(MUBUF, uint64_t, MUBUF_FIELDS)

#define SMRD_FIELDS(X)                                                                                                                                         \
  X(OFFSET, 0, 8)                                                                                                                                              \
  X(IMM, 8, 1)                                                                                                                                                 \
  X(SBASE, 9, 6)                                                                                                                                               \
  X(SDST, 15, 7)                                                                                                                                               \
  X(OP, 22, 5)
DEFINE_BITFIELD_STRUCT(SMRD, uint32_t, SMRD_FIELDS)

} // namespace compiler::frontend::translate