#pragma once

#include "bitfield.h"

#include <stdint.h>

namespace compiler::frontend::parser {
#define SPI_SHADER_PGM_RSRC1_ES_FIELDS(X)                                                                                                                      \
  X(VGPRS, 0, 6)                                                                                                                                               \
  X(SGPRS, 6, 4)                                                                                                                                               \
  X(PRIORITY, 10, 2)                                                                                                                                           \
  X(FLOAT_MODE, 12, 8)                                                                                                                                         \
  X(PRIV, 20, 1)                                                                                                                                               \
  X(DX10_CLAMP, 21, 1)                                                                                                                                         \
  X(DEBUG_MODE, 22, 1)                                                                                                                                         \
  X(IEEE_MODE, 23, 1)                                                                                                                                          \
  X(VGPR_COMP_CNT, 24, 2)                                                                                                                                      \
  X(CU_GROUP_ENABLE, 26, 1)                                                                                                                                    \
  X(CACHE_CTL__CI__VI, 27, 3)                                                                                                                                  \
  X(CDBG_USER__CI__VI, 30, 1)

DEFINE_BITFIELD_STRUCT(SPI_SHADER_PGM_RSRC1_ES, uint32_t, SPI_SHADER_PGM_RSRC1_ES_FIELDS)

#define SPI_SHADER_PGM_RSRC1_GS_FIELDS(X)                                                                                                                      \
  X(VGPRS, 0, 6)                                                                                                                                               \
  X(SGPRS, 6, 4)                                                                                                                                               \
  X(PRIORITY, 10, 2)                                                                                                                                           \
  X(FLOAT_MODE, 12, 8)                                                                                                                                         \
  X(PRIV, 20, 1)                                                                                                                                               \
  X(DX10_CLAMP, 21, 1)                                                                                                                                         \
  X(DEBUG_MODE, 22, 1)                                                                                                                                         \
  X(IEEE_MODE, 23, 1)                                                                                                                                          \
  X(CU_GROUP_ENABLE, 24, 1)                                                                                                                                    \
  X(CACHE_CTL__CI__VI, 25, 3)                                                                                                                                  \
  X(CDBG_USER__CI__VI, 28, 1)

DEFINE_BITFIELD_STRUCT(SPI_SHADER_PGM_RSRC1_GS, uint32_t, SPI_SHADER_PGM_RSRC1_GS_FIELDS)

#define SPI_SHADER_PGM_RSRC1_HS_FIELDS(X)                                                                                                                      \
  X(VGPRS, 0, 6)                                                                                                                                               \
  X(SGPRS, 6, 4)                                                                                                                                               \
  X(PRIORITY, 10, 2)                                                                                                                                           \
  X(FLOAT_MODE, 12, 8)                                                                                                                                         \
  X(PRIV, 20, 1)                                                                                                                                               \
  X(DX10_CLAMP, 21, 1)                                                                                                                                         \
  X(DEBUG_MODE, 22, 1)                                                                                                                                         \
  X(IEEE_MODE, 23, 1)                                                                                                                                          \
  X(CACHE_CTL__CI__VI, 24, 3)                                                                                                                                  \
  X(CDBG_USER__CI__VI, 27, 1)

DEFINE_BITFIELD_STRUCT(SPI_SHADER_PGM_RSRC1_HS, uint32_t, SPI_SHADER_PGM_RSRC1_HS_FIELDS)

#define SPI_SHADER_PGM_RSRC1_LS_FIELDS(X)                                                                                                                      \
  X(VGPRS, 0, 6)                                                                                                                                               \
  X(SGPRS, 6, 4)                                                                                                                                               \
  X(PRIORITY, 10, 2)                                                                                                                                           \
  X(FLOAT_MODE, 12, 8)                                                                                                                                         \
  X(PRIV, 20, 1)                                                                                                                                               \
  X(DX10_CLAMP, 21, 1)                                                                                                                                         \
  X(DEBUG_MODE, 22, 1)                                                                                                                                         \
  X(IEEE_MODE, 23, 1)                                                                                                                                          \
  X(VGPR_COMP_CNT, 24, 2)                                                                                                                                      \
  X(CACHE_CTL__CI__VI, 26, 3)                                                                                                                                  \
  X(CDBG_USER__CI__VI, 29, 1)

DEFINE_BITFIELD_STRUCT(SPI_SHADER_PGM_RSRC1_LS, uint32_t, SPI_SHADER_PGM_RSRC1_LS_FIELDS)

#define SPI_SHADER_PGM_RSRC1_PS_FIELDS(X)                                                                                                                      \
  X(VGPRS, 0, 6)                                                                                                                                               \
  X(SGPRS, 6, 4)                                                                                                                                               \
  X(PRIORITY, 10, 2)                                                                                                                                           \
  X(FLOAT_MODE, 12, 8)                                                                                                                                         \
  X(PRIV, 20, 1)                                                                                                                                               \
  X(DX10_CLAMP, 21, 1)                                                                                                                                         \
  X(DEBUG_MODE, 22, 1)                                                                                                                                         \
  X(IEEE_MODE, 23, 1)                                                                                                                                          \
  X(CU_GROUP_DISABLE, 24, 1)                                                                                                                                   \
  X(CACHE_CTL__CI__VI, 25, 3)                                                                                                                                  \
  X(CDBG_USER__CI__VI, 28, 1)

DEFINE_BITFIELD_STRUCT(SPI_SHADER_PGM_RSRC1_PS, uint32_t, SPI_SHADER_PGM_RSRC1_PS_FIELDS)

#define SPI_SHADER_PGM_RSRC1_VS_FIELDS(X)                                                                                                                      \
  X(VGPRS, 0, 6)                                                                                                                                               \
  X(SGPRS, 6, 4)                                                                                                                                               \
  X(PRIORITY, 10, 2)                                                                                                                                           \
  X(FLOAT_MODE, 12, 8)                                                                                                                                         \
  X(PRIV, 20, 1)                                                                                                                                               \
  X(DX10_CLAMP, 21, 1)                                                                                                                                         \
  X(DEBUG_MODE, 22, 1)                                                                                                                                         \
  X(IEEE_MODE, 23, 1)                                                                                                                                          \
  X(VGPR_COMP_CNT, 24, 2)                                                                                                                                      \
  X(CU_GROUP_ENABLE, 26, 1)                                                                                                                                    \
  X(CACHE_CTL__CI__VI, 27, 3)                                                                                                                                  \
  X(CDBG_USER__CI__VI, 30, 1)

DEFINE_BITFIELD_STRUCT(SPI_SHADER_PGM_RSRC1_VS, uint32_t, SPI_SHADER_PGM_RSRC1_VS_FIELDS)

#define COMPUTE_PGM_RSRC1_FIELDS(X)                                                                                                                            \
  X(VGPRS, 0, 6)                                                                                                                                               \
  X(SGPRS, 6, 4)                                                                                                                                               \
  X(PRIORITY, 10, 2)                                                                                                                                           \
  X(FLOAT_MODE, 12, 8)                                                                                                                                         \
  X(PRIV, 20, 1)                                                                                                                                               \
  X(DX10_CLAMP, 21, 1)                                                                                                                                         \
  X(DEBUG_MODE, 22, 1)                                                                                                                                         \
  X(IEEE_MODE, 23, 1)                                                                                                                                          \
  X(BULKY__CI__VI, 24, 1)                                                                                                                                      \
  X(CDBG_USER__CI__VI, 25, 1)

DEFINE_BITFIELD_STRUCT(COMPUTE_PGM_RSRC1, uint32_t, COMPUTE_PGM_RSRC1_FIELDS)

#define COMPUTE_PGM_RSRC2_FIELDS(X)                                                                                                                            \
  X(SCRATCH_EN, 0, 1)                                                                                                                                          \
  X(USER_SGPR, 1, 5)                                                                                                                                           \
  X(TRAP_PRESENT, 6, 1)                                                                                                                                        \
  X(TGID_X_EN, 7, 1)                                                                                                                                           \
  X(TGID_Y_EN, 8, 1)                                                                                                                                           \
  X(TGID_Z_EN, 9, 1)                                                                                                                                           \
  X(TG_SIZE_EN, 10, 1)                                                                                                                                         \
  X(TIDIG_COMP_CNT, 11, 2)                                                                                                                                     \
  X(EXCP_EN_MSB__CI__VI, 13, 2)                                                                                                                                \
  X(LDS_SIZE, 15, 9)                                                                                                                                           \
  X(EXCP_EN, 24, 7)

DEFINE_BITFIELD_STRUCT(COMPUTE_PGM_RSRC2, uint32_t, COMPUTE_PGM_RSRC2_FIELDS)

#define SPI_SHADER_PGM_RSRC2_ES_FIELDS(X)                                                                                                                      \
  X(SCRATCH_EN, 0, 1)                                                                                                                                          \
  X(USER_SGPR, 1, 5)                                                                                                                                           \
  X(TRAP_PRESENT, 6, 1)                                                                                                                                        \
  X(OC_LDS_EN, 7, 1)                                                                                                                                           \
  X(EXCP_EN, 8, 9)                                                                                                                                             \
  X(LDS_SIZE, 20, 9)

DEFINE_BITFIELD_STRUCT(SPI_SHADER_PGM_RSRC2_ES, uint32_t, SPI_SHADER_PGM_RSRC2_ES_FIELDS)

#define SPI_SHADER_PGM_RSRC2_GS_FIELDS(X)                                                                                                                      \
  X(SCRATCH_EN, 0, 1)                                                                                                                                          \
  X(USER_SGPR, 1, 5)                                                                                                                                           \
  X(TRAP_PRESENT, 6, 1)                                                                                                                                        \
  X(EXCP_EN, 7, 9)

DEFINE_BITFIELD_STRUCT(SPI_SHADER_PGM_RSRC2_GS, uint32_t, SPI_SHADER_PGM_RSRC2_GS_FIELDS)

#define SPI_SHADER_PGM_RSRC2_HS_FIELDS(X)                                                                                                                      \
  X(SCRATCH_EN, 0, 1)                                                                                                                                          \
  X(USER_SGPR, 1, 5)                                                                                                                                           \
  X(TRAP_PRESENT, 6, 1)                                                                                                                                        \
  X(OC_LDS_EN, 7, 1)                                                                                                                                           \
  X(TG_SIZE_EN, 8, 1)                                                                                                                                          \
  X(EXCP_EN, 9, 9)

DEFINE_BITFIELD_STRUCT(SPI_SHADER_PGM_RSRC2_HS, uint32_t, SPI_SHADER_PGM_RSRC2_HS_FIELDS)

#define SPI_SHADER_PGM_RSRC2_LS_FIELDS(X)                                                                                                                      \
  X(SCRATCH_EN, 0, 1)                                                                                                                                          \
  X(USER_SGPR, 1, 5)                                                                                                                                           \
  X(TRAP_PRESENT, 6, 1)                                                                                                                                        \
  X(LDS_SIZE, 7, 9)                                                                                                                                            \
  X(EXCP_EN, 16, 9)

DEFINE_BITFIELD_STRUCT(SPI_SHADER_PGM_RSRC2_LS, uint32_t, SPI_SHADER_PGM_RSRC2_LS_FIELDS)

#define SPI_SHADER_PGM_RSRC2_PS_FIELDS(X)                                                                                                                      \
  X(SCRATCH_EN, 0, 1)                                                                                                                                          \
  X(USER_SGPR, 1, 5)                                                                                                                                           \
  X(TRAP_PRESENT, 6, 1)                                                                                                                                        \
  X(WAVE_CNT_EN, 7, 1)                                                                                                                                         \
  X(EXTRA_LDS_SIZE, 8, 8)                                                                                                                                      \
  X(EXCP_EN, 16, 9)

DEFINE_BITFIELD_STRUCT(SPI_SHADER_PGM_RSRC2_PS, uint32_t, SPI_SHADER_PGM_RSRC2_PS_FIELDS)
#define SPI_SHADER_PGM_RSRC2_VS_FIELDS(X)                                                                                                                      \
  X(SCRATCH_EN, 0, 1)                                                                                                                                          \
  X(USER_SGPR, 1, 5)                                                                                                                                           \
  X(TRAP_PRESENT, 6, 1)                                                                                                                                        \
  X(OC_LDS_EN, 7, 1)                                                                                                                                           \
  X(SO_BASE0_EN, 8, 1)                                                                                                                                         \
  X(SO_BASE1_EN, 9, 1)                                                                                                                                         \
  X(SO_BASE2_EN, 10, 1)                                                                                                                                        \
  X(SO_BASE3_EN, 11, 1)                                                                                                                                        \
  X(SO_EN, 12, 1)                                                                                                                                              \
  X(EXCP_EN, 13, 9)                                                                                                                                            \
  X(DISPATCH_DRAW_EN__VI, 24, 1)

DEFINE_BITFIELD_STRUCT(SPI_SHADER_PGM_RSRC2_VS, uint32_t, SPI_SHADER_PGM_RSRC2_VS_FIELDS)

} // namespace compiler::frontend::parser