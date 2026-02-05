#include <array>
#include <stdint.h>

const std::array<uint32_t, 35> shader_ps_exec_ifelse = {
    0xBEEB03FF, 0x00000012, // 00000000: s_mov_b32     vcc_hi, #0x00000012
    0xBE82047E,             // 00000008: s_mov_b64     s[2:3], exec
    0xBEFC0300,             // 0000000C: s_mov_b32     m0, s0
    0xC8100000,             // 00000010: v_interp_p1_f32 v4, v0, attr0.x
    0xC8110001,             // 00000014: v_interp_p2_f32 v4, v1, attr0.x
    0x7E0A0304,             // 00000018: v_mov_b32     v5, v4
    0xC8080100,             // 0000001C: v_interp_p1_f32 v2, v0, attr0.y
    0xC8090101,             // 00000020: v_interp_p2_f32 v2, v1, attr0.y
    0x7E060302,             // 00000024: v_mov_b32     v3, v2
    0xC8080200,             // 00000028: v_interp_p1_f32 v2, v0, attr0.z
    0xC8090201,             // 0000002C: v_interp_p2_f32 v2, v1, attr0.z
    0xC8000300,             // 00000030: v_interp_p1_f32 v0, v0, attr0.w
    0xC8010301,             // 00000034: v_interp_p2_f32 v0, v1, attr0.w
    0x7C1A08FF, 0x00000000, // 00000038: v_cmp_neq_f32 vcc, #0x00000000, v4
    0xBEEA246A,             // 00000040: s_and_saveexec_b64 vcc, vcc
    0xBF880004,             // 00000044: s_cbranch_execz label_main_0058
    0x10020AF6,             // 00000048: v_mul_f32     v1, 4.0, v5
    0x100606F6,             // 0000004C: v_mul_f32     v3, 4.0, v3
    0x100404F6,             // 00000050: v_mul_f32     v2, 4.0, v2
    0x100000F6,             // 00000054: v_mul_f32     v0, 4.0, v0
    0x8AFE7E6A,             // 00000058: s_andn2_b64   exec, vcc, exec
    0xBF880004,             // 0000005C: s_cbranch_execz label_main_0070
    0x10020AF4,             // 00000060: v_mul_f32     v1, 2.0, v5
    0x100606F4,             // 00000064: v_mul_f32     v3, 2.0, v3
    0x100404F4,             // 00000068: v_mul_f32     v2, 2.0, v2
    0x8AFE7E6A,             // 00000070: s_mov_b64     exec, vcc
    0x5E020701,             // 00000074: v_cvt_pkrtz_f16_f32 v1, v1, v3
    0x5E000102,             // 00000078: v_cvt_pkrtz_f16_f32 v0, v2, v0
    0xBEFE0402,             // 0000007C: s_mov_b64     exec, s[2:3]
    0xF8001C0F, 0x00000001, // 00000080: exp           mrt_color0, v1, v0 compr vm done
    0xBF810000,             // 00000088:  s_endpgm
};

const std::array<uint32_t, 55> shader_ps_forloop = {
    0xBEEB03FF, 0x0000001C, // 00000000: s_mov_b32     vcc_hi, #0x0000001c
    0xBE82047E,             // 00000008: s_mov_b64     s[2:3], exec
    0xBEFC0300,             // 0000000C: s_mov_b32     m0, s0
    0xC8140000,             // 00000010: v_interp_p1_f32 v5, v0, attr0.x
    0xC8150001,             // 00000014: v_interp_p2_f32 v5, v1, attr0.x
    0x7E040305,             // 00000018: v_mov_b32     v2, v5
    0xC80C0100,             // 0000001C: v_interp_p1_f32 v3, v0, attr0.y
    0xC80D0101,             // 00000020: v_interp_p2_f32 v3, v1, attr0.y
    0x7E080303,             // 00000024: v_mov_b32     v4, v3
    0xC80C0200,             // 00000028: v_interp_p1_f32 v3, v0, attr0.z
    0xC80D0201,             // 0000002C: v_interp_p2_f32 v3, v1, attr0.z
    0xC8000300,             // 00000030: v_interp_p1_f32 v0, v0, attr0.w
    0xC8010301,             // 00000034: v_interp_p2_f32 v0, v1, attr0.w
    0x7E020300,             // 00000038: v_mov_b32     v1, v0
    0x7C1A0AFF, 0x00000000, // 0000003C: v_cmp_neq_f32 vcc, #0x00000000, v5
    0xBE80246A,             // 00000044: s_and_saveexec_b64 s[0:1], vcc
    0xBF880014,             // 00000048: s_cbranch_execz label_main_009C
    0xBEEA0380,             // 0000004C: s_mov_b32     vcc_lo, 0
    0x7E0A0280,             // 00000050: v_mov_b32     v5, 0
    0xBE84036A,             // 00000054: s_mov_b32     s4, vcc_lo
    0x7E020305,             // 00000058: v_mov_b32     v1, v5
    0x7E000280,             // 0000005C: v_mov_b32     v0, 0
    0x7E080280,             // 00000060: v_mov_b32     v4, 0
    0x7E060280,             // 00000064: v_mov_b32     v3, 0
    0xD102006A, 0x0001086A, // 00000068: v_cmp_lt_i32  vcc, vcc_lo, 4
    0xBF86000A,             // 00000070: s_cbranch_vccz label_main_009C
    0x7E000A04,             // 00000074: v_cvt_f32_i32 v0, s4
    0x060000F2,             // 00000078: v_add_f32     v0, 1.0, v0
    0x7E005500,             // 0000007C: v_rcp_f32     v0, v0
    0x100000FF, 0x3E800000, // 00000080: v_mul_f32     v0, #0x3e800000, v0
    0xD2820001, 0x04060102, // 00000088: v_mad_f32     v1, v2, v0, v1
    0x7E0A0301,             // 00000090: v_mov_b32     v5, v1
    0x816A8104,             // 00000094: s_add_i32     vcc_lo, s4, 1
    0xBF82FFEE,             // 00000098: s_branch      label_main_0054
    0x8AFE7E00,             // 0000009C: s_andn2_b64   exec, s[0:1], exec
    0xBF880007,             // 000000A0: s_cbranch_execz label_main_00C0
    0x100004F4,             // 000000A4: v_mul_f32     v0, 2.0, v2
    0x100408F4,             // 000000A8: v_mul_f32     v2, 2.0, v4
    0x7E080302,             // 000000AC: v_mov_b32     v4, v2
    0x100406F4,             // 000000B0: v_mul_f32     v2, 2.0, v3
    0x7E060302,             // 000000B4: v_mov_b32     v3, v2
    0x100202F4,             // 000000B8: v_mul_f32     v1, 2.0, v1
    0x7E0A0301,             // 000000BC: v_mov_b32     v5, v1
    0xBEFE0400,             // 000000C0: s_mov_b64     exec, s[0:1]
    0x5E000900,             // 000000C4: v_cvt_pkrtz_f16_f32 v0, v0, v4
    0x5E020B03,             // 000000C8: v_cvt_pkrtz_f16_f32 v1, v3, v5
    0xBEFE0402,             // 000000CC: s_mov_b64     exec, s[2:3]
    0xF8001C0F, 0x00000100, // 000000D0: exp           mrt_color0, v0, v1 compr vm done
    0xBF810000,             // 000000D8: s_endpgm
};
