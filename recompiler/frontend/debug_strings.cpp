#include "debug_strings.h"

#include "frontend.h"
#include "ir/ir.h"
#include "shader_types.h"

#include <iostream>

namespace compiler::frontend::debug {

void dumpResource(ResourceVBuffer const& res) {
  std::cout << "=== V# Buffer ===\n";
  std::cout << "  base:          0x" << std::hex << res.base << std::dec << "\n";
  std::cout << "  mtype_L1s:     " << res.mtype_L1s << "\n";
  std::cout << "  mtype_L2:      " << res.mtype_L2 << "\n";
  std::cout << "  stride:        " << res.stride << "\n";
  std::cout << "  cache_swizzle: " << res.cache_swizzle << "\n";
  std::cout << "  swizzle_en:    " << res.swizzle_en << "\n";

  std::cout << "Data Layout:\n";
  std::cout << "  num_records:   " << res.num_records << "\n";
  std::cout << "  dst_sel:       x=" << getDebug(getSwizzle(res.dst_sel_xyzw, 0)) << ", y=" << getDebug(getSwizzle(res.dst_sel_xyzw, 1))
            << ", z=" << getDebug(getSwizzle(res.dst_sel_xyzw, 2)) << ", w=" << getDebug(getSwizzle(res.dst_sel_xyzw, 3)) << "\n";
  std::cout << "  nfmt:          " << res.nfmt << "\n";
  std::cout << "  dfmt:          " << res.dfmt << "\n";
  std::cout << "  element_size:  " << res.element_size << "\n";
  std::cout << "  index_stride:  " << res.index_stride << "\n";

  std::cout << "Flags:\n";
  std::cout << "  addtid_en:     " << res.addtid_en << "\n";
  std::cout << "  hash_en:       " << res.hash_en << "\n";
  std::cout << "  mtype:         " << res.mtype << "\n";
  std::cout << "  type:          " << res.type << "\n";
}

void dumpResource(ResourceTBuffer const& res) {
  std::cout << "=== T# Image ===\n";
  std::cout << "  baseaddr256:   0x" << std::hex << res.baseaddr256 << std::dec << "\n";
  std::cout << "  mtype_L1s:     " << ((res.mtype_L1_msb << 2) | res.mtype_L1_lsb) << "\n";
  std::cout << "  mtype_L2:      " << res.mtype_L2 << "\n";
  std::cout << "  min_lod:       " << (res.min_lod / 256.0) << "\n";

  std::cout << "Geometry:\n";
  std::cout << "  width:         " << res.width << "\n";
  std::cout << "  height:        " << res.height << "\n";
  std::cout << "  depth:         " << res.depth << "\n";
  std::cout << "  pitch:         " << res.pitch << "\n";
  std::cout << "  interlaced:    " << res.interlaced << "\n";

  std::cout << "Data Layout:\n";
  std::cout << "  dst_sel:       x=" << getDebug(getSwizzle(res.dst_sel_xyzw, 0)) << ", y=" << getDebug(getSwizzle(res.dst_sel_xyzw, 1))
            << ", z=" << getDebug(getSwizzle(res.dst_sel_xyzw, 2)) << ", w=" << getDebug(getSwizzle(res.dst_sel_xyzw, 3)) << "\n";
  std::cout << "  nfmt:          " << res.nfmt << "\n";
  std::cout << "  dfmt:          " << res.dfmt << "\n";

  std::cout << "Mipmapping:\n";
  std::cout << "  base_level:    " << res.base_level << "\n";
  std::cout << "  last_level:    " << res.last_level << "\n";
  std::cout << "  min_lod_warn:  " << (res.min_lod_warn / 256.0) << "\n";

  std::cout << "Arrays:\n";
  std::cout << "  base_array:    " << res.base_array << "\n";
  std::cout << "  last_array:    " << res.last_array << "\n";

  std::cout << "Flags:\n";
  std::cout << "  tiling_idx:    " << res.tiling_idx << "\n";
  std::cout << "  pow2pad:       " << res.pow2pad << "\n";
  std::cout << "  perf_mod:      " << res.perf_mod << "\n";
  std::cout << "  counter_bank:  " << res.counter_bank_id << "\n";
  std::cout << "  LOD_cnt_en:    " << res.LOD_hdw_cnt_en << "\n";
  std::cout << "  type:          " << res.type << "\n";
}

void dumpResource(ResourceSampler const& res) {
  std::cout << "=== Sampler ===\n";
  std::cout << "Clamping:\n";
  std::cout << "  clamp_x:           " << res.clamp_x << "\n";
  std::cout << "  clamp_y:           " << res.clamp_y << "\n";
  std::cout << "  clamp_z:           " << res.clamp_z << "\n";

  std::cout << "Filtering:\n";
  std::cout << "  xy_mag_filter:     " << res.xy_mag_filter << "\n";
  std::cout << "  xy_min_filter:     " << res.xy_min_filter << "\n";
  std::cout << "  z_filter:          " << res.z_filter << "\n";
  std::cout << "  mip_filter:        " << res.mip_filter << "\n";
  std::cout << "  filter_mode:       " << res.filter_mode << "\n";

  std::cout << "LOD:\n";
  std::cout << "  min_lod:           " << (res.min_lod / 256.0) << "\n";
  std::cout << "  max_lod:           " << (res.max_lod / 256.0) << "\n";
  std::cout << "  lod_bias:          " << (static_cast<int16_t>(res.lod_bias) / 256.0) << "\n";
  std::cout << "  lod_bias_sec:      " << (static_cast<int8_t>(res.lod_bias_sec << 4) / 256.0) << "\n"; // signed 1.4 fixed-point

  std::cout << "Anisotropy:\n";
  std::cout << "  max_aniso_ratio:   " << res.max_aniso_ratio << "\n";
  std::cout << "  aniso_thresh:      " << res.aniso_thresh << "\n";
  std::cout << "  aniso_bias:        " << res.aniso_bias << "\n";

  std::cout << "Control Flags:\n";
  std::cout << "  force_unorm_coords:" << res.force_unorm_coords << "\n";
  std::cout << "  mc_coord_trunc:    " << res.mc_coord_trunc << "\n";
  std::cout << "  force_degamma:     " << res.force_degamma << "\n";
  std::cout << "  trunc_coord:       " << res.trunc_coord << "\n";
  std::cout << "  disable_cube_wrap: " << res.disable_cube_wrap << "\n";

  std::cout << "Performance:\n";
  std::cout << "  perf_mip:          " << res.perf_mip << "\n";
  std::cout << "  perf_z:            " << res.perf_z << "\n";

  std::cout << "Border:\n";
  std::cout << "  border_color_ptr:  0x" << std::hex << res.border_color_ptr << std::dec << "\n";
  std::cout << "  border_color_type: " << res.border_color_type << "\n";

  std::cout << "Depth Compare:\n";
  std::cout << "  depth_cmp_func:    " << res.depth_cmp_func << "\n";
}

std::string_view getDebug(eSwizzle swizzle) {
  switch (swizzle) {
    case eSwizzle::Zero: return "Zero";
    case eSwizzle::One: return "One";
    case eSwizzle::R: return "R";
    case eSwizzle::G: return "G";
    case eSwizzle::B: return "B";
    case eSwizzle::A: return "A";
  }
}

static void printKind(std::ostream& os, eOperandKind kind, uint16_t numdw, bool is64bit) {
  if (kind.isConstI()) {
    os << kind.getConstI();
  } else if (kind.isConstF()) {
    os << kind.getConstF();
  } else if (kind.isSGPR()) {
    auto const start = kind.getSGPR();
    if (numdw == 1)
      os << "s" << std::dec << start;
    else {
      auto const end = start + numdw - 1;
      os << "s[" << std::dec << start << ":" << end << "]";
    }
  } else if (kind.isVGPR()) {
    auto const start = kind.getVGPR();
    if (numdw == 1)
      os << "v" << std::dec << start;
    else {
      auto const end = start + numdw - 1;
      os << "v[" << std::dec << start << ":" << end << "]";
    }
  } else {
    if (is64bit) {
      switch (kind.base()) {
        case eOperandKind::eBase::VccLo: os << "VCC"; break;
        case eOperandKind::eBase::CustomTemp0Lo: os << "TEMP0"; break;
        case eOperandKind::eBase::CustomTemp1Lo: os << "TEMP1"; break;
        case eOperandKind::eBase::ExecLo: os << "EXEC"; break;
        case eOperandKind::eBase::Literal: os << "LITERAL"; break;
        default: os << "UNK" << std::dec << (uint16_t)kind.base(); break;
      }
    } else {
      switch (kind.base()) {
        case eOperandKind::eBase::VccLo: os << "VCC_LO"; break;
        case eOperandKind::eBase::VccHi: os << "VCC_HI"; break;
        case eOperandKind::eBase::CustomTemp0Lo: os << "TEMP0_LO"; break;
        case eOperandKind::eBase::CustomTemp0Hi: os << "TEMP0_HI"; break;
        case eOperandKind::eBase::CustomTemp1Lo: os << "TEMP1_LO"; break;
        case eOperandKind::eBase::CustomTemp1Hi: os << "TEMP1_HI"; break;
        case eOperandKind::eBase::M0: os << "M0"; break;
        case eOperandKind::eBase::CustomVskip: os << "VSKIP"; break;
        case eOperandKind::eBase::ExecLo: os << "EXEC_LO"; break;
        case eOperandKind::eBase::ExecHi: os << "EXEC_HI"; break;
        case eOperandKind::eBase::INV_2PI: os << "INV_2PI"; break;
        case eOperandKind::eBase::SDWA: os << "SDWA"; break;
        case eOperandKind::eBase::DPP: os << "DPP"; break;
        case eOperandKind::eBase::VccZ: os << "VCCZ"; break;
        case eOperandKind::eBase::ExecZ: os << "EXECZ"; break;
        case eOperandKind::eBase::Scc: os << "SCC"; break;
        case eOperandKind::eBase::LdsDirect: os << "DIRECT"; break;
        case eOperandKind::eBase::Literal: os << "LITERAL"; break;
        default: os << "UNK" << std::dec << (uint16_t)kind.base(); break;
      }
    }
  }
}

void printType(std::ostream& os, ir::OperandType type) {
  if (type.is_array()) os << "arr";
  if (type.is_vector()) os << "vec";
  if (type.cols() > 1) {
    os << type.cols() << "x";
  }
  switch (type.base()) {
    case ir::OperandType::eBase::i1: os << "i1"; break;
    case ir::OperandType::eBase::i8: os << "i8"; break;
    case ir::OperandType::eBase::i16: os << "i16"; break;
    case ir::OperandType::eBase::i32: os << "i32"; break;
    case ir::OperandType::eBase::i64: os << "i64"; break;
    case ir::OperandType::eBase::f32: os << "f32"; break;
    case ir::OperandType::eBase::f64: os << "f64"; break;
  }
}

static uint16_t getNumDw(ir::OperandType type) {
  switch (type.base()) {
    case ir::OperandType::eBase::i1: return 1 * type.cols();
    case ir::OperandType::eBase::i8: return 1 * type.cols();
    case ir::OperandType::eBase::i16: return 1 * type.cols();
    case ir::OperandType::eBase::i32: return 1 * type.cols();
    case ir::OperandType::eBase::i64: return 2 * type.cols();
    case ir::OperandType::eBase::f32: return 1 * type.cols();
    case ir::OperandType::eBase::f64: return 2 * type.cols();
  }
}

void printOperandDst(std::ostream& os, ir::Operand const& op) {
  auto const flags = OperandFlagsDst(op.flags);
  auto       kind  = eOperandKind::import(op.kind);

  if (flags.getNegate()) { // custom type
    os << "-(";
  }
  if (flags.getClamp()) {
    os << "clamp{";
  }
  if (flags.hasMultiply()) {
    os << flags.getMultiply() << "*";
  }
  printKind(os, kind, getNumDw(op.type), op.type.is_64bit());

  if (flags.getClamp()) {
    os << "}";
  }
  if (flags.getNegate()) { // custom type
    os << ")";
  }
}

void printOperandSrc(std::ostream& os, ir::Operand const& op) {
  auto const flags = OperandFlagsSrc(op.flags);
  auto       kind  = eOperandKind::import(op.kind);

  if (flags.getNegate()) {
    os << "-(";
  }
  if (flags.getAbsolute()) {
    os << "abs(";
  }

  printKind(os, kind, getNumDw(op.type), op.type.is_64bit());

  if (flags.getAbsolute()) {
    os << ")";
  }
  if (flags.getNegate()) {
    os << ")";
  }
}

} // namespace compiler::frontend::debug