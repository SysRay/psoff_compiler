#pragma once
#include <cstdint>

namespace compiler::frontend::translate {
union ENC_VOP1 {
  struct {
    uint32_t SRC0 : 9;
    uint32_t OP   : 8;
    uint32_t VDST : 8;
    uint32_t      : 7;
  };

  uint32_t raw;
};

union ENC_VOP2 {
  struct {
    uint32_t SRC0  : 9;
    uint32_t VSRC1 : 8;
    uint32_t VDST  : 8;
    uint32_t OP    : 6;
    uint32_t       : 1;
  };

  uint32_t raw;
};

union ENC_VOPC {
  struct {
    uint32_t SRC0  : 9;
    uint32_t VSRC1 : 8;
    uint32_t OP    : 8;
    uint32_t       : 7;
  };

  uint32_t raw;
};

union ENC_VOP3 {
  struct {
    uint64_t VDST  : 8;
    uint64_t ABS   : 3;
    uint64_t CLAMP : 1;
    uint64_t       : 5;
    uint64_t OP    : 9;
    uint64_t       : 6;

    uint64_t SRC0 : 9;
    uint64_t SRC1 : 9;
    uint64_t SRC2 : 9;
    uint64_t OMOD : 2;
    uint64_t NEG  : 3;
  };

  uint64_t raw;
};

union ENC_VOP3_SDST {
  struct {
    uint64_t VDST : 8;
    uint64_t SDST : 7;
    uint64_t      : 2;
    uint64_t OP   : 9;
    uint64_t      : 6;

    uint64_t SRC0 : 9;
    uint64_t SRC1 : 9;
    uint64_t SRC2 : 9;
    uint64_t OMOD : 2;
    uint64_t NEG  : 3;
  };

  uint64_t raw;
};

union ENC_VINTRP {
  struct {
    uint32_t VSRC     : 8;
    uint32_t ATTRCHAN : 2;
    uint32_t ATTR     : 6;
    uint32_t OP       : 2;
    uint32_t VDST     : 8;
    uint32_t          : 6;
  };

  uint32_t raw;
};

union ENC_SOP1 {
  struct {
    uint32_t SSRC0 : 8;
    uint32_t OP    : 8;
    uint32_t SDST  : 7;
    uint32_t       : 9;
  };

  uint32_t raw;
};

union ENC_SOP2 {
  struct {
    uint32_t SSRC0 : 8;
    uint32_t SSRC1 : 8;
    uint32_t SDST  : 7;
    uint32_t OP    : 7;
    uint32_t       : 2;
  };

  uint32_t raw;
};

union ENC_SOPC {
  struct {
    uint32_t SSRC0 : 8;
    uint32_t SSRC1 : 8;
    uint32_t OP    : 7;
    uint32_t       : 9;
  };

  uint32_t raw;
};

union ENC_SOPK {
  struct {
    uint32_t SIMM16 : 16;
    uint32_t SDST   : 7;
    uint32_t OP     : 5;
    uint32_t        : 4;
  };

  uint32_t raw;
};

union ENC_SOPP {
  struct {
    uint32_t SIMM16 : 16;
    uint32_t OP     : 7;
    uint32_t        : 9;
  };

  uint32_t raw;
};

union ENC_EXP {
  struct {
    uint64_t EN    : 4;
    uint64_t TGT   : 6;
    uint64_t COMPR : 1;
    uint64_t DONE  : 1;
    uint64_t VM    : 1;
    uint64_t       : 13;
    uint64_t       : 6;
    uint64_t VSRC0 : 8;
    uint64_t VSRC1 : 8;
    uint64_t VSRC2 : 8;
    uint64_t VSRC3 : 8;
  };

  uint64_t raw;
};

union ENC_DS {
  struct {
    uint64_t OFFSET0 : 8;
    uint64_t OFFSET1 : 8;
    uint64_t         : 1;
    uint64_t GDS     : 1;
    uint64_t OP      : 8;
    uint64_t         : 6;
    uint64_t ADDR    : 8;
    uint64_t DATA0   : 8;
    uint64_t DATA1   : 8;
    uint64_t VDST    : 8;
  };

  uint64_t raw;
};

union ENC_MIMG {
  struct {
    uint64_t       : 8;
    uint64_t DMASK : 4;
    uint64_t UNORM : 1;
    uint64_t GLC   : 1;
    uint64_t DA    : 1;
    uint64_t R128  : 1;
    uint64_t TFE   : 1;
    uint64_t LWE   : 1;
    uint64_t OP    : 7;
    uint64_t SLC   : 1;
    uint64_t       : 6;

    uint64_t VADDR : 8;
    uint64_t VDATA : 8;
    uint64_t SRSRC : 5;
    uint64_t SSAMP : 5;
    uint64_t       : 5;
    uint64_t       : 1;
  };

  uint64_t raw;
};

union ENC_MTBUF {
  struct {
    uint64_t OFFSET  : 12;
    uint64_t OFFEN   : 1;
    uint64_t IDXEN   : 1;
    uint64_t GLC     : 1;
    uint64_t         : 1;
    uint64_t OP      : 3;
    uint64_t DFMT    : 4;
    uint64_t NFMT    : 3;
    uint64_t         : 6;
    uint64_t VADDR   : 8;
    uint64_t VDATA   : 8;
    uint64_t SRSRC   : 5;
    uint64_t         : 1;
    uint64_t SLC     : 1;
    uint64_t TFE     : 1;
    uint64_t SOFFSET : 8;
  };

  uint64_t raw;
};

union ENC_MUBUF {
  struct {
    uint64_t OFFSET  : 12;
    uint64_t OFFEN   : 1;
    uint64_t IDXEN   : 1;
    uint64_t GLC     : 1;
    uint64_t         : 1;
    uint64_t LDS     : 1;
    uint64_t         : 1;
    uint64_t OP      : 7;
    uint64_t         : 1;
    uint64_t         : 6;
    uint64_t VADDR   : 8;
    uint64_t VDATA   : 8;
    uint64_t SRSRC   : 5;
    uint64_t         : 1;
    uint64_t SLC     : 1;
    uint64_t TFE     : 1;
    uint64_t SOFFSET : 8;
  };

  uint64_t raw;
};

union ENC_SMRD {
  struct {
    uint32_t OFFSET : 8;
    uint32_t IMM    : 1;
    uint32_t SBASE  : 6;
    uint32_t SDST   : 7;
    uint32_t OP     : 5;
    uint32_t        : 5;
  };

  uint32_t raw;
};
} // namespace compiler::frontend::translate