#define IN(a, b, c) {a, b, c}

#define OPCODE_LIST                                                                                                                     \
  X(NOP, 0, RC(RegClass::Special), IN(RC(RegClass::Special), 0, 0), kNone)                                                              \
  X(ADD_V, 2, RC(RegClass::VGPR), IN(RC(RegClass::VGPR), RC(RegClass::VGPR), 0), kWritesVCC)                                            \
  X(MUL_F, 2, RC(RegClass::VGPR), IN(RC(RegClass::VGPR), RC(RegClass::VGPR), 0), kNone)                                                 \
  X(LD_GLOB, 1, RC(RegClass::VGPR), IN(RC(RegClass::VGPR) | RC(RegClass::SGPR), 0, 0), kMayLoad | kHasSideEffects)                      \
  X(ST_GLOB, 2, RC(RegClass::Special), IN(RC(RegClass::VGPR), RC(RegClass::VGPR) | RC(RegClass::SGPR), 0), kMayStore | kHasSideEffects) \
  X(S_MOV_B32, 1, RC(RegClass::SGPR), IN(RC(RegClass::SGPR), 0, 0), kNone)                                                              \
  X(S_CMP_EQ, 2, RC(RegClass::Special), IN(RC(RegClass::SGPR), RC(RegClass::SGPR), 0), kWritesSCC)                                      \
  X(PHI, 0, RC(RegClass::Special), IN(0, 0, 0), kNone)