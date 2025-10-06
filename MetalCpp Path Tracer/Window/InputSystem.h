#ifndef INPUT_SYSTEM_H
#define INPUT_SYSTEM_H

#include <simd/simd.h>

namespace MetalCppPathTracer {

namespace InputSystem
{

inline simd::float3 movementInput {0};
inline simd::float2 rotationInput {0};
inline float zoomInput = 0;
inline bool resetInput = 0;
inline int debugAS = 0; // 0:none, 1:TLAS, 2:BLAS

inline void clearInputs()
{
    rotationInput = 0;
    zoomInput = 0;
    resetInput = 0;
}

};

};

#endif
