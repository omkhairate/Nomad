#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <simd/simd.h>

namespace MetalCppPathTracer {

struct Triangle {
    simd::float3 v0;
    simd::float3 v1;
    simd::float3 v2;
    simd::float2 uv0;
    simd::float2 uv1;
    simd::float2 uv2;
};

}

#endif
