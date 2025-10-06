#ifndef RECTANGLE_H
#define RECTANGLE_H

#include <simd/simd.h>

namespace MetalCppPathTracer {

struct Rectangle {
    simd::float3 center;
    simd::float3 u;
    simd::float3 v;
};

}

#endif
