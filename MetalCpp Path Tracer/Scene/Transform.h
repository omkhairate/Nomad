#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <simd/simd.h>

namespace MetalCppPathTracer {

struct Transform
{
    simd::float3 position;
    float scale;

    Transform()
        : position(simd::make_float3(0,0,0)), scale(1.0f)
    {}

    Transform(const simd::float3& p, float s)
        : position(p), scale(s)
    {}
};

}

#endif
