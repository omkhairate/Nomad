#ifndef SPHERE_H
#define SPHERE_H

#include <simd/simd.h>
#include <cstdint>

namespace NomadPathTracer {

struct Sphere {
    simd::float3 center;
    float radius;
    uint32_t supportsNormalMap = 0;
};

}

#endif
