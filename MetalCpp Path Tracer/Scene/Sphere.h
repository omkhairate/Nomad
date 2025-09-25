#ifndef SPHERE_H
#define SPHERE_H

#include <simd/simd.h>

namespace MetalCppPathTracer {

struct Sphere {
    simd::float3 center;
    float radius;
};

}

#endif
