#ifndef BVHNODE_H
#define BVHNODE_H

#include <simd/simd.h>

namespace MetalCppPathTracer {

struct BVHNode {
    simd::float3 boundsMin;
    simd::float3 boundsMax;
    int leftFirst;
    int count; // >0: number of primitives (leaf). <=0: -right child index (internal)
};

} // namespace MetalCppPathTracer

#endif
