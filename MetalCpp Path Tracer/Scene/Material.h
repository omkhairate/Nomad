#ifndef MATERIAL_H
#define MATERIAL_H

#include <cstdint>
#include <simd/simd.h>

namespace MetalCppPathTracer {

struct Material
{
    simd::float3 albedo;
    float materialType;
    simd::float3 emissionColor;
    float emissionPower;
    int32_t diffuseTextureIndex = -1;
};

} // namespace MetalCppPathTracer

#endif
