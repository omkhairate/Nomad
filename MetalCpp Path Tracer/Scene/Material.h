#ifndef MATERIAL_H
#define MATERIAL_H

#include <array>
#include <simd/simd.h>

namespace MetalCppPathTracer {

inline constexpr size_t kMaterialFloat4Count = 5;

struct Material
{
    simd::float3 diffuseColor = {0.8f, 0.8f, 0.8f};
    float opacity = 1.0f;

    simd::float3 specularColor = {0.04f, 0.04f, 0.04f};
    float shininess = 32.0f;

    simd::float3 emissionColor = {0.0f, 0.0f, 0.0f};
    float emissionPower = 0.0f;

    simd::float3 transmissionColor = {0.0f, 0.0f, 0.0f};
    float indexOfRefraction = 1.0f;

    float roughness = 0.5f;
    int diffuseTextureIndex = -1;
    int specularTextureIndex = -1;
    int normalTextureIndex = -1;
    int padding = 0;
};

inline std::array<simd::float4, kMaterialFloat4Count> encodeMaterial(const Material &m)
{
    std::array<simd::float4, kMaterialFloat4Count> data{};
    data[0] = simd::make_float4(m.diffuseColor, m.opacity);
    data[1] = simd::make_float4(m.specularColor, m.shininess);
    data[2] = simd::make_float4(m.emissionColor, m.emissionPower);
    data[3] = simd::make_float4(m.transmissionColor, m.indexOfRefraction);
    data[4] = simd::make_float4(static_cast<float>(m.diffuseTextureIndex),
                                 static_cast<float>(m.specularTextureIndex),
                                 static_cast<float>(m.normalTextureIndex),
                                 m.roughness);
    return data;
}

};


#endif
