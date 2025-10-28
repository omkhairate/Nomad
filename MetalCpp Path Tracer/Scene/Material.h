#ifndef MATERIAL_H
#define MATERIAL_H

#include <simd/simd.h>

namespace MetalCppPathTracer {

struct Material {
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

inline constexpr int kPackedMaterialFlagHasNormalMap = 1 << 0;
inline constexpr int kPackedMaterialFlagPrimitiveShift = 1;
inline constexpr int kPackedMaterialFlagPrimitiveMask = 0x6;

struct PackedMaterial {
    simd::float3 diffuseColor = {0.0f, 0.0f, 0.0f};
    float opacity = 1.0f;

    simd::float3 emissionColor = {0.0f, 0.0f, 0.0f};
    float emissionPower = 0.0f;

    simd::float3 specularColor = {0.0f, 0.0f, 0.0f};
    float shininess = 1.0f;

    simd::float3 transmissionColor = {0.0f, 0.0f, 0.0f};
    float indexOfRefraction = 1.0f;

    simd::float3 normal = {0.0f, 1.0f, 0.0f};
    float roughness = 0.5f;

    int diffuseTextureIndex = -1;
    int specularTextureIndex = -1;
    int normalTextureIndex = -1;
    int flags = 0;
};

inline PackedMaterial encodeMaterial(const Material &m) {
    PackedMaterial data{};
    data.diffuseColor = m.diffuseColor;
    data.opacity = m.opacity;
    data.specularColor = m.specularColor;
    data.shininess = m.shininess;
    data.emissionColor = m.emissionColor;
    data.emissionPower = m.emissionPower;
    data.transmissionColor = m.transmissionColor;
    data.indexOfRefraction = m.indexOfRefraction;
    data.roughness = m.roughness;
    data.diffuseTextureIndex = m.diffuseTextureIndex;
    data.specularTextureIndex = m.specularTextureIndex;
    data.normalTextureIndex = m.normalTextureIndex;
    data.flags = 0;
    data.normal = simd::make_float3(0.0f, 1.0f, 0.0f);
    return data;
}

} // namespace MetalCppPathTracer

#endif
