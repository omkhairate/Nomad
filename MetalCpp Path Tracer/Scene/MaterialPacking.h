#ifndef MATERIAL_PACKING_H
#define MATERIAL_PACKING_H

#include <cmath>

#include "Primitive.h"

namespace MetalCppPathTracer {

inline bool isFiniteVector(const simd::float3 &v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

inline simd::float3 normalizeOrDefault(const simd::float3 &v,
                                       const simd::float3 &fallback) {
    float len = simd::length(v);
    if (len > 0.0f && isFiniteVector(v))
        return v / len;
    return fallback;
}

inline PackedMaterial packPrimitiveMaterial(const Primitive &primitive) {
    PackedMaterial packed = encodeMaterial(primitive.material);

    simd::float3 fallbackNormal = simd::make_float3(0.0f, 1.0f, 0.0f);
    simd::float3 normal = fallbackNormal;
    bool geometrySupportsNormalMap = false;

    switch (primitive.type) {
    case PrimitiveType::Sphere:
        normal = fallbackNormal;
        geometrySupportsNormalMap = false;
        break;
    case PrimitiveType::Rectangle:
        normal = primitive.rectangle.normal;
        geometrySupportsNormalMap = primitive.rectangle.supportsNormalMap;
        break;
    case PrimitiveType::Triangle:
        normal = primitive.triangle.normal;
        geometrySupportsNormalMap = true;
        break;
    }

    packed.normal = normalizeOrDefault(normal, fallbackNormal);

    int flags = (static_cast<int>(primitive.type) <<
                kPackedMaterialFlagPrimitiveShift) &
               kPackedMaterialFlagPrimitiveMask;
    if (geometrySupportsNormalMap && primitive.material.normalTextureIndex >= 0)
        flags |= kPackedMaterialFlagHasNormalMap;

    packed.flags = flags;
    return packed;
}

} // namespace MetalCppPathTracer

#endif
