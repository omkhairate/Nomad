#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <simd/simd.h>
#include <cmath>

namespace MetalCppPathTracer {

struct Triangle {
    simd::float3 v0;
    simd::float3 v1;
    simd::float3 v2;
    simd::float2 uv0 = simd::make_float2(0.0f, 0.0f);
    simd::float2 uv1 = simd::make_float2(0.0f, 0.0f);
    simd::float2 uv2 = simd::make_float2(0.0f, 0.0f);
    simd::float3 tangent = simd::make_float3(1.0f, 0.0f, 0.0f);
    simd::float3 bitangent = simd::make_float3(0.0f, 1.0f, 0.0f);
    simd::float3 normal = simd::make_float3(0.0f, 0.0f, 1.0f);

    void computeFrame() {
        constexpr float kEpsilon = 1.0e-6f;

        auto isFinite = [](const simd::float3& v) {
            return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
        };

        auto safeNormalize = [&](const simd::float3& v) {
            float len = simd::length(v);
            if (len > kEpsilon && isFinite(v))
                return v / len;
            return simd::make_float3(0.0f, 0.0f, 0.0f);
        };

        simd::float3 edge1 = v1 - v0;
        simd::float3 edge2 = v2 - v0;

        simd::float3 computedNormal = simd::cross(edge1, edge2);
        computedNormal = safeNormalize(computedNormal);
        if (simd::length(computedNormal) <= kEpsilon || !isFinite(computedNormal)) {
            computedNormal = simd::make_float3(0.0f, 1.0f, 0.0f);
        }

        simd::float2 duv1 = uv1 - uv0;
        simd::float2 duv2 = uv2 - uv0;
        float det = duv1.x * duv2.y - duv1.y * duv2.x;

        simd::float3 computedTangent = simd::make_float3(0.0f, 0.0f, 0.0f);
        simd::float3 computedBitangent = simd::make_float3(0.0f, 0.0f, 0.0f);

        if (std::fabs(det) > kEpsilon) {
            float invDet = 1.0f / det;
            computedTangent = safeNormalize((edge1 * duv2.y - edge2 * duv1.y) * invDet);
            computedBitangent = safeNormalize((edge2 * duv1.x - edge1 * duv2.x) * invDet);
        }

        bool tangentValid = simd::length(computedTangent) > kEpsilon &&
                            isFinite(computedTangent);
        bool bitangentValid = simd::length(computedBitangent) > kEpsilon &&
                              isFinite(computedBitangent);

        if (!tangentValid || !bitangentValid) {
            computedTangent = safeNormalize(edge1);
            if (simd::length(computedTangent) <= kEpsilon ||
                !isFinite(computedTangent)) {
                computedTangent = safeNormalize(edge2);
            }

            if (simd::length(computedTangent) <= kEpsilon ||
                !isFinite(computedTangent)) {
                simd::float3 up = std::fabs(computedNormal.y) < 0.999f
                                      ? simd::make_float3(0.0f, 1.0f, 0.0f)
                                      : simd::make_float3(1.0f, 0.0f, 0.0f);
                computedTangent = safeNormalize(simd::cross(up, computedNormal));
            }

            if (simd::length(computedTangent) <= kEpsilon ||
                !isFinite(computedTangent)) {
                computedTangent = simd::make_float3(1.0f, 0.0f, 0.0f);
            }

            computedBitangent = safeNormalize(simd::cross(computedNormal, computedTangent));
            if (simd::length(computedBitangent) <= kEpsilon ||
                !isFinite(computedBitangent)) {
                computedBitangent = safeNormalize(simd::cross(computedTangent, computedNormal));
            }
        } else {
            computedTangent = safeNormalize(
                computedTangent - computedNormal * simd::dot(computedNormal, computedTangent));
            computedBitangent = safeNormalize(
                computedBitangent - computedNormal * simd::dot(computedNormal, computedBitangent));
        }

        if (simd::length(computedBitangent) <= kEpsilon || !isFinite(computedBitangent)) {
            computedBitangent = safeNormalize(simd::cross(computedNormal, computedTangent));
        }

        if (simd::length(computedBitangent) <= kEpsilon || !isFinite(computedBitangent)) {
            computedBitangent = simd::make_float3(0.0f, 1.0f, 0.0f);
        }

        tangent = computedTangent;
        bitangent = computedBitangent;
        normal = safeNormalize(computedNormal);
        if (simd::length(normal) <= kEpsilon || !isFinite(normal)) {
            normal = simd::make_float3(0.0f, 0.0f, 1.0f);
        }
    }
};

}

#endif
