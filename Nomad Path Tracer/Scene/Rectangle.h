#ifndef RECTANGLE_H
#define RECTANGLE_H

#include <simd/simd.h>
#include <cmath>

namespace NomadPathTracer {

struct Rectangle {
    simd::float3 center;
    simd::float3 u;
    simd::float3 v;
    simd::float3 tangent = simd::make_float3(1.0f, 0.0f, 0.0f);
    simd::float3 bitangent = simd::make_float3(0.0f, 1.0f, 0.0f);
    simd::float3 normal = simd::make_float3(0.0f, 0.0f, 1.0f);
    bool supportsNormalMap = false;

    void computeFrame() {
        constexpr float kEpsilon = 1.0e-6f;

        auto isFinite = [](const simd::float3 &v) {
            return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
        };

        auto safeNormalize = [&](const simd::float3 &v) {
            float len = simd::length(v);
            if (len > kEpsilon && isFinite(v))
                return v / len;
            return simd::make_float3(0.0f, 0.0f, 0.0f);
        };

        simd::float3 computedTangent = safeNormalize(u);
        simd::float3 computedBitangent = safeNormalize(v);
        simd::float3 computedNormal = safeNormalize(simd::cross(computedTangent, computedBitangent));

        if (simd::length(computedTangent) <= kEpsilon || !isFinite(computedTangent)) {
            computedTangent = simd::make_float3(1.0f, 0.0f, 0.0f);
        }

        if (simd::length(computedBitangent) <= kEpsilon || !isFinite(computedBitangent)) {
            simd::float3 refAxis = (std::fabs(computedTangent.y) < 0.999f)
                                       ? simd::make_float3(0.0f, 1.0f, 0.0f)
                                       : simd::make_float3(0.0f, 0.0f, 1.0f);
            computedBitangent = safeNormalize(simd::cross(refAxis, computedTangent));
        }

        if (simd::length(computedNormal) <= kEpsilon || !isFinite(computedNormal)) {
            computedNormal = safeNormalize(simd::cross(u, v));
        }

        if (simd::length(computedNormal) <= kEpsilon || !isFinite(computedNormal)) {
            computedNormal = simd::make_float3(0.0f, 1.0f, 0.0f);
        }

        computedTangent = safeNormalize(computedTangent -
                                        computedNormal * simd::dot(computedNormal, computedTangent));
        computedBitangent = safeNormalize(simd::cross(computedNormal, computedTangent));

        bool tangentValid = simd::length(computedTangent) > kEpsilon && isFinite(computedTangent);
        bool bitangentValid = simd::length(computedBitangent) > kEpsilon && isFinite(computedBitangent);
        bool normalValid = simd::length(computedNormal) > kEpsilon && isFinite(computedNormal);

        supportsNormalMap = tangentValid && bitangentValid && normalValid;
        if (!supportsNormalMap) {
            computedTangent = simd::make_float3(1.0f, 0.0f, 0.0f);
            computedBitangent = simd::make_float3(0.0f, 1.0f, 0.0f);
            computedNormal = simd::make_float3(0.0f, 0.0f, 1.0f);
        }

        tangent = computedTangent;
        bitangent = computedBitangent;
        normal = computedNormal;
    }
};

}

#endif
