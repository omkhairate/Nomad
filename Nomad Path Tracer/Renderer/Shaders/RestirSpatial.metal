#include <metal_stdlib>

using namespace metal;

#include "Structs.h"

inline uint hashUint(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

inline float hashToFloat(uint x) {
    return float(hashUint(x) & 0x00ffffffu) / float(0x01000000u);
}

inline bool isFinite3(float3 v) {
    return all(isfinite(v));
}

inline void mergeReservoir(thread RestirReservoir &current,
                           thread const RestirReservoir &neighbor,
                           float neighborWeight,
                           float xi) {
    if (neighbor.m == 0u || neighborWeight <= 0.0f || !isfinite(neighborWeight)) {
        return;
    }
    float currentWeight = max(current.wSum, 0.0f);
    float totalWeight = currentWeight + neighborWeight;
    if (totalWeight <= 0.0f || !isfinite(totalWeight)) {
        return;
    }
    float neighborProb = neighborWeight / totalWeight;
    if (xi < neighborProb) {
        current.sampleRadiance = neighbor.sampleRadiance;
        current.wi = neighbor.wi;
        current.pdf = neighbor.pdf;
        current.packedLightId = neighbor.packedLightId;
    }
    current.wSum = totalWeight;
    current.m += neighbor.m;
}

kernel void restirSpatialMain(
    device RestirReservoir *reservoirBuffer [[buffer(0)]],
    constant UniformsData &uniforms [[buffer(1)]],
    texture2d<float, access::read> positionAccum [[texture(0)]],
    texture2d<float, access::read> normalAccum [[texture(1)]],
    texture2d<float, access::read> albedoAccum [[texture(2)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (uniforms.restirEnableSpatial == 0u) {
        return;
    }

    uint width = positionAccum.get_width();
    uint height = positionAccum.get_height();
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    uint index = gid.y * width + gid.x;
    RestirReservoir current = reservoirBuffer[index];

    float3 currentPosition = positionAccum.read(gid).xyz;
    float3 currentNormal = normalAccum.read(gid).xyz;
    float3 currentAlbedo = albedoAccum.read(gid).xyz;

    if (!isFinite3(currentPosition) || !isFinite3(currentNormal) ||
        !isFinite3(currentAlbedo)) {
        reservoirBuffer[index] = current;
        return;
    }

    uint radius = uniforms.restirSpatialRadius;
    uint neighborCount = uniforms.restirSpatialNeighborCount;
    if (radius == 0u || neighborCount == 0u) {
        reservoirBuffer[index] = current;
        return;
    }

    float3 currentNormalUnit = normalize(currentNormal);
    uint seed = hashUint(index ^ uint(fabs(uniforms.randomSeed.y) * 4096.0f));

    for (uint i = 0u; i < neighborCount; ++i) {
        seed = hashUint(seed + 0x9e3779b9u + i);
        uint span = radius * 2u + 1u;
        int offsetX = int(seed % span) - int(radius);
        seed = hashUint(seed + 0x85ebca6bu);
        int offsetY = int(seed % span) - int(radius);
        if (offsetX == 0 && offsetY == 0) {
            continue;
        }

        int sampleX = int(gid.x) + offsetX;
        int sampleY = int(gid.y) + offsetY;
        if (sampleX < 0 || sampleY < 0 || sampleX >= int(width) ||
            sampleY >= int(height)) {
            continue;
        }

        uint2 neighborPixel = uint2(sampleX, sampleY);
        uint neighborIndex = neighborPixel.y * width + neighborPixel.x;
        RestirReservoir neighbor = reservoirBuffer[neighborIndex];
        if (neighbor.m == 0u || neighbor.wSum <= 0.0f || !isfinite(neighbor.wSum)) {
            continue;
        }

        float3 neighborPosition = positionAccum.read(neighborPixel).xyz;
        float3 neighborNormal = normalAccum.read(neighborPixel).xyz;
        float3 neighborAlbedo = albedoAccum.read(neighborPixel).xyz;

        if (!isFinite3(neighborPosition) || !isFinite3(neighborNormal) ||
            !isFinite3(neighborAlbedo)) {
            continue;
        }

        float normalWeight = max(dot(currentNormalUnit, normalize(neighborNormal)), 0.0f);
        float positionDelta = length(currentPosition - neighborPosition);
        float positionWeight = 1.0f / (1.0f + positionDelta);
        float albedoDiff = length(currentAlbedo - neighborAlbedo);
        float albedoWeight = 1.0f / (1.0f + albedoDiff * 4.0f);
        float similarity = normalWeight * positionWeight * albedoWeight;

        if (similarity <= 0.0f || !isfinite(similarity)) {
            continue;
        }

        float neighborWeight = max(neighbor.wSum, 0.0f) * similarity;
        float xi = hashToFloat(seed);
        mergeReservoir(current, neighbor, neighborWeight, xi);
    }

    reservoirBuffer[index] = current;
}
