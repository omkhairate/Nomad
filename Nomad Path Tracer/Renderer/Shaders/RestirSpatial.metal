#include <metal_stdlib>

using namespace metal;

#include "PathTracing.h"

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

struct RestirSampleData {
    float3 radiance = float3(0.0f);
    float3 wi = float3(0.0f);
    float pdf = 0.0f;
    float geometryFactor = 0.0f;
    uint packedLightId = 0u;
    float3 lightPosition = float3(0.0f);
    float3 lightNormal = float3(0.0f);
    float lightArea = 0.0f;
    float lightPdf = 0.0f;
};

inline bool reevaluateReservoirSample(
    thread const RestirReservoir &source,
    float3 currentPosition,
    float3 currentNormal,
    float3 currentAlbedo,
    constant UniformsData &uniforms,
    device const float4 *materials,
    device const float4 *tlasNodes,
    device const float4 *bvhNodes,
    device const float4 *primitives,
    device const int *primitiveIndices,
    device const uchar *activeMask,
    device const InstanceRecord *instanceRecords,
    device const uint *primitiveRemap,
    device atomic_uint *primitiveRayStats,
    thread RestirSampleData &outSample,
    thread float &outWeight) {
    if (source.m == 0u || source.packedLightId == 0xffffffffu) {
        return false;
    }

    uint primitiveCount = uint(uniforms.primitiveCount);
    uint totalPrimitiveCount = uint(uniforms.totalPrimitiveCount);
    uint tlasNodeCount = uint(uniforms.tlasNodeCount);

    float3 toLight = source.lightPosition - currentPosition;
    float dist2 = dot(toLight, toLight);
    if (dist2 <= RAY_EPS || !isfinite(dist2)) {
        return false;
    }

    float dist = sqrt(dist2);
    float3 wi = toLight / dist;
    float3 normalUnit = normalize(currentNormal);
    float3 lightNormal = normalize(source.lightNormal);
    float cosTheta = max(dot(normalUnit, wi), 0.0f);
    float cosLight = max(dot(lightNormal, -wi), 0.0f);
    if (cosTheta <= 0.0f || cosLight <= 0.0f) {
        return false;
    }

    float lightArea = source.lightArea;
    float lightPdf = source.lightPdf;
    if (lightArea <= 0.0f || lightPdf <= 0.0f) {
        return false;
    }

    float totalPdf = lightPdf * dist2 / (cosLight * lightArea);
    if (totalPdf <= 0.0f || !isfinite(totalPdf)) {
        return false;
    }

    TlasLeafCache cache;
    bool visible = isLightVisible(
        currentPosition, normalUnit, wi, dist, source.packedLightId, cache,
        tlasNodes, tlasNodeCount, bvhNodes, primitives,
        primitiveIndices, activeMask, instanceRecords, primitiveRemap,
        primitiveCount, totalPrimitiveCount,
        primitiveRayStats);
    if (!visible) {
        return false;
    }

    if (source.packedLightId >= primitiveCount) {
        return false;
    }

    int lightMatIndex = int(source.packedLightId) * int(kMaterialFloat4Count);
    int totalEntries = int(primitiveCount) * int(kMaterialFloat4Count);
    if (lightMatIndex < 0 ||
        lightMatIndex + int(kMaterialFloat4Count) > totalEntries) {
        return false;
    }

    MaterialPayload lightMaterial = decodeMaterial(lightMatIndex, materials, 1.0f);
    float3 lightRadiance =
        lightMaterial.emissionColor * lightMaterial.emissionPower;
    float3 throughput = directLightingBsdfFromAlbedo(currentAlbedo);
    float3 radiance = throughput * lightRadiance * cosTheta;
    float geometryFactor = cosLight / max(dist2, RAY_EPS);
    float weight = restirTargetWeight(radiance, geometryFactor, totalPdf);
    if (weight <= 0.0f || !isfinite(weight)) {
        return false;
    }

    outSample.radiance = radiance;
    outSample.wi = wi;
    outSample.pdf = totalPdf;
    outSample.geometryFactor = geometryFactor;
    outSample.packedLightId = source.packedLightId;
    outSample.lightPosition = source.lightPosition;
    outSample.lightNormal = source.lightNormal;
    outSample.lightArea = source.lightArea;
    outSample.lightPdf = source.lightPdf;
    outWeight = weight;
    return true;
}

inline void mergeReservoir(thread RestirReservoir &current,
                           thread const RestirSampleData &candidate,
                           float weight,
                           uint candidateCount,
                           float xi) {
    if (candidateCount == 0u || weight <= 0.0f || !isfinite(weight)) {
        return;
    }
    float scaledWeight = weight * float(candidateCount);
    if (scaledWeight <= 0.0f || !isfinite(scaledWeight)) {
        return;
    }
    float currentWeight = max(current.wSum, 0.0f);
    float totalWeight = currentWeight + scaledWeight;
    if (totalWeight <= 0.0f || !isfinite(totalWeight)) {
        return;
    }
    float candidateProb = scaledWeight / totalWeight;
    if (xi < candidateProb) {
        current.sampleRadiance = candidate.radiance;
        current.wi = candidate.wi;
        current.pdf = candidate.pdf;
        current.geometryFactor = candidate.geometryFactor;
        current.packedLightId = candidate.packedLightId;
        current.lightPosition = candidate.lightPosition;
        current.lightNormal = candidate.lightNormal;
        current.lightArea = candidate.lightArea;
        current.lightPdf = candidate.lightPdf;
    }
    current.wSum = totalWeight;
    current.m += candidateCount;
}

kernel void restirSpatialMain(
    device RestirReservoir *reservoirBuffer [[buffer(0)]],
    constant UniformsData &uniforms [[buffer(1)]],
    device const float4 *bvhNodes [[buffer(2)]],
    device const float4 *primitives [[buffer(3)]],
    device const float4 *materials [[buffer(4)]],
    device const int *primitiveIndices [[buffer(5)]],
    device const float4 *tlasNodes [[buffer(6)]],
    device const uchar *activeMask [[buffer(7)]],
    device const uint *primitiveRemap [[buffer(8)]],
    device atomic_uint *primitiveRayStats [[buffer(9)]],
    device const InstanceRecord *instanceRecords [[buffer(10)]],
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

        RestirSampleData candidate;
        float candidateWeight = 0.0f;
        if (reevaluateReservoirSample(
                neighbor, currentPosition, currentNormal, currentAlbedo,
                uniforms, materials, tlasNodes, bvhNodes, primitives,
                primitiveIndices, activeMask, instanceRecords, primitiveRemap,
                primitiveRayStats, candidate, candidateWeight)) {
            float xi = hashToFloat(seed);
            mergeReservoir(current, candidate, candidateWeight * similarity,
                           neighbor.m, xi);
        }
    }

    reservoirBuffer[index] = current;
}
