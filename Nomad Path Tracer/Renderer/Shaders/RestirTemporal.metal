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

inline float depthFromClip(float4 clip) {
    if (clip.w == 0.0f) {
        return 0.0f;
    }
    return clip.z / clip.w;
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
    float currentRoughness,
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
    MaterialPayload surfaceMaterial =
        restirMaterialFromGBuffer(currentAlbedo, currentRoughness);
    float3 viewDir = uniforms.cameraPosition - currentPosition;
    if (dot(viewDir, viewDir) <= RAY_EPS) {
        viewDir = normalUnit;
    }
    float3 throughput =
        evaluateDirectLightingBsdf(surfaceMaterial, normalUnit, viewDir, wi);
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

kernel void restirTemporalMain(
    device const RestirReservoir *historyReservoir [[buffer(0)]],
    device RestirReservoir *currentReservoir [[buffer(1)]],
    constant UniformsData &uniforms [[buffer(2)]],
    device const float4 *bvhNodes [[buffer(3)]],
    device const float4 *primitives [[buffer(4)]],
    device const float4 *materials [[buffer(5)]],
    device const int *primitiveIndices [[buffer(6)]],
    device const float4 *tlasNodes [[buffer(7)]],
    device const uchar *activeMask [[buffer(8)]],
    device const uint *primitiveRemap [[buffer(9)]],
    device atomic_uint *primitiveRayStats [[buffer(10)]],
    device const InstanceRecord *instanceRecords [[buffer(11)]],
    texture2d<float, access::read> positionAccum [[texture(0)]],
    texture2d<float, access::read> normalAccum [[texture(1)]],
    texture2d<float, access::read> albedoAccum [[texture(2)]],
    texture2d<float, access::read> historyPosition [[texture(3)]],
    texture2d<float, access::read> historyNormal [[texture(4)]],
    texture2d<float, access::read> historyAlbedo [[texture(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (uniforms.restirEnableTemporal == 0u) {
        return;
    }

    uint width = positionAccum.get_width();
    uint height = positionAccum.get_height();
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    uint index = gid.y * width + gid.x;
    RestirReservoir current = currentReservoir[index];

    float3 currentPosition = positionAccum.read(gid).xyz;
    float4 currentNormalSample = normalAccum.read(gid);
    float3 currentNormal = currentNormalSample.xyz;
    float currentRoughness = clamp(currentNormalSample.w, 0.0f, 1.0f);
    float3 currentAlbedo = albedoAccum.read(gid).xyz;

    if (!isFinite3(currentPosition) || !isFinite3(currentNormal) ||
        !isFinite3(currentAlbedo)) {
        currentReservoir[index] = current;
        return;
    }

    float4 currentClip =
        uniforms.currentViewProjection * float4(currentPosition, 1.0f);
    if (currentClip.w <= 0.0f) {
        currentReservoir[index] = current;
        return;
    }

    float4 prevClip = uniforms.prevViewProjection * float4(currentPosition, 1.0f);
    if (prevClip.w <= 0.0f) {
        currentReservoir[index] = current;
        return;
    }

    float2 prevNdc = prevClip.xy / prevClip.w;
    float2 prevUv = prevNdc * 0.5f + 0.5f;
    if (prevUv.x < 0.0f || prevUv.x > 1.0f || prevUv.y < 0.0f || prevUv.y > 1.0f) {
        currentReservoir[index] = current;
        return;
    }

    uint2 prevPixel = uint2(prevUv * float2(width, height));
    if (prevPixel.x >= width || prevPixel.y >= height) {
        currentReservoir[index] = current;
        return;
    }

    float3 historyPositionValue = historyPosition.read(prevPixel).xyz;
    float3 historyNormalValue = historyNormal.read(prevPixel).xyz;
    float3 historyAlbedoValue = historyAlbedo.read(prevPixel).xyz;

    if (!isFinite3(historyPositionValue) || !isFinite3(historyNormalValue) ||
        !isFinite3(historyAlbedoValue)) {
        currentReservoir[index] = current;
        return;
    }

    float motion = clamp(uniforms.cameraMotionMetric, 0.0f, 1.0f);
    float normalLow = uniforms.restirNormalDepthThresholds.x;
    float normalHigh = uniforms.restirNormalDepthThresholds.y;
    float depthLow = uniforms.restirNormalDepthThresholds.z;
    float depthHigh = uniforms.restirNormalDepthThresholds.w;
    float depthThreshold = mix(depthLow, depthHigh, motion);
    float normalThreshold = mix(normalLow, normalHigh, motion);
    float albedoThreshold = mix(0.25f, 0.1f, motion);

    float4 historyClip =
        uniforms.currentViewProjection * float4(historyPositionValue, 1.0f);
    if (historyClip.w <= 0.0f) {
        currentReservoir[index] = current;
        return;
    }

    float depthCurrent = depthFromClip(currentClip);
    float depthHistory = depthFromClip(historyClip);
    float depthDiff = fabs(depthCurrent - depthHistory);

    float3 currentNormalUnit = normalize(currentNormal);
    float3 historyNormalUnit = normalize(historyNormalValue);
    float normalDot = dot(currentNormalUnit, historyNormalUnit);

    float albedoDiff = length(currentAlbedo - historyAlbedoValue);

    bool valid = depthDiff <= depthThreshold &&
                 normalDot >= normalThreshold &&
                 albedoDiff <= albedoThreshold &&
                 motion < 0.95f;

    if (valid) {
        uint prevIndex = prevPixel.y * width + prevPixel.x;
        RestirReservoir history = historyReservoir[prevIndex];
        uint seed = hashUint(index ^
                             uint(fabs(uniforms.randomSeed.x) * 4096.0f));
        float xi = hashToFloat(seed);
        RestirSampleData candidate;
        float candidateWeight = 0.0f;
        if (reevaluateReservoirSample(
                history, currentPosition, currentNormal, currentAlbedo,
                currentRoughness,
                uniforms, materials, tlasNodes, bvhNodes, primitives,
                primitiveIndices, activeMask, instanceRecords, primitiveRemap,
                primitiveRayStats, candidate, candidateWeight)) {
            mergeReservoir(current, candidate, candidateWeight, history.m, xi);
        }
        if (uniforms.restirMaxTemporalM > 0u) {
            current.m = min(current.m, uniforms.restirMaxTemporalM);
        }
    }

    currentReservoir[index] = current;
}
