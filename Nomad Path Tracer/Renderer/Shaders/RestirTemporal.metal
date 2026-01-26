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

inline float depthFromClip(float4 clip) {
    if (clip.w == 0.0f) {
        return 0.0f;
    }
    return clip.z / clip.w;
}

inline void mergeReservoir(thread RestirReservoir &current,
                           thread const RestirReservoir &history,
                           float xi) {
    if (history.m == 0u || history.wSum <= 0.0f || !isfinite(history.wSum)) {
        return;
    }
    float currentWeight = max(current.wSum, 0.0f);
    float historyWeight = max(history.wSum, 0.0f);
    float totalWeight = currentWeight + historyWeight;
    if (totalWeight <= 0.0f || !isfinite(totalWeight)) {
        return;
    }
    float historyProb = historyWeight / totalWeight;
    if (xi < historyProb) {
        current.sampleRadiance = history.sampleRadiance;
        current.wi = history.wi;
        current.pdf = history.pdf;
        current.packedLightId = history.packedLightId;
    }
    current.wSum = totalWeight;
    current.m += history.m;
}

kernel void restirTemporalMain(
    device const RestirReservoir *historyReservoir [[buffer(0)]],
    device RestirReservoir *currentReservoir [[buffer(1)]],
    constant UniformsData &uniforms [[buffer(2)]],
    texture2d<float, access::read> positionAccum [[texture(0)]],
    texture2d<float, access::read> normalAccum [[texture(1)]],
    texture2d<float, access::read> albedoAccum [[texture(2)]],
    texture2d<float, access::read> historyPosition [[texture(3)]],
    texture2d<float, access::read> historyNormal [[texture(4)]],
    texture2d<float, access::read> historyAlbedo [[texture(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint width = positionAccum.get_width();
    uint height = positionAccum.get_height();
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    uint index = gid.y * width + gid.x;
    RestirReservoir current = currentReservoir[index];

    float3 currentPosition = positionAccum.read(gid).xyz;
    float3 currentNormal = normalAccum.read(gid).xyz;
    float3 currentAlbedo = albedoAccum.read(gid).xyz;

    if (!isFinite3(currentPosition) || !isFinite3(currentNormal) ||
        !isFinite3(currentAlbedo)) {
        currentReservoir[index] = current;
        return;
    }

    float4 currentClip = uniforms.viewProjection * float4(currentPosition, 1.0f);
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
    float depthThreshold = mix(0.02f, 0.005f, motion);
    float normalThreshold = mix(0.8f, 0.95f, motion);
    float albedoThreshold = mix(0.25f, 0.1f, motion);

    float4 historyClip = uniforms.viewProjection * float4(historyPositionValue, 1.0f);
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
        mergeReservoir(current, history, xi);
    }

    currentReservoir[index] = current;
}
