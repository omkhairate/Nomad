#include <metal_stdlib>

using namespace metal;

struct DenoiserUniforms {
    uint radius;
    uint historyValid;
    float temporalBlendBase;
    float sampleCountForStableHistory;
    float normalSigma;
    float albedoSigma;
    float luminanceSigma;
    float spatialSigma;
};

inline float gaussianWeight(float distance, float sigma) {
    float safeSigma = max(sigma, 1e-3f);
    return exp(-distance / (2.0f * safeSigma * safeSigma));
}

inline float3 sanitizeNormal(float3 normal) {
    float lenSq = dot(normal, normal);
    if (lenSq < 1e-6f) {
        return float3(0.0f);
    }
    return normal / sqrt(lenSq);
}

inline float luminance(float3 color) {
    const float3 weights = float3(0.299f, 0.587f, 0.114f);
    return dot(color, weights);
}

kernel void denoiseMain(texture2d<half, access::read> currentFrame [[texture(0)]],
                        texture2d<half, access::read> lastFrame [[texture(1)]],
                        texture2d<half, access::read> albedoAccum [[texture(2)]],
                        texture2d<half, access::read> normalAccum [[texture(3)]],
                        texture2d<half, access::read> sampleCountTex [[texture(4)]],
                        texture2d<half, access::write> outputFrame [[texture(5)]],
                        constant DenoiserUniforms &params [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint width = outputFrame.get_width();
    uint height = outputFrame.get_height();
    if (gid.x >= width || gid.y >= height)
        return;

    half4 currentSample = currentFrame.read(gid);
    float3 currentColor = float3(currentSample.xyz);

    half4 historySample = lastFrame.read(gid);
    float3 historyColor = float3(historySample.xyz);

    half4 albedoSample = albedoAccum.read(gid);
    float3 baseAlbedo = float3(albedoSample.xyz);

    half4 normalSample = normalAccum.read(gid);
    float3 baseNormal = sanitizeNormal(float3(normalSample.xyz));

    float sampleCount = float(sampleCountTex.read(gid).x);
    float sampleFactor = clamp(sampleCount / max(params.sampleCountForStableHistory, 1.0f), 0.0f, 1.0f);

    float historyWeight = 0.0f;
    if (params.historyValid != 0) {
        float lumCurr = luminance(currentColor);
        float lumHist = luminance(historyColor);
        float diff = fabs(lumCurr - lumHist);
        float luminanceWeight = gaussianWeight(diff * diff, max(params.luminanceSigma, 1e-3f));
        historyWeight = params.temporalBlendBase * sampleFactor * luminanceWeight;
    }
    historyWeight = clamp(historyWeight, 0.0f, 0.95f);
    float3 temporalColor = mix(currentColor, historyColor, historyWeight);

    int radius = int(params.radius);
    float3 accumColor = float3(0.0f);
    float accumWeight = 0.0f;
    float spatialSigma = max(params.spatialSigma, 0.5f);
    float normalSigma = max(params.normalSigma, 1e-3f);
    float albedoSigma = max(params.albedoSigma, 1e-3f);

    for (int y = -radius; y <= radius; ++y) {
        int iy = clamp(int(gid.y) + y, 0, int(height) - 1);
        for (int x = -radius; x <= radius; ++x) {
            int ix = clamp(int(gid.x) + x, 0, int(width) - 1);
            uint2 coord = uint2(ix, iy);

            float distance2 = float(x * x + y * y);
            float spatialWeight = gaussianWeight(distance2, spatialSigma);

            half4 neighborAlbedoSample = albedoAccum.read(coord);
            float3 neighborAlbedo = float3(neighborAlbedoSample.xyz);
            float3 albedoDiff = neighborAlbedo - baseAlbedo;
            float albedoWeight = gaussianWeight(dot(albedoDiff, albedoDiff), albedoSigma);

            half4 neighborNormalSample = normalAccum.read(coord);
            float3 neighborNormal = sanitizeNormal(float3(neighborNormalSample.xyz));
            float3 normalDiff = neighborNormal - baseNormal;
            float normalWeight = gaussianWeight(dot(normalDiff, normalDiff), normalSigma);

            half4 neighborCurrentSample = currentFrame.read(coord);
            float3 neighborCurrent = float3(neighborCurrentSample.xyz);
            half4 neighborHistorySample = lastFrame.read(coord);
            float3 neighborHistory = float3(neighborHistorySample.xyz);
            float neighborSamples = float(sampleCountTex.read(coord).x);
            float neighborFactor = clamp(neighborSamples / max(params.sampleCountForStableHistory, 1.0f), 0.0f, 1.0f);
            float neighborHistoryWeight = 0.0f;
            if (params.historyValid != 0) {
                float lumCurr = luminance(neighborCurrent);
                float lumHist = luminance(neighborHistory);
                float diff = fabs(lumCurr - lumHist);
                float lumWeight = gaussianWeight(diff * diff, max(params.luminanceSigma, 1e-3f));
                neighborHistoryWeight = params.temporalBlendBase * neighborFactor * lumWeight;
            }
            neighborHistoryWeight = clamp(neighborHistoryWeight, 0.0f, 0.95f);
            float3 neighborTemporal = mix(neighborCurrent, neighborHistory, neighborHistoryWeight);

            float weight = spatialWeight * normalWeight * albedoWeight;
            accumColor += neighborTemporal * weight;
            accumWeight += weight;
        }
    }

    float3 filtered = accumWeight > 0.0f ? accumColor / accumWeight : temporalColor;
    filtered = clamp(filtered, 0.0f, 1.0f);
    outputFrame.write(half4(float4(filtered, 1.0f)), gid);
}
