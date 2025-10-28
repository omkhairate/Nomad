#include <metal_stdlib>

using namespace metal;

#include "Structs.h"

inline float luminance(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

inline float fetchLuminance(texture2d<half, access::read> accumulation,
                            uint width, uint height, int2 coord)
{
    int x = clamp(coord.x, 0, int(width - 1));
    int y = clamp(coord.y, 0, int(height - 1));
    return luminance(float3(accumulation.read(uint2(x, y)).xyz));
}

inline float safeFetchLuminance(texture2d<half, access::read> source,
                               uint width,
                               uint height,
                               int2 coord,
                               float fallback)
{
    if (width == 0 || height == 0)
    {
        return fallback;
    }
    return fetchLuminance(source, width, height, coord);
}

inline float4 safeFetchSampleState(texture2d<float, access::read> source,
                                   uint width,
                                   uint height,
                                   uint2 coord)
{
    if (width == 0 || height == 0)
    {
        return float4(0.0f);
    }

    uint clampedX = min(coord.x, width - 1);
    uint clampedY = min(coord.y, height - 1);

    float4 value = source.read(uint2(clampedX, clampedY));
    if (!(isfinite(value.x) && isfinite(value.y) && isfinite(value.z) && isfinite(value.w)))
    {
        return float4(0.0f);
    }

    value.x = max(value.x, 0.0f);
    value.y = max(value.y, 0.0f);
    value.z = max(value.z, 0.0f);
    value.w = max(value.w, 0.0f);
    return value;
}

kernel void adaptiveSamplingMain(
    texture2d<half, access::read> accumulation [[texture(0)]],
    texture2d<half, access::write> importance [[texture(1)]],
    texture2d<float, access::read> sampleCountTexture [[texture(2)]],
    texture2d<half, access::read> historyTexture [[texture(3)]],
    constant UniformsData& uniforms [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = accumulation.get_width();
    uint height = accumulation.get_height();

    if (gid.x >= width || gid.y >= height)
    {
        return;
    }

    int2 coord = int2(gid);

    float center = fetchLuminance(accumulation, width, height, coord);
    float left = fetchLuminance(accumulation, width, height, coord + int2(-1, 0));
    float right = fetchLuminance(accumulation, width, height, coord + int2(1, 0));
    float up = fetchLuminance(accumulation, width, height, coord + int2(0, -1));
    float down = fetchLuminance(accumulation, width, height, coord + int2(0, 1));
    float diag0 = fetchLuminance(accumulation, width, height, coord + int2(-1, -1));
    float diag1 = fetchLuminance(accumulation, width, height, coord + int2(1, -1));
    float diag2 = fetchLuminance(accumulation, width, height, coord + int2(-1, 1));
    float diag3 = fetchLuminance(accumulation, width, height, coord + int2(1, 1));

    float gradient = fabs(right - left) + fabs(down - up);
    float diagonal = fabs(diag0 - diag3) + fabs(diag1 - diag2);
    float contrast = fabs(center - 0.25f * (left + right + up + down));

    float detail = gradient + 0.5f * diagonal + 4.0f * contrast;
    if (!isfinite(detail))
    {
        detail = 0.0f;
    }

    uint sampleWidth = sampleCountTexture.get_width();
    uint sampleHeight = sampleCountTexture.get_height();
    float4 sampleState = safeFetchSampleState(sampleCountTexture, sampleWidth, sampleHeight, gid);
    float totalSamples = sampleState.x;
    float runningMean = sampleState.y;
    float variance = sampleState.w;
    if (totalSamples > 1.0f && variance <= 0.0f)
    {
        float denom = max(totalSamples - 1.0f, 1.0f);
        variance = max(sampleState.z / denom, 0.0f);
    }

    float minSamples = float(max(uniforms.minSamplesPerPixel, 1u));
    float maxSamples = float(max(uniforms.maxSamplesPerPixel, uniforms.minSamplesPerPixel));

    float completion = 0.0f;
    float sampleCompletion = 0.0f;
    if (maxSamples > 0.0f)
    {
        sampleCompletion = clamp(totalSamples / maxSamples, 0.0f, 1.0f);
    }

    constexpr float kAbsoluteErrorThreshold = 1.0f / 512.0f;
    constexpr float kRelativeErrorThreshold = 0.02f;
    if (totalSamples >= 2.0f && kAbsoluteErrorThreshold > 0.0f && kRelativeErrorThreshold > 0.0f)
    {
        variance = max(variance, 0.0f);
        float error = sqrt(variance);
        if (isfinite(error))
        {
            float normalizedAbsolute = error / kAbsoluteErrorThreshold;
            float safeMean = max(runningMean, 1e-4f);
            float normalizedRelative = error / (safeMean * kRelativeErrorThreshold);
            float absoluteCompletion = 1.0f - normalizedAbsolute;
            float relativeCompletion = 1.0f - normalizedRelative;
            if (!isfinite(absoluteCompletion))
            {
                absoluteCompletion = 0.0f;
            }
            if (!isfinite(relativeCompletion))
            {
                relativeCompletion = 0.0f;
            }
            completion = max(absoluteCompletion, relativeCompletion);
        }
    }

    if (totalSamples < 2.0f && minSamples > 0.0f)
    {
        completion = max(completion, clamp(totalSamples / minSamples, 0.0f, 1.0f));
    }

    completion = max(completion, sampleCompletion);
    completion = clamp(completion, 0.0f, 1.0f);

    uint historyWidth = historyTexture.get_width();
    uint historyHeight = historyTexture.get_height();
    float historyCenter = safeFetchLuminance(historyTexture, historyWidth, historyHeight, coord, center);
    float motionMagnitude = fabs(center - historyCenter);

    float historyLeft = safeFetchLuminance(historyTexture, historyWidth, historyHeight,
                                           coord + int2(-1, 0), left);
    float historyRight = safeFetchLuminance(historyTexture, historyWidth, historyHeight,
                                            coord + int2(1, 0), right);
    float historyUp = safeFetchLuminance(historyTexture, historyWidth, historyHeight,
                                         coord + int2(0, -1), up);
    float historyDown = safeFetchLuminance(historyTexture, historyWidth, historyHeight,
                                           coord + int2(0, 1), down);

    motionMagnitude = max(motionMagnitude, fabs(left - historyLeft));
    motionMagnitude = max(motionMagnitude, fabs(right - historyRight));
    motionMagnitude = max(motionMagnitude, fabs(up - historyUp));
    motionMagnitude = max(motionMagnitude, fabs(down - historyDown));
    motionMagnitude = clamp(motionMagnitude, 0.0f, 1.0f);

    float globalMotion = clamp(uniforms.motionIntensity, 0.0f, 1.0f);
    float motionFactor = clamp(motionMagnitude * 4.0f + globalMotion * 0.5f, 0.0f, 1.0f);

    float detailFloorBase = 0.05f * (1.0f - completion);
    float detailFloor = detailFloorBase * (1.0f - motionFactor);
    detailFloor = clamp(detailFloor, 0.0f, 1.0f);

    detail = clamp(max(detail, detailFloor), 0.0f, 1.0f);

    float sampleBudget = mix(minSamples, maxSamples, detail);
    if (!isfinite(sampleBudget))
    {
        sampleBudget = minSamples;
    }

    importance.write(half4(sampleBudget, half(0.0), half(0.0), half(0.0)), gid);
}
