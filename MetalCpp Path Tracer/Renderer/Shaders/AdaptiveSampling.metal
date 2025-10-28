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

kernel void adaptiveSamplingMain(
    texture2d<half, access::read> accumulation [[texture(0)]],
    texture2d<half, access::write> importance [[texture(1)]],
    texture2d<half, access::read> sampleCountTex [[texture(2)]],
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

    float mean = (center + left + right + up + down + diag0 + diag1 + diag2 + diag3) / 9.0f;
    float safeMean = max(mean, 1e-4f);
    float invMean = 1.0f / safeMean;

    float gradient = (fabs(right - left) + fabs(down - up)) * invMean;
    float diagonal = (fabs(diag0 - diag3) + fabs(diag1 - diag2)) * invMean;
    float contrast =
        fabs(center - 0.25f * (left + right + up + down)) * invMean;

    float variance = 0.0f;
    variance += (center - mean) * (center - mean);
    variance += (left - mean) * (left - mean);
    variance += (right - mean) * (right - mean);
    variance += (up - mean) * (up - mean);
    variance += (down - mean) * (down - mean);
    variance += (diag0 - mean) * (diag0 - mean);
    variance += (diag1 - mean) * (diag1 - mean);
    variance += (diag2 - mean) * (diag2 - mean);
    variance += (diag3 - mean) * (diag3 - mean);
    variance /= 9.0f;

    float normalizedVariance = variance / (safeMean * safeMean + 1e-8f);
    float varianceClamped = clamp(normalizedVariance, 0.0f, 4.0f);
    float varianceSqrt = sqrt(varianceClamped) * 0.5f;

    float spatialDetail = gradient + 0.5f * diagonal + contrast;
    float normalizedSpatial = clamp(spatialDetail / 3.0f, 0.0f, 1.0f);
    float baseDetail = clamp(normalizedSpatial + 0.6f * varianceSqrt, 0.0f, 1.0f);

    float historySamples = float(sampleCountTex.read(gid).x);
    float minSamples = float(max(uniforms.minSamplesPerPixel, 1u));
    float maxSamples =
        float(max(uniforms.maxSamplesPerPixel, uniforms.minSamplesPerPixel));
    float completion = (maxSamples > 0.0f)
                           ? clamp(historySamples / maxSamples, 0.0f, 1.0f)
                           : 0.0f;

    float varianceHistory =
        varianceSqrt * rsqrt(max(historySamples, 1.0f));
    varianceHistory = clamp(varianceHistory * 1.5f, 0.0f, 1.0f);

    float stability = smoothstep(0.3f, 0.9f, completion);
    float blendedDetail = mix(baseDetail, varianceHistory, stability);
    float detailFloor = 0.05f * (1.0f - completion);
    float finalDetail = clamp(max(blendedDetail, detailFloor), 0.0f, 1.0f);

    float sampleBudget = mix(minSamples, maxSamples, finalDetail);

    importance.write(
        half4(sampleBudget, half(finalDetail), half(varianceSqrt), half(completion)),
        gid);
}
