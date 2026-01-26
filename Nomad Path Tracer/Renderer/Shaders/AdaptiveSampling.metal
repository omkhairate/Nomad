#include <metal_stdlib>

using namespace metal;

#include "Structs.h"

inline float luminance(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

inline float fetchLuminance(texture2d<float, access::read> currentFrame,
                            uint width, uint height, int2 coord)
{
    int x = clamp(coord.x, 0, int(width - 1));
    int y = clamp(coord.y, 0, int(height - 1));
    return luminance(float3(currentFrame.read(uint2(x, y)).xyz));
}

kernel void adaptiveSamplingMain(
    texture2d<float, access::read> currentFrame [[texture(0)]],
    texture2d<half, access::write> importance [[texture(1)]],
    constant UniformsData& uniforms [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint width = currentFrame.get_width();
    uint height = currentFrame.get_height();

    if (gid.x >= width || gid.y >= height)
    {
        return;
    }

    int2 coord = int2(gid);

    float center = fetchLuminance(currentFrame, width, height, coord);
    float left = fetchLuminance(currentFrame, width, height, coord + int2(-1, 0));
    float right = fetchLuminance(currentFrame, width, height, coord + int2(1, 0));
    float up = fetchLuminance(currentFrame, width, height, coord + int2(0, -1));
    float down = fetchLuminance(currentFrame, width, height, coord + int2(0, 1));
    float diag0 = fetchLuminance(currentFrame, width, height, coord + int2(-1, -1));
    float diag1 = fetchLuminance(currentFrame, width, height, coord + int2(1, -1));
    float diag2 = fetchLuminance(currentFrame, width, height, coord + int2(-1, 1));
    float diag3 = fetchLuminance(currentFrame, width, height, coord + int2(1, 1));

    float gradient = fabs(right - left) + fabs(down - up);
    float diagonal = fabs(diag0 - diag3) + fabs(diag1 - diag2);
    float contrast = fabs(center - 0.25f * (left + right + up + down));

    float detail = gradient + 0.5f * diagonal + 4.0f * contrast;
    detail = clamp(detail, 0.0f, 1.0f);

    float minSamples = float(max(uniforms.minSamplesPerPixel, 1u));
    float maxSamples = float(max(uniforms.maxSamplesPerPixel, uniforms.minSamplesPerPixel));
    float sampleBudget = mix(minSamples, maxSamples, detail);

    importance.write(half4(sampleBudget, half(0.0), half(0.0), half(0.0)), gid);
}
