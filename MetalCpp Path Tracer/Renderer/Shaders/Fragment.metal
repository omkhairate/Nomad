#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using metal::raytracing::ray;

#include "PathTracing.h"

float4 fragment fragmentMain(
    v2f in [[stage_in]],
    device const UniformsData* uniforms [[buffer(0)]],
    device const float4* tlasNodes [[buffer(1)]],
    constant InstanceMetadata* instanceMetadata [[buffer(2)]],
    device InstanceArgumentBuffer& instanceArgs [[buffer(3)]],
    device const int* tlasInstanceIndices [[buffer(4)]],
    texture2d<float, access::read_write> lastFrame [[texture(0)]],
    texture2d<float, access::read_write> currentFrame [[texture(1)]])

{
    const device UniformsData& u = *uniforms;

    if (u.frameCount == 0)
    {
        uint2 coord = uint2(in.uv * u.screenSize);
        lastFrame.write(0, coord);
    }

    uint32_t seed = random(in.uv, u.randomSeed.xyz) * ((uint32_t)-1);

    float xOff = (randomFloat(seed) - 0.5) / u.screenSize.x;
    seed = random(seed);
    float yOff = (randomFloat(seed) - 0.5) / u.screenSize.y;
    seed = random(seed);

    float3 rayDir = (
        u.firstPixelPosition +
        (in.uv.x + xOff) * u.viewportU +
        (in.uv.y + yOff) * u.viewportV
    ) - u.cameraPosition;

    ray r{u.cameraPosition, normalize(rayDir)};
    r.min_distance = 0.0001;
    r.max_distance = INFINITY; // or INFINITY

    float3 rayDx = u.rayDx;
    float3 rayDy = u.rayDy;

    float4 color = rayColor(
        r,
        rayDx,
        rayDy,
        tlasNodes,
        tlasInstanceIndices,
        u.tlasNodeCount,
        instanceArgs,
        instanceMetadata,
        seed,
        u.maxRayDepth,
        u.debugAS,
        u.blasNodeCount
    );



    uint2 coord = uint2(in.uv * u.screenSize);
    uint64_t frameCount = u.frameCount + 1;

    color += lastFrame.read(coord) * (float)(frameCount - 1);
    color /= frameCount;
    color = clamp(color, 0.0, 1.0);

    currentFrame.write(color, coord);
    color.w = 1;
    return color;
}
