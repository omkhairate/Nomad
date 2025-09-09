#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using metal::raytracing::ray;

#include "PathTracing.h"

float4 fragment fragmentMain(
    v2f in [[stage_in]],
    device const float4* bvhNodes [[buffer(0)]],          // <--- ADD THIS LINE
    device const float4* primitives [[buffer(1)]],
    device const float4* materials [[buffer(2)]],
    device const UniformsData* uniforms [[buffer(3)]],
    device const float3* vertexBuffer [[buffer(4)]],
    device const uint3* indexBuffer [[buffer(5)]],
    device const int* primitiveIndices [[buffer(6)]],
    device const float4* tlasNodes [[buffer(7)]],
    device const uchar* activeMask [[buffer(8)]],
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
        u.tlasNodeCount,
        bvhNodes,
        primitives,       // <- Each primitive is 3 float4s
        materials,
        u.primitiveCount,
        u.primitiveIndex,
        primitiveIndices,
        activeMask,
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
