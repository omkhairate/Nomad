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
    device const uint* lightIndices [[buffer(9)]],
    device const float* lightCdf [[buffer(10)]],
    device const uint* primitiveRemap [[buffer(11)]],
    device atomic_uint* primitiveHitCounts [[buffer(12)]],
    device const InstanceRecord* instanceRecords [[buffer(13)]],
    device const float* tileBudgets [[buffer(14)]],
    texture2d<float, access::read_write> lastFrame [[texture(0)]],
    texture2d<float, access::read_write> currentFrame [[texture(1)]],
    texture2d<float, access::read_write> sampleCount [[texture(2)]],
    texture2d<float, access::read> sampleImportance [[texture(3)]])

{
    const device UniformsData& u = *uniforms;

    uint32_t seed = random(in.uv, u.randomSeed.xyz) * ((uint32_t)-1);

    uint2 coord = uint2(in.uv * u.screenSize);

    float desiredSamples = sampleImportance.read(coord).x;

    float tileMultiplier = 1.0f;
    if (u.tileCountX > 0 && u.tileCountY > 0 && u.tileSizeX > 0 && u.tileSizeY > 0 && tileBudgets)
    {
        uint tileX = min(coord.x / u.tileSizeX, u.tileCountX - 1);
        uint tileY = min(coord.y / u.tileSizeY, u.tileCountY - 1);
        uint tileIndex = tileY * u.tileCountX + tileX;
        tileMultiplier = tileBudgets[tileIndex];
        tileMultiplier = clamp(tileMultiplier, u.tileBudgetMinMultiplier, u.tileBudgetMaxMultiplier);
    }
    desiredSamples *= tileMultiplier;
    if (!isfinite(desiredSamples))
    {
        desiredSamples = float(u.minSamplesPerPixel);
    }

    uint samplesThisFrame = uint(round(desiredSamples));
    samplesThisFrame = clamp(samplesThisFrame, u.minSamplesPerPixel, u.maxSamplesPerPixel);
    samplesThisFrame = max(samplesThisFrame, 1u);

    float previousSampleCount = sampleCount.read(coord).x;
    float4 previousColor = lastFrame.read(coord);

    float4 accumulatedColor = float4(0.0);

    float3 rayDx = u.rayDx;
    float3 rayDy = u.rayDy;

    for (uint sample = 0; sample < samplesThisFrame; ++sample)
    {
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
        r.max_distance = INFINITY;

        accumulatedColor += rayColor(
            r,
            rayDx,
            rayDy,
            tlasNodes,
            u.tlasNodeCount,
            bvhNodes,
            primitives,
            materials,
            u.primitiveCount,
            primitiveIndices,
            activeMask,
            instanceRecords,
            lightIndices,
            lightCdf,
            primitiveRemap,
            primitiveHitCounts,
            seed,
            u.maxRayDepth,
            u.debugAS,
            u.blasNodeCount,
            u.lightCount,
            u.lightTotalWeight,
            static_cast<uint>(u.totalPrimitiveCount)
        );
    }

    float totalSamples = previousSampleCount + float(samplesThisFrame);
    float3 previousSum = previousColor.xyz * previousSampleCount;
    float3 combinedSum = previousSum + accumulatedColor.xyz;
    float3 averaged = (totalSamples > 0.0f) ? combinedSum / totalSamples : float3(0.0);
    averaged = clamp(averaged, 0.0, 1.0);

    float4 result = float4(averaged, 1.0);
    currentFrame.write(result, coord);
    sampleCount.write(totalSamples, coord);

    return result;
}
