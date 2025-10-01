#include <metal_raytracing>
#include <metal_stdlib>

using namespace metal;

#include "PathTracing.h"

kernel void pathTraceKernel(
    instance_acceleration_structure tlas [[buffer(0)]],
    device const GeometryHandle *geometryHandles [[buffer(1)]],
    device const float4 *primitives [[buffer(2)]],
    device const float4 *materials [[buffer(3)]],
    device const UniformsData *uniforms [[buffer(4)]],
    device const uchar *activeMask [[buffer(5)]],
    device const uint *lightIndices [[buffer(6)]],
    device const float *lightCdf [[buffer(7)]],
    device const uint *primitiveRemap [[buffer(8)]],
    device atomic_uint *primitiveHitCounts [[buffer(9)]],
    device const InstanceRecord *instanceRecords [[buffer(10)]],
    texture2d<half, access::read> lastFrame [[texture(0)]],
    texture2d<half, access::write> currentFrame [[texture(1)]],
    texture2d<half, access::read_write> sampleCount [[texture(2)]],
    texture2d<half, access::read> sampleImportance [[texture(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (!uniforms)
    return;

  uint width = currentFrame.get_width();
  uint height = currentFrame.get_height();
  if (gid.x >= width || gid.y >= height)
    return;

  const device UniformsData &u = *uniforms;
  float2 screenSize = float2(u.screenSize);
  if (screenSize.x <= 0.0f || screenSize.y <= 0.0f)
    return;

  float2 uv = (float2(gid) + 0.5f) / screenSize;
  uint32_t seed = random(uv, u.randomSeed.xyz) * ((uint32_t)-1);

  float desiredSamples = float(sampleImportance.read(gid).x);
  if (!isfinite(desiredSamples))
    desiredSamples = float(u.minSamplesPerPixel);

  uint samplesThisFrame = uint(round(desiredSamples));
  samplesThisFrame = clamp(samplesThisFrame, u.minSamplesPerPixel, u.maxSamplesPerPixel);
  samplesThisFrame = max(samplesThisFrame, 1u);

  float previousSampleCount = float(sampleCount.read(gid).x);
  float4 previousColor = float4(lastFrame.read(gid));

  float4 accumulatedColor = float4(0.0);

  float3 rayDx = u.rayDx;
  float3 rayDy = u.rayDy;

  for (uint sample = 0; sample < samplesThisFrame; ++sample) {
    float xOff = (randomFloat(seed) - 0.5f) / screenSize.x;
    seed = random(seed);
    float yOff = (randomFloat(seed) - 0.5f) / screenSize.y;
    seed = random(seed);

    float3 rayDir = (u.firstPixelPosition + (uv.x + xOff) * u.viewportU +
                     (uv.y + yOff) * u.viewportV) -
                    u.cameraPosition;

    Ray r;
    r.origin = u.cameraPosition;
    r.direction = normalize(rayDir);
    r.minDistance = 0.0001f;
    r.maxDistance = INFINITY;

    accumulatedColor += rayColor(r, rayDx, rayDy, tlas, geometryHandles, primitives,
                                 materials, u.primitiveCount, activeMask, instanceRecords,
                                 lightIndices, lightCdf, primitiveRemap, primitiveHitCounts,
                                 seed, u.maxRayDepth, u.debugAS, u.lightCount,
                                 u.lightTotalWeight, static_cast<uint>(u.totalPrimitiveCount),
                                 u.tlasNodeCount, u.blasNodeCount);
  }

  float totalSamples = previousSampleCount + float(samplesThisFrame);
  float3 previousSum = previousColor.xyz * previousSampleCount;
  float3 combinedSum = previousSum + accumulatedColor.xyz;
  float3 averaged = (totalSamples > 0.0f) ? combinedSum / totalSamples : float3(0.0f);
  averaged = clamp(averaged, 0.0f, 1.0f);

  float4 result = float4(averaged, 1.0f);
  currentFrame.write(half4(result), gid);
  sampleCount.write(half4(totalSamples, half(0.0f), half(0.0f), half(0.0f)), gid);
}
