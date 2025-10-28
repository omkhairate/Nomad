#include <metal_stdlib>

using namespace metal;

#include "PathTracing.h"

kernel void pathTraceKernel(
    device const float4 *bvhNodes [[buffer(0)]],
    device const float4 *primitives [[buffer(1)]],
    device const float4 *materials [[buffer(2)]],
    device const UniformsData *uniforms [[buffer(3)]],
    device const float3 *vertexBuffer [[buffer(4)]],
    device const uint3 *indexBuffer [[buffer(5)]],
    device const int *primitiveIndices [[buffer(6)]],
    device const float4 *tlasNodes [[buffer(7)]],
    device const uchar *activeMask [[buffer(8)]],
    device const uint *lightIndices [[buffer(9)]],
    device const float *lightCdf [[buffer(10)]],
    device const uint *primitiveRemap [[buffer(11)]],
    device atomic_uint *primitiveHitCounts [[buffer(12)]],
    device const InstanceRecord *instanceRecords [[buffer(13)]],
    texture2d<half, access::read> lastFrame [[texture(0)]],
    texture2d<half, access::write> currentFrame [[texture(1)]],
    texture2d<half, access::read_write> sampleCount [[texture(2)]],
    texture2d<half, access::read> sampleImportance [[texture(3)]],
    texture2d<half, access::read_write> albedoAccum [[texture(4)]],
    texture2d<half, access::read_write> normalAccum [[texture(5)]],
    array<texture2d<float, access::sample>, kMaxMaterialTextures>
        materialTextures [[texture(6)]],
    constant TileRegion &tile [[buffer(16)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (!uniforms)
    return;

  if (vertexBuffer) {
  }
  if (indexBuffer) {
  }

  uint width = currentFrame.get_width();
  uint height = currentFrame.get_height();
  if (gid.x >= tile.size.x || gid.y >= tile.size.y)
    return;

  uint2 pixel = gid + tile.origin;
  if (pixel.x >= width || pixel.y >= height)
    return;

  const device UniformsData &u = *uniforms;
  float2 screenSize = float2(u.screenSize);
  if (screenSize.x <= 0.0f || screenSize.y <= 0.0f)
    return;

  float2 uv = (float2(pixel) + 0.5f) / screenSize;
  uint32_t seed = random(uv, u.randomSeed.xyz) * ((uint32_t)-1);

  float desiredSamples = float(sampleImportance.read(pixel).x);
  if (!isfinite(desiredSamples))
    desiredSamples = float(u.minSamplesPerPixel);

  desiredSamples = clamp(desiredSamples, float(u.minSamplesPerPixel),
                         float(u.maxSamplesPerPixel));

  uint requestedSamples = uint(round(desiredSamples));
  requestedSamples = clamp(requestedSamples, u.minSamplesPerPixel,
                           u.maxSamplesPerPixel);

  float storedSampleCount = float(sampleCount.read(pixel).x);
  uint previousSamples = uint(round(storedSampleCount));
  uint targetSamples = max(u.maxSamplesPerPixel, u.minSamplesPerPixel);
  previousSamples = min(previousSamples, targetSamples);
  uint remainingTargetSamples = 0u;
  if (previousSamples < targetSamples)
    remainingTargetSamples = targetSamples - previousSamples;

  uint dispatchBudget = max(u.maxSamplesPerDispatch, 1u);
  uint samplesThisFrame = requestedSamples;
  samplesThisFrame = min(samplesThisFrame, remainingTargetSamples);
  samplesThisFrame = min(samplesThisFrame, dispatchBudget);

  if (samplesThisFrame == 0u && remainingTargetSamples > 0u &&
      dispatchBudget > 0u)
    samplesThisFrame = min(remainingTargetSamples, dispatchBudget);

  float previousSampleCount = float(previousSamples);
  float4 previousColor = float4(lastFrame.read(pixel));

  float3 accumulatedColor = float3(0.0);
  float3 accumulatedAlbedo = float3(0.0);
  float3 accumulatedNormal = float3(0.0);

  float3 rayDx = u.rayDx;
  float3 rayDy = u.rayDy;

  for (uint sampleIdx = 0; sampleIdx < samplesThisFrame; ++sampleIdx) {
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

    PathTraceSample sample = rayColor(r, rayDx, rayDy, tlasNodes, u.tlasNodeCount,
                                      bvhNodes, primitives, materials,
                                      u.primitiveCount, primitiveIndices,
                                      activeMask, instanceRecords, lightIndices,
                                      lightCdf, primitiveRemap, primitiveHitCounts,
                                      seed, u.maxRayDepth, u.debugAS, u.blasNodeCount,
                                      u.lightCount, u.lightTotalWeight,
                                      static_cast<uint>(u.totalPrimitiveCount),
                                      materialTextures, u.textureCount);
    accumulatedColor += sample.radiance;
    accumulatedAlbedo += sample.albedo;
    accumulatedNormal += sample.normal;
  }

  float totalSamples = previousSampleCount + float(samplesThisFrame);
  float3 previousSum = previousColor.xyz * previousSampleCount;
  float3 combinedSum = previousSum + accumulatedColor;
  float3 averaged = (totalSamples > 0.0f) ? combinedSum / totalSamples : float3(0.0f);
  averaged = clamp(averaged, 0.0f, 1.0f);

  float3 previousAlbedo =
      float3(float4(albedoAccum.read(pixel)).xyz);
  float3 previousNormal =
      float3(float4(normalAccum.read(pixel)).xyz);
  float3 albedoSum = previousAlbedo * previousSampleCount + accumulatedAlbedo;
  float3 normalSum = previousNormal * previousSampleCount + accumulatedNormal;
  float3 averagedAlbedo =
      (totalSamples > 0.0f) ? albedoSum / totalSamples : float3(0.0f);
  averagedAlbedo = clamp(averagedAlbedo, 0.0f, 1.0f);
  float3 averagedNormal =
      (totalSamples > 0.0f) ? normalSum / totalSamples : float3(0.0f);
  averagedNormal = clamp(averagedNormal, -1.0f, 1.0f);

  float4 result = float4(averaged, 1.0f);
  currentFrame.write(half4(result), pixel);
  albedoAccum.write(half4(float4(averagedAlbedo, 1.0f)), pixel);
  normalAccum.write(half4(float4(averagedNormal, 1.0f)), pixel);
  sampleCount.write(half4(totalSamples, half(0.0f), half(0.0f), half(0.0f)), pixel);
}
