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
    device atomic_uint *primitiveRayStats [[buffer(12)]],
    device const InstanceRecord *instanceRecords [[buffer(13)]],
    texture2d<float, access::write> currentFrame [[texture(0)]],
    texture2d<float, access::write> albedoAccum [[texture(1)]],
    texture2d<float, access::write> normalAccum [[texture(2)]],
    texture2d<float, access::write> positionAccum [[texture(3)]],
    texture2d<float, access::write> restirData0Out [[texture(4)]],
    texture2d<float, access::write> restirData1Out [[texture(5)]],
    texture2d<float, access::write> restirData2Out [[texture(6)]],
    texture2d<float, access::read> restirData0Prev [[texture(7)]],
    texture2d<float, access::read> restirData1Prev [[texture(8)]],
    texture2d<float, access::read> restirData2Prev [[texture(9)]],
    array<texture2d<float, access::sample>, kMaxMaterialTextures>
        materialTextures [[texture(13)]],
    texture2d<float, access::sample> environmentMap
        [[texture(13 + kMaxMaterialTextures)]],
    sampler environmentSampler [[sampler(0)]],
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

  uint samplesThisFrame = max(u.maxSamplesPerPixel, 1u);

  float3 accumulatedColor = float3(0.0);
  float3 accumulatedAlbedo = float3(0.0);
  float3 accumulatedNormal = float3(0.0);
  float3 accumulatedPosition = float3(0.0);
  float accumulatedRoughness = 0.0f;

  float3 rayDx = u.rayDx;
  float3 rayDy = u.rayDy;
  float footprint = length(cross(rayDx, rayDy));
  float lodAtten = 1.0 / (1.0 + footprint);

  for (uint sampleIdx = 0; sampleIdx < samplesThisFrame; ++sampleIdx) {
    float xOff = (randomFloat(seed) - 0.5f) / screenSize.x;
    seed = random(seed);
    float yOff = (randomFloat(seed) - 0.5f) / screenSize.y;
    seed = random(seed);

    float3 rayDir = (u.firstPixelPosition + (uv.x + xOff) * u.viewportU +
                     (uv.y + yOff) * u.viewportV) -
                    u.cameraPosition;
    float3 rayOrigin = u.cameraPosition;
    float3 rayDirection = normalize(rayDir);
    if (u.aperture > 0.0f) {
      float lensU = randomFloat(seed);
      seed = random(seed);
      float lensV = randomFloat(seed);
      seed = random(seed);
      float2 diskSample = concentricSampleDisk(float2(lensU, lensV));
      float lensRadius = u.aperture;
      float3 cameraRight = normalize(u.viewportU);
      float3 cameraUp = normalize(u.viewportV);
      float3 lensOffset =
          lensRadius * (diskSample.x * cameraRight + diskSample.y * cameraUp);
      float focusDistance = max(u.focusDistance, 0.0001f);
      float3 focusPoint = u.cameraPosition + rayDirection * focusDistance;
      rayOrigin = u.cameraPosition + lensOffset;
      rayDirection = normalize(focusPoint - rayOrigin);
    }

    Ray r;
    r.origin = rayOrigin;
    r.direction = rayDirection;
    r.minDistance = 0.0001f;
    r.maxDistance = INFINITY;

    bool restirWriteEnabled = (sampleIdx == 0);
    PathTraceSample sample = rayColor(
        r, rayDx, rayDy, tlasNodes, u.tlasNodeCount, bvhNodes, primitives,
        materials, u.primitiveCount, primitiveIndices, activeMask,
        instanceRecords, lightIndices, lightCdf, primitiveRemap,
        primitiveRayStats, seed, u.maxRayDepth, u.debugAS, u.blasNodeCount,
        u.lightCount, u.lightTotalWeight,
        static_cast<uint>(u.totalPrimitiveCount), materialTextures,
        u.textureCount, environmentMap, environmentSampler,
        u.environmentMapEnabled, u.environmentMapIntensity, pixel,
        restirData0Out, restirData1Out, restirData2Out, restirData0Prev,
        restirData1Prev, restirData2Prev, u.restirEnabled,
        u.restirCandidateCount, u.restirTemporalReuse, u.prevViewProjection,
        u.cameraMotionMetric, restirWriteEnabled);
    accumulatedColor += sample.radiance;
    accumulatedAlbedo += sample.albedo;
    accumulatedNormal += sample.normal;
    accumulatedPosition += sample.position;
    accumulatedRoughness += sample.roughness;
  }

  float totalSamples = float(samplesThisFrame);
  float3 averaged =
      (totalSamples > 0.0f) ? accumulatedColor / totalSamples : float3(0.0f);
  averaged = clamp(averaged, 0.0f, 1.0f);

  float3 averagedAlbedo =
      (totalSamples > 0.0f) ? accumulatedAlbedo / totalSamples : float3(0.0f);
  averagedAlbedo = clamp(averagedAlbedo, 0.0f, 1.0f);
  float3 averagedNormal =
      (totalSamples > 0.0f) ? accumulatedNormal / totalSamples : float3(0.0f);
  averagedNormal = clamp(averagedNormal, -1.0f, 1.0f);
  float3 averagedPosition =
      (totalSamples > 0.0f) ? accumulatedPosition / totalSamples : float3(0.0f);
  float averagedRoughness =
      (totalSamples > 0.0f) ? accumulatedRoughness / totalSamples : 0.0f;
  averagedRoughness = clamp(averagedRoughness, 0.0f, 1.0f);

  float4 result = float4(averaged, 1.0f);
  currentFrame.write(result, pixel);
  albedoAccum.write(float4(averagedAlbedo, 1.0f), pixel);
  normalAccum.write(float4(averagedNormal, averagedRoughness), pixel);
  positionAccum.write(float4(averagedPosition, 1.0f), pixel);
}
