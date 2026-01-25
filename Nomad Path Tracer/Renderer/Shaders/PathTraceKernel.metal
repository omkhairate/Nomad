#include <metal_stdlib>

using namespace metal;

#include "PathTracing.h"

inline bool loadRestirReservoir(texture2d<float, access::read> restirSample,
                                texture2d<float, access::read> restirNormal,
                                texture2d<float, access::read> restirState,
                                uint2 pixel,
                                thread RestirReservoir &reservoir) {
  float4 state = restirState.read(pixel);
  float4 sample = restirSample.read(pixel);
  float4 normal = restirNormal.read(pixel);
  if (isfinite(state.x) && isfinite(state.y) && state.x > 0.0f &&
      state.y > 0.0f && isfinite(sample.w) && sample.w >= 0.0f) {
    reservoir.valid = 1u;
    reservoir.W = state.x;
    reservoir.M = state.y;
    reservoir.sample.position = sample.xyz;
    reservoir.sample.normal = normal.xyz;
    reservoir.sample.area = normal.w;
    reservoir.sample.lightIndex = uint(sample.w);
    reservoir.selectedFromTemporal = 0u;
    return true;
  }
  return false;
}

inline void writeRestirReservoir(texture2d<float, access::write> restirSample,
                                 texture2d<float, access::write> restirNormal,
                                 texture2d<float, access::write> restirState,
                                 uint2 pixel,
                                 thread const RestirReservoir &reservoir) {
  float4 outSample = float4(0.0f);
  float4 outNormal = float4(0.0f);
  float4 outState = float4(0.0f);
  if (reservoir.valid != 0u && reservoir.W > 0.0f && reservoir.M > 0.0f) {
    outSample = float4(reservoir.sample.position,
                       float(reservoir.sample.lightIndex));
    outNormal = float4(reservoir.sample.normal, reservoir.sample.area);
    outState = float4(reservoir.W, reservoir.M, 0.0f, 1.0f);
  }
  restirSample.write(outSample, pixel);
  restirNormal.write(outNormal, pixel);
  restirState.write(outState, pixel);
}

kernel void restirSpatialKernel(
    device const UniformsData *uniforms [[buffer(0)]],
    device const float4 *bvhNodes [[buffer(1)]],
    device const float4 *primitives [[buffer(2)]],
    device const float4 *materials [[buffer(3)]],
    device const int *primitiveIndices [[buffer(4)]],
    device const float4 *tlasNodes [[buffer(5)]],
    device const uchar *activeMask [[buffer(6)]],
    device const uint *lightIndices [[buffer(7)]],
    device const float *lightCdf [[buffer(8)]],
    device const uint *primitiveRemap [[buffer(9)]],
    device atomic_uint *primitiveRayStats [[buffer(10)]],
    device const InstanceRecord *instanceRecords [[buffer(11)]],
    texture2d<float, access::read> albedoAccum [[texture(0)]],
    texture2d<float, access::read> normalAccum [[texture(1)]],
    texture2d<float, access::read> positionAccum [[texture(2)]],
    texture2d<float, access::read> restirInSample [[texture(3)]],
    texture2d<float, access::read> restirInNormal [[texture(4)]],
    texture2d<float, access::read> restirInState [[texture(5)]],
    texture2d<float, access::write> restirOutSample [[texture(6)]],
    texture2d<float, access::write> restirOutNormal [[texture(7)]],
    texture2d<float, access::write> restirOutState [[texture(8)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (!uniforms)
    return;
  if (!bvhNodes || !primitives || !materials || !primitiveIndices || !tlasNodes ||
      !activeMask || !lightIndices || !lightCdf || !primitiveRemap ||
      !instanceRecords)
    return;

  uint width = restirInSample.get_width();
  uint height = restirInSample.get_height();
  if (gid.x >= width || gid.y >= height)
    return;

  const device UniformsData &u = *uniforms;
  float2 screenSize = float2(u.screenSize);
  if (screenSize.x <= 0.0f || screenSize.y <= 0.0f)
    return;

  float2 uv = (float2(gid) + 0.5f) / screenSize;
  uint32_t seed = random(uv, u.randomSeed.xyz) * ((uint32_t)-1);
  seed ^= uint32_t(u.frameCount + gid.x * 1973u + gid.y * 9277u);

  RestirReservoir combined{};
  combined.valid = 0u;
  combined.W = 0.0f;
  combined.M = 0.0f;

  if (u.lightCount == 0 || u.lightTotalWeight <= 0.0f) {
    writeRestirReservoir(restirOutSample, restirOutNormal, restirOutState, gid,
                         combined);
    return;
  }

  float3 shadingPosition = positionAccum.read(gid).xyz;
  float3 shadingNormal = normalAccum.read(gid).xyz;
  float3 diffuseColor = albedoAccum.read(gid).xyz;
  if (!all(isfinite(shadingPosition)) || !all(isfinite(shadingNormal)) ||
      !all(isfinite(diffuseColor))) {
    writeRestirReservoir(restirOutSample, restirOutNormal, restirOutState, gid,
                         combined);
    return;
  }

  float normalLen = length(shadingNormal);
  if (normalLen <= 1e-4f) {
    writeRestirReservoir(restirOutSample, restirOutNormal, restirOutState, gid,
                         combined);
    return;
  }
  shadingNormal = shadingNormal / normalLen;

  float diffuseLum = luminance(diffuseColor);
  if (diffuseLum <= 0.0f) {
    writeRestirReservoir(restirOutSample, restirOutNormal, restirOutState, gid,
                         combined);
    return;
  }

  intersection bestHit{};
  bestHit.point = shadingPosition;
  bestHit.normal = shadingNormal;
  bestHit.primitiveId = -1;
  TlasLeafCache bounceCache;
  bounceCache.valid = false;
  float4 absorption = float4(1.0f);

  constexpr int kSpatialTapCount = 5;
  const int2 offsets[kSpatialTapCount] = {
      int2(0, 0), int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1)};
  constexpr float kSpatialNormalThreshold = 0.85f;
  constexpr float kSpatialDepthThreshold = 0.25f;
  constexpr float kSpatialPositionThreshold = 0.5f;

  for (int tap = 0; tap < kSpatialTapCount; ++tap) {
    int2 offset = offsets[tap];
    int2 sampleCoord = int2(gid) + offset;
    sampleCoord.x = clamp(sampleCoord.x, 0, int(width) - 1);
    sampleCoord.y = clamp(sampleCoord.y, 0, int(height) - 1);
    uint2 pixel = uint2(sampleCoord);

    float3 neighborPosition = positionAccum.read(pixel).xyz;
    float3 neighborNormal = normalAccum.read(pixel).xyz;
    if (!all(isfinite(neighborPosition)) || !all(isfinite(neighborNormal)))
      continue;
    float neighborNormalLen = length(neighborNormal);
    if (neighborNormalLen <= 1e-4f)
      continue;
    neighborNormal /= neighborNormalLen;

    float normalSimilarity = dot(shadingNormal, neighborNormal);
    if (normalSimilarity < kSpatialNormalThreshold)
      continue;
    float3 delta = neighborPosition - shadingPosition;
    float depthDelta = abs(dot(delta, shadingNormal));
    if (depthDelta > kSpatialDepthThreshold)
      continue;
    if (length(delta) > kSpatialPositionThreshold)
      continue;

    RestirReservoir candidate{};
    if (!loadRestirReservoir(restirInSample, restirInNormal, restirInState,
                             pixel, candidate))
      continue;
    if (candidate.W <= 0.0f || candidate.M <= 0.0f)
      continue;
    uint lightOffset = candidate.sample.lightIndex;
    if (lightOffset >= u.lightCount)
      continue;
    if (!isfinite(candidate.sample.area) || candidate.sample.area <= 0.0f)
      continue;
    uint lightPrimIndex = lightIndices[lightOffset];
    RestirEvaluation eval = evaluateLightCandidate(
        lightOffset, lightPrimIndex, candidate.sample.position,
        candidate.sample.normal, candidate.sample.area, shadingNormal,
        diffuseColor, absorption, bestHit, bounceCache, tlasNodes,
        uint(u.tlasNodeCount), bvhNodes, primitives, primitiveIndices, activeMask,
        instanceRecords, primitiveRemap, uint(u.primitiveCount),
        uint(u.totalPrimitiveCount), primitiveRayStats, materials, lightCdf,
        u.lightTotalWeight);
    if (!eval.valid || eval.pdf <= 0.0f)
      continue;

    float candidateM = candidate.M;
    if (!isfinite(candidateM) || candidateM <= 0.0f)
      continue;
    float weight = (eval.target / eval.pdf) * candidateM;
    if (weight <= 0.0f || !isfinite(weight))
      continue;
    updateRestirReservoir(combined, candidate.sample, weight, false, seed);
    if (candidateM > 1.0f)
      combined.M += candidateM - 1.0f;
  }

  writeRestirReservoir(restirOutSample, restirOutNormal, restirOutState, gid,
                       combined);
}

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
    device atomic_uint *restirStats [[buffer(14)]],
    texture2d<float, access::read> lastFrame [[texture(0)]],
    texture2d<float, access::write> currentFrame [[texture(1)]],
    texture2d<float, access::read_write> sampleCount [[texture(2)]],
    texture2d<half, access::read> sampleImportance [[texture(3)]],
    texture2d<float, access::read_write> albedoAccum [[texture(4)]],
    texture2d<float, access::read_write> normalAccum [[texture(5)]],
    texture2d<float, access::read_write> positionAccum [[texture(6)]],
    texture2d<float, access::read> restirPrevSample [[texture(7)]],
    texture2d<float, access::read> restirPrevNormal [[texture(8)]],
    texture2d<float, access::read> restirPrevState [[texture(9)]],
    texture2d<float, access::write> restirOutSample [[texture(10)]],
    texture2d<float, access::write> restirOutNormal [[texture(11)]],
    texture2d<float, access::write> restirOutState [[texture(12)]],
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

  float desiredSamples = float(sampleImportance.read(pixel).x);
  if (!isfinite(desiredSamples))
    desiredSamples = float(u.minSamplesPerPixel);

  uint samplesThisFrame = uint(round(desiredSamples));
  samplesThisFrame = clamp(samplesThisFrame, u.minSamplesPerPixel, u.maxSamplesPerPixel);
  samplesThisFrame = max(samplesThisFrame, 1u);

  float previousSampleCount = float(sampleCount.read(pixel).x);
  float4 previousColor = float4(lastFrame.read(pixel));

  float3 accumulatedColor = float3(0.0);
  float3 accumulatedAlbedo = float3(0.0);
  float3 accumulatedNormal = float3(0.0);
  float3 accumulatedPosition = float3(0.0);

  float3 rayDx = u.rayDx;
  float3 rayDy = u.rayDy;

  bool restirEnabled = (u.restirSamplingEnabled != 0u);
  RestirReservoir restir{};
  if (restirEnabled) {
    float4 prevState = restirPrevState.read(pixel);
    float4 prevSample = restirPrevSample.read(pixel);
    float4 prevNormal = restirPrevNormal.read(pixel);
    if (isfinite(prevState.x) && isfinite(prevState.y) &&
        prevState.x > 0.0f && prevState.y > 0.0f && isfinite(prevSample.w) &&
        prevSample.w >= 0.0f) {
      restir.valid = 1u;
      restir.W = prevState.x;
      restir.M = prevState.y;
      restir.sample.position = prevSample.xyz;
      restir.sample.normal = prevNormal.xyz;
      restir.sample.area = prevNormal.w;
      restir.sample.lightIndex = uint(prevSample.w);
      restir.selectedFromTemporal = 1u;
    }
  }

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

    PathTraceSample sample = rayColor(r, rayDx, rayDy, tlasNodes, u.tlasNodeCount,
                                      bvhNodes, primitives, materials,
                                      u.primitiveCount, primitiveIndices,
                                      activeMask, instanceRecords, lightIndices,
                                      lightCdf, primitiveRemap, primitiveRayStats,
                                      seed, u.maxRayDepth, u.debugAS, u.blasNodeCount,
                                      u.lightCount, u.lightTotalWeight,
                                      static_cast<uint>(u.totalPrimitiveCount),
                                      materialTextures, u.textureCount,
                                      environmentMap, environmentSampler,
                                      u.environmentMapEnabled,
                                      u.environmentMapIntensity,
                                      restirEnabled && sampleIdx == 0
                                          ? &restir
                                          : nullptr,
                                      restirEnabled && sampleIdx == 0);
    accumulatedColor += sample.radiance;
    accumulatedAlbedo += sample.albedo;
    accumulatedNormal += sample.normal;
    accumulatedPosition += sample.position;
  }

  if (restirEnabled) {
    float4 outSample = float4(0.0f);
    float4 outNormal = float4(0.0f);
    float4 outState = float4(0.0f);
    if (restir.valid != 0u && restir.W > 0.0f && restir.M > 0.0f) {
      outSample = float4(restir.sample.position,
                         float(restir.sample.lightIndex));
      outNormal = float4(restir.sample.normal, restir.sample.area);
      outState = float4(restir.W, restir.M, 0.0f, 1.0f);
    }
    restirOutSample.write(outSample, pixel);
    restirOutNormal.write(outNormal, pixel);
    restirOutState.write(outState, pixel);

    if (restirStats) {
      atomic_fetch_add_explicit(&restirStats[0], 1u, memory_order_relaxed);
      if (restir.valid != 0u) {
        if (restir.selectedFromTemporal != 0u) {
          atomic_fetch_add_explicit(&restirStats[1], 1u, memory_order_relaxed);
        } else {
          atomic_fetch_add_explicit(&restirStats[2], 1u, memory_order_relaxed);
        }
      }
    }
  } else {
    restirOutSample.write(float4(0.0f), pixel);
    restirOutNormal.write(float4(0.0f), pixel);
    restirOutState.write(float4(0.0f), pixel);
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
  float3 previousPosition =
      float3(float4(positionAccum.read(pixel)).xyz);
  float3 albedoSum = previousAlbedo * previousSampleCount + accumulatedAlbedo;
  float3 normalSum = previousNormal * previousSampleCount + accumulatedNormal;
  float3 positionSum =
      previousPosition * previousSampleCount + accumulatedPosition;
  float3 averagedAlbedo =
      (totalSamples > 0.0f) ? albedoSum / totalSamples : float3(0.0f);
  averagedAlbedo = clamp(averagedAlbedo, 0.0f, 1.0f);
  float3 averagedNormal =
      (totalSamples > 0.0f) ? normalSum / totalSamples : float3(0.0f);
  averagedNormal = clamp(averagedNormal, -1.0f, 1.0f);
  float3 averagedPosition =
      (totalSamples > 0.0f) ? positionSum / totalSamples : float3(0.0f);

  float4 result = float4(averaged, 1.0f);
  currentFrame.write(result, pixel);
  albedoAccum.write(float4(averagedAlbedo, 1.0f), pixel);
  normalAccum.write(float4(averagedNormal, 1.0f), pixel);
  positionAccum.write(float4(averagedPosition, 1.0f), pixel);
  sampleCount.write(float4(totalSamples, 0.0f, 0.0f, 0.0f), pixel);
}
