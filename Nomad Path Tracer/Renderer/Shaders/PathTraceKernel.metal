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
    device RestirReservoir *reservoirBuffer [[buffer(14)]],
    texture2d<float, access::write> currentFrame [[texture(0)]],
    texture2d<float, access::write> albedoAccum [[texture(1)]],
    texture2d<float, access::write> normalAccum [[texture(2)]],
    texture2d<float, access::write> positionAccum [[texture(3)]],
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
  RestirReservoir reservoir = initReservoir();
  uint restirCandidateCount = max(u.restirCandidateCount, 1u);
  bool useRestir = (u.restirEnabled != 0u);

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

    if (!useRestir) {
      PathTraceSample sample = rayColor(
          r, rayDx, rayDy, tlasNodes, u.tlasNodeCount, bvhNodes, primitives,
          materials, u.primitiveCount, primitiveIndices, activeMask,
          instanceRecords, lightIndices, lightCdf, primitiveRemap,
          primitiveRayStats, seed, u.maxRayDepth, u.debugAS, u.blasNodeCount,
          u.lightCount, u.lightTotalWeight,
          static_cast<uint>(u.totalPrimitiveCount), materialTextures,
          u.textureCount, environmentMap, environmentSampler,
          u.environmentMapEnabled, u.environmentMapIntensity);
      accumulatedColor += sample.radiance;
      accumulatedAlbedo += sample.albedo;
      accumulatedNormal += sample.normal;
      accumulatedPosition += sample.position;
      accumulatedRoughness += sample.roughness;
      continue;
    }

    TlasLeafCache bounceCache;
    bounceCache.valid = false;
    intersection bestHit = firstHitTLAS(
        r, tlasNodes, u.tlasNodeCount, bvhNodes, primitives, primitiveIndices,
        activeMask, instanceRecords, primitiveRemap, u.primitiveCount,
        static_cast<uint>(u.totalPrimitiveCount), primitiveRayStats, &bounceCache);

    if (bestHit.primitiveId != -1) {
      int matBase = bestHit.primitiveId * int(kMaterialFloat4Count);
      int totalEntries = int(u.primitiveCount) * int(kMaterialFloat4Count);
      if (matBase >= 0 && matBase + int(kMaterialFloat4Count) <= totalEntries) {
        MaterialPayload material = decodeMaterial(matBase, materials, lodAtten);
        if (!bestHit.supportsNormalMap)
          material.normalTextureIndex = -1;

        float2 surfaceUV = float2(0.0f);
        bool haveUV = false;
        int primBase = bestHit.primitiveId * int(kPrimitiveFloat4Count);
        int primEntries = int(u.primitiveCount) * int(kPrimitiveFloat4Count);
        if (primBase + int(kPrimitiveFloat4Count) <= primEntries && primBase >= 0) {
          float4 gp0 = primitives[primBase + 0];
          float4 gp1 = primitives[primBase + 1];
          float4 gp2 = primitives[primBase + 2];
          float4 gp3 = primitives[primBase + 3];
          int primitiveType = int(gp0.w);
          if (primitiveType == 1) {
            float2 uv0 = float2(gp1.w, gp2.w);
            float2 uv1 = gp3.xy;
            float2 uv2 = gp3.zw;
            float2 bary = bestHit.barycentric;
            float w = 1.0f - bary.x - bary.y;
            surfaceUV = uv0 * w + uv1 * bary.x + uv2 * bary.y;
            haveUV = true;
          } else if (primitiveType == 2) {
            surfaceUV = bestHit.uv;
            haveUV = true;
          } else if (primitiveType == 0) {
            float3 center = gp0.xyz;
            float3 dir = normalize(bestHit.point - center);
            if (all(isfinite(dir))) {
              float u = 0.5f + atan2(dir.z, dir.x) / (2.0f * M_PI);
              float v = 0.5f - asin(clamp(dir.y, -1.0f, 1.0f)) / M_PI;
              surfaceUV = float2(u, v);
              haveUV = true;
            }
          }
        }

        if (haveUV) {
          if (material.diffuseTextureIndex >= 0) {
            float4 sample = samplePackedTexture(material.diffuseTextureIndex,
                                                surfaceUV, materialTextures,
                                                u.textureCount);
            material.diffuseColor *= sample.xyz;
            material.opacity *= clamp(sample.w, 0.0f, 1.0f);
          }
          if (material.specularTextureIndex >= 0) {
            float4 sample = samplePackedTexture(material.specularTextureIndex,
                                                surfaceUV, materialTextures,
                                                u.textureCount);
            material.specularColor *= sample.xyz;
          }
        }

        float3 shadingNormal = bestHit.normal;
        if (!all(isfinite(shadingNormal)) ||
            dot(shadingNormal, shadingNormal) <= RAY_EPS) {
          shadingNormal = -rayDirection;
        }
        if (haveUV && material.normalTextureIndex >= 0 &&
            bestHit.supportsNormalMap) {
          float4 normalSample =
              samplePackedTexture(material.normalTextureIndex, surfaceUV,
                                  materialTextures, u.textureCount);
          float3 sampledNormal = normalSample.xyz;
          float3 localNormal = sampledNormal * 2.0f - float3(1.0f);

          float3 geomNormal = bestHit.normal;
          float3 tangent = bestHit.tangent;
          float3 bitangent = bestHit.bitangent;

          bool validFrame =
              dot(tangent, tangent) > RAY_EPS &&
              dot(bitangent, bitangent) > RAY_EPS &&
              dot(geomNormal, geomNormal) > RAY_EPS &&
              all(isfinite(tangent)) && all(isfinite(bitangent)) &&
              all(isfinite(geomNormal));

          if (validFrame && all(isfinite(localNormal))) {
            float3 t = normalize(tangent);
            float3 b = normalize(bitangent);
            float3 n = normalize(geomNormal);
            float3 worldNormal = localNormal.x * t + localNormal.y * b +
                                 localNormal.z * n;
            if (dot(worldNormal, worldNormal) > RAY_EPS &&
                all(isfinite(worldNormal))) {
              shadingNormal = normalize(worldNormal);
              if (dot(shadingNormal, n) < 0.0f)
                shadingNormal = -shadingNormal;
            }
          }
        }

        accumulatedAlbedo += material.diffuseColor * material.opacity;
        accumulatedNormal += shadingNormal;
        accumulatedPosition += bestHit.point;
        accumulatedRoughness += clamp(material.roughness, 0.0f, 1.0f);

        float3 directLightingBsdf = directLightingBsdfFromMaterial(material);
        if (luminance(directLightingBsdf) > 0.0f && u.lightCount > 0 &&
            u.lightTotalWeight > 0.0f) {
          for (uint candidateIdx = 0u; candidateIdx < restirCandidateCount;
               ++candidateIdx) {
            LightSampleCandidate candidate;
            if (sampleDirectLightCandidate(
                    bestHit.point, shadingNormal, directLightingBsdf,
                    bestHit.primitiveId, tlasNodes, u.tlasNodeCount, bvhNodes,
                    primitives, materials, u.primitiveCount, primitiveIndices,
                    activeMask, instanceRecords, lightIndices, lightCdf,
                    primitiveRemap, primitiveRayStats, bounceCache, seed,
                    u.lightCount, u.lightTotalWeight,
                    static_cast<uint>(u.totalPrimitiveCount), candidate)) {
              float weight = luminance(restirTargetContribution(candidate)) /
                             max(candidate.pdf, RAY_EPS);
              float xi = randomFloat(seed);
              seed = random(seed);
              updateReservoir(reservoir, candidate, weight, xi);
            }
          }
        }
      }
    }
  }

  float totalSamples = float(samplesThisFrame);
  float3 averaged =
      (totalSamples > 0.0f) ? accumulatedColor / totalSamples : float3(0.0f);
  averaged = clamp(averaged, 0.0f, 1.0f);
  float3 reservoirEstimate = finalizeReservoir(reservoir);
  reservoirEstimate = clamp(reservoirEstimate, 0.0f, 1.0f);

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

  float3 outputColor = useRestir ? reservoirEstimate : averaged;
  float4 result = float4(outputColor, 1.0f);
  currentFrame.write(result, pixel);
  albedoAccum.write(float4(averagedAlbedo, 1.0f), pixel);
  normalAccum.write(float4(averagedNormal, averagedRoughness), pixel);
  positionAccum.write(float4(averagedPosition, 1.0f), pixel);

  if (useRestir && reservoirBuffer) {
    uint pixelIndex = pixel.y * width + pixel.x;
    reservoirBuffer[pixelIndex] = reservoir;
  }
}
