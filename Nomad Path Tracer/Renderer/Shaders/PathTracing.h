#ifndef PATH_TRACING_H
#define PATH_TRACING_H

#include <metal_atomic>
#include <metal_stdlib>

#define M_PI 3.14159265358979323846

using namespace metal;

constant uint kMaterialFloat4Count = 5;
constant uint kPrimitiveFloat4Count = 7;
constant uint kMaxMaterialTextures = 64;

constant sampler kMaterialTextureSampler(address::repeat, filter::linear);

constexpr float RAY_EPS = 1e-4f;

struct PathTraceSample {
  float3 radiance;
  float3 albedo;
  float3 normal;
  float3 position;
  float roughness;
};

constant float kVisibilityRayEpsilon = RAY_EPS;
constant float kVisibilityRayOffset = 5.0f * RAY_EPS;

struct Ray {
  float3 origin;
  float3 direction;
  float minDistance = 0.0f;
  float maxDistance = 0.0f;
};

struct TlasLeafCache {
  bool valid = false;
  int instanceId = -1;
  int blasRootIndex = -1;
  float3 boundsMin = float3(0.0f);
  float3 boundsMax = float3(0.0f);
};

struct LightSampleCandidate {
  float3 radiance = float3(0.0f);
  float3 wi = float3(0.0f);
  float pdf = 0.0f;
  float geometryFactor = 0.0f;
  uint lightId = 0u;
  float3 lightPosition = float3(0.0f);
  float3 lightNormal = float3(0.0f);
  float lightArea = 0.0f;
  float lightPdf = 0.0f;
};

#include "Intersect.h"
#include "Random.h"
#include "Scatter.h"
#include "Structs.h"

template <typename T> inline void swap(thread T &a, thread T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}

inline float3 randomUnitVector(thread uint32_t &seed) {
  float z = 2.0 * randomFloat(seed) - 1.0;
  float t = 2.0 * M_PI * randomFloat(seed);
  float r = sqrt(1.0 - z * z);
  return float3(r * cos(t), r * sin(t), z);
}

// Helper: Random point in unit sphere
inline float3 randomInUnitSphere(thread uint32_t &seed) {
  while (true) {
    float3 p =
        2.0 * float3(randomFloat(seed), randomFloat(seed), randomFloat(seed)) -
        float3(1.0);

    if (length_squared(p) < 1.0)
      return p;
  }
}

inline float2 concentricSampleDisk(float2 u) {
  float2 uOffset = 2.0 * u - float2(1.0);
  if (uOffset.x == 0.0f && uOffset.y == 0.0f) {
    return float2(0.0f);
  }

  float r;
  float theta;
  if (abs(uOffset.x) > abs(uOffset.y)) {
    r = uOffset.x;
    theta = (M_PI / 4.0f) * (uOffset.y / uOffset.x);
  } else {
    r = uOffset.y;
    theta = (M_PI / 2.0f) - (M_PI / 4.0f) * (uOffset.x / uOffset.y);
  }

  return r * float2(cos(theta), sin(theta));
}

inline int decodeTextureIndex(float value) {
  return (value >= -0.5f) ? int(floor(value + 0.5f)) : -1;
}

inline MaterialPayload decodeMaterial(int baseIndex,
                                      device const float4 *materials,
                                      float lodAttenuation) {
  MaterialPayload payload;
  float4 m0 = materials[baseIndex + 0];
  float4 m1 = materials[baseIndex + 1];
  float4 m2 = materials[baseIndex + 2];
  float4 m3 = materials[baseIndex + 3];
  float4 m4 = materials[baseIndex + 4];

  payload.diffuseColor = max(m0.xyz * lodAttenuation, float3(0.0f));
  payload.opacity = clamp(m0.w, 0.0f, 1.0f);
  payload.specularColor = max(m1.xyz, float3(0.0f));
  payload.shininess = max(m1.w, 1.0f);
  payload.emissionColor = max(m2.xyz, float3(0.0f));
  payload.emissionPower = max(m2.w, 0.0f);
  payload.transmissionColor = max(m3.xyz, float3(0.0f));
  payload.indexOfRefraction = max(m3.w, 1e-3f);
  payload.roughness = clamp(m4.w, 0.0f, 1.0f);
  payload.diffuseTextureIndex = decodeTextureIndex(m4.x);
  payload.specularTextureIndex = decodeTextureIndex(m4.y);
  payload.normalTextureIndex = decodeTextureIndex(m4.z);
  payload.pad0 = 0.0f;
  payload.pad1 = 0;

  if (payload.opacity < 1.0f && luminance(payload.transmissionColor) <= 0.0f) {
    float trans = 1.0f - payload.opacity;
    payload.transmissionColor = float3(trans);
  }

  return payload;
}

template <typename TextureArray>
inline float4 samplePackedTexture(int index, float2 uv,
                                  thread const TextureArray &textures,
                                  uint textureCount) {
  if (index < 0)
    return float4(1.0f);

  uint texIndex = uint(index);
  if (texIndex >= textureCount)
    return float4(1.0f);

  texture2d<float, access::sample> tex = textures[texIndex];
  uint width = tex.get_width();
  uint height = tex.get_height();
  if (width == 0 || height == 0)
    return float4(1.0f);

  float2 coord = float2(uv.x - floor(uv.x), uv.y - floor(uv.y));
  return tex.sample(kMaterialTextureSampler, coord, level(0.0f));
}

inline uint selectLightOffset(float xi, device const float *lightCdf,
                              uint lightCount, float totalWeight) {
  if (lightCount == 0 || totalWeight <= 0.0f)
    return lightCount;

  float target = xi * totalWeight;
  uint low = 0;
  uint high = lightCount;
  while (low < high) {
    uint mid = (low + high) / 2;
    if (lightCdf[mid] < target)
      low = mid + 1;
    else
      high = mid;
  }
  return min(low, lightCount - 1);
}

inline float lightPdfFromCdf(uint offset, device const float *lightCdf,
                             float lightTotalWeight) {
  if (lightTotalWeight <= 0.0f)
    return 0.0f;
  float currentCdf = lightCdf[offset];
  float prevCdf = (offset > 0) ? lightCdf[offset - 1] : 0.0f;
  float weight = currentCdf - prevCdf;
  if (weight <= 0.0f)
    return 0.0f;
  return weight / lightTotalWeight;
}

inline RestirReservoir initReservoir() {
  RestirReservoir reservoir;
  reservoir.sampleRadiance = float3(0.0f);
  reservoir.wi = float3(0.0f);
  reservoir.pdf = 0.0f;
  reservoir.geometryFactor = 0.0f;
  reservoir.wSum = 0.0f;
  reservoir.m = 0u;
  reservoir.packedLightId = 0xffffffffu;
  reservoir.lightPosition = float3(0.0f);
  reservoir.lightNormal = float3(0.0f);
  reservoir.lightArea = 0.0f;
  reservoir.lightPdf = 0.0f;
  return reservoir;
}

inline float restirTargetContribution(float3 radiance, float geometryFactor) {
  float3 contribution = radiance * max(geometryFactor, 0.0f);
  return max(luminance(contribution), 0.0f);
}

inline float restirTargetContribution(thread const LightSampleCandidate &candidate) {
  return restirTargetContribution(candidate.radiance, candidate.geometryFactor);
}

inline float restirTargetContribution(thread const RestirReservoir &reservoir) {
  return restirTargetContribution(reservoir.sampleRadiance, reservoir.geometryFactor);
}

inline float restirTargetWeight(float3 radiance, float geometryFactor, float pdf) {
  float target = restirTargetContribution(radiance, geometryFactor);
  return target / max(pdf, RAY_EPS);
}

inline float3 evaluateDirectLightingBsdf(thread const MaterialPayload &material,
                                         float3 normal, float3 viewDir,
                                         float3 lightDir) {
  float3 n = normalize(normal);
  float3 v = normalize(viewDir);
  float3 l = normalize(lightDir);
  if (!all(isfinite(n)) || !all(isfinite(v)) || !all(isfinite(l))) {
    return float3(0.0f);
  }
  float nDotL = max(dot(n, l), 0.0f);
  float nDotV = max(dot(n, v), 0.0f);
  if (nDotL <= 0.0f || nDotV <= 0.0f) {
    return float3(0.0f);
  }

  float3 diffuseColor =
      max(material.diffuseColor, float3(0.0f)) * material.opacity;
  float3 specularColor =
      clamp(material.specularColor * material.opacity, 0.0f, 1.0f);

  float roughness = clamp(material.roughness, 0.0f, 1.0f);
  float alpha = max(roughness * roughness, 0.001f);
  float alpha2 = alpha * alpha;

  float3 h = normalize(v + l);
  float nDotH = max(dot(n, h), 0.0f);
  float vDotH = max(dot(v, h), 0.0f);

  float denom = nDotH * nDotH * (alpha2 - 1.0f) + 1.0f;
  float D = alpha2 / max(M_PI * denom * denom, RAY_EPS);

  float k = (alpha + 1.0f);
  k = (k * k) / 8.0f;
  float Gv = nDotV / (nDotV * (1.0f - k) + k);
  float Gl = nDotL / (nDotL * (1.0f - k) + k);
  float G = Gv * Gl;

  float Fc = pow(max(1.0f - vDotH, 0.0f), 5.0f);
  float3 F = specularColor + (float3(1.0f) - specularColor) * Fc;

  float3 specular =
      (D * G) * F / max(4.0f * nDotL * nDotV, RAY_EPS);
  float3 diffuse = diffuseColor / M_PI;
  return diffuse + specular;
}

inline MaterialPayload restirMaterialFromGBuffer(float3 albedo,
                                                 float roughness) {
  MaterialPayload material;
  material.diffuseColor = max(albedo, float3(0.0f));
  material.opacity = 1.0f;
  material.specularColor = float3(0.04f);
  material.shininess = 1.0f;
  material.emissionColor = float3(0.0f);
  material.emissionPower = 0.0f;
  material.transmissionColor = float3(0.0f);
  material.indexOfRefraction = 1.0f;
  material.roughness = clamp(roughness, 0.0f, 1.0f);
  material.pad0 = 0.0f;
  material.diffuseTextureIndex = -1;
  material.specularTextureIndex = -1;
  material.normalTextureIndex = -1;
  material.pad1 = 0;
  return material;
}

inline void updateReservoir(thread RestirReservoir &reservoir,
                            thread const LightSampleCandidate &candidate,
                            float weight, float xi) {
  reservoir.m += 1u;
  if (weight <= 0.0f || !isfinite(weight)) {
    return;
  }
  reservoir.wSum += weight;
  float threshold = weight / reservoir.wSum;
  if (xi < threshold) {
    reservoir.sampleRadiance = candidate.radiance;
    reservoir.wi = candidate.wi;
    reservoir.pdf = candidate.pdf;
    reservoir.geometryFactor = candidate.geometryFactor;
    reservoir.packedLightId = candidate.lightId;
    reservoir.lightPosition = candidate.lightPosition;
    reservoir.lightNormal = candidate.lightNormal;
    reservoir.lightArea = candidate.lightArea;
    reservoir.lightPdf = candidate.lightPdf;
  }
}

inline float3 finalizeReservoir(thread const RestirReservoir &reservoir) {
  if (reservoir.m == 0u || reservoir.pdf <= 0.0f) {
    return float3(0.0f);
  }
  float target = restirTargetContribution(reservoir);
  float weightSelected = target / max(reservoir.pdf, RAY_EPS);
  if (weightSelected <= 0.0f) {
    return float3(0.0f);
  }
  float normalization = reservoir.wSum / (float(reservoir.m) * weightSelected);
  return reservoir.sampleRadiance * normalization;
}

inline bool sampleLightPoint(int primitiveType, float4 p0, float4 p1, float4 p2,
                             thread uint32_t &seed, thread float3 &position,
                             thread float3 &normal, thread float &area);
inline bool isLightVisible(
    float3 origin, float3 offsetNormal, float3 wi, float dist,
    uint lightPrimIndex, thread TlasLeafCache &bounceCache,
    device const float4 *tlasNodes, uint tlasNodeCount,
    device const float4 *bvhNodes, device const float4 *primitives,
    device const int *primitiveIndices, device const uchar *activeMask,
    device const InstanceRecord *instanceRecords,
    device const uint *primitiveRemap, uint primitiveCount,
    uint totalPrimitiveCount, device atomic_uint *primitiveRayStats);

inline bool sampleDirectLightCandidate(
    float3 hitPoint, float3 offsetNormal, float3 viewDir,
    thread const MaterialPayload &material,
    int hitPrimitiveId, device const float4 *tlasNodes, uint tlasNodeCount,
    device const float4 *bvhNodes, device const float4 *primitives,
    device const float4 *materials, uint primitiveCount,
    device const int *primitiveIndices, device const uchar *activeMask,
    device const InstanceRecord *instanceRecords,
    device const uint *lightIndices, device const float *lightCdf,
    device const uint *primitiveRemap, device atomic_uint *primitiveRayStats,
    thread TlasLeafCache &bounceCache, thread uint32_t &seed, uint lightCount,
    float lightTotalWeight, uint totalPrimitiveCount,
    thread LightSampleCandidate &candidate) {
  candidate = LightSampleCandidate();
  if (lightCount == 0 || lightTotalWeight <= 0.0f)
    return false;

  float lightXi = randomFloat(seed);
  seed = random(seed);
  uint selectedOffset =
      selectLightOffset(lightXi, lightCdf, lightCount, lightTotalWeight);
  if (selectedOffset >= lightCount)
    return false;

  uint lightPrimIndex = lightIndices[selectedOffset];
  if (int(lightPrimIndex) == hitPrimitiveId)
    return false;

  int base = int(lightPrimIndex) * int(kPrimitiveFloat4Count);
  int primEntries = int(primitiveCount) * int(kPrimitiveFloat4Count);
  if (base < 0 || base + 2 >= primEntries)
    return false;

  float4 lp0 = primitives[base + 0];
  float4 lp1 = primitives[base + 1];
  float4 lp2 = primitives[base + 2];
  float3 lightPoint;
  float3 lightNormal;
  float area = 0.0f;
  if (!sampleLightPoint(int(lp0.w), lp0, lp1, lp2, seed, lightPoint,
                        lightNormal, area) ||
      area <= 0.0f) {
    return false;
  }

  float3 toLight = lightPoint - hitPoint;
  float dist2 = dot(toLight, toLight);
  if (dist2 <= RAY_EPS)
    return false;

  float dist = sqrt(dist2);
  float3 wi = toLight / dist;
  float cosTheta = max(dot(offsetNormal, wi), 0.0f);
  float cosLight = max(dot(lightNormal, -wi), 0.0f);
  if (cosTheta <= 0.0f || cosLight <= 0.0f)
    return false;

  float lightPdf = lightPdfFromCdf(selectedOffset, lightCdf, lightTotalWeight);
  if (lightPdf <= 0.0f)
    return false;

  float pdfArea = 1.0f / area;
  float pdfSolid = pdfArea * dist2 / cosLight;
  float totalPdf = lightPdf * pdfSolid;
  if (totalPdf <= 0.0f)
    return false;

  bool visible = isLightVisible(
      hitPoint, offsetNormal, wi, dist, lightPrimIndex, bounceCache, tlasNodes,
      tlasNodeCount, bvhNodes, primitives, primitiveIndices, activeMask,
      instanceRecords, primitiveRemap, primitiveCount, totalPrimitiveCount,
      primitiveRayStats);
  if (!visible)
    return false;

  int lightMatIndex = int(lightPrimIndex) * int(kMaterialFloat4Count);
  int totalEntries = int(primitiveCount) * int(kMaterialFloat4Count);
  if (lightMatIndex < 0 || lightMatIndex + int(kMaterialFloat4Count) > totalEntries)
    return false;

  MaterialPayload lightMaterial = decodeMaterial(lightMatIndex, materials, 1.0f);
  float3 lightRadiance =
      lightMaterial.emissionColor * lightMaterial.emissionPower;
  float3 throughput =
      evaluateDirectLightingBsdf(material, offsetNormal, viewDir, wi);
  float geometryFactor = cosLight / max(dist2, RAY_EPS);
  candidate.radiance = throughput * lightRadiance * cosTheta;
  candidate.geometryFactor = geometryFactor;
  candidate.wi = wi;
  candidate.pdf = totalPdf;
  candidate.lightId = lightPrimIndex;
  candidate.lightPosition = lightPoint;
  candidate.lightNormal = lightNormal;
  candidate.lightArea = area;
  candidate.lightPdf = lightPdf;
  return true;
}

inline bool intersectAABB(thread const Ray &r, float3 bmin, float3 bmax,
                          float tMin, float tMax);
inline intersection firstHitBVH(thread const Ray &r,
                                device const float4 *bvhNodes,
                                device const float4 *primitives,
                                device const int *primitiveIndices,
                                device const uchar *activeMask,
                                device const uint *primitiveRemap,
                                uint residentPrimitiveCount,
                                uint totalPrimitiveCount,
                                device atomic_uint *primitiveRayStats,
                                int startNode);
inline intersection firstHitTLAS(
    thread const Ray &r, device const float4 *tlasNodes, uint tlasNodeCount,
    device const float4 *bvhNodes, device const float4 *primitives,
    device const int *primitiveIndices, device const uchar *activeMask,
    device const InstanceRecord *instanceRecords,
    device const uint *primitiveRemap, uint residentPrimitiveCount,
    uint totalPrimitiveCount, device atomic_uint *primitiveRayStats,
    thread TlasLeafCache *cache);

inline bool isLightVisible(
    float3 origin, float3 offsetNormal, float3 wi, float dist,
    uint lightPrimIndex, thread TlasLeafCache &bounceCache,
    device const float4 *tlasNodes, uint tlasNodeCount,
    device const float4 *bvhNodes, device const float4 *primitives,
    device const int *primitiveIndices, device const uchar *activeMask,
    device const InstanceRecord *instanceRecords,
    device const uint *primitiveRemap, uint primitiveCount,
    uint totalPrimitiveCount, device atomic_uint *primitiveRayStats) {
  if (dist <= kVisibilityRayEpsilon || !isfinite(dist))
    return false;

  Ray shadowRay;
  shadowRay.origin = origin + kVisibilityRayOffset * offsetNormal;
  shadowRay.direction = wi;
  shadowRay.minDistance = kVisibilityRayEpsilon;
  shadowRay.maxDistance = dist - kVisibilityRayEpsilon;
  if (shadowRay.maxDistance < shadowRay.minDistance)
    shadowRay.maxDistance = shadowRay.minDistance;

  intersection shadowHit;
  bool usedCache = false;
  if (bounceCache.valid) {
    bool intersectsCached =
        intersectAABB(shadowRay, bounceCache.boundsMin, bounceCache.boundsMax,
                      shadowRay.minDistance, shadowRay.maxDistance);
    if (intersectsCached && bounceCache.blasRootIndex >= 0) {
      intersection cachedHit =
          firstHitBVH(shadowRay, bvhNodes, primitives, primitiveIndices,
                      activeMask, primitiveRemap, primitiveCount,
                      totalPrimitiveCount, primitiveRayStats,
                      bounceCache.blasRootIndex);
      if (cachedHit.primitiveId != -1 &&
          cachedHit.primitiveId != int(lightPrimIndex) &&
          cachedHit.t <= shadowRay.maxDistance) {
        shadowHit = cachedHit;
        usedCache = true;
      }
    }
  }
  if (!usedCache) {
    shadowHit = firstHitTLAS(
        shadowRay, tlasNodes, tlasNodeCount, bvhNodes, primitives,
        primitiveIndices, activeMask, instanceRecords, primitiveRemap,
        primitiveCount, totalPrimitiveCount, primitiveRayStats, nullptr);
  }

  if (shadowHit.primitiveId == -1)
    return true;
  if (shadowHit.primitiveId == int(lightPrimIndex) &&
      shadowHit.t >= dist - kVisibilityRayEpsilon)
    return true;
  return false;
}

inline bool isVisibleToLight(
    float3 hitPoint, float3 hitNormal, float3 lightPos,
    device const float4 *bvhNodes, device const float4 *primitives,
    device const int *primitiveIndices, device const uchar *activeMask,
    device const uint *primitiveRemap, uint residentPrimitiveCount,
    uint totalPrimitiveCount, device atomic_uint *primitiveRayStats,
    int startNode) {
  float3 dir = lightPos - hitPoint;
  float dist = length(dir);
  if (dist <= RAY_EPS || !isfinite(dist))
    return false;

  Ray shadowRay;
  shadowRay.origin = hitPoint + hitNormal * RAY_EPS;
  shadowRay.direction = dir / dist;
  shadowRay.minDistance = RAY_EPS;
  shadowRay.maxDistance = dist - RAY_EPS;
  intersection shadowHit =
      firstHitBVH(shadowRay, bvhNodes, primitives, primitiveIndices, activeMask,
                  primitiveRemap, residentPrimitiveCount, totalPrimitiveCount,
                  primitiveRayStats, startNode);
  return shadowHit.primitiveId == -1;
}

inline bool sampleLightPoint(int primitiveType, float4 p0, float4 p1, float4 p2,
                             thread uint32_t &seed, thread float3 &position,
                             thread float3 &normal, thread float &area) {
  if (primitiveType == 0) {
    float radius = p1.x;
    if (radius <= 0.0f)
      return false;
    float u1 = randomFloat(seed);
    seed = random(seed);
    float u2 = randomFloat(seed);
    seed = random(seed);
    float z = 2.0 * u1 - 1.0;
    float phi = 2.0 * M_PI * u2;
    float r = sqrt(max(0.0f, 1.0f - z * z));
    float3 dir = float3(r * cos(phi), r * sin(phi), z);
    position = p0.xyz + radius * dir;
    normal = dir;
    area = 4.0f * M_PI * radius * radius;
    return area > 0.0f;
  } else if (primitiveType == 1) {
    float u1 = randomFloat(seed);
    seed = random(seed);
    float u2 = randomFloat(seed);
    seed = random(seed);
    float su1 = sqrt(u1);
    float b0 = 1.0 - su1;
    float b1 = su1 * (1.0 - u2);
    float b2 = su1 * u2;
    float3 v0 = p0.xyz;
    float3 v1 = p1.xyz;
    float3 v2 = p2.xyz;
    position = b0 * v0 + b1 * v1 + b2 * v2;
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 n = cross(e1, e2);
    float lenN = length(n);
    if (lenN <= 0.0f)
      return false;
    normal = n / lenN;
    area = 0.5f * lenN;
    return area > 0.0f;
  } else if (primitiveType == 2) {
    float u1 = randomFloat(seed);
    seed = random(seed);
    float u2 = randomFloat(seed);
    seed = random(seed);
    float su = u1 * 2.0 - 1.0;
    float sv = u2 * 2.0 - 1.0;
    float3 center = p0.xyz;
    float3 e1 = p1.xyz;
    float3 e2 = p2.xyz;
    position = center + su * e1 + sv * e2;
    float3 n = cross(e1, e2);
    float lenN = length(n);
    if (lenN <= 0.0f)
      return false;
    normal = n / lenN;
    area = 4.0f * lenN;
    return area > 0.0f;
  }
  return false;
}

// BVH intersection helper
inline bool intersectAABB(thread const Ray &r, float3 bmin, float3 bmax,
                          float tMin, float tMax) {
  for (int i = 0; i < 3; ++i) {
    float dir = r.direction[i];
    if (abs(dir) < RAY_EPS) {
      if (r.origin[i] < bmin[i] || r.origin[i] > bmax[i])
        return false;
      continue;
    }
    float invD = 1.0 / dir;
    float t0 = (bmin[i] - r.origin[i]) * invD;
    float t1 = (bmax[i] - r.origin[i]) * invD;
    if (invD < 0.0)
      swap(t0, t1);
    tMin = max(tMin, t0);
    tMax = min(tMax, t1);
    // Use a strict comparison to allow zero-width slabs (e.g., thin rectangles)
    if (tMax < tMin)
      return false;
  }
  return true;
}

// BVH traversal version of firstHit
inline intersection firstHitBVH(thread const Ray &r,
                                device const float4 *bvhNodes,
                                device const float4 *primitives,
                                device const int *primitiveIndices,
                                device const uchar *activeMask,
                                device const uint *primitiveRemap,
                                uint residentPrimitiveCount,
                                uint totalPrimitiveCount,
                                device atomic_uint *primitiveRayStats,
                                int startNode) {
  intersection in;
  in.t = INFINITY;
  in.primitiveId = -1;
  in.isTriangle = 0;
  in.nodeIndex = -1;
  in.uv = float2(0.0f);
  in.barycentric = float2(0.0f);

  constexpr int stackSize = 64;
  int stack[stackSize];
  int stackPtr = 0;
  stack[stackPtr++] = startNode;

  while (stackPtr > 0) {
    int nodeIdx = stack[--stackPtr];

    float3 bmin = bvhNodes[2 * nodeIdx + 0].xyz;
    float3 bmax = bvhNodes[2 * nodeIdx + 1].xyz;
    int leftFirst = as_type<int>(bvhNodes[2 * nodeIdx + 0].w);
    int second = as_type<int>(bvhNodes[2 * nodeIdx + 1].w);

    if (!intersectAABB(r, bmin, bmax, RAY_EPS, in.t))
      continue;

    if (second > 0) {
      int count = second;
      if (count <= 0 || leftFirst < 0)
        continue;
      for (int i = 0; i < count; ++i) {
        int primIdx = primitiveIndices[leftFirst + i];
        if (primIdx < 0)
          continue;

        uint remapIdx = static_cast<uint>(primIdx);
        uint globalId = remapIdx;
        if (primitiveRemap && remapIdx < residentPrimitiveCount)
          globalId = primitiveRemap[remapIdx];
        if (primitiveRayStats && globalId < totalPrimitiveCount) {
          uint statsIndex = globalId * 2u + 1u;
          atomic_fetch_add_explicit(&primitiveRayStats[statsIndex], 1u,
                                    memory_order_relaxed);
        }

        if (!activeMask[primIdx])
          continue;
        int base = primIdx * int(kPrimitiveFloat4Count);
        float4 p0 = primitives[base + 0];
        float4 p1 = primitives[base + 1];
        float4 p2 = primitives[base + 2];

        int primitiveType = int(p0.w);
        float tHit = INFINITY;
        float3 n = float3(0);
        float3 hit = float3(0);
        bool hitThis = false;

        float2 localUV = float2(0.0f);
        float2 bary = float2(0.0f);
        float3 candidateTangent = float3(0.0f);
        float3 candidateBitangent = float3(0.0f);

        float4 packedTangent = primitives[base + 4];
        float4 packedBitangent = primitives[base + 5];
        float4 packedNormal = primitives[base + 6];

        float3 storedTangent = packedTangent.xyz;
        float3 storedBitangent = packedBitangent.xyz;
        float3 storedNormal = packedNormal.xyz;
        int storedSupportsNormalMap = (packedNormal.w > 0.5f) ? 1 : 0;
        int candidateSupportsNormalMap = 0;

        if (primitiveType == 0) {
          float3 center = p0.xyz;
          float radius = p1.x;
          float3 oc = r.origin - center;
          float a = dot(r.direction, r.direction);
          float b = dot(oc, r.direction);
          float c = dot(oc, oc) - radius * radius;
          float discriminant = b * b - a * c;

          if (discriminant > 0.0) {
            float sqrtD = sqrt(discriminant);
            float temp = (-b - sqrtD) / a;
            if (temp < in.t && temp > RAY_EPS) {
              tHit = temp;
              hit = r.origin + tHit * r.direction;
              n = normalize(hit - center);
              hitThis = true;
              candidateSupportsNormalMap = storedSupportsNormalMap;
            }
          }
        } else if (primitiveType == 1) {
          float3 v0 = p0.xyz;
          float3 v1 = p1.xyz;
          float3 v2 = p2.xyz;

          float3 edge1 = v1 - v0;
          float3 edge2 = v2 - v0;

          float3 fallbackNormal = cross(edge1, edge2);
          float fallbackLenSq = dot(fallbackNormal, fallbackNormal);
          if (fallbackLenSq > 0.0f)
            fallbackNormal = normalize(fallbackNormal);
          else
            fallbackNormal = float3(0.0f, 1.0f, 0.0f);

          float3 h = cross(r.direction, edge2);
          float a = dot(edge1, h);
          if (abs(a) > 1e-5) {
            float f = 1.0 / a;
            float3 s = r.origin - v0;
            float u = f * dot(s, h);
            if (u >= 0.0 && u <= 1.0) {
              float3 q = cross(s, edge1);
              float v = f * dot(r.direction, q);
              if (v >= 0.0 && u + v <= 1.0) {
                float tt = f * dot(edge2, q);
                if (tt > RAY_EPS && tt < in.t) {
                  tHit = tt;
                  hit = r.origin + tHit * r.direction;
                  float storedLenSq = dot(storedNormal, storedNormal);
                  if (storedLenSq > RAY_EPS && all(isfinite(storedNormal)))
                    n = normalize(storedNormal);
                  else
                    n = fallbackNormal;
                  if (!all(isfinite(n)) || dot(n, n) <= RAY_EPS)
                    n = fallbackNormal;
                  float tanLenSq = dot(storedTangent, storedTangent);
                  bool tangentValid =
                      tanLenSq > RAY_EPS && all(isfinite(storedTangent));
                  if (tangentValid)
                    candidateTangent = normalize(storedTangent);
                  float bitLenSq = dot(storedBitangent, storedBitangent);
                  bool bitangentValid =
                      bitLenSq > RAY_EPS && all(isfinite(storedBitangent));
                  if (bitangentValid)
                    candidateBitangent = normalize(storedBitangent);
                  if (dot(candidateTangent, candidateTangent) <= RAY_EPS ||
                      !all(isfinite(candidateTangent))) {
                    float3 refAxis = (abs(fallbackNormal.y) < 0.999f)
                                         ? float3(0.0f, 1.0f, 0.0f)
                                         : float3(1.0f, 0.0f, 0.0f);
                    candidateTangent = normalize(cross(refAxis, fallbackNormal));
                  }
                  if (dot(candidateBitangent, candidateBitangent) <= RAY_EPS ||
                      !all(isfinite(candidateBitangent)))
                    candidateBitangent = normalize(cross(fallbackNormal,
                                                          candidateTangent));
                  if (dot(candidateTangent, candidateTangent) > RAY_EPS &&
                      all(isfinite(candidateTangent))) {
                    candidateTangent = normalize(candidateTangent);
                    float3 orthBitangent = cross(n, candidateTangent);
                    if (dot(orthBitangent, orthBitangent) > RAY_EPS &&
                        all(isfinite(orthBitangent)))
                      candidateBitangent = normalize(orthBitangent);
                  }
                  candidateSupportsNormalMap =
                      (storedSupportsNormalMap && tangentValid &&
                       bitangentValid &&
                       (dot(n, n) > RAY_EPS))
                          ? 1
                          : 0;
                  hitThis = true;
                  in.isTriangle = 1;
                  bary = float2(u, v);
                }
              }
            }
          }
        } else if (primitiveType == 2) {
          float3 center = p0.xyz;
          float3 e1 = p1.xyz;
          float3 e2 = p2.xyz;
          float3 fallbackNormal = cross(e1, e2);
          float fallbackLenSq = dot(fallbackNormal, fallbackNormal);
          if (fallbackLenSq > RAY_EPS && all(isfinite(fallbackNormal)))
            fallbackNormal = normalize(fallbackNormal);
          else
            fallbackNormal = float3(0.0f, 1.0f, 0.0f);
          float3 normalCandidate = fallbackNormal;
          float normalLenSq = dot(storedNormal, storedNormal);
          bool normalValid = normalLenSq > RAY_EPS && all(isfinite(storedNormal));
          if (normalValid)
            normalCandidate = normalize(storedNormal);
          float denom = dot(normalCandidate, r.direction);
          if (fabs(denom) > 1e-5) {
            float tt = dot(center - r.origin, normalCandidate) / denom;
            if (tt > RAY_EPS && tt < in.t) {
              float3 hitPoint = r.origin + tt * r.direction;
              float3 rel = hitPoint - center;
              float u = dot(rel, e1) / dot(e1, e1);
              float v = dot(rel, e2) / dot(e2, e2);
              if (fabs(u) <= 1.0 && fabs(v) <= 1.0) {
                tHit = tt;
                hit = hitPoint;
                n = normalCandidate;
                bool tangentValid =
                    dot(storedTangent, storedTangent) > RAY_EPS &&
                    all(isfinite(storedTangent));
                bool bitangentValid =
                    dot(storedBitangent, storedBitangent) > RAY_EPS &&
                    all(isfinite(storedBitangent));
                if (tangentValid)
                  candidateTangent = normalize(storedTangent);
                if (bitangentValid)
                  candidateBitangent = normalize(storedBitangent);
                if (dot(candidateTangent, candidateTangent) <= RAY_EPS ||
                    !all(isfinite(candidateTangent))) {
                  float3 refAxis = (abs(fallbackNormal.y) < 0.999f)
                                       ? float3(0.0f, 1.0f, 0.0f)
                                       : float3(1.0f, 0.0f, 0.0f);
                  candidateTangent = normalize(cross(refAxis, fallbackNormal));
                }
                if (dot(candidateBitangent, candidateBitangent) <= RAY_EPS ||
                    !all(isfinite(candidateBitangent)))
                  candidateBitangent = normalize(
                      cross(fallbackNormal, candidateTangent));
                if (dot(candidateTangent, candidateTangent) > RAY_EPS &&
                    all(isfinite(candidateTangent))) {
                  float3 orthBitangent = cross(n, candidateTangent);
                  if (dot(orthBitangent, orthBitangent) > RAY_EPS &&
                      all(isfinite(orthBitangent)))
                    candidateBitangent = normalize(orthBitangent);
                }
                candidateSupportsNormalMap =
                    (storedSupportsNormalMap && normalValid && tangentValid &&
                     bitangentValid)
                        ? 1
                        : 0;
                hitThis = true;
                localUV = float2(0.5f * (u + 1.0f), 0.5f * (v + 1.0f));
              }
            }
          }
        }

        if (hitThis && tHit < in.t) {
          in.t = tHit;
          in.primitiveId = primIdx;
          in.normal = n;
          in.point = hit;
          in.isTriangle = primitiveType;
          in.nodeIndex = nodeIdx;
          in.barycentric = bary;
          in.uv = localUV;
          in.tangent = candidateTangent;
          in.bitangent = candidateBitangent;
          in.supportsNormalMap = candidateSupportsNormalMap;
        }
      }
    } else {
      int rightChild = -second;
      bool leftValid = leftFirst >= 0;
      bool rightValid = rightChild >= 0;

      if (!leftValid && !rightValid)
        continue;

      if (leftValid && stackPtr < stackSize)
        stack[stackPtr++] = leftFirst;
      if (rightValid && stackPtr < stackSize)
        stack[stackPtr++] = rightChild;
    }
  }

  if (in.primitiveId != -1) {
    in.frontFace = dot(in.normal, r.direction) < 0.0;
    if (!in.frontFace) {
      in.normal = -in.normal;
      if (dot(in.tangent, in.tangent) > RAY_EPS && all(isfinite(in.tangent)))
        in.tangent = -in.tangent;
      float3 adjustedBitangent = cross(in.normal, in.tangent);
      if (dot(adjustedBitangent, adjustedBitangent) > RAY_EPS &&
          all(isfinite(adjustedBitangent)))
        in.bitangent = normalize(adjustedBitangent);
    }
  }

  return in;
}

inline intersection firstHitTLAS(
    thread const Ray &r, device const float4 *tlasNodes, uint tlasNodeCount,
    device const float4 *bvhNodes, device const float4 *primitives,
    device const int *primitiveIndices, device const uchar *activeMask,
    device const InstanceRecord *instanceRecords,
    device const uint *primitiveRemap, uint residentPrimitiveCount,
    uint totalPrimitiveCount, device atomic_uint *primitiveRayStats,
    thread TlasLeafCache *cache) {
  intersection bestHit;
  bestHit.t = INFINITY;
  bestHit.primitiveId = -1;
  bestHit.nodeIndex = -1;
  bestHit.isTriangle = 0;

  if (cache) {
    cache->valid = false;
    cache->instanceId = -1;
    cache->blasRootIndex = -1;
    cache->boundsMin = float3(0.0f);
    cache->boundsMax = float3(0.0f);
  }

  if (tlasNodeCount == 0)
    return bestHit;

  constexpr int stackSize = 64;
  int stack[stackSize];
  int stackPtr = 0;
  stack[stackPtr++] = 0;

  while (stackPtr > 0) {
    int nodeIdx = stack[--stackPtr];
    float3 bmin = tlasNodes[2 * nodeIdx + 0].xyz;
    float3 bmax = tlasNodes[2 * nodeIdx + 1].xyz;

    if (!intersectAABB(r, bmin, bmax, RAY_EPS, bestHit.t))
      continue;

    int leftChild = as_type<int>(tlasNodes[2 * nodeIdx + 0].w);
    int rightChild = as_type<int>(tlasNodes[2 * nodeIdx + 1].w);

      if (leftChild < 0) {
        int instanceId = -(leftChild + 1);
        InstanceRecord record = instanceRecords[instanceId];
        int blasRoot = record.blasRootIndex;
        if (blasRoot >= 0) {
          intersection hit = firstHitBVH(
              r, bvhNodes, primitives, primitiveIndices, activeMask,
              primitiveRemap, residentPrimitiveCount, totalPrimitiveCount,
              primitiveRayStats, blasRoot);
          if (hit.primitiveId != -1 && hit.t < bestHit.t) {
            bestHit = hit;
            if (cache) {
              cache->valid = true;
              cache->instanceId = instanceId;
              cache->blasRootIndex = blasRoot;
              cache->boundsMin = bmin;
              cache->boundsMax = bmax;
            }
          }
        }
        continue;
      }

    bool leftHit = false;
    bool rightHit = false;
    float leftKey = INFINITY;
    float rightKey = INFINITY;

    if (leftChild >= 0) {
      float3 leftMin = tlasNodes[2 * leftChild + 0].xyz;
      float3 leftMax = tlasNodes[2 * leftChild + 1].xyz;
      leftHit = intersectAABB(r, leftMin, leftMax, RAY_EPS, bestHit.t);
      leftKey = dot((leftMin + leftMax) * 0.5 - r.origin, r.direction);
    }

    if (rightChild >= 0) {
      float3 rightMin = tlasNodes[2 * rightChild + 0].xyz;
      float3 rightMax = tlasNodes[2 * rightChild + 1].xyz;
      rightHit = intersectAABB(r, rightMin, rightMax, RAY_EPS, bestHit.t);
      rightKey = dot((rightMin + rightMax) * 0.5 - r.origin, r.direction);
    }

    if (leftHit && rightHit) {
      if (stackPtr + 2 <= stackSize) {
        if (leftKey < rightKey) {
          stack[stackPtr++] = rightChild;
          stack[stackPtr++] = leftChild;
        } else {
          stack[stackPtr++] = leftChild;
          stack[stackPtr++] = rightChild;
        }
      } else if (stackPtr < stackSize) {
        stack[stackPtr++] = (leftKey < rightKey) ? leftChild : rightChild;
      }
    } else if (leftHit) {
      if (stackPtr < stackSize)
        stack[stackPtr++] = leftChild;
    } else if (rightHit) {
      if (stackPtr < stackSize)
        stack[stackPtr++] = rightChild;
    }
  }

  return bestHit;
}

template <typename TextureArray>
inline PathTraceSample rayColor(Ray r, float3 rayDx, float3 rayDy,
                                device const float4 *tlasNodes,
                                uint tlasNodeCount, device const float4 *bvhNodes,
                                device const float4 *primitives,
                                device const float4 *materials,
                                uint primitiveCount,
                                device const int *primitiveIndices,
                                device const uchar *activeMask,
                                device const InstanceRecord *instanceRecords,
                                device const uint *lightIndices,
                                device const float *lightCdf,
                                device const uint *primitiveRemap,
                                device atomic_uint *primitiveRayStats,
                                thread uint32_t &seed, uint maxRayDepth,
                                uint debugAS, uint blasNodeCount, uint lightCount,
                                float lightTotalWeight, uint totalPrimitiveCount,
                                thread const TextureArray &textures,
                                uint textureCount,
                                texture2d<float, access::sample> environmentMap,
                                sampler environmentSampler,
                                uint environmentEnabled,
                                float environmentIntensity) {
  PathTraceSample sampleResult;
  sampleResult.radiance = float3(0.0f);
  sampleResult.albedo = float3(0.0f);
  sampleResult.normal = float3(0.0f);
  sampleResult.position = float3(0.0f);
  sampleResult.roughness = 0.0f;
  bool recordedFirstHit = false;

  float footprint = length(cross(rayDx, rayDy));
  float lodAtten = 1.0 / (1.0 + footprint);
  if (debugAS == 1) {
    for (uint i = 0; i < tlasNodeCount; ++i) {
      float3 bmin = tlasNodes[2 * i + 0].xyz;
      float3 bmax = tlasNodes[2 * i + 1].xyz;
      if (intersectAABB(r, bmin, bmax, RAY_EPS, INFINITY)) {
        float t = (tlasNodeCount > 1) ? float(i) / float(tlasNodeCount - 1) : 0.0;
        sampleResult.radiance = float3(t, 1.0 - t, 0.0);
        return sampleResult;
      }
    }
    return sampleResult;
  } else if (debugAS == 2) {
    intersection bestHit = firstHitTLAS(
        r, tlasNodes, tlasNodeCount, bvhNodes, primitives, primitiveIndices,
        activeMask, instanceRecords, nullptr, 0u, 0u, nullptr, nullptr);
    if (bestHit.primitiveId != -1) {
      float t = (blasNodeCount > 1)
                    ? float(bestHit.nodeIndex) / float(blasNodeCount - 1)
                    : 0.0;
      sampleResult.radiance = float3(t, 0.0, 1.0 - t);
      return sampleResult;
    }
    return sampleResult;
  }

  float4 absorption = float4(1.0);
  float4 light = float4(0.0);

  uint globalStatsBase = totalPrimitiveCount * 2u;
  uint envHitIndex = globalStatsBase;
  uint totalRayIndex = globalStatsBase + 1u;

  for (uint depth = 0; depth < maxRayDepth; ++depth) {
    if (primitiveRayStats) {
      atomic_fetch_add_explicit(&primitiveRayStats[totalRayIndex], 1u,
                                memory_order_relaxed);
    }

    TlasLeafCache bounceCache;
    bounceCache.valid = false;
    intersection bestHit = firstHitTLAS(
        r, tlasNodes, tlasNodeCount, bvhNodes, primitives, primitiveIndices,
        activeMask, instanceRecords, primitiveRemap, primitiveCount,
        totalPrimitiveCount, primitiveRayStats, &bounceCache);

    if (bestHit.primitiveId == -1) {
      if (primitiveRayStats) {
        atomic_fetch_add_explicit(&primitiveRayStats[envHitIndex], 1u,
                                  memory_order_relaxed);
      }

      float3 unitDir = normalize(r.direction);
      if (environmentEnabled > 0 && environmentIntensity > 0.0f) {
        uint envWidth = environmentMap.get_width();
        uint envHeight = environmentMap.get_height();
        if (envWidth > 0 && envHeight > 0) {
          float cappedY = clamp(unitDir.y, -1.0f, 1.0f);
          float azimuth = atan2(unitDir.z, unitDir.x);
          float elevation = asin(cappedY);
          float uCoord = 0.5f + azimuth / (2.0f * M_PI);
          float vCoord = 0.5f - elevation / M_PI;
          float2 envUV = float2(uCoord, vCoord);
          float4 envSample = environmentMap.sample(environmentSampler, envUV);
          float3 envColor = envSample.xyz * environmentIntensity;
          light += absorption * float4(envColor, 1.0f);
          break;
        }
      }
      float t = 0.5 * (unitDir.y + 1.0);
      float3 skyColor = mix(float3(1.0), float3(0.6, 0.7, 1.0), t);
      light += absorption * float4(skyColor, 1.0);
      break;
    }
    if (primitiveCount > 0 && bestHit.primitiveId >= 0) {
      uint localIndex = static_cast<uint>(bestHit.primitiveId);
      if (localIndex < primitiveCount) {
        uint globalId = primitiveRemap[localIndex];
        if (globalId < totalPrimitiveCount && primitiveRayStats) {
          uint statsIndex = globalId * 2u;
          atomic_fetch_add_explicit(&primitiveRayStats[statsIndex], 1u,
                                    memory_order_relaxed);
        }
      }
    }
    if (debugAS == 3) {
      float3 debugLightPos = float3(0.0f, 5.0f, 0.0f);
      float3 hitNormal = bestHit.normal;
      if (!all(isfinite(hitNormal)) || dot(hitNormal, hitNormal) <= RAY_EPS) {
        if (dot(r.direction, r.direction) > RAY_EPS &&
            all(isfinite(r.direction))) {
          hitNormal = -normalize(r.direction);
        } else {
          hitNormal = float3(0.0f, 1.0f, 0.0f);
        }
      }
      int rootIndex =
          (bounceCache.valid && bounceCache.blasRootIndex >= 0)
              ? bounceCache.blasRootIndex
              : 0;
      bool visible = isVisibleToLight(
          bestHit.point, hitNormal, debugLightPos, bvhNodes, primitives,
          primitiveIndices, activeMask, primitiveRemap, primitiveCount,
          totalPrimitiveCount, primitiveRayStats, rootIndex);
      sampleResult.radiance =
          visible ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
      return sampleResult;
    }
    int matBase = bestHit.primitiveId * int(kMaterialFloat4Count);
    int totalEntries = int(primitiveCount) * int(kMaterialFloat4Count);
    if (matBase + int(kMaterialFloat4Count) > totalEntries)
      break;

    MaterialPayload material = decodeMaterial(matBase, materials, lodAtten);
    if (!bestHit.supportsNormalMap)
      material.normalTextureIndex = -1;

    float2 surfaceUV = float2(0.0f);
    bool haveUV = false;
    int primBase = bestHit.primitiveId * int(kPrimitiveFloat4Count);
    int primEntries = int(primitiveCount) * int(kPrimitiveFloat4Count);
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
                                            surfaceUV, textures, textureCount);
        material.diffuseColor *= sample.xyz;
        material.opacity *= clamp(sample.w, 0.0f, 1.0f);
      }
      if (material.specularTextureIndex >= 0) {
        float4 sample = samplePackedTexture(material.specularTextureIndex,
                                            surfaceUV, textures, textureCount);
        material.specularColor *= sample.xyz;
      }
    }

    float emissionStrength = material.emissionPower *
                             luminance(material.emissionColor);
    if (emissionStrength > 0.0f) {
      light += absorption * float4(material.emissionColor, 1.0) *
               material.emissionPower;
    }

    float3 shadingNormal = bestHit.normal;
    if (haveUV && material.normalTextureIndex >= 0 &&
        bestHit.supportsNormalMap) {
      float4 normalSample =
          samplePackedTexture(material.normalTextureIndex, surfaceUV, textures,
                              textureCount);
      float3 sampledNormal = normalSample.xyz;
      float3 localNormal = sampledNormal * 2.0f - float3(1.0f);

      float3 geomNormal = bestHit.normal;
      float3 tangent = bestHit.tangent;
      float3 bitangent = bestHit.bitangent;

      bool validFrame =
          dot(tangent, tangent) > RAY_EPS && dot(bitangent, bitangent) > RAY_EPS &&
          dot(geomNormal, geomNormal) > RAY_EPS && all(isfinite(tangent)) &&
          all(isfinite(bitangent)) && all(isfinite(geomNormal));

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

    float3 offsetNormal = shadingNormal;
    if (!recordedFirstHit) {
      sampleResult.albedo = material.diffuseColor;
      sampleResult.normal = offsetNormal;
      sampleResult.position = bestHit.point;
      sampleResult.roughness = clamp(material.roughness, 0.0f, 1.0f);
      recordedFirstHit = true;
    }

    float3 diffuseColor = material.diffuseColor * material.opacity;
    float diffuseLum = luminance(diffuseColor);
    if (diffuseLum > 0.0f && lightCount > 0 && lightTotalWeight > 0.0f) {
      float lightXi = randomFloat(seed);
      seed = random(seed);
      uint selectedOffset =
          selectLightOffset(lightXi, lightCdf, lightCount, lightTotalWeight);
      if (selectedOffset < lightCount) {
        uint lightPrimIndex = lightIndices[selectedOffset];
        if (int(lightPrimIndex) != bestHit.primitiveId) {
          int base = int(lightPrimIndex) * int(kPrimitiveFloat4Count);
          float4 lp0 = primitives[base + 0];
          float4 lp1 = primitives[base + 1];
          float4 lp2 = primitives[base + 2];
          float3 lightPoint;
          float3 lightNormal;
          float area = 0.0f;
          if (sampleLightPoint(int(lp0.w), lp0, lp1, lp2, seed, lightPoint,
                               lightNormal, area) && area > 0.0f) {
            float3 toLight = lightPoint - bestHit.point;
            float dist2 = dot(toLight, toLight);
            if (dist2 > RAY_EPS) {
              float dist = sqrt(dist2);
              float3 wi = toLight / dist;
              float cosTheta = max(dot(offsetNormal, wi), 0.0f);
              float cosLight = max(dot(lightNormal, -wi), 0.0f);
              if (cosTheta > 0.0f && cosLight > 0.0f) {
                float currentCdf = lightCdf[selectedOffset];
                float prevCdf =
                    (selectedOffset > 0) ? lightCdf[selectedOffset - 1] : 0.0f;
                float weight = currentCdf - prevCdf;
                float lightPdf =
                    (lightTotalWeight > 0.0f && weight > 0.0f)
                        ? weight / lightTotalWeight
                        : 0.0f;
                if (lightPdf > 0.0f) {
                  float pdfArea = 1.0f / area;
                  float pdfSolid = pdfArea * dist2 / cosLight;
                  float totalPdf = lightPdf * pdfSolid;
                  if (totalPdf > 0.0f) {
                    Ray shadowRay;
                    shadowRay.origin = bestHit.point + 0.0005f * offsetNormal;
                    shadowRay.direction = wi;
                    shadowRay.minDistance = RAY_EPS;
                    shadowRay.maxDistance = dist - RAY_EPS;
                    if (shadowRay.maxDistance < shadowRay.minDistance)
                      shadowRay.maxDistance = shadowRay.minDistance;
                    intersection shadowHit;
                    bool usedCache = false;
                    if (bounceCache.valid) {
                      bool intersectsCached = intersectAABB(
                          shadowRay, bounceCache.boundsMin,
                          bounceCache.boundsMax, shadowRay.minDistance,
                          shadowRay.maxDistance);
                      if (intersectsCached && bounceCache.blasRootIndex >= 0) {
                        intersection cachedHit = firstHitBVH(
                            shadowRay, bvhNodes, primitives, primitiveIndices,
                            activeMask, primitiveRemap, primitiveCount,
                            totalPrimitiveCount, primitiveRayStats,
                            bounceCache.blasRootIndex);
                        if (cachedHit.primitiveId != -1 &&
                            cachedHit.primitiveId != int(lightPrimIndex) &&
                            cachedHit.t <= shadowRay.maxDistance) {
                          shadowHit = cachedHit;
                          usedCache = true;
                        }
                      }
                    }
                    if (!usedCache) {
                      shadowHit = firstHitTLAS(
                          shadowRay, tlasNodes, tlasNodeCount, bvhNodes,
                          primitives, primitiveIndices, activeMask,
                          instanceRecords, primitiveRemap, primitiveCount,
                          totalPrimitiveCount, primitiveRayStats, nullptr);
                    }
                    bool visible = false;
                    if (shadowHit.primitiveId == -1) {
                      visible = true;
                    } else if (shadowHit.primitiveId == int(lightPrimIndex) &&
                               shadowHit.t >= dist - 0.001f) {
                      visible = true;
                    }
                    if (visible) {
                      int lightMatIndex = int(lightPrimIndex) *
                                          int(kMaterialFloat4Count);
                      if (lightMatIndex + int(kMaterialFloat4Count) <=
                          int(primitiveCount) * int(kMaterialFloat4Count)) {
                        MaterialPayload lightMaterial =
                            decodeMaterial(lightMatIndex, materials, 1.0f);
                        float3 lightRadiance = lightMaterial.emissionColor *
                                              lightMaterial.emissionPower;
                        float3 viewDir = normalize(-r.direction);
                        float3 bsdf = evaluateDirectLightingBsdf(
                            material, offsetNormal, viewDir, wi);
                        float3 throughput = absorption.xyz * bsdf;
                        float3 contribution =
                            throughput * lightRadiance * (cosTheta / totalPdf);
                        light.xyz += contribution;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    r.minDistance = RAY_EPS;
    r.maxDistance = INFINITY;
    intersection shadingHit = bestHit;
    shadingHit.normal = shadingNormal;
    float3 scatterWeight = scatter(r, shadingHit, material, seed);
    seed = random(seed);
    float3 biasNormal =
        (dot(r.direction, offsetNormal) < 0.0f) ? -offsetNormal : offsetNormal;
    r.origin = bestHit.point + RAY_EPS * biasNormal;
    absorption.xyz *= scatterWeight;
    absorption.w = 1.0f;

    if (depth >= 5) {
      float p = max(absorption.x, max(absorption.y, absorption.z));
      float randVal = randomFloat(seed);
      seed = random(seed);
      if (randVal >= p)
        break;
      absorption /= p;
    }
  }

  sampleResult.radiance = clamp(light.xyz, 0.0, 1.0);
  return sampleResult;
}

#endif
