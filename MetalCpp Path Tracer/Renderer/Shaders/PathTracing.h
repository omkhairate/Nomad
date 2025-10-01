#ifndef PATH_TRACING_H
#define PATH_TRACING_H

#include <metal_atomic>
#include <metal_raytracing>
#include <metal_stdlib>

#define M_PI 3.14159265358979323846

using namespace metal;

struct Ray {
  float3 origin;
  float3 direction;
  float minDistance = 0.0f;
  float maxDistance = 0.0f;
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

namespace mtlrt = metal::raytracing;

struct RayQueryResult {
  bool valid = false;
  intersection hit;
  uint instanceId = 0;
  uint primitiveLocal = 0;
  float2 barycentrics = float2(0.0f);
};

inline bool loadTriangleFromHandle(thread const GeometryHandle &handle,
                                   uint primitiveLocal,
                                   thread float3 &v0, thread float3 &v1,
                                   thread float3 &v2) {
  if (handle.vertexBufferAddress == 0 || handle.indexBufferAddress == 0)
    return false;

  uint indexStride = handle.indexStride > 0 ? handle.indexStride
                                            : static_cast<uint>(sizeof(uint));
  uint vertexStride =
      handle.vertexStride > 0 ? handle.vertexStride
                              : static_cast<uint>(sizeof(float3));

  uint baseIndex = primitiveLocal * 3u;
  if (baseIndex + 2u >= handle.indexCount)
    return false;

  device const uchar *indexBytes =
      reinterpret_cast<device const uchar *>(handle.indexBufferAddress);
  uint i0 = *reinterpret_cast<device const uint *>(indexBytes + indexStride * (baseIndex + 0u));
  uint i1 = *reinterpret_cast<device const uint *>(indexBytes + indexStride * (baseIndex + 1u));
  uint i2 = *reinterpret_cast<device const uint *>(indexBytes + indexStride * (baseIndex + 2u));

  if (i0 >= handle.vertexCount || i1 >= handle.vertexCount ||
      i2 >= handle.vertexCount)
    return false;

  device const uchar *vertexBytes =
      reinterpret_cast<device const uchar *>(handle.vertexBufferAddress);
  v0 = *reinterpret_cast<device const float3 *>(vertexBytes + vertexStride * i0);
  v1 = *reinterpret_cast<device const float3 *>(vertexBytes + vertexStride * i1);
  v2 = *reinterpret_cast<device const float3 *>(vertexBytes + vertexStride * i2);
  return true;
}

inline RayQueryResult traceRay(mtlrt::instance_acceleration_structure tlas,
                               device const GeometryHandle *geometryHandles,
                               device const InstanceRecord *instanceRecords,
                               device const uchar *activeMask,
                               uint primitiveCount, thread const Ray &r) {
  RayQueryResult result;
  result.hit.t = INFINITY;
  result.hit.primitiveId = -1;
  result.hit.nodeIndex = -1;
  result.hit.isTriangle = 0;

  // Note: Some SDKs don't expose metal::raytracing::is_null; assume TLAS is valid.

  mtlrt::ray rtRay(r.origin, r.direction, r.minDistance, r.maxDistance);
  mtlrt::intersector<mtlrt::triangle_data, mtlrt::instancing> intersector;
  intersector.assume_geometry_type(mtlrt::geometry_type::triangle);
  auto hit = intersector.intersect(rtRay, tlas);
  if (hit.type != mtlrt::intersection_type::triangle)
    return result;

  uint instanceId = hit.instance_id;
  uint primitiveLocal = hit.primitive_id;
  float distance = hit.distance;
  float2 bary = hit.triangle_barycentric_coord;

  if (!instanceRecords)
    return result;

  InstanceRecord record = instanceRecords[instanceId];
  if (record.primitiveCount == 0 || primitiveLocal >= record.primitiveCount)
    return result;

  uint primitiveIndex = record.primitiveBase + primitiveLocal;
  if (primitiveIndex >= primitiveCount)
    return result;

  if (activeMask && !activeMask[primitiveIndex])
    return result;

  uint geometryIndex = instanceId + 1u;
  GeometryHandle handle =
      geometryHandles ? geometryHandles[geometryIndex] : GeometryHandle{};
  float3 v0, v1, v2;
  if (!loadTriangleFromHandle(handle, primitiveLocal, v0, v1, v2))
    return result;

  float3 edge1 = v1 - v0;
  float3 edge2 = v2 - v0;
  float3 normal = cross(edge1, edge2);
  float normalLen = length(normal);
  if (normalLen <= 0.0f)
    return result;
  normal /= normalLen;

  float b0 = 1.0f - bary.x - bary.y;
  float b1 = bary.x;
  float b2 = bary.y;
  float3 point = b0 * v0 + b1 * v1 + b2 * v2;

  bool frontFace = dot(normal, r.direction) < 0.0f;
  float3 orientedNormal = frontFace ? normal : -normal;

  result.valid = true;
  result.instanceId = instanceId;
  result.primitiveLocal = primitiveLocal;
  result.barycentrics = bary;
  result.hit.t = distance;
  result.hit.point = point;
  result.hit.normal = orientedNormal;
  result.hit.frontFace = frontFace;
  result.hit.primitiveId = static_cast<int>(primitiveIndex);
  result.hit.isTriangle = 1;
  result.hit.nodeIndex = static_cast<int>(primitiveLocal);
  return result;
}

inline float4 rayColor(Ray r, float3 rayDx, float3 rayDy,
                       mtlrt::instance_acceleration_structure tlas,
                       device const GeometryHandle *geometryHandles,
                       device const float4 *primitives,
                       device const float4 *materials, uint primitiveCount,
                       device const uchar *activeMask,
                       device const InstanceRecord *instanceRecords,
                       device const uint *lightIndices,
                       device const float *lightCdf,
                       device const uint *primitiveRemap,
                       device atomic_uint *primitiveHitCounts,
                       thread uint32_t &seed, uint maxRayDepth,
                       uint debugAS, uint lightCount, float lightTotalWeight,
                       uint totalPrimitiveCount, uint tlasNodeCount,
                       uint blasNodeCount) {
  float footprint = length(cross(rayDx, rayDy));
  float lodAtten = 1.0 / (1.0 + footprint);

  if (debugAS == 1) {
    RayQueryResult debugHit = traceRay(tlas, geometryHandles, instanceRecords,
                                       activeMask, primitiveCount, r);
    if (debugHit.valid) {
      float denom = (tlasNodeCount > 1) ? float(tlasNodeCount - 1) : 1.0f;
      float t = denom > 0.0f ? clamp(float(debugHit.instanceId) / denom, 0.0f, 1.0f)
                             : 0.0f;
      return float4(t, 1.0f - t, 0.0f, 1.0f);
    }
    return float4(0.0, 0.0, 0.0, 1.0);
  } else if (debugAS == 2) {
    RayQueryResult debugHit = traceRay(tlas, geometryHandles, instanceRecords,
                                       activeMask, primitiveCount, r);
    if (debugHit.valid) {
      float denom = (blasNodeCount > 1) ? float(blasNodeCount - 1) : 1.0f;
      float key = denom > 0.0f
                      ? clamp(float(debugHit.hit.nodeIndex) / denom, 0.0f, 1.0f)
                      : 0.0f;
      return float4(key, 0.0f, 1.0f - key, 1.0f);
    }
    return float4(0.0, 0.0, 0.0, 1.0);
  }

  float4 absorption = float4(1.0);
  float4 light = float4(0.0);

  for (uint depth = 0; depth < maxRayDepth; ++depth) {
    RayQueryResult hit = traceRay(tlas, geometryHandles, instanceRecords,
                                  activeMask, primitiveCount, r);

    if (!hit.valid || hit.hit.primitiveId < 0) {
      float3 unitDir = normalize(r.direction);
      float t = 0.5 * (unitDir.y + 1.0);
      float3 skyColor = mix(float3(1.0), float3(0.6, 0.7, 1.0), t);
      light += absorption * float4(skyColor, 1.0);
      break;
    }

    uint localIndex = static_cast<uint>(hit.hit.primitiveId);
    if (primitiveCount > 0 && localIndex < primitiveCount) {
      uint globalId = primitiveRemap[localIndex];
      if (globalId < totalPrimitiveCount)
        atomic_fetch_add_explicit(&primitiveHitCounts[globalId], 1u,
                                  memory_order_relaxed);
    }

    int matIndex = hit.hit.primitiveId * 2;
    if (matIndex + 1 >= int(primitiveCount) * 2)
      break;

    float4 m0 = materials[matIndex + 0];
    float4 m1 = materials[matIndex + 1];

    float3 albedo = m0.xyz * lodAtten;
    float materialType = m0.w;
    float3 emissionColor = m1.xyz;
    float emissionPower = m1.w;

    if (emissionPower > 0.0 || materialType == 2) {
      light += absorption * float4(emissionColor, 1.0) * emissionPower;
    }

    float3 offsetNormal = hit.hit.frontFace ? hit.hit.normal : -hit.hit.normal;

    if (materialType == 0.0 && lightCount > 0 && lightTotalWeight > 0.0f) {
      float lightXi = randomFloat(seed);
      seed = random(seed);
      uint selectedOffset =
          selectLightOffset(lightXi, lightCdf, lightCount, lightTotalWeight);
      if (selectedOffset < lightCount) {
        uint lightPrimIndex = lightIndices[selectedOffset];
        if (int(lightPrimIndex) != hit.hit.primitiveId) {
          int base = int(lightPrimIndex) * 3;
          float4 lp0 = primitives[base + 0];
          float4 lp1 = primitives[base + 1];
          float4 lp2 = primitives[base + 2];
          float3 lightPoint;
          float3 lightNormal;
          float area = 0.0f;
          if (sampleLightPoint(int(lp0.w), lp0, lp1, lp2, seed, lightPoint,
                               lightNormal, area) && area > 0.0f) {
            float3 toLight = lightPoint - hit.hit.point;
            float dist2 = dot(toLight, toLight);
            if (dist2 > 1e-6f) {
              float dist = sqrt(dist2);
              float3 wi = toLight / dist;
              float cosTheta = max(dot(hit.hit.normal, wi), 0.0f);
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
                    shadowRay.origin = hit.hit.point + 0.0005f * offsetNormal;
                    shadowRay.direction = wi;
                    shadowRay.minDistance = 0.0001f;
                    shadowRay.maxDistance = dist - 0.0001f;
                    if (shadowRay.maxDistance < shadowRay.minDistance)
                      shadowRay.maxDistance = shadowRay.minDistance;
                    RayQueryResult shadowHit = traceRay(
                        tlas, geometryHandles, instanceRecords, activeMask,
                        primitiveCount, shadowRay);
                    bool visible = false;
                    if (!shadowHit.valid || shadowHit.hit.primitiveId == -1) {
                      visible = true;
                    } else if (shadowHit.hit.primitiveId == int(lightPrimIndex) &&
                               shadowHit.hit.t >= dist - 0.001f) {
                      visible = true;
                    }
                    if (visible) {
                      int lightMatIndex = int(lightPrimIndex) * 2;
                      if (lightMatIndex + 1 < int(primitiveCount) * 2) {
                        float4 lightM1 = materials[lightMatIndex + 1];
                        float3 lightRadiance = lightM1.xyz * lightM1.w;
                        float3 throughput = absorption.xyz * (albedo / M_PI);
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

    r.origin = hit.hit.point + 0.0001 * offsetNormal;
    r.minDistance = 0.0001f;
    r.maxDistance = INFINITY;
    scatter(r, hit.hit, materialType, seed);
    seed = random(seed);
    absorption *= float4(albedo, 1.0);

    if (depth >= 5) {
      float p = max(absorption.x, max(absorption.y, absorption.z));
      float randVal = randomFloat(seed);
      seed = random(seed);
      if (randVal >= p)
        break;
      absorption /= p;
    }
  }

  return clamp(light, 0.0, 1.0);
}

#endif



