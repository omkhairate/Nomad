#ifndef PATH_TRACING_H
#define PATH_TRACING_H

#include <metal_atomic>
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

inline bool fetchTriangleFromHandle(const device GeometryHandle &handle,
                                    uint triangleIndex, thread float3 &v0,
                                    thread float3 &v1, thread float3 &v2) {
  if (!handle.vertexBytes || !handle.indexBytes)
    return false;

  uint triangleCount = handle.indexCount / 3;
  if (triangleIndex >= triangleCount)
    return false;

  uint vertexStride = (handle.vertexStride > 0) ? handle.vertexStride
                                                : static_cast<uint>(sizeof(float3));
  uint indexStride = (handle.indexStride > 0) ? handle.indexStride
                                              : static_cast<uint>(sizeof(uint));

  device const uchar *indexBytes = handle.indexBytes;
  uint baseIndex = triangleIndex * 3u;
  uint i0 = 0;
  uint i1 = 0;
  uint i2 = 0;

  if (indexStride == sizeof(uint)) {
    device const uint *indices =
        reinterpret_cast<device const uint *>(indexBytes);
    if (!indices || baseIndex + 2u >= handle.indexCount)
      return false;
    i0 = indices[baseIndex + 0u];
    i1 = indices[baseIndex + 1u];
    i2 = indices[baseIndex + 2u];
  } else if (indexStride == sizeof(ushort)) {
    device const ushort *indices =
        reinterpret_cast<device const ushort *>(indexBytes);
    if (!indices || baseIndex + 2u >= handle.indexCount)
      return false;
    i0 = static_cast<uint>(indices[baseIndex + 0u]);
    i1 = static_cast<uint>(indices[baseIndex + 1u]);
    i2 = static_cast<uint>(indices[baseIndex + 2u]);
  } else {
    return false;
  }

  if (i0 >= handle.vertexCount || i1 >= handle.vertexCount ||
      i2 >= handle.vertexCount)
    return false;

  device const uchar *vertexBytes = handle.vertexBytes;
  auto loadVertex = [&](uint index) -> float3 {
    device const float3 *ptr =
        reinterpret_cast<device const float3 *>(vertexBytes + index * vertexStride);
    return ptr ? *ptr : float3(0.0f);
  };

  v0 = loadVertex(i0);
  v1 = loadVertex(i1);
  v2 = loadVertex(i2);
  return true;
}

inline const device GeometryHandle *findGeometryHandleForInstance(
    device const GeometryHandle *geometryHandles, uint geometryHandleCount,
    uint instanceIndex) {
  if (!geometryHandles || geometryHandleCount == 0)
    return nullptr;

  uint targetSlot = instanceIndex + 1u;
  for (uint handleIndex = 1; handleIndex < geometryHandleCount; ++handleIndex) {
    const device GeometryHandle &handle = geometryHandles[handleIndex];
    if (handle.instanceSlot == targetSlot)
      return &handle;
  }
  return nullptr;
}

inline bool mapPrimitiveToInstance(uint primitiveIndex,
                                   device const InstanceRecord *instanceRecords,
                                   uint instanceCount,
                                   thread uint &instanceIndex,
                                   thread uint &localPrimitiveIndex) {
  if (!instanceRecords || instanceCount == 0)
    return false;

  for (uint idx = 0; idx < instanceCount; ++idx) {
    const device InstanceRecord &record = instanceRecords[idx];
    if (record.primitiveCount == 0)
      continue;
    uint start = record.primitiveBase;
    uint end = start + record.primitiveCount;
    if (primitiveIndex >= start && primitiveIndex < end) {
      instanceIndex = idx;
      localPrimitiveIndex = primitiveIndex - start;
      return true;
    }
  }
  return false;
}

inline bool fetchTriangleForPrimitive(
    uint primitiveIndex, device const InstanceRecord *instanceRecords,
    uint instanceCount, device const GeometryHandle *geometryHandles,
    uint geometryHandleCount, thread float3 &v0, thread float3 &v1,
    thread float3 &v2) {
  uint instanceIndex = 0;
  uint localIndex = 0;
  if (!mapPrimitiveToInstance(primitiveIndex, instanceRecords, instanceCount,
                              instanceIndex, localIndex))
    return false;

  const device GeometryHandle *handle =
      findGeometryHandleForInstance(geometryHandles, geometryHandleCount,
                                    instanceIndex);
  if (!handle)
    return false;

  return fetchTriangleFromHandle(*handle, localIndex, v0, v1, v2);
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

inline bool sampleLightPoint(
    int primitiveType, uint primitiveIndex, float4 p0, float4 p1, float4 p2,
    device const GeometryHandle *geometryHandles, uint geometryHandleCount,
    device const InstanceRecord *instanceRecords, uint instanceCount,
    thread uint32_t &seed, thread float3 &position, thread float3 &normal,
    thread float &area) {
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
    float3 v0 = p0.xyz;
    float3 v1 = p1.xyz;
    float3 v2 = p2.xyz;
    float3 baseEdge1 = v1 - v0;
    float3 baseEdge2 = v2 - v0;
    float3 baseCross = cross(baseEdge1, baseEdge2);
    float baseLen = length(baseCross);
    float3 triNormal = (baseLen > 0.0f) ? baseCross / baseLen : float3(0.0f);
    float triArea = 0.5f * baseLen;
    bool fetched = fetchTriangleForPrimitive(primitiveIndex, instanceRecords,
                                             instanceCount, geometryHandles,
                                             geometryHandleCount, v0, v1, v2);
    if (fetched) {
      float3 e1 = v1 - v0;
      float3 e2 = v2 - v0;
      float3 n = cross(e1, e2);
      float lenN = length(n);
      if (lenN <= 0.0f)
        return false;
      triNormal = n / lenN;
      triArea = 0.5f * lenN;
    }

    float u1 = randomFloat(seed);
    seed = random(seed);
    float u2 = randomFloat(seed);
    seed = random(seed);
    float su1 = sqrt(u1);
    float b0 = 1.0 - su1;
    float b1 = su1 * (1.0 - u2);
    float b2 = su1 * u2;
    position = b0 * v0 + b1 * v1 + b2 * v2;
    normal = triNormal;
    area = triArea;
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

inline intersection firstHitScene(
    thread const Ray &r, device const GeometryHandle *geometryHandles,
    uint geometryHandleCount, device const float4 *primitives,
    device const InstanceRecord *instanceRecords, uint instanceCount,
    device const uchar *activeMask, uint totalPrimitiveCount) {
  intersection bestHit;
  bestHit.t = INFINITY;
  bestHit.primitiveId = -1;
  bestHit.nodeIndex = -1;
  bestHit.isTriangle = 0;

  if (!primitives || !instanceRecords || totalPrimitiveCount == 0)
    return bestHit;

  if (instanceCount == 0)
    return bestHit;

  for (uint instanceIdx = 0; instanceIdx < instanceCount; ++instanceIdx) {
    const device InstanceRecord &record = instanceRecords[instanceIdx];
    if (record.primitiveCount == 0)
      continue;

    const device GeometryHandle *handle =
        findGeometryHandleForInstance(geometryHandles, geometryHandleCount,
                                      instanceIdx);

    for (uint local = 0; local < record.primitiveCount; ++local) {
      uint primitiveIndex = record.primitiveBase + local;
      if (primitiveIndex >= totalPrimitiveCount)
        continue;
      if (activeMask && !activeMask[primitiveIndex])
        continue;

      uint base = primitiveIndex * 3u;
      float4 p0 = primitives[base + 0u];
      float4 p1 = primitives[base + 1u];
      float4 p2 = primitives[base + 2u];

      int primitiveType = static_cast<int>(p0.w);
      float tHit = INFINITY;
      float3 hitPoint = float3(0.0f);
      float3 normal = float3(0.0f);
      bool hitThis = false;

      if (primitiveType == 0) {
        float3 center = p0.xyz;
        float radius = p1.x;
        float3 oc = r.origin - center;
        float a = dot(r.direction, r.direction);
        float b = dot(oc, r.direction);
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - a * c;
        if (discriminant > 0.0f) {
          float sqrtD = sqrt(discriminant);
          float temp = (-b - sqrtD) / a;
          float minT = max(r.minDistance, 0.0001f);
          float maxT = (r.maxDistance > 0.0f) ? r.maxDistance : INFINITY;
          if (temp > minT && temp < maxT && temp < bestHit.t) {
            tHit = temp;
            hitPoint = r.origin + tHit * r.direction;
            normal = normalize(hitPoint - center);
            hitThis = true;
          } else {
            temp = (-b + sqrtD) / a;
            if (temp > minT && temp < maxT && temp < bestHit.t) {
              tHit = temp;
              hitPoint = r.origin + tHit * r.direction;
              normal = normalize(hitPoint - center);
              hitThis = true;
            }
          }
        }
      } else if (primitiveType == 1) {
        float3 v0 = p0.xyz;
        float3 v1 = p1.xyz;
        float3 v2 = p2.xyz;
        if (handle)
          fetchTriangleFromHandle(*handle, local, v0, v1, v2);

        float3 barycentric = float3(0.0f);
        if (triangleIntersection(r, v0, v1, v2, tHit, barycentric) &&
            tHit < bestHit.t) {
          float3 edge1 = v1 - v0;
          float3 edge2 = v2 - v0;
          normal = normalize(cross(edge1, edge2));
          hitPoint = r.origin + tHit * r.direction;
          hitThis = true;
        }
      } else if (primitiveType == 2) {
        float3 center = p0.xyz;
        float3 e1 = p1.xyz;
        float3 e2 = p2.xyz;
        float3 rectNormal;
        if (rectangleIntersection(r, center, e1, e2, tHit, rectNormal) &&
            tHit < bestHit.t) {
          hitPoint = r.origin + tHit * r.direction;
          normal = rectNormal;
          hitThis = true;
        }
      }

      if (hitThis && tHit < bestHit.t) {
        bestHit.t = tHit;
        bestHit.primitiveId = static_cast<int>(primitiveIndex);
        bestHit.point = hitPoint;
        bestHit.normal = normal;
        bestHit.isTriangle = primitiveType;
        bestHit.nodeIndex = static_cast<int>(instanceIdx);
      }
    }
  }

  if (bestHit.primitiveId != -1) {
    bestHit.frontFace = dot(bestHit.normal, r.direction) < 0.0;
    if (!bestHit.frontFace)
      bestHit.normal = -bestHit.normal;
  }

  return bestHit;
}

inline float4 rayColor(Ray r, float3 rayDx, float3 rayDy,
                       device const GeometryHandle *geometryHandles,
                       uint geometryHandleCount,
                       device const float4 *primitives,
                       device const float4 *materials, uint primitiveCount,
                       device const uchar *activeMask,
                       device const InstanceRecord *instanceRecords,
                       uint instanceCount, device const uint *lightIndices,
                       device const float *lightCdf,
                       device const uint *primitiveRemap,
                       device atomic_uint *primitiveHitCounts,
                       thread uint32_t &seed, uint maxRayDepth,
                       uint debugAS, uint lightCount,
                       float lightTotalWeight, uint totalPrimitiveCount) {
  float footprint = length(cross(rayDx, rayDy));
  float lodAtten = 1.0 / (1.0 + footprint);
  if (debugAS == 1) {
    intersection debugHit = firstHitScene(r, geometryHandles, geometryHandleCount,
                                          primitives, instanceRecords,
                                          instanceCount, activeMask,
                                          totalPrimitiveCount);
    if (debugHit.primitiveId != -1) {
      uint instanceIdx = (debugHit.nodeIndex >= 0)
                             ? static_cast<uint>(debugHit.nodeIndex)
                             : 0u;
      float denom = (instanceCount > 1) ? float(instanceCount - 1) : 1.0f;
      float t = min(1.0f, float(instanceIdx) / denom);
      return float4(t, 1.0f - t, 0.0f, 1.0f);
    }
    return float4(0.0, 0.0, 0.0, 1.0);
  } else if (debugAS == 2) {
    intersection debugHit = firstHitScene(r, geometryHandles, geometryHandleCount,
                                          primitives, instanceRecords,
                                          instanceCount, activeMask,
                                          totalPrimitiveCount);
    if (debugHit.primitiveId != -1) {
      float denom = (totalPrimitiveCount > 1)
                        ? float(totalPrimitiveCount - 1)
                        : 1.0f;
      float t = min(1.0f, float(debugHit.primitiveId) / denom);
      return float4(t, 0.0f, 1.0f - t, 1.0f);
    }
    return float4(0.0, 0.0, 0.0, 1.0);
  }

  float4 absorption = float4(1.0);
  float4 light = float4(0.0);

  for (uint depth = 0; depth < maxRayDepth; ++depth) {
    intersection bestHit = firstHitScene(r, geometryHandles, geometryHandleCount,
                                         primitives, instanceRecords,
                                         instanceCount, activeMask,
                                         totalPrimitiveCount);

    if (bestHit.primitiveId == -1) {
      float3 unitDir = normalize(r.direction);
      float t = 0.5 * (unitDir.y + 1.0);
      float3 skyColor = mix(float3(1.0), float3(0.6, 0.7, 1.0), t);
      light += absorption * float4(skyColor, 1.0);
      break;
    }
    if (primitiveCount > 0 && bestHit.primitiveId >= 0) {
      uint localIndex = static_cast<uint>(bestHit.primitiveId);
      if (localIndex < primitiveCount) {
        uint globalId = primitiveRemap[localIndex];
        if (globalId < totalPrimitiveCount)
          atomic_fetch_add_explicit(&primitiveHitCounts[globalId], 1u,
                                    memory_order_relaxed);
      }
    }
    int matIndex = bestHit.primitiveId * 2;
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

    float3 offsetNormal = bestHit.frontFace ? bestHit.normal : -bestHit.normal;

    if (materialType == 0.0 && lightCount > 0 && lightTotalWeight > 0.0f) {
      float lightXi = randomFloat(seed);
      seed = random(seed);
      uint selectedOffset =
          selectLightOffset(lightXi, lightCdf, lightCount, lightTotalWeight);
      if (selectedOffset < lightCount) {
        uint lightPrimIndex = lightIndices[selectedOffset];
        if (int(lightPrimIndex) != bestHit.primitiveId) {
          int base = int(lightPrimIndex) * 3;
          float4 lp0 = primitives[base + 0];
          float4 lp1 = primitives[base + 1];
          float4 lp2 = primitives[base + 2];
          float3 lightPoint;
          float3 lightNormal;
          float area = 0.0f;
          if (sampleLightPoint(int(lp0.w), lightPrimIndex, lp0, lp1, lp2,
                               geometryHandles, geometryHandleCount,
                               instanceRecords, instanceCount, seed, lightPoint,
                               lightNormal, area) &&
              area > 0.0f) {
            float3 toLight = lightPoint - bestHit.point;
            float dist2 = dot(toLight, toLight);
            if (dist2 > 1e-6f) {
              float dist = sqrt(dist2);
              float3 wi = toLight / dist;
              float cosTheta = max(dot(bestHit.normal, wi), 0.0f);
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
                    shadowRay.minDistance = 0.0001f;
                    shadowRay.maxDistance = dist - 0.0001f;
                    if (shadowRay.maxDistance < shadowRay.minDistance)
                      shadowRay.maxDistance = shadowRay.minDistance;
                    intersection shadowHit = firstHitScene(
                        shadowRay, geometryHandles, geometryHandleCount,
                        primitives, instanceRecords, instanceCount, activeMask,
                        totalPrimitiveCount);
                    bool visible = false;
                    if (shadowHit.primitiveId == -1) {
                      visible = true;
                    } else if (shadowHit.primitiveId == int(lightPrimIndex) &&
                               shadowHit.t >= dist - 0.001f) {
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

    r.origin = bestHit.point + 0.0001 * offsetNormal;
    r.minDistance = 0.0001f;
    r.maxDistance = INFINITY;
    scatter(r, bestHit, materialType, seed);
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
