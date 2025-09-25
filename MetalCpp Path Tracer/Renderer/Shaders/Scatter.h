#ifndef SCATTER_H
#define SCATTER_H

#include <metal_stdlib>
#include <metal_raytracing>

#include "Random.h"
#include "Structs.h"

using namespace metal;

struct BSDFSample {
  float3 direction;
  float3 weight; // bsdf * cos(theta) / pdf contribution
  float pdf;
  bool isDelta;
};

inline bool mirrorAngle(float refractionIndex, thread const float3 &normal,
                        thread const float3 &rayDir, thread uint32_t &seed) {
  const float cosTheta = dot(-1 * rayDir, normal);
  const float sinTheta = sqrt(fmax(0.0, 1.0 - cosTheta * cosTheta));

  float r0 = (1 - refractionIndex) / (1 + refractionIndex);
  r0 = r0 * r0;
  const float reflectance = r0 + (1 - r0) * pow((1 - cosTheta), 5.0);

  return ((refractionIndex * sinTheta > 1.0) ||
          (reflectance > randomFloat(seed)));
}

inline void buildOrthonormalBasis(thread const float3 &n, thread float3 &t,
                                  thread float3 &b) {
  if (fabs(n.z) < 0.999) {
    t = normalize(cross(n, float3(0.0, 0.0, 1.0)));
  } else {
    t = normalize(cross(n, float3(0.0, 1.0, 0.0)));
  }
  b = cross(n, t);
}

inline BSDFSample sampleBSDF(thread const float3 &inDir,
                             thread const float3 &normal, bool frontFace,
                             float materialType, thread const float3 &albedo,
                             thread uint32_t &seed) {
  BSDFSample s;
  s.direction = normal;
  s.weight = float3(0.0);
  s.pdf = 0.0;
  s.isDelta = false;

  if (materialType == 0.0) {
    // Diffuse (Lambertian)
    float u1 = randomFloat(seed);
    seed = random(seed);
    float u2 = randomFloat(seed);
    seed = random(seed);

    float r = sqrt(u1);
    float theta = 2.0 * M_PI * u2;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(fmax(0.0, 1.0 - u1));

    float3 tangent, bitangent;
    buildOrthonormalBasis(normal, tangent, bitangent);
    float3 sampled = tangent * x + bitangent * y + normal * z;
    s.direction = normalize(sampled);

    float cosTheta = fmax(0.0, dot(s.direction, normal));
    s.pdf = cosTheta / M_PI;
    if (s.pdf > 0.0) {
      s.weight = albedo;
    }
  } else if (materialType < 0.0) {
    // Perfect mirror
    float3 dir = reflect(inDir, normal);
    s.direction = normalize(dir);
    s.pdf = 1.0;
    s.weight = albedo;
    s.isDelta = true;
  } else {
    // Dielectric with simple Fresnel reflection/refraction
    float eta = materialType;
    eta = frontFace ? (1.0 / eta) : eta;
    bool reflectDir = mirrorAngle(eta, normal, inDir, seed);
    float3 dir = reflectDir ? reflect(inDir, normal)
                            : refract(inDir, normal, eta);
    s.direction = normalize(dir);
    s.pdf = 1.0;
    s.weight = albedo;
    s.isDelta = true;
    seed = random(seed);
  }

  return s;
}

inline float3 evaluateBSDF(thread const float3 &inDir,
                           thread const float3 &outDir,
                           thread const float3 &normal, float materialType,
                           thread const float3 &albedo) {
  if (materialType == 0.0) {
    float cosTheta = fmax(0.0, dot(outDir, normal));
    if (cosTheta <= 0.0)
      return float3(0.0);
    return albedo / M_PI;
  }
  // Delta materials have zero contribution for arbitrary directions
  return float3(0.0);
}

inline float schlickReflectance(float cosine, float refIdx) {
  float r0 = (1.0 - refIdx) / (1.0 + refIdx);
  r0 = r0 * r0;
  return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

inline void scatter(thread ray &r, thread const intersection &hit,
                    float materialType, thread uint32_t &seed) {
  float3 normal = hit.frontFace ? hit.normal : -hit.normal;

  if (materialType == 0.0) {
    float3 scatterDir = normal + randomUnitVector(seed);
    seed = random(seed);
    if (length_squared(scatterDir) < 1e-6)
      scatterDir = normal;
    r.direction = normalize(scatterDir);
  } else if (materialType < 0.0) {
    r.direction = normalize(reflect(r.direction, normal));
  } else {
    float refractionRatio = hit.frontFace ? (1.0 / materialType) : materialType;
    float cosTheta = fmin(dot(-r.direction, normal), 1.0);
    float sinTheta = sqrt(fmax(0.0, 1.0 - cosTheta * cosTheta));
    bool cannotRefract = refractionRatio * sinTheta > 1.0;
    float reflectProb = schlickReflectance(cosTheta, materialType);
    bool shouldReflect = cannotRefract || reflectProb > randomFloat(seed);
    seed = random(seed);
    float3 dir = shouldReflect ? reflect(r.direction, normal)
                               : refract(r.direction, normal, refractionRatio);
    r.direction = normalize(dir);
  }

  r.min_distance = 0.0001;
  r.max_distance = INFINITY;
}

#endif
