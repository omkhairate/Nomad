#ifndef SCATTER_H
#define SCATTER_H

#include <metal_stdlib>

#include "Random.h"
#include "Structs.h"

struct MaterialPayload {
    float3 diffuseColor;
    float opacity;
    float3 specularColor;
    float shininess;
    float3 emissionColor;
    float emissionPower;
    float3 transmissionColor;
    float indexOfRefraction;
    float roughness;
    float pad0;
    int diffuseTextureIndex;
    int specularTextureIndex;
    int normalTextureIndex;
    int pad1;
};

inline float luminance(float3 c) {
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

inline bool mirrorAngle(float refractionIndex, thread const float3 &normal,
                        thread const float3 &rayDir, thread uint32_t &seed) {
    const float cosTheta = dot(-rayDir, normal);
    const float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));

    float r0 = (1.0f - refractionIndex) / (1.0f + refractionIndex);
    r0 = r0 * r0;
    const float reflectance = r0 + (1.0f - r0) * pow((1.0f - cosTheta), 5.0f);

    float xi = randomFloat(seed);
    seed = random(seed);

    return ((refractionIndex * sinTheta > 1.0f) || (reflectance > xi));
}

inline void buildOrthonormalBasis(thread const float3 &n,
                                  thread float3 &t, thread float3 &b) {
    float3 up = (fabs(n.z) < 0.999f) ? float3(0.0f, 0.0f, 1.0f)
                                     : float3(0.0f, 1.0f, 0.0f);
    t = normalize(cross(up, n));
    b = normalize(cross(n, t));
}

inline float3 sampleCosineHemisphere(thread const float3 &normal,
                                     thread uint32_t &seed) {
    float u1 = randomFloat(seed);
    seed = random(seed);
    float u2 = randomFloat(seed);
    seed = random(seed);

    float r = sqrt(max(0.0f, u1));
    float theta = 2.0f * M_PI * u2;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0f, 1.0f - u1));

    float3 tangent;
    float3 bitangent;
    buildOrthonormalBasis(normal, tangent, bitangent);
    float3 local = tangent * x + bitangent * y + normal * z;
    return normalize(local);
}

inline float3 scatter(thread Ray &r, thread const intersection &i,
                      thread const MaterialPayload &material,
                      thread uint32_t &seed) {
    float3 normal = i.frontFace ? i.normal : -i.normal;
    float3 shadingNormal = i.normal;

    float diffuseWeight = max(luminance(material.diffuseColor) * material.opacity,
                              0.0f);
    float specularWeight = max(luminance(material.specularColor) * material.opacity,
                               0.0f);
    float dielectricWeight = max(
        luminance(material.transmissionColor) * (1.0f - material.opacity), 0.0f);

    if (dielectricWeight > 0.0f) {
        specularWeight = 0.0f;
    }

    float totalWeight = diffuseWeight + specularWeight + dielectricWeight;
    if (totalWeight <= 0.0f) {
        r.direction = normal;
        return float3(0.0f);
    }

    float xi = randomFloat(seed) * totalWeight;
    seed = random(seed);

    if (xi < diffuseWeight) {
        float3 direction = sampleCosineHemisphere(normal, seed);
        r.direction = direction;
        float probability = max(diffuseWeight / totalWeight, 1e-4f);
        return (material.diffuseColor * material.opacity) / probability;
    }

    xi -= diffuseWeight;
    if (xi < specularWeight) {
        float3 reflectDir = reflect(r.direction, normal);
        float exponent = max(material.shininess, 1.0f);
        if (material.roughness > 0.0f) {
            float smoothness = clamp(1.0f - material.roughness, 0.0f, 1.0f);
            float roughExponent = max(1.0f, smoothness * smoothness * 256.0f);
            exponent = max(exponent, roughExponent);
        }

        float u1 = randomFloat(seed);
        seed = random(seed);
        float u2 = randomFloat(seed);
        seed = random(seed);

        float cosTheta = pow(u1, 1.0f / (exponent + 1.0f));
        float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
        float phi = 2.0f * M_PI * u2;

        float3 tangent;
        float3 bitangent;
        buildOrthonormalBasis(reflectDir, tangent, bitangent);
        float3 specDir = normalize(tangent * (sinTheta * cos(phi)) +
                                   bitangent * (sinTheta * sin(phi)) +
                                   reflectDir * cosTheta);
        r.direction = specDir;
        float probability = max(specularWeight / totalWeight, 1e-4f);
        return material.specularColor / probability;
    }

    float probability = max(dielectricWeight / totalWeight, 1e-4f);
    float eta = max(material.indexOfRefraction, 1e-3f);
    float etaRatio = i.frontFace ? (1.0f / eta) : eta;
    bool reflectSample = mirrorAngle(etaRatio, shadingNormal, r.direction, seed);
    if (reflectSample) {
        r.direction = reflect(r.direction, shadingNormal);
        return material.specularColor / probability;
    }

    float3 refracted = refract(r.direction, shadingNormal, etaRatio);
    if (!all(isfinite(refracted))) {
        r.direction = reflect(r.direction, shadingNormal);
        return material.specularColor / probability;
    }

    r.direction = normalize(refracted);
    return material.transmissionColor / probability;
}

#endif
