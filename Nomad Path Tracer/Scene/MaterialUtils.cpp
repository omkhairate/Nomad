#include "MaterialUtils.h"

#include "Scene.h"
#include "TextureLoader.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <simd/simd.h>
#include <tinyxml2.h>

namespace NomadPathTracer {
namespace MaterialUtils {
namespace {

float clamp01(float value) { return std::clamp(value, 0.0f, 1.0f); }

simd::float3 parseVec3(const char *str) {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  if (str)
    std::sscanf(str, "%f,%f,%f", &x, &y, &z);
  return simd::make_float3(x, y, z);
}

simd::float3 toFloat3(const tinyobj::real_t values[3]) {
  return simd::make_float3(static_cast<float>(values[0]),
                           static_cast<float>(values[1]),
                           static_cast<float>(values[2]));
}

float luminance(const simd::float3 &c) {
  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

float computeRoughness(float shininess, float explicitRoughness) {
  if (explicitRoughness > 0.0f)
    return clamp01(explicitRoughness);
  if (shininess <= 0.0f)
    return 1.0f;
  const float converted = std::sqrt(2.0f / (shininess + 2.0f));
  return clamp01(converted);
}

int ResolveTextureIndex(const std::string &texName, const std::string &baseDir,
                        Scene *scene, MaterialTextureCache &cache,
                        TextureUsage usage = TextureUsage::Color) {
  if (!scene || texName.empty())
    return -1;

  std::filesystem::path resolved(texName);
  if (resolved.is_relative())
    resolved = std::filesystem::path(baseDir) / resolved;
  resolved = resolved.lexically_normal();

  const std::string key = resolved.string();
  const TextureColorSpace colorSpace = DetermineTextureColorSpace(key, usage);
  std::string cacheKey = key;
  cacheKey += (colorSpace == TextureColorSpace::Linear) ? "|linear" : "|srgb";

  auto it = cache.find(cacheKey);
  if (it != cache.end())
    return it->second;

  LoadedTextureImage image;
  if (!LoadTextureImage(key, image, usage)) {
    cache.emplace(cacheKey, -1);
    return -1;
  }

  Texture texture;
  texture.width = static_cast<uint32_t>(image.width);
  texture.height = static_cast<uint32_t>(image.height);
  texture.pixels = std::move(image.pixels);

  const int index = scene->registerTexture(cacheKey, key, std::move(texture));
  cache.emplace(cacheKey, index);
  return index;
}

} // namespace

MaterialTextureCache &GetTextureCache() {
  static MaterialTextureCache cache;
  return cache;
}

void ClearTextureCache() { GetTextureCache().clear(); }

bool HasOverrideAttributes(const tinyxml2::XMLElement *element) {
  if (!element)
    return false;

  return element->Attribute("diffuse") || element->Attribute("albedo") ||
         element->Attribute("emission") ||
         element->FindAttribute("emissionPower") ||
         element->Attribute("specular") ||
         element->Attribute("transmission") ||
         element->FindAttribute("opacity") ||
         element->FindAttribute("shininess") ||
         element->FindAttribute("roughness") ||
         element->FindAttribute("ior") ||
         element->FindAttribute("materialType") ||
         element->Attribute("diffuseTexture") ||
         element->Attribute("albedoTexture") ||
         element->Attribute("specularTexture") ||
         element->Attribute("normalTexture") ||
         element->Attribute("normalMap") ||
         element->Attribute("bumpTexture");
}

bool NeedsFallbackMaterial(const Material &material) {
  constexpr float kMinMaterialLuminance = 1.0e-4f;
  const bool hasTextures = material.diffuseTextureIndex >= 0 ||
                           material.specularTextureIndex >= 0 ||
                           material.normalTextureIndex >= 0;
  if (hasTextures)
    return false;
  return luminance(material.diffuseColor) <= kMinMaterialLuminance &&
         luminance(material.specularColor) <= kMinMaterialLuminance;
}

Material ConvertTinyMaterial(const tinyobj::material_t &src, Scene *scene,
                             MaterialTextureCache &textureCache,
                             const std::string &baseDir) {
  Material material{};
  material.diffuseColor = toFloat3(src.diffuse);
  material.specularColor = toFloat3(src.specular);
  material.transmissionColor = toFloat3(src.transmittance);
  material.emissionColor = toFloat3(src.emission);

  const float emissionLum = luminance(material.emissionColor);
  material.emissionPower = (emissionLum > 0.0f) ? 1.0f : 0.0f;

  material.shininess = static_cast<float>(src.shininess);
  material.indexOfRefraction =
      (src.ior != 0.0f) ? static_cast<float>(src.ior) : 1.0f;
  material.opacity = clamp01(static_cast<float>(src.dissolve));
  material.roughness =
      computeRoughness(material.shininess, static_cast<float>(src.roughness));

  if (luminance(material.transmissionColor) <= 0.0f && material.opacity < 1.0f) {
    const float trans = 1.0f - material.opacity;
    material.transmissionColor = simd::make_float3(trans, trans, trans);
  }

  material.diffuseTextureIndex = ResolveTextureIndex(src.diffuse_texname, baseDir,
                                                     scene, textureCache,
                                                     TextureUsage::Color);
  if (material.diffuseTextureIndex >= 0) {
    constexpr float kMinDiffuseLuminance = 1.0e-4f;
    if (luminance(material.diffuseColor) <= kMinDiffuseLuminance)
      material.diffuseColor = simd::make_float3(1.0f, 1.0f, 1.0f);
  }

  material.specularTextureIndex = ResolveTextureIndex(src.specular_texname, baseDir,
                                                      scene, textureCache,
                                                      TextureUsage::Color);

  const std::string normalTex =
      src.normal_texname.empty() ? src.bump_texname : src.normal_texname;
  material.normalTextureIndex = ResolveTextureIndex(normalTex, baseDir, scene,
                                                    textureCache,
                                                    TextureUsage::NormalMap);

  return material;
}

void ApplyTextureAttributes(const tinyxml2::XMLElement *element,
                            const std::filesystem::path &baseDir, Scene *scene,
                            Material &material,
                            MaterialTextureCache &textureCache) {
  if (!element || !scene)
    return;

  const std::string baseDirStr = baseDir.string();
  auto resolveTexture = [&](const char *attrValue, TextureUsage usage) {
    if (!attrValue || !attrValue[0])
      return -1;
    return ResolveTextureIndex(attrValue, baseDirStr, scene, textureCache, usage);
  };

  const char *diffuseAttr = element->Attribute("diffuseTexture");
  if (!diffuseAttr)
    diffuseAttr = element->Attribute("albedoTexture");
  const int diffuseIndex = resolveTexture(diffuseAttr, TextureUsage::Color);
  if (diffuseIndex >= 0) {
    material.diffuseTextureIndex = diffuseIndex;
    constexpr float kMinDiffuseLuminance = 1.0e-4f;
    if (luminance(material.diffuseColor) <= kMinDiffuseLuminance)
      material.diffuseColor = simd::make_float3(1.0f, 1.0f, 1.0f);
  }

  const char *specularAttr = element->Attribute("specularTexture");
  const int specularIndex = resolveTexture(specularAttr, TextureUsage::Color);
  if (specularIndex >= 0)
    material.specularTextureIndex = specularIndex;

  const char *normalAttr = element->Attribute("normalTexture");
  if (!normalAttr)
    normalAttr = element->Attribute("normalMap");
  if (!normalAttr)
    normalAttr = element->Attribute("bumpTexture");
  const int normalIndex = resolveTexture(normalAttr, TextureUsage::NormalMap);
  if (normalIndex >= 0)
    material.normalTextureIndex = normalIndex;
}

Material ParseMaterialAttributes(const tinyxml2::XMLElement *element) {
  Material material{};

  bool hasDiffuseAttr = false;
  if (const char *diffuseAttr = element->Attribute("diffuse")) {
    material.diffuseColor = parseVec3(diffuseAttr);
    hasDiffuseAttr = true;
  }
  if (const char *albedoAttr = element->Attribute("albedo")) {
    material.diffuseColor = parseVec3(albedoAttr);
    hasDiffuseAttr = true;
  }

  if (const char *emissionAttr = element->Attribute("emission"))
    material.emissionColor = parseVec3(emissionAttr);
  material.emissionPower =
      element->FloatAttribute("emissionPower", material.emissionPower);

  bool hasSpecularAttr = false;
  if (const char *specAttr = element->Attribute("specular")) {
    material.specularColor = parseVec3(specAttr);
    hasSpecularAttr = true;
  }

  bool hasTransmissionAttr = false;
  if (const char *transAttr = element->Attribute("transmission")) {
    material.transmissionColor = parseVec3(transAttr);
    hasTransmissionAttr = true;
  }

  const bool hasOpacityAttr = element->FindAttribute("opacity") != nullptr;
  material.opacity = element->FloatAttribute("opacity", material.opacity);
  material.shininess = element->FloatAttribute("shininess", material.shininess);
  material.roughness = element->FloatAttribute("roughness", material.roughness);
  material.indexOfRefraction =
      element->FloatAttribute("ior", material.indexOfRefraction);

  if (!hasTransmissionAttr && material.opacity < 1.0f) {
    const float trans = 1.0f - material.opacity;
    material.transmissionColor = simd::make_float3(trans, trans, trans);
  }

  if (const tinyxml2::XMLAttribute *typeAttr =
          element->FindAttribute("materialType")) {
    const float materialType = typeAttr->FloatValue();
    if (materialType < 0.0f) {
      if (!hasSpecularAttr)
        material.specularColor = simd::make_float3(1.0f, 1.0f, 1.0f);
      material.opacity = 1.0f;
      material.transmissionColor = simd::make_float3(0.0f, 0.0f, 0.0f);
      material.shininess = std::max(material.shininess, 256.0f);
      material.roughness = std::min(material.roughness, 0.02f);
    } else if (materialType > 0.0f) {
      if (!hasSpecularAttr)
        material.specularColor = simd::make_float3(1.0f, 1.0f, 1.0f);
      if (!hasTransmissionAttr)
        material.transmissionColor = simd::make_float3(1.0f, 1.0f, 1.0f);
      material.indexOfRefraction = materialType;
      if (!hasOpacityAttr)
        material.opacity = 0.0f;
      material.shininess = std::max(material.shininess, 96.0f);
      material.roughness = std::min(material.roughness, 0.1f);
    }
  }

  material.opacity = clamp01(material.opacity);
  material.roughness = clamp01(material.roughness);
  material.shininess = std::max(material.shininess, 1.0f);

  if (!hasDiffuseAttr && material.opacity < 1.0f &&
      luminance(material.diffuseColor) <= 0.0f) {
    const float base = 1.0f - material.opacity;
    material.diffuseColor = simd::make_float3(base, base, base);
  }

  return material;
}

} // namespace MaterialUtils
} // namespace NomadPathTracer
