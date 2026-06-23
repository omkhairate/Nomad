#ifndef MATERIAL_UTILS_H
#define MATERIAL_UTILS_H

#include "Material.h"
#include "TextureLoader.h"
#include "../tiny_obj_loader.h"
#include <filesystem>
#include <string>
#include <unordered_map>

namespace tinyxml2 {
class XMLElement;
}

namespace NomadPathTracer {

class Scene;

using MaterialTextureCache = std::unordered_map<std::string, int>;

namespace MaterialUtils {

MaterialTextureCache &GetTextureCache();
void ClearTextureCache();

bool HasOverrideAttributes(const tinyxml2::XMLElement *element);
bool NeedsFallbackMaterial(const Material &material);

Material ConvertTinyMaterial(const tinyobj::material_t &src, Scene *scene,
                             MaterialTextureCache &textureCache,
                             const std::string &baseDir);
void ApplyTextureAttributes(const tinyxml2::XMLElement *element,
                            const std::filesystem::path &baseDir, Scene *scene,
                            Material &material,
                            MaterialTextureCache &textureCache);
Material ParseMaterialAttributes(const tinyxml2::XMLElement *element);

} // namespace MaterialUtils
} // namespace NomadPathTracer

#endif
