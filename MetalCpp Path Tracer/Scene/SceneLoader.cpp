#include "SceneLoader.h"
#include "Material.h"
#include <tinyxml2.h>
#include <simd/simd.h>
#include <simd/quaternion.h>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <limits>
#include <cmath>
#include "tiny_obj_loader.h"

using namespace tinyxml2;

namespace MetalCppPathTracer {

constexpr float kDegToRad = 0.0174532925199432957692f;

static simd::float3 parseVec3(const char* str) {
    float x = 0, y = 0, z = 0;
    sscanf(str, "%f,%f,%f", &x, &y, &z);
    return simd::make_float3(x, y, z);
}

// Rotates a vector by Euler angles applied in X->Y->Z order.
static simd::float3 rotateVectorEulerXYZ(const simd::float3& v,
                                         const simd::float3& radians) {
    simd::float3 result = v;

    if (radians.x != 0.0f) {
        simd::quatf qx(radians.x, simd::make_float3(1.0f, 0.0f, 0.0f));
        result = simd_act(qx, result);
    }
    if (radians.y != 0.0f) {
        simd::quatf qy(radians.y, simd::make_float3(0.0f, 1.0f, 0.0f));
        result = simd_act(qy, result);
    }
    if (radians.z != 0.0f) {
        simd::quatf qz(radians.z, simd::make_float3(0.0f, 0.0f, 1.0f));
        result = simd_act(qz, result);
    }

    return result;
}

namespace {

static float clamp01(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

static simd::float3 toFloat3(const tinyobj::real_t values[3]) {
    return simd::make_float3(static_cast<float>(values[0]),
                             static_cast<float>(values[1]),
                             static_cast<float>(values[2]));
}

static float luminance(const simd::float3& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

static float computeRoughness(float shininess, float explicitRoughness) {
    if (explicitRoughness > 0.0f) {
        return clamp01(explicitRoughness);
    }
    if (shininess <= 0.0f) {
        return 1.0f;
    }
    float converted = std::sqrt(2.0f / (shininess + 2.0f));
    return clamp01(converted);
}

static Material ConvertTinyMaterial(const tinyobj::material_t& src) {
    Material material{};
    material.diffuseColor = toFloat3(src.diffuse);
    material.specularColor = toFloat3(src.specular);
    material.transmissionColor = toFloat3(src.transmittance);
    material.emissionColor = toFloat3(src.emission);

    float emissionLum = luminance(material.emissionColor);
    material.emissionPower = (emissionLum > 0.0f) ? 1.0f : 0.0f;

    material.shininess = static_cast<float>(src.shininess);
    material.indexOfRefraction = (src.ior != 0.0f) ? static_cast<float>(src.ior)
                                                   : 1.0f;
    material.opacity = clamp01(static_cast<float>(src.dissolve));
    material.roughness = computeRoughness(material.shininess,
                                          static_cast<float>(src.roughness));

    if (luminance(material.transmissionColor) <= 0.0f && material.opacity < 1.0f) {
        float trans = 1.0f - material.opacity;
        material.transmissionColor = simd::make_float3(trans, trans, trans);
    }

    return material;
}

static Material ParseMaterialAttributes(const XMLElement* element) {
    Material material{};

    bool hasDiffuseAttr = false;
    if (const char* diffuseAttr = element->Attribute("diffuse")) {
        material.diffuseColor = parseVec3(diffuseAttr);
        hasDiffuseAttr = true;
    }
    if (const char* albedoAttr = element->Attribute("albedo")) {
        material.diffuseColor = parseVec3(albedoAttr);
        hasDiffuseAttr = true;
    }

    if (const char* emissionAttr = element->Attribute("emission")) {
        material.emissionColor = parseVec3(emissionAttr);
    }
    material.emissionPower = element->FloatAttribute("emissionPower",
                                                     material.emissionPower);

    bool hasSpecularAttr = false;
    if (const char* specAttr = element->Attribute("specular")) {
        material.specularColor = parseVec3(specAttr);
        hasSpecularAttr = true;
    }

    bool hasTransmissionAttr = false;
    if (const char* transAttr = element->Attribute("transmission")) {
        material.transmissionColor = parseVec3(transAttr);
        hasTransmissionAttr = true;
    }

    bool hasOpacityAttr = element->FindAttribute("opacity") != nullptr;
    material.opacity = element->FloatAttribute("opacity", material.opacity);
    material.shininess = element->FloatAttribute("shininess", material.shininess);
    material.roughness = element->FloatAttribute("roughness", material.roughness);
    material.indexOfRefraction = element->FloatAttribute("ior",
                                                         material.indexOfRefraction);

    if (!hasTransmissionAttr && material.opacity < 1.0f) {
        float trans = 1.0f - material.opacity;
        material.transmissionColor = simd::make_float3(trans, trans, trans);
    }

    if (const XMLAttribute* typeAttr = element->FindAttribute("materialType")) {
        float materialType = typeAttr->FloatValue();
        if (materialType < 0.0f) {
            if (!hasSpecularAttr) {
                material.specularColor = simd::make_float3(1.0f, 1.0f, 1.0f);
            }
            material.opacity = 1.0f;
            material.transmissionColor = simd::make_float3(0.0f, 0.0f, 0.0f);
            material.shininess = std::max(material.shininess, 256.0f);
            material.roughness = std::min(material.roughness, 0.02f);
        } else if (materialType > 0.0f) {
            if (!hasSpecularAttr) {
                material.specularColor = simd::make_float3(1.0f, 1.0f, 1.0f);
            }
            if (!hasTransmissionAttr) {
                material.transmissionColor = simd::make_float3(1.0f, 1.0f, 1.0f);
            }
            material.indexOfRefraction = materialType;
            if (!hasOpacityAttr) {
                material.opacity = 0.0f;
            }
            material.shininess = std::max(material.shininess, 96.0f);
            material.roughness = std::min(material.roughness, 0.1f);
        }
    }

    material.opacity = clamp01(material.opacity);
    material.roughness = clamp01(material.roughness);
    material.shininess = std::max(material.shininess, 1.0f);

    if (!hasDiffuseAttr && material.opacity < 1.0f && luminance(material.diffuseColor) <= 0.0f) {
        float base = 1.0f - material.opacity;
        material.diffuseColor = simd::make_float3(base, base, base);
    }

    return material;
}

static float AxisValue(const simd::float3& v, int axis) {
    switch (axis) {
        case 0: return v.x;
        case 1: return v.y;
        default: return v.z;
    }
}

static void ComputeBounds(const std::vector<Primitive>& prims, size_t start,
                          size_t end, simd::float3& outMin,
                          simd::float3& outMax) {
    const float maxFloat = std::numeric_limits<float>::max();
    const float minFloat = -std::numeric_limits<float>::max();
    simd::float3 bmin = simd::make_float3(maxFloat, maxFloat, maxFloat);
    simd::float3 bmax = simd::make_float3(minFloat, minFloat, minFloat);

    for (size_t i = start; i < end; ++i) {
        const Primitive& p = prims[i];
        const auto& tri = p.triangle;

        bmin.x = std::min(bmin.x, tri.v0.x);
        bmin.y = std::min(bmin.y, tri.v0.y);
        bmin.z = std::min(bmin.z, tri.v0.z);
        bmax.x = std::max(bmax.x, tri.v0.x);
        bmax.y = std::max(bmax.y, tri.v0.y);
        bmax.z = std::max(bmax.z, tri.v0.z);

        bmin.x = std::min(bmin.x, tri.v1.x);
        bmin.y = std::min(bmin.y, tri.v1.y);
        bmin.z = std::min(bmin.z, tri.v1.z);
        bmax.x = std::max(bmax.x, tri.v1.x);
        bmax.y = std::max(bmax.y, tri.v1.y);
        bmax.z = std::max(bmax.z, tri.v1.z);

        bmin.x = std::min(bmin.x, tri.v2.x);
        bmin.y = std::min(bmin.y, tri.v2.y);
        bmin.z = std::min(bmin.z, tri.v2.z);
        bmax.x = std::max(bmax.x, tri.v2.x);
        bmax.y = std::max(bmax.y, tri.v2.y);
        bmax.z = std::max(bmax.z, tri.v2.z);
    }

    outMin = bmin;
    outMax = bmax;
}

static float PrimitiveCentroidAxis(const Primitive& prim, int axis) {
    const auto& tri = prim.triangle;
    float c0 = AxisValue(tri.v0, axis);
    float c1 = AxisValue(tri.v1, axis);
    float c2 = AxisValue(tri.v2, axis);
    return (c0 + c1 + c2) / 3.0f;
}

static void PartitionTrianglesRecursive(std::vector<Primitive>& prims,
                                        size_t start, size_t end,
                                        size_t maxTriangles, float maxExtent,
                                        std::vector<std::pair<size_t, size_t>>&
                                            partitions) {
    size_t count = end - start;
    if (count == 0) {
        return;
    }

    simd::float3 bmin, bmax;
    ComputeBounds(prims, start, end, bmin, bmax);
    simd::float3 extents = bmax - bmin;

    float longest = extents.x;
    int axis = 0;
    if (extents.y > longest) {
        longest = extents.y;
        axis = 1;
    }
    if (extents.z > longest) {
        longest = extents.z;
        axis = 2;
    }

    bool withinTriangleBudget = (maxTriangles == 0 || count <= maxTriangles);
    bool withinExtentBudget = (maxExtent <= 0.0f || longest <= maxExtent);

    if ((withinTriangleBudget && withinExtentBudget) || count <= 1) {
        partitions.emplace_back(start, end);
        return;
    }

    size_t mid = start + count / 2;
    if (mid == start || mid == end) {
        partitions.emplace_back(start, end);
        return;
    }

    std::nth_element(
        prims.begin() + start, prims.begin() + mid, prims.begin() + end,
        [axis](const Primitive& a, const Primitive& b) {
            return PrimitiveCentroidAxis(a, axis) <
                   PrimitiveCentroidAxis(b, axis);
        });

    PartitionTrianglesRecursive(prims, start, mid, maxTriangles, maxExtent,
                                partitions);
    PartitionTrianglesRecursive(prims, mid, end, maxTriangles, maxExtent,
                                partitions);
}

static void EmitMeshPartitions(Scene* scene, std::vector<Primitive>& prims,
                               size_t maxTriangles, float maxExtent,
                               int meshGroupId) {
    if (prims.empty()) {
        return;
    }

    if ((maxTriangles == 0 && maxExtent <= 0.0f) || prims.size() <= 1) {
        scene->addObject(prims, meshGroupId);
        return;
    }

    std::vector<std::pair<size_t, size_t>> partitions;
    PartitionTrianglesRecursive(prims, 0, prims.size(), maxTriangles,
                                maxExtent, partitions);

    for (const auto& range : partitions) {
        std::vector<Primitive> chunk(prims.begin() + range.first,
                                     prims.begin() + range.second);
        scene->addObject(chunk, meshGroupId);
    }
}

// Stores previously loaded mesh vertex/index buffers keyed by resolved file
// path so repeated references to the same mesh can reuse geometry data within a
// single scene load.
struct CachedMesh {
    std::vector<simd::float3> vertices;
    std::vector<simd::uint3> indices;
    std::vector<int> faceMaterialIndices;
    std::vector<Material> materials;
};

using MeshCache = std::unordered_map<std::string, CachedMesh>;

MeshCache& GetMeshCache() {
    static MeshCache cache;
    return cache;
}

static void LoadOBJ(const std::string& path, CachedMesh& mesh) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> objMaterials;
    std::string warn, err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &objMaterials, &warn, &err,
                                path.c_str());

    if (!warn.empty()) {
        printf("tinyobjloader warning: %s\n", warn.c_str());
    }
    if (!err.empty()) {
        printf("tinyobjloader error: %s\n", err.c_str());
    }
    if (!ret) {
        printf("Failed to load OBJ: %s\n", path.c_str());
        return;
    }

    mesh.vertices.clear();
    mesh.indices.clear();
    mesh.faceMaterialIndices.clear();
    mesh.materials.clear();

    mesh.vertices.reserve(attrib.vertices.size() / 3);
    for (size_t v = 0; v < attrib.vertices.size(); v += 3) {
        mesh.vertices.push_back(simd::make_float3(
            attrib.vertices[v],
            attrib.vertices[v + 1],
            attrib.vertices[v + 2]
        ));
    }

    mesh.materials.reserve(objMaterials.size());
    for (const auto& srcMat : objMaterials) {
        mesh.materials.push_back(ConvertTinyMaterial(srcMat));
    }

    for (const auto& shape : shapes) {
        size_t indexOffset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shape.mesh.num_face_vertices[f]);
            if (fv != 3) {
                indexOffset += fv;
                continue;
            }

            uint32_t idx0 = shape.mesh.indices[indexOffset + 0].vertex_index;
            uint32_t idx1 = shape.mesh.indices[indexOffset + 1].vertex_index;
            uint32_t idx2 = shape.mesh.indices[indexOffset + 2].vertex_index;

            if (idx0 >= mesh.vertices.size() || idx1 >= mesh.vertices.size() ||
                idx2 >= mesh.vertices.size()) {
                printf("Invalid triangle indices\n");
                indexOffset += fv;
                continue;
            }

            mesh.indices.push_back(simd::make_uint3(idx0, idx1, idx2));

            int materialId = -1;
            if (f < shape.mesh.material_ids.size()) {
                materialId = shape.mesh.material_ids[f];
            }
            if (materialId >= 0 &&
                static_cast<size_t>(materialId) < mesh.materials.size()) {
                mesh.faceMaterialIndices.push_back(materialId);
            } else {
                mesh.faceMaterialIndices.push_back(-1);
            }
            indexOffset += fv;
        }
    }

    printf("Loaded OBJ: %zu vertices, %zu triangles, %zu materials\n",
           mesh.vertices.size(), mesh.indices.size(), mesh.materials.size());
}

} // namespace

void SceneLoader::ClearCache() {
    GetMeshCache().clear();
}

bool SceneLoader::LoadSceneFromXML(const std::string& path, Scene* scene) {
    // Ensure stale mesh data does not leak between scene loads.
    ClearCache();

    XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != XML_SUCCESS) {
        printf("Failed to load scene XML: %s\n", path.c_str());
        return false;
    }

    scene->clear();

    auto* root = doc.FirstChildElement("Scene");
    if (!root) {
        printf("No <Scene> root.\n");
        return false;
    }

    // Resolve relative paths against the directory containing the scene file
    std::filesystem::path baseDir = std::filesystem::path(path).parent_path();

    scene->screenSize.x = root->FloatAttribute("width", scene->screenSize.x);
    scene->screenSize.y = root->FloatAttribute("height", scene->screenSize.y);
    scene->maxRayDepth = root->UnsignedAttribute("maxRayDepth", scene->maxRayDepth);

    if (const char* residencyAttr = root->Attribute("residencyStrategy")) {
        std::string value = residencyAttr;
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });

        if (value == "energy" || value == "importance" || value == "energyimportance" ||
            value == "energy_importance") {
            scene->setResidencyStrategy(ResidencyStrategy::EnergyImportance);
        } else if (value == "distance" || value == "lod" || value == "distance_lod" ||
                   value == "distancelod" || value == "distancebased") {
            scene->setResidencyStrategy(ResidencyStrategy::DistanceLOD);
        } else if (value == "rayhitbudget" || value == "ray_hit_budget" ||
                   value == "rayhits" || value == "ray_hit" ||
                   value == "hitbudget") {
            scene->setResidencyStrategy(ResidencyStrategy::RayHitBudget);
        } else if (value == "screenspacefootprint" ||
                   value == "screen_space_footprint" || value == "footprint" ||
                   value == "screenspace") {
            scene->setResidencyStrategy(ResidencyStrategy::ScreenSpaceFootprint);
        } else if (value == "alwaysresident" || value == "always_resident" ||
                   value == "always" || value == "none" || value == "off") {
            scene->setResidencyStrategy(ResidencyStrategy::AlwaysResident);
        } else {
            printf("Unknown residency strategy '%s', defaulting to distance LOD.\n",
                   residencyAttr);
            scene->setResidencyStrategy(ResidencyStrategy::DistanceLOD);
        }
    }

    ResidencyParameters params = scene->getResidencyParameters();
    params.lodEnterDistance = root->FloatAttribute("lodEnterDistance", params.lodEnterDistance);
    params.lodExitDistance = root->FloatAttribute("lodExitDistance", params.lodExitDistance);
    params.stateCooldownFrames =
        root->UnsignedAttribute("residencyCooldown", params.stateCooldownFrames);
    params.lodMaxTogglesPerFrame =
        root->UnsignedAttribute("lodToggleBudget", params.lodMaxTogglesPerFrame);

    params.energyTargetFraction =
        root->FloatAttribute("energyTargetFraction", params.energyTargetFraction);
    params.energyMinActivePrimitives = static_cast<size_t>(root->Unsigned64Attribute(
        "energyMinActive", static_cast<uint64_t>(params.energyMinActivePrimitives)));
    params.energyMaxTogglesPerFrame = static_cast<size_t>(root->Unsigned64Attribute(
        "energyToggleBudget", static_cast<uint64_t>(params.energyMaxTogglesPerFrame)));
    params.energyVisibilityBoost =
        root->FloatAttribute("energyVisibilityBoost", params.energyVisibilityBoost);

    params.rayHitDecay = root->FloatAttribute("rayHitDecay", params.rayHitDecay);
    params.rayHitTargetFraction =
        root->FloatAttribute("rayHitTargetFraction", params.rayHitTargetFraction);
    params.rayHitMinActivePrimitives = static_cast<size_t>(root->Unsigned64Attribute(
        "rayHitMinActive", static_cast<uint64_t>(params.rayHitMinActivePrimitives)));
    params.rayHitMaxTogglesPerFrame = static_cast<size_t>(root->Unsigned64Attribute(
        "rayHitToggleBudget", static_cast<uint64_t>(params.rayHitMaxTogglesPerFrame)));
    params.rayHitRebuildCooldownFrames = root->UnsignedAttribute(
        "rayHitCooldown", params.rayHitRebuildCooldownFrames);

    params.screenFootprintTargetFraction = root->FloatAttribute(
        "screenTargetFraction", params.screenFootprintTargetFraction);
    params.screenFootprintMinPixelCoverage = root->FloatAttribute(
        "screenMinPixelCoverage", params.screenFootprintMinPixelCoverage);
    params.screenFootprintMinActivePrimitives = static_cast<size_t>(root->Unsigned64Attribute(
        "screenMinActive",
        static_cast<uint64_t>(params.screenFootprintMinActivePrimitives)));
    params.screenFootprintMaxTogglesPerFrame = static_cast<size_t>(root->Unsigned64Attribute(
        "screenToggleBudget",
        static_cast<uint64_t>(params.screenFootprintMaxTogglesPerFrame)));

    scene->setResidencyParameters(params);

    bool startCompacted = root->BoolAttribute("startCompacted", scene->getStartCompacted());
    scene->setStartCompacted(startCompacted);

    double textureCap = root->DoubleAttribute(
        "textureResidencyMemoryCapMB", scene->getTextureResidencyMemoryCapMB());
    scene->setTextureResidencyMemoryCapMB(textureCap);

    int nextMeshGroupId = 0;

    for (auto* e = root->FirstChildElement(); e; e = e->NextSiblingElement()) {
        std::string tag = e->Name();
        if (tag == "Sphere") {
            Primitive p{};
            p.type = PrimitiveType::Sphere;
            p.sphere.center = parseVec3(e->Attribute("position"));
            p.sphere.radius = e->FloatAttribute("radius", 1.0f);

            p.material = ParseMaterialAttributes(e);

            scene->addPrimitive(p);
        }
        else if (tag == "Rectangle") {
            Primitive p{};
            p.type = PrimitiveType::Rectangle;
            p.rectangle.center = parseVec3(e->Attribute("position"));
            p.rectangle.u = parseVec3(e->Attribute("u"));
            p.rectangle.v = parseVec3(e->Attribute("v"));

            if (const char* rotationAttr = e->Attribute("rotation")) {
                simd::float3 rotationRadians = parseVec3(rotationAttr) * kDegToRad;
                p.rectangle.u = rotateVectorEulerXYZ(p.rectangle.u, rotationRadians);
                p.rectangle.v = rotateVectorEulerXYZ(p.rectangle.v, rotationRadians);
            }

            p.material = ParseMaterialAttributes(e);

            scene->addPrimitive(p);
        }
        else if (tag == "Mesh") {
            // Build full path to mesh file relative to the scene's directory
            const char* fileAttr = e->Attribute("file");
            std::filesystem::path meshPath = baseDir / (fileAttr ? fileAttr : "");
            std::string normalizedPath = meshPath.lexically_normal().string();

            auto& cache = GetMeshCache();
            auto it = cache.find(normalizedPath);
            if (it == cache.end()) {
                // First time encountering this mesh path in the current load.
                CachedMesh entry;
                LoadOBJ(normalizedPath, entry);
                it = cache.emplace(normalizedPath, std::move(entry)).first;
            }

            const CachedMesh& meshData = it->second;

            simd::float3 pos = parseVec3(e->Attribute("position"));
            float scale = e->FloatAttribute("scale", 1.0f);

            simd::float3 basisX = simd::make_float3(scale, 0.0f, 0.0f);
            simd::float3 basisY = simd::make_float3(0.0f, scale, 0.0f);
            simd::float3 basisZ = simd::make_float3(0.0f, 0.0f, scale);

            simd::float3 rotationDegrees = simd::make_float3(0.0f, 0.0f, 0.0f);
            if (const char* rotationAttr = e->Attribute("rotation")) {
                rotationDegrees = parseVec3(rotationAttr);
            }

            if (const char* bxAttr = e->Attribute("basisX")) {
                basisX = parseVec3(bxAttr);
            }
            if (const char* byAttr = e->Attribute("basisY")) {
                basisY = parseVec3(byAttr);
            }
            if (const char* bzAttr = e->Attribute("basisZ")) {
                basisZ = parseVec3(bzAttr);
            }

            if (rotationDegrees.x != 0.0f || rotationDegrees.y != 0.0f ||
                rotationDegrees.z != 0.0f) {
                simd::float3 rotationRadians = rotationDegrees * kDegToRad;
                basisX = rotateVectorEulerXYZ(basisX, rotationRadians);
                basisY = rotateVectorEulerXYZ(basisY, rotationRadians);
                basisZ = rotateVectorEulerXYZ(basisZ, rotationRadians);
            }

            Material overrideMaterial = ParseMaterialAttributes(e);

            size_t clusterMaxTriangles = static_cast<size_t>(
                e->Unsigned64Attribute("clusterMaxTriangles", 0));
            float clusterMaxExtent =
                e->FloatAttribute("clusterMaxExtent", 0.0f);

            std::vector<Primitive> meshPrimitives;
            meshPrimitives.reserve(meshData.indices.size());
            auto transformVertex = [&](const simd::float3& v) {
                return pos + basisX * v.x + basisY * v.y + basisZ * v.z;
            };
            for (size_t triIndex = 0; triIndex < meshData.indices.size(); ++triIndex) {
                const auto& tri = meshData.indices[triIndex];
                Primitive p{};
                p.type = PrimitiveType::Triangle;
                p.triangle.v0 = transformVertex(meshData.vertices[tri.x]);
                p.triangle.v1 = transformVertex(meshData.vertices[tri.y]);
                p.triangle.v2 = transformVertex(meshData.vertices[tri.z]);

                const Material* chosenMaterial = &overrideMaterial;
                if (triIndex < meshData.faceMaterialIndices.size()) {
                    int materialId = meshData.faceMaterialIndices[triIndex];
                    if (materialId >= 0 &&
                        static_cast<size_t>(materialId) < meshData.materials.size()) {
                        chosenMaterial = &meshData.materials[materialId];
                    }
                }

                p.material = *chosenMaterial;
                meshPrimitives.push_back(p);
            }
            int meshGroupId = nextMeshGroupId++;
            EmitMeshPartitions(scene, meshPrimitives, clusterMaxTriangles,
                               clusterMaxExtent, meshGroupId);
        }
        else if (tag == "CameraPath") {
            for (auto* kf = e->FirstChildElement("Keyframe"); kf; kf = kf->NextSiblingElement("Keyframe")) {
                CameraKeyframe key{};
                key.frame = kf->UnsignedAttribute("frame", 0);
                key.position = parseVec3(kf->Attribute("position"));
                key.lookAt = parseVec3(kf->Attribute("lookAt"));
                scene->cameraPath.push_back(key);
            }
        }
        else if (tag == "ObserverCamera") {
            ObserverCamera observer{};
            if (const char* posAttr = e->Attribute("position")) {
                observer.position = parseVec3(posAttr);
            }
            if (const char* lookAttr = e->Attribute("lookAt")) {
                observer.lookAt = parseVec3(lookAttr);
            }
            observer.verticalFov = e->FloatAttribute("verticalFov", observer.verticalFov);
            scene->setObserverCamera(observer);
        }
    }

    return true;
}

}
