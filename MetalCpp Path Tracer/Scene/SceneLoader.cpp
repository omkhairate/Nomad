#include "SceneLoader.h"
#include "Material.h"
#include "ImageLoader.h"
#include <tinyxml2.h>
#include <simd/simd.h>
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
#include <functional>
#include "tiny_obj_loader.h"

using namespace tinyxml2;

namespace MetalCppPathTracer {

static simd::float3 parseVec3(const char* str) {
    float x=0,y=0,z=0;
    sscanf(str, "%f,%f,%f", &x,&y,&z);
    return simd::make_float3(x,y,z);
}

namespace {

static Material DefaultMaterial() {
    Material m;
    m.albedo = simd::make_float3(0.8f, 0.8f, 0.8f);
    m.materialType = 0.0f;
    m.emissionColor = simd::make_float3(0.0f, 0.0f, 0.0f);
    m.emissionPower = 0.0f;
    m.diffuseTextureIndex = -1;
    return m;
}

static Material ConvertMaterial(const tinyobj::material_t &src,
                                int textureIndex) {
    Material m = DefaultMaterial();
    m.albedo = simd::make_float3(src.diffuse[0], src.diffuse[1], src.diffuse[2]);
    m.emissionColor =
        simd::make_float3(src.emission[0], src.emission[1], src.emission[2]);
    m.diffuseTextureIndex = textureIndex;
    return m;
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
    std::vector<simd::float2> uvs;
    std::vector<simd::uint3> indices;
    std::vector<int32_t> triangleTextureIndices;
};

using MeshCache = std::unordered_map<std::string, CachedMesh>;

MeshCache& GetMeshCache() {
    static MeshCache cache;
    return cache;
}

static void LoadOBJ(const std::filesystem::path &path, CachedMesh &outMesh) {
    outMesh.vertices.clear();
    outMesh.uvs.clear();
    outMesh.indices.clear();
    outMesh.triangleTextureIndices.clear();

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    std::string pathStr = path.string();
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                                pathStr.c_str());

    if (!warn.empty()) {
        printf("tinyobjloader warning: %s\n", warn.c_str());
    }
    if (!err.empty()) {
        printf("tinyobjloader error: %s\n", err.c_str());
    }
    if (!ret) {
        printf("Failed to load OBJ: %s\n", pathStr.c_str());
        return;
    }

    std::filesystem::path baseDir = path.parent_path();
    ImageLoader &imageLoader = GetGlobalImageLoader();
    std::vector<int> materialTextureIndices(materials.size(), -1);
    for (size_t i = 0; i < materials.size(); ++i) {
        int textureIndex = -1;
        if (!materials[i].diffuse_texname.empty()) {
            textureIndex = imageLoader.loadImage(baseDir, materials[i].diffuse_texname);
        }
        Material converted = ConvertMaterial(materials[i], textureIndex);
        materialTextureIndices[i] = converted.diffuseTextureIndex;
    }

    struct VertexKey {
        int positionIndex;
        int texcoordIndex;

        bool operator==(const VertexKey &other) const {
            return positionIndex == other.positionIndex &&
                   texcoordIndex == other.texcoordIndex;
        }
    };

    struct VertexKeyHash {
        size_t operator()(const VertexKey &key) const {
            size_t h1 = std::hash<int>{}(key.positionIndex);
            size_t h2 = std::hash<int>{}(key.texcoordIndex);
            return h1 ^ (h2 << 1);
        }
    };

    std::unordered_map<VertexKey, uint32_t, VertexKeyHash> remap;
    remap.reserve(attrib.vertices.size());

    auto getPosition = [&](int index) -> simd::float3 {
        size_t base = static_cast<size_t>(index) * 3;
        if (base + 2 >= attrib.vertices.size())
            return simd::make_float3(0.0f, 0.0f, 0.0f);
        return simd::make_float3(attrib.vertices[base + 0],
                                 attrib.vertices[base + 1],
                                 attrib.vertices[base + 2]);
    };

    auto getUV = [&](int index) -> simd::float2 {
        if (index < 0)
            return simd::make_float2(0.0f, 0.0f);
        size_t base = static_cast<size_t>(index) * 2;
        if (base + 1 >= attrib.texcoords.size())
            return simd::make_float2(0.0f, 0.0f);
        return simd::make_float2(attrib.texcoords[base + 0],
                                 attrib.texcoords[base + 1]);
    };

    for (const auto &shape : shapes) {
        size_t indexOffset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            size_t fv = static_cast<size_t>(shape.mesh.num_face_vertices[f]);
            if (fv != 3) {
                indexOffset += fv;
                continue;
            }

            simd::uint3 tri = {0, 0, 0};
            bool valid = true;
            for (size_t v = 0; v < 3; ++v) {
                const tinyobj::index_t &idx = shape.mesh.indices[indexOffset + v];
                if (idx.vertex_index < 0) {
                    valid = false;
                    break;
                }

                size_t vertexBase = static_cast<size_t>(idx.vertex_index) * 3;
                if (vertexBase + 2 >= attrib.vertices.size()) {
                    valid = false;
                    break;
                }

                VertexKey key{idx.vertex_index, idx.texcoord_index};
                auto it = remap.find(key);
                if (it == remap.end()) {
                    simd::float3 position = getPosition(idx.vertex_index);
                    remap[key] = static_cast<uint32_t>(outMesh.vertices.size());
                    outMesh.vertices.push_back(position);
                    outMesh.uvs.push_back(getUV(idx.texcoord_index));
                    tri[v] = static_cast<uint32_t>(outMesh.vertices.size() - 1);
                } else {
                    tri[v] = it->second;
                }
            }

            if (!valid) {
                indexOffset += fv;
                continue;
            }

            outMesh.indices.push_back(tri);
            int materialId = -1;
            if (f < shape.mesh.material_ids.size())
                materialId = shape.mesh.material_ids[f];
            int textureIndex = -1;
            if (materialId >= 0 &&
                materialId < static_cast<int>(materialTextureIndices.size()))
                textureIndex = materialTextureIndices[static_cast<size_t>(materialId)];
            outMesh.triangleTextureIndices.push_back(textureIndex);

            indexOffset += fv;
        }
    }

    printf("Loaded OBJ: %zu vertices, %zu triangles\n", outMesh.vertices.size(),
           outMesh.indices.size());
}

} // namespace

void SceneLoader::ClearCache() {
    GetMeshCache().clear();
    GetGlobalImageLoader().clear();
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

            p.material = DefaultMaterial();
            p.material.albedo = parseVec3(e->Attribute("albedo"));
            p.material.emissionColor = parseVec3(e->Attribute("emission"));
            p.material.materialType = e->FloatAttribute("materialType", 0);
            p.material.emissionPower = e->FloatAttribute("emissionPower", 0);

            scene->addPrimitive(p);
        }
        else if (tag == "Rectangle") {
            Primitive p{};
            p.type = PrimitiveType::Rectangle;
            p.rectangle.center = parseVec3(e->Attribute("position"));
            p.rectangle.u = parseVec3(e->Attribute("u"));
            p.rectangle.v = parseVec3(e->Attribute("v"));

            p.material = DefaultMaterial();
            p.material.albedo = parseVec3(e->Attribute("albedo"));
            p.material.emissionColor = parseVec3(e->Attribute("emission"));
            p.material.materialType = e->FloatAttribute("materialType", 0);
            p.material.emissionPower = e->FloatAttribute("emissionPower", 0);

            scene->addPrimitive(p);
        }
        else if (tag == "Mesh") {
            // Build full path to mesh file relative to the scene's directory
            const char* fileAttr = e->Attribute("file");
            std::filesystem::path meshPath = baseDir / (fileAttr ? fileAttr : "");
            std::filesystem::path normalizedMeshPath = meshPath.lexically_normal();
            std::string cacheKey = normalizedMeshPath.string();

            auto& cache = GetMeshCache();
            auto it = cache.find(cacheKey);
            if (it == cache.end()) {
                // First time encountering this mesh path in the current load.
                CachedMesh entry;
                LoadOBJ(normalizedMeshPath, entry);
                it = cache.emplace(cacheKey, std::move(entry)).first;
            }

            const CachedMesh& meshData = it->second;

            simd::float3 pos = parseVec3(e->Attribute("position"));
            float scale = e->FloatAttribute("scale", 1.0f);

            simd::float3 basisX = simd::make_float3(scale, 0.0f, 0.0f);
            simd::float3 basisY = simd::make_float3(0.0f, scale, 0.0f);
            simd::float3 basisZ = simd::make_float3(0.0f, 0.0f, scale);

            if (const char* bxAttr = e->Attribute("basisX")) {
                basisX = parseVec3(bxAttr);
            }
            if (const char* byAttr = e->Attribute("basisY")) {
                basisY = parseVec3(byAttr);
            }
            if (const char* bzAttr = e->Attribute("basisZ")) {
                basisZ = parseVec3(bzAttr);
            }

            Material m = DefaultMaterial();
            m.albedo = parseVec3(e->Attribute("albedo"));
            m.emissionColor = parseVec3(e->Attribute("emission"));
            m.materialType = e->FloatAttribute("materialType", 0);
            m.emissionPower = e->FloatAttribute("emissionPower", 0);

            size_t clusterMaxTriangles = static_cast<size_t>(
                e->Unsigned64Attribute("clusterMaxTriangles", 0));
            float clusterMaxExtent =
                e->FloatAttribute("clusterMaxExtent", 0.0f);

            std::vector<Primitive> meshPrimitives;
            meshPrimitives.reserve(meshData.indices.size());
            auto transformVertex = [&](const simd::float3& v) {
                return pos + basisX * v.x + basisY * v.y + basisZ * v.z;
            };
            auto fetchUV = [&](uint32_t index) {
                if (index < meshData.uvs.size())
                    return meshData.uvs[index];
                return simd::make_float2(0.0f, 0.0f);
            };

            size_t triangleIndex = 0;
            for (const auto& tri : meshData.indices) {
                Primitive p{};
                p.type = PrimitiveType::Triangle;
                p.triangle.v0 = transformVertex(meshData.vertices[tri.x]);
                p.triangle.v1 = transformVertex(meshData.vertices[tri.y]);
                p.triangle.v2 = transformVertex(meshData.vertices[tri.z]);
                p.triangle.uv0 = fetchUV(tri.x);
                p.triangle.uv1 = fetchUV(tri.y);
                p.triangle.uv2 = fetchUV(tri.z);
                p.material = m;
                if (triangleIndex < meshData.triangleTextureIndices.size())
                    p.material.diffuseTextureIndex =
                        meshData.triangleTextureIndices[triangleIndex];
                meshPrimitives.push_back(p);
                ++triangleIndex;
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
