#include "SceneLoader.h"
#include "Material.h"
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
#include "tiny_obj_loader.h"

using namespace tinyxml2;

namespace MetalCppPathTracer {

static simd::float3 parseVec3(const char* str) {
    float x=0,y=0,z=0;
    sscanf(str, "%f,%f,%f", &x,&y,&z);
    return simd::make_float3(x,y,z);
}

namespace {

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
                               size_t maxTriangles, float maxExtent) {
    if (prims.empty()) {
        return;
    }

    if ((maxTriangles == 0 && maxExtent <= 0.0f) || prims.size() <= 1) {
        scene->addObject(prims);
        return;
    }

    std::vector<std::pair<size_t, size_t>> partitions;
    PartitionTrianglesRecursive(prims, 0, prims.size(), maxTriangles,
                                maxExtent, partitions);

    for (const auto& range : partitions) {
        std::vector<Primitive> chunk(prims.begin() + range.first,
                                     prims.begin() + range.second);
        scene->addObject(chunk);
    }
}

// Stores previously loaded mesh vertex/index buffers keyed by resolved file
// path so repeated references to the same mesh can reuse geometry data within a
// single scene load.
struct CachedMesh {
    std::vector<simd::float3> vertices;
    std::vector<simd::uint3> indices;
};

using MeshCache = std::unordered_map<std::string, CachedMesh>;

MeshCache& GetMeshCache() {
    static MeshCache cache;
    return cache;
}

static void LoadOBJ(const std::string& path, std::vector<simd::float3>& verts, std::vector<simd::uint3>& tris) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str());

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

    verts.reserve(attrib.vertices.size() / 3);
    for (size_t v = 0; v < attrib.vertices.size(); v += 3) {
        verts.push_back(simd::make_float3(
            attrib.vertices[v],
            attrib.vertices[v + 1],
            attrib.vertices[v + 2]
        ));
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

            if (idx0 >= verts.size() || idx1 >= verts.size() || idx2 >= verts.size()) {
                printf("Invalid triangle indices\n");
                indexOffset += fv;
                continue;
            }

            tris.push_back(simd::make_uint3(idx0, idx1, idx2));
            indexOffset += fv;
        }
    }

    printf("Loaded OBJ: %zu vertices, %zu triangles\n", verts.size(), tris.size());
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

    for (auto* e = root->FirstChildElement(); e; e = e->NextSiblingElement()) {
        std::string tag = e->Name();
        if (tag == "Sphere") {
            Primitive p{};
            p.type = PrimitiveType::Sphere;
            p.sphere.center = parseVec3(e->Attribute("position"));
            p.sphere.radius = e->FloatAttribute("radius", 1.0f);

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
            std::string normalizedPath = meshPath.lexically_normal().string();

            auto& cache = GetMeshCache();
            auto it = cache.find(normalizedPath);
            if (it == cache.end()) {
                // First time encountering this mesh path in the current load.
                CachedMesh entry;
                LoadOBJ(normalizedPath, entry.vertices, entry.indices);
                it = cache.emplace(normalizedPath, std::move(entry)).first;
            }

            const CachedMesh& meshData = it->second;

            simd::float3 pos = parseVec3(e->Attribute("position"));
            float scale = e->FloatAttribute("scale", 1.0f);

            Material m;
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
            for (const auto& tri : meshData.indices) {
                Primitive p{};
                p.type = PrimitiveType::Triangle;
                p.triangle.v0 = pos + scale * meshData.vertices[tri.x];
                p.triangle.v1 = pos + scale * meshData.vertices[tri.y];
                p.triangle.v2 = pos + scale * meshData.vertices[tri.z];
                p.material = m;
                meshPrimitives.push_back(p);
            }
            EmitMeshPartitions(scene, meshPrimitives, clusterMaxTriangles,
                               clusterMaxExtent);
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
    }

    return true;
}

}
