#ifndef SCENELOADER_H
#define SCENELOADER_H

#include "Scene.h"
#include <string>

namespace MetalCppPathTracer {

// SceneLoader caches mesh vertex/index data during scene loading to avoid
// re-reading the same meshes from disk. Call ClearCache before beginning a new
// load (done automatically by Scene::clear and LoadSceneFromXML) to release
// cached data.
class SceneLoader {
public:
    // Returns true on success, false if the XML could not be loaded or parsed
    static bool LoadSceneFromXML(const std::string& path, Scene* scene);

    // Clears any cached mesh data so future loads re-read meshes from disk.
    static void ClearCache();
};

}

#endif
