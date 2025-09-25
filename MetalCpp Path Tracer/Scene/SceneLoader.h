#ifndef SCENELOADER_H
#define SCENELOADER_H

#include "Scene.h"
#include <string>

namespace MetalCppPathTracer {

class SceneLoader {
public:
    // Returns true on success, false if the XML could not be loaded or parsed
    static bool LoadSceneFromXML(const std::string& path, Scene* scene);
};

}

#endif
