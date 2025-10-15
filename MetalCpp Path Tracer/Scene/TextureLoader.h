#ifndef TEXTURE_LOADER_H
#define TEXTURE_LOADER_H

#include <string>
#include <vector>

namespace MetalCppPathTracer {

struct LoadedTextureImage {
    int width = 0;
    int height = 0;
    std::vector<float> pixels; // RGBA, linear [0,1]
};

bool LoadTextureImage(const std::string &path, LoadedTextureImage &outImage);

} // namespace MetalCppPathTracer

#endif
