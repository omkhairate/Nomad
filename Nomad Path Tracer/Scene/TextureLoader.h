#ifndef TEXTURE_LOADER_H
#define TEXTURE_LOADER_H

#include <string>
#include <vector>

namespace NomadPathTracer {

struct LoadedTextureImage {
    int width = 0;
    int height = 0;
    std::vector<float> pixels; // RGBA, linear [0,1]
};

enum class TextureUsage {
    Color,
    NormalMap,
    Data,
};

enum class TextureColorSpace {
    sRGB,
    Linear,
};

TextureColorSpace DetermineTextureColorSpace(const std::string &path,
                                            TextureUsage usage);

bool LoadTextureImage(const std::string &path, LoadedTextureImage &outImage,
                      TextureUsage usage = TextureUsage::Color);

} // namespace NomadPathTracer

#endif
