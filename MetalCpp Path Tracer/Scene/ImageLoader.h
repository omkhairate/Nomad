#ifndef IMAGELOADER_H
#define IMAGELOADER_H

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace MetalCppPathTracer {

struct LoadedImage {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<uint8_t> pixels;
};

class ImageLoader {
public:
    int loadImage(const std::filesystem::path &baseDir, const std::string &filename);
    const LoadedImage *getImage(int index) const;
    const std::vector<LoadedImage> &images() const { return _images; }
    void clear();

private:
    bool loadImageFile(const std::filesystem::path &path, LoadedImage &outImage);

    std::unordered_map<std::string, int> _pathToIndex;
    std::vector<LoadedImage> _images;
};

ImageLoader &GetGlobalImageLoader();

} // namespace MetalCppPathTracer

#endif
