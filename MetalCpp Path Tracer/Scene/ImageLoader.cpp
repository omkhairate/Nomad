#include "ImageLoader.h"

#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#include <CoreFoundation/CoreFoundation.h>
#include <cstdio>
#include <utility>

namespace MetalCppPathTracer {

namespace {

std::string NormalizePath(const std::filesystem::path &path) {
    std::filesystem::path normalized = path.lexically_normal();
    normalized.make_preferred();
    return normalized.string();
}

} // namespace

ImageLoader &GetGlobalImageLoader() {
    static ImageLoader loader;
    return loader;
}

int ImageLoader::loadImage(const std::filesystem::path &baseDir,
                           const std::string &filename) {
    if (filename.empty())
        return -1;

    std::filesystem::path resolved = baseDir / filename;
    std::string key = NormalizePath(resolved);

    auto it = _pathToIndex.find(key);
    if (it != _pathToIndex.end())
        return it->second;

    LoadedImage image;
    if (!loadImageFile(resolved, image)) {
        std::printf("Failed to load texture: %s\n", key.c_str());
        return -1;
    }

    int index = static_cast<int>(_images.size());
    _images.push_back(std::move(image));
    _pathToIndex.emplace(std::move(key), index);
    return index;
}

const LoadedImage *ImageLoader::getImage(int index) const {
    if (index < 0 || static_cast<size_t>(index) >= _images.size())
        return nullptr;
    return &_images[static_cast<size_t>(index)];
}

void ImageLoader::clear() {
    _pathToIndex.clear();
    _images.clear();
}

bool ImageLoader::loadImageFile(const std::filesystem::path &path,
                                LoadedImage &outImage) {
    std::string utf8 = NormalizePath(path);
    CFStringRef cfPath = CFStringCreateWithCString(nullptr, utf8.c_str(),
                                                   kCFStringEncodingUTF8);
    if (!cfPath)
        return false;

    CFURLRef url = CFURLCreateWithFileSystemPath(nullptr, cfPath,
                                                 kCFURLPOSIXPathStyle, false);
    CFRelease(cfPath);
    if (!url)
        return false;

    CGImageSourceRef source = CGImageSourceCreateWithURL(url, nullptr);
    CFRelease(url);
    if (!source)
        return false;

    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
    CFRelease(source);
    if (!image)
        return false;

    size_t width = CGImageGetWidth(image);
    size_t height = CGImageGetHeight(image);
    if (width == 0 || height == 0) {
        CGImageRelease(image);
        return false;
    }

    outImage.width = static_cast<int>(width);
    outImage.height = static_cast<int>(height);
    outImage.channels = 4;
    outImage.pixels.resize(width * height * 4);

    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    if (!colorSpace) {
        CGImageRelease(image);
        return false;
    }

    CGContextRef context = CGBitmapContextCreate(
        outImage.pixels.data(), width, height, 8, width * 4, colorSpace,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Little);
    CGColorSpaceRelease(colorSpace);
    if (!context) {
        CGImageRelease(image);
        return false;
    }

    CGRect rect = CGRectMake(0, 0, static_cast<double>(width),
                             static_cast<double>(height));
    CGContextDrawImage(context, rect, image);

    CGContextRelease(context);
    CGImageRelease(image);
    return true;
}

} // namespace MetalCppPathTracer
