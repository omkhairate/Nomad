#include "TextureLoader.h"

#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#include <CoreFoundation/CoreFoundation.h>

#include <cstdio>
#include <vector>

namespace MetalCppPathTracer {

namespace {

CFURLRef CreateFileURL(const std::string &path) {
    return CFURLCreateFromFileSystemRepresentation(
        nullptr, reinterpret_cast<const UInt8 *>(path.c_str()), path.size(), false);
}

} // namespace

bool LoadTextureImage(const std::string &path, LoadedTextureImage &outImage) {
    CFURLRef url = CreateFileURL(path);
    if (!url) {
        std::printf("Failed to create URL for texture '%s'\n", path.c_str());
        return false;
    }

    CGImageSourceRef source = CGImageSourceCreateWithURL(url, nullptr);
    if (!source) {
        std::printf("Failed to create image source for texture '%s'\n", path.c_str());
        CFRelease(url);
        return false;
    }

    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
    if (!image) {
        std::printf("Failed to decode texture '%s'\n", path.c_str());
        CFRelease(source);
        CFRelease(url);
        return false;
    }

    size_t width = CGImageGetWidth(image);
    size_t height = CGImageGetHeight(image);
    if (width == 0 || height == 0) {
        std::printf("Texture '%s' has zero size\n", path.c_str());
        CGImageRelease(image);
        CFRelease(source);
        CFRelease(url);
        return false;
    }

    std::vector<uint8_t> rawData(width * height * 4);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(
        rawData.data(), width, height, 8, width * 4, colorSpace,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);

    if (!context) {
        std::printf("Failed to create bitmap context for texture '%s'\n", path.c_str());
        CGColorSpaceRelease(colorSpace);
        CGImageRelease(image);
        CFRelease(source);
        CFRelease(url);
        return false;
    }

    CGRect rect = CGRectMake(0, 0, static_cast<double>(width),
                             static_cast<double>(height));
    CGContextDrawImage(context, rect, image);

    CGColorSpaceRelease(colorSpace);
    CGContextRelease(context);
    CGImageRelease(image);
    CFRelease(source);
    CFRelease(url);

    outImage.width = static_cast<int>(width);
    outImage.height = static_cast<int>(height);
    outImage.pixels.resize(width * height * 4);

    const float inv255 = 1.0f / 255.0f;
    for (size_t i = 0; i < width * height; ++i) {
        outImage.pixels[4 * i + 0] = rawData[4 * i + 0] * inv255;
        outImage.pixels[4 * i + 1] = rawData[4 * i + 1] * inv255;
        outImage.pixels[4 * i + 2] = rawData[4 * i + 2] * inv255;
        outImage.pixels[4 * i + 3] = rawData[4 * i + 3] * inv255;
    }

    return true;
}

} // namespace MetalCppPathTracer
