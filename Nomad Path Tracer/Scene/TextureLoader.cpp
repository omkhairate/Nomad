#include "TextureLoader.h"

#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <vector>

namespace NomadPathTracer {

namespace {

CFURLRef CreateFileURL(const std::string &path) {
    return CFURLCreateFromFileSystemRepresentation(
        nullptr, reinterpret_cast<const UInt8 *>(path.c_str()), path.size(), false);
}

} // namespace

namespace {

bool ContainsDelimitedToken(const std::string &value, const std::string &token) {
    size_t pos = value.find(token);
    while (pos != std::string::npos) {
        bool startOk = (pos == 0) ||
                       !std::isalnum(static_cast<unsigned char>(value[pos - 1]));
        size_t end = pos + token.size();
        bool endOk = (end >= value.size()) ||
                      !std::isalnum(static_cast<unsigned char>(value[end]));
        if (startOk && endOk) {
            return true;
        }
        pos = value.find(token, pos + token.size());
    }
    return false;
}

bool LooksLikeNormalMapFilename(const std::string &path) {
    std::filesystem::path fsPath(path);
    std::string filename = fsPath.filename().string();
    std::string lower = filename;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    static const char *kTokens[] = {"normalmap", "normal", "nrm", "nmap", "tangent"};
    for (const char *token : kTokens) {
        if (ContainsDelimitedToken(lower, token)) {
            return true;
        }
    }

    if (lower.find(".normal.") != std::string::npos) {
        return true;
    }

    std::string extension = fsPath.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (extension == ".nrm" || extension == ".nm" || extension == ".normal") {
        return true;
    }

    return false;
}

} // namespace

TextureColorSpace DetermineTextureColorSpace(const std::string &path,
                                            TextureUsage usage) {
    if (usage == TextureUsage::NormalMap || usage == TextureUsage::Data) {
        return TextureColorSpace::Linear;
    }

    if (LooksLikeNormalMapFilename(path)) {
        return TextureColorSpace::Linear;
    }

    return TextureColorSpace::sRGB;
}

bool LoadTextureImage(const std::string &path, LoadedTextureImage &outImage,
                      TextureUsage usage) {
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

    TextureColorSpace targetSpace = DetermineTextureColorSpace(path, usage);
    CGColorSpaceRef colorSpace = nullptr;
#if defined(kCGColorSpaceGenericRGBLinear)
    if (targetSpace == TextureColorSpace::Linear) {
        colorSpace = CGColorSpaceCreateWithName(kCGColorSpaceGenericRGBLinear);
    }
#endif
#if defined(kCGColorSpaceSRGB)
    if (!colorSpace && targetSpace == TextureColorSpace::sRGB) {
        colorSpace = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    }
#endif
    if (!colorSpace) {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    CGContextRef context = CGBitmapContextCreate(
        rawData.data(), width, height, 8, width * 4, colorSpace,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);

    if (!context) {
        std::printf("Failed to create bitmap context for texture '%s'\n", path.c_str());
        if (colorSpace) {
            CGColorSpaceRelease(colorSpace);
        }
        CGImageRelease(image);
        CFRelease(source);
        CFRelease(url);
        return false;
    }

    CGRect rect = CGRectMake(0, 0, static_cast<double>(width),
                             static_cast<double>(height));
    CGContextDrawImage(context, rect, image);

    if (colorSpace) {
        CGColorSpaceRelease(colorSpace);
    }
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

} // namespace NomadPathTracer
