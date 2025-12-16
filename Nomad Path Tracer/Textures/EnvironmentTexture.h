#ifndef METALCPP_PATH_TRACER_TEXTURES_ENVIRONMENTTEXTURE_H
#define METALCPP_PATH_TRACER_TEXTURES_ENVIRONMENTTEXTURE_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if __has_include(<Metal/Metal.hpp>)
#include <Metal/Metal.hpp>
#define METALCPP_ENVIRONMENT_HAS_METAL 1
#else
#define METALCPP_ENVIRONMENT_HAS_METAL 0
#endif

#include "../tinyexr.h"

namespace NomadPathTracer {

struct EnvironmentTextureData {
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<float> pixels;
};

inline bool LoadEnvironmentTextureData(const std::string &path,
                                       EnvironmentTextureData &outData) {
  outData = EnvironmentTextureData{};
  if (path.empty())
    return false;

  float *rgba = nullptr;
  int width = 0;
  int height = 0;
  const char *err = nullptr;
  int ret = LoadEXR(&rgba, &width, &height, path.c_str(), &err);
  if (ret != TINYEXR_SUCCESS) {
    if (err) {
      std::printf("Failed to load environment EXR '%s': %s\n", path.c_str(), err);
      FreeEXRErrorMessage(err);
    } else {
      std::printf("Failed to load environment EXR '%s' (error %d)\n", path.c_str(),
                  ret);
    }
    return false;
  }

  if (!rgba || width <= 0 || height <= 0) {
    if (rgba)
      free(rgba);
    std::printf("Environment EXR '%s' contained no pixel data.\n", path.c_str());
    return false;
  }

  size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
  outData.width = static_cast<uint32_t>(width);
  outData.height = static_cast<uint32_t>(height);
  outData.pixels.assign(rgba, rgba + pixelCount);
  free(rgba);
  return true;
}

#if METALCPP_ENVIRONMENT_HAS_METAL

inline MTL::Texture *CreateEnvironmentTexture(MTL::Device *device,
                                              const EnvironmentTextureData &data) {
  if (!device || data.width == 0 || data.height == 0 || data.pixels.empty())
    return nullptr;

  MTL::TextureDescriptor *descriptor = MTL::TextureDescriptor::alloc()->init();
  descriptor->setTextureType(MTL::TextureType::TextureType2D);
  descriptor->setPixelFormat(MTL::PixelFormat::PixelFormatRGBA32Float);
  descriptor->setWidth(data.width);
  descriptor->setHeight(data.height);
  descriptor->setUsage(MTL::TextureUsageShaderRead);
  descriptor->setStorageMode(MTL::StorageMode::StorageModeManaged);

  MTL::Texture *texture = device->newTexture(descriptor);
  descriptor->release();
  if (!texture)
    return nullptr;

  NS::UInteger bytesPerRow =
      static_cast<NS::UInteger>(data.width * sizeof(float) * 4);
  MTL::Region region = MTL::Region::Make2D(0, 0, data.width, data.height);
  texture->replaceRegion(region, 0, data.pixels.data(), bytesPerRow);
  return texture;
}

inline MTL::SamplerState *CreateEnvironmentSampler(MTL::Device *device) {
  if (!device)
    return nullptr;

  MTL::SamplerDescriptor *descriptor = MTL::SamplerDescriptor::alloc()->init();
  descriptor->setMinFilter(MTL::SamplerMinMagFilterLinear);
  descriptor->setMagFilter(MTL::SamplerMinMagFilterLinear);
  descriptor->setMipFilter(MTL::SamplerMipFilterNotMipmapped);
  descriptor->setSAddressMode(MTL::SamplerAddressModeRepeat);
  descriptor->setTAddressMode(MTL::SamplerAddressModeClampToEdge);
  descriptor->setRAddressMode(MTL::SamplerAddressModeClampToEdge);
  descriptor->setSupportArgumentBuffers(true);

  MTL::SamplerState *sampler = device->newSamplerState(descriptor);
  descriptor->release();
  return sampler;
}

#endif // METALCPP_ENVIRONMENT_HAS_METAL

} // namespace NomadPathTracer

#endif // METALCPP_PATH_TRACER_TEXTURES_ENVIRONMENTTEXTURE_H
