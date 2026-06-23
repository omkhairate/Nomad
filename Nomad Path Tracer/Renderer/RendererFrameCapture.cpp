#include "Renderer.h"

#include <Metal/Metal.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define TINYEXR_IMPLEMENTATION
#include "../tinyexr.h"

namespace {
struct OidnApi {
  using Device = void *;
  using Filter = void *;
  using ErrorCode = int;
  using NewDeviceFn = Device (*)(int);
  using CommitDeviceFn = void (*)(Device);
  using GetDeviceErrorFn = const char *(*)(Device, ErrorCode *);
  using NewFilterFn = Filter (*)(Device, const char *);
  using SetSharedFilterImageFn = void (*)(Filter, const char *, void *, int,
                                          size_t, size_t, size_t, size_t,
                                          size_t);
  using SetFilterBoolFn = void (*)(Filter, const char *, bool);
  using SetFilter1bFn = void (*)(Filter, const char *, bool);
  using CommitFilterFn = void (*)(Filter);
  using ExecuteFilterFn = void (*)(Filter);
  using ReleaseFilterFn = void (*)(Filter);
  using ReleaseDeviceFn = void (*)(Device);

  void *libraryHandle = nullptr;
  NewDeviceFn newDevice = nullptr;
  CommitDeviceFn commitDevice = nullptr;
  GetDeviceErrorFn getDeviceError = nullptr;
  NewFilterFn newFilter = nullptr;
  SetSharedFilterImageFn setSharedFilterImage = nullptr;
  SetFilterBoolFn setFilterBool = nullptr;
  SetFilter1bFn setFilter1b = nullptr;
  CommitFilterFn commitFilter = nullptr;
  ExecuteFilterFn executeFilter = nullptr;
  ReleaseFilterFn releaseFilter = nullptr;
  ReleaseDeviceFn releaseDevice = nullptr;
  bool loadAttempted = false;
  bool loaded = false;
  std::string loadError;
};

constexpr int kOidnDeviceTypeDefault = 0;
constexpr int kOidnFormatFloat3 = 3;

OidnApi &oidnApi() {
  static OidnApi api;
  if (api.loadAttempted)
    return api;

  api.loadAttempted = true;
  const std::array<const char *, 5> candidateLibraries = {
      "libOpenImageDenoise.dylib",
      "/opt/homebrew/lib/libOpenImageDenoise.dylib",
      "/usr/local/lib/libOpenImageDenoise.dylib",
      "libOpenImageDenoise.2.dylib",
      "libOpenImageDenoise.1.dylib",
  };

  for (const char *candidate : candidateLibraries) {
    api.libraryHandle = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
    if (api.libraryHandle)
      break;
  }

  if (!api.libraryHandle) {
    const char *err = dlerror();
    api.loadError = err ? err : "OIDN library not found.";
    return api;
  }

  auto loadSymbol = [&](auto &target, const char *symbol) -> bool {
    target = reinterpret_cast<std::remove_reference_t<decltype(target)>>(
        dlsym(api.libraryHandle, symbol));
    return target != nullptr;
  };

  const bool requiredSymbolsLoaded =
      loadSymbol(api.newDevice, "oidnNewDevice") &&
      loadSymbol(api.commitDevice, "oidnCommitDevice") &&
      loadSymbol(api.getDeviceError, "oidnGetDeviceError") &&
      loadSymbol(api.newFilter, "oidnNewFilter") &&
      loadSymbol(api.setSharedFilterImage, "oidnSetSharedFilterImage") &&
      loadSymbol(api.commitFilter, "oidnCommitFilter") &&
      loadSymbol(api.executeFilter, "oidnExecuteFilter") &&
      loadSymbol(api.releaseFilter, "oidnReleaseFilter") &&
      loadSymbol(api.releaseDevice, "oidnReleaseDevice");

  if (!requiredSymbolsLoaded) {
    const char *err = dlerror();
    api.loadError = err ? err : "Missing required OIDN symbols.";
    dlclose(api.libraryHandle);
    api.libraryHandle = nullptr;
    return api;
  }

  loadSymbol(api.setFilterBool, "oidnSetFilterBool");
  loadSymbol(api.setFilter1b, "oidnSetFilter1b");
  if (!api.setFilterBool && !api.setFilter1b) {
    api.loadError = "OIDN bool setter symbol not found.";
    dlclose(api.libraryHandle);
    api.libraryHandle = nullptr;
    return api;
  }

  api.loaded = true;
  return api;
}

void oidnSetBoolOption(OidnApi &api, OidnApi::Filter filter, const char *name,
                       bool value) {
  if (api.setFilterBool) {
    api.setFilterBool(filter, name, value);
    return;
  }
  if (api.setFilter1b)
    api.setFilter1b(filter, name, value);
}

bool applyOidnOfflineDenoise(std::vector<float> &rgba,
                             const std::vector<float> &albedoData,
                             const std::vector<float> &normalData,
                             size_t width, size_t height,
                             uint64_t frameIndex) {
  const size_t pixelCount = width * height;
  if (pixelCount == 0 || rgba.size() < pixelCount * 4)
    return false;
  if (albedoData.size() < pixelCount * 4 || normalData.size() < pixelCount * 4)
    return false;

  OidnApi &api = oidnApi();
  if (!api.loaded) {
    std::printf("OIDN unavailable for frame %llu: %s\n",
                static_cast<unsigned long long>(frameIndex),
                api.loadError.empty() ? "library load failed"
                                      : api.loadError.c_str());
    return false;
  }

  std::vector<float> color(pixelCount * 3, 0.0f);
  std::vector<float> output(pixelCount * 3, 0.0f);
  std::vector<float> albedo(pixelCount * 3, 0.0f);
  std::vector<float> normal(pixelCount * 3, 0.0f);
  for (size_t i = 0; i < pixelCount; ++i) {
    color[i * 3 + 0] = rgba[i * 4 + 0];
    color[i * 3 + 1] = rgba[i * 4 + 1];
    color[i * 3 + 2] = rgba[i * 4 + 2];

    albedo[i * 3 + 0] = albedoData[i * 4 + 0];
    albedo[i * 3 + 1] = albedoData[i * 4 + 1];
    albedo[i * 3 + 2] = albedoData[i * 4 + 2];

    float nx = normalData[i * 4 + 0];
    float ny = normalData[i * 4 + 1];
    float nz = normalData[i * 4 + 2];
    float len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 1e-6f) {
      nx /= len;
      ny /= len;
      nz /= len;
    }
    normal[i * 3 + 0] = nx;
    normal[i * 3 + 1] = ny;
    normal[i * 3 + 2] = nz;
  }

  OidnApi::Device device = api.newDevice(kOidnDeviceTypeDefault);
  if (!device) {
    std::printf("OIDN failed to create device for frame %llu.\n",
                static_cast<unsigned long long>(frameIndex));
    return false;
  }
  api.commitDevice(device);

  OidnApi::Filter filter = api.newFilter(device, "RT");
  if (!filter) {
    std::printf("OIDN failed to create RT filter for frame %llu.\n",
                static_cast<unsigned long long>(frameIndex));
    api.releaseDevice(device);
    return false;
  }

  const size_t pixelStride = sizeof(float) * 3;
  const size_t rowStride = width * pixelStride;
  api.setSharedFilterImage(filter, "color", color.data(), kOidnFormatFloat3,
                           width, height, 0, pixelStride, rowStride);
  api.setSharedFilterImage(filter, "output", output.data(), kOidnFormatFloat3,
                           width, height, 0, pixelStride, rowStride);
  api.setSharedFilterImage(filter, "albedo", albedo.data(), kOidnFormatFloat3,
                           width, height, 0, pixelStride, rowStride);
  api.setSharedFilterImage(filter, "normal", normal.data(), kOidnFormatFloat3,
                           width, height, 0, pixelStride, rowStride);
  oidnSetBoolOption(api, filter, "hdr", true);

  api.commitFilter(filter);
  api.executeFilter(filter);

  OidnApi::ErrorCode errorCode = 0;
  const char *errorMessage = api.getDeviceError(device, &errorCode);
  bool success = errorCode == 0;
  if (!success) {
    std::printf("OIDN error for frame %llu: %s (code %d)\n",
                static_cast<unsigned long long>(frameIndex),
                errorMessage ? errorMessage : "unknown",
                static_cast<int>(errorCode));
  } else {
    for (size_t i = 0; i < pixelCount; ++i) {
      rgba[i * 4 + 0] = output[i * 3 + 0];
      rgba[i * 4 + 1] = output[i * 3 + 1];
      rgba[i * 4 + 2] = output[i * 3 + 2];
      rgba[i * 4 + 3] = 1.0f;
    }
  }

  api.releaseFilter(filter);
  api.releaseDevice(device);
  return success;
}



size_t alignTo(size_t value, size_t alignment) {
  if (alignment == 0)
    return value;
  return (value + alignment - 1) & ~(alignment - 1);
}

size_t bytesPerPixel(MTL::PixelFormat format) {
  switch (format) {
  case MTL::PixelFormat::PixelFormatRGBA16Float:
    return sizeof(uint16_t) * 4;
  case MTL::PixelFormat::PixelFormatRGBA32Float:
    return sizeof(float) * 4;
  case MTL::PixelFormat::PixelFormatRGBA8Unorm:
  case MTL::PixelFormat::PixelFormatRGBA8Unorm_sRGB:
    return sizeof(uint8_t) * 4;
  case MTL::PixelFormat::PixelFormatRG16Float:
    return sizeof(uint16_t) * 2;
  case MTL::PixelFormat::PixelFormatR16Float:
    return sizeof(uint16_t);
  default:
    return 0;
  }
}
} // namespace

namespace NomadPathTracer {

void Renderer::ensureFrameCaptureDirectory() {
  if (_frameCaptureDirectoryInitialized)
    return;

  std::filesystem::path baseDir;
  if (const char *runsPath = std::getenv("MPT_RUNS_PATH");
      runsPath && runsPath[0]) {
    baseDir = std::filesystem::path(runsPath);
  } else {
    baseDir = std::filesystem::current_path() / "Benchmarks";
  }
  std::filesystem::path framesDir = baseDir / _frameCaptureSubdirectory;

  std::error_code ec;
  std::filesystem::create_directories(framesDir, ec);
  if (ec) {
    std::error_code fallbackError;
    std::filesystem::create_directories(baseDir, fallbackError);
    if (fallbackError) {
      std::printf(
          "Failed to initialize frame capture directory '%s': %s\n",
          framesDir.string().c_str(), ec.message().c_str());
      _frameCaptureDirectory.clear();
      _frameCaptureDirectoryInitialized = true;
      return;
    }
    _frameCaptureDirectory = baseDir.string();
  } else {
    _frameCaptureDirectory = framesDir.string();
  }

  _frameCaptureDirectoryInitialized = true;
}

std::shared_ptr<Renderer::FrameCaptureRequest>
Renderer::encodeFrameCapture(MTL::Texture *colorTexture,
                             MTL::Texture *albedoTexture,
                             MTL::Texture *normalTexture, uint64_t frameIndex,
                             MTL::CommandBuffer *cmd,
                             MTL::BlitCommandEncoder *&blit) {
  if (!colorTexture || !cmd)
    return nullptr;

  ensureFrameCaptureDirectory();
  if (_frameCaptureDirectory.empty())
    return nullptr;

  NS::UInteger width = colorTexture->width();
  NS::UInteger height = colorTexture->height();
  if (width == 0 || height == 0)
    return nullptr;

  size_t pixelBytes = bytesPerPixel(colorTexture->pixelFormat());
  if (pixelBytes == 0)
    return nullptr;

  size_t alignedRowBytes =
      alignTo(static_cast<size_t>(width) * pixelBytes, size_t(256));
  size_t totalBytes = alignedRowBytes * static_cast<size_t>(height);
  if (totalBytes == 0)
    return nullptr;

  MTL::Buffer *readback =
      allocateBuffer(static_cast<NS::UInteger>(totalBytes),
                     MTL::ResourceStorageModeShared,
                     GpuMemoryTracker::Category::Staging, "FrameCaptureReadback");
  if (!readback)
    return nullptr;

  if (!blit)
    blit = cmd->blitCommandEncoder();
  if (!blit) {
    releaseTrackedResource(readback);
    return nullptr;
  }

  NS::UInteger bytesPerImage =
      static_cast<NS::UInteger>(alignedRowBytes * static_cast<size_t>(height));
  MTL::Size size = MTL::Size::Make(width, height, 1);
  MTL::Origin origin = MTL::Origin::Make(0, 0, 0);
  blit->copyFromTexture(colorTexture, 0, 0, origin, size, readback, 0,
                        static_cast<NS::UInteger>(alignedRowBytes),
                        bytesPerImage);

  std::ostringstream nameBuilder;
  nameBuilder << "frame_" << std::setfill('0') << std::setw(6) << frameIndex
              << ".exr";

  std::filesystem::path outputPath;
  try {
    outputPath = std::filesystem::path(_frameCaptureDirectory) / nameBuilder.str();
  } catch (const std::exception &e) {
    std::printf("Failed to build frame capture path: %s\n", e.what());
    releaseTrackedResource(readback);
    return nullptr;
  }

  auto capture = std::make_shared<FrameCaptureRequest>();
  capture->frameIndex = frameIndex;
  capture->filePath = outputPath.string();
  capture->buffer = readback;
  capture->width = static_cast<size_t>(width);
  capture->height = static_cast<size_t>(height);
  capture->alignedRowBytes = alignedRowBytes;
  capture->format = colorTexture->pixelFormat();

  auto captureAuxiliaryTexture = [&](MTL::Texture *src, MTL::Buffer *&dstBuffer,
                                     size_t &rowBytes, MTL::PixelFormat &format,
                                     const std::string &pathSuffix) {
    if (!src)
      return;
    if (src->width() != width || src->height() != height)
      return;
    size_t auxPixelBytes = bytesPerPixel(src->pixelFormat());
    if (auxPixelBytes == 0)
      return;
    rowBytes =
        alignTo(static_cast<size_t>(width) * auxPixelBytes, static_cast<size_t>(256));
    size_t auxTotalBytes = rowBytes * static_cast<size_t>(height);
    if (auxTotalBytes == 0)
      return;
    dstBuffer =
        allocateBuffer(static_cast<NS::UInteger>(auxTotalBytes),
                       MTL::ResourceStorageModeShared,
                       GpuMemoryTracker::Category::Staging,
                       "FrameCaptureAuxReadback");
    if (!dstBuffer) {
      rowBytes = 0;
      return;
    }
    if (!blit)
      blit = cmd->blitCommandEncoder();
    if (!blit) {
      releaseTrackedResource(dstBuffer);
      dstBuffer = nullptr;
      rowBytes = 0;
      return;
    }
    blit->copyFromTexture(src, 0, 0, origin, size, dstBuffer, 0,
                          static_cast<NS::UInteger>(rowBytes), bytesPerImage);
    format = src->pixelFormat();
    try {
      std::filesystem::path auxPath = outputPath;
      auxPath.replace_filename(auxPath.stem().string() + pathSuffix +
                               auxPath.extension().string());
      if (pathSuffix == "_albedo")
        capture->albedoPath = auxPath.string();
      else if (pathSuffix == "_normal")
        capture->normalPath = auxPath.string();
    } catch (const std::exception &e) {
      std::printf("Failed to build auxiliary capture path: %s\n", e.what());
    }
  };

  captureAuxiliaryTexture(albedoTexture, capture->albedoBuffer,
                          capture->albedoAlignedRowBytes, capture->albedoFormat,
                          std::string("_albedo"));
  captureAuxiliaryTexture(normalTexture, capture->normalBuffer,
                          capture->normalAlignedRowBytes, capture->normalFormat,
                          std::string("_normal"));

  std::printf("Queued frame %llu for EXR capture: %s\n",
              static_cast<unsigned long long>(frameIndex),
              capture->filePath.c_str());

  return capture;
}

void Renderer::finalizeFrameCapture(
    const std::shared_ptr<FrameCaptureRequest> &capture) {
  if (!capture)
    return;

  auto releaseCaptureBuffers = [&]() {
    if (capture->buffer) {
      releaseTrackedResource(capture->buffer);
      capture->buffer = nullptr;
    }
    if (capture->albedoBuffer) {
      releaseTrackedResource(capture->albedoBuffer);
      capture->albedoBuffer = nullptr;
    }
    if (capture->normalBuffer) {
      releaseTrackedResource(capture->normalBuffer);
      capture->normalBuffer = nullptr;
    }
  };

  if (!capture->buffer) {
    std::printf("Frame capture buffer missing for frame %llu\n",
               static_cast<unsigned long long>(capture->frameIndex));
    releaseCaptureBuffers();
    return;
  }

  {
    std::lock_guard<std::mutex> lock(_captureMutex);
    _completedCaptures.push_back(capture);
  }
  _captureOutputsPending.store(true, std::memory_order_release);

  std::printf("Frame %llu queued for offline EXR denoising: %s\n",
             static_cast<unsigned long long>(capture->frameIndex),
             capture->filePath.c_str());
}

void Renderer::processPendingCapturedFrames() {
  while (true) {
    std::vector<std::shared_ptr<FrameCaptureRequest>> captures;
    {
      std::lock_guard<std::mutex> lock(_captureMutex);
      if (_completedCaptures.empty()) {
        _captureOutputsPending.store(false, std::memory_order_release);
        break;
      }
      captures.swap(_completedCaptures);
    }

    for (const auto &capture : captures) {
      if (!capture)
        continue;

      auto releaseCaptureBuffers = [&]() {
        if (capture->buffer) {
          releaseTrackedResource(capture->buffer);
          capture->buffer = nullptr;
        }
        if (capture->albedoBuffer) {
          releaseTrackedResource(capture->albedoBuffer);
          capture->albedoBuffer = nullptr;
        }
        if (capture->normalBuffer) {
          releaseTrackedResource(capture->normalBuffer);
          capture->normalBuffer = nullptr;
        }
      };

      if (!capture->buffer) {
        std::printf("Frame capture buffer missing for frame %llu\n",
                   static_cast<unsigned long long>(capture->frameIndex));
        releaseCaptureBuffers();
        continue;
      }

      void *contents = capture->buffer->contents();
    if (!contents) {
      std::printf("Failed to map frame capture buffer for frame %llu\n",
                 static_cast<unsigned long long>(capture->frameIndex));
      releaseCaptureBuffers();
      continue;
    }

    size_t pixelCount = capture->width * capture->height;
    if (pixelCount == 0) {
      releaseCaptureBuffers();
      continue;
    }

    auto convertBuffer = [&](const void *src, size_t rowBytes,
                             MTL::PixelFormat format, std::vector<float> &dst) {
      if (!src)
        return false;
      dst.assign(pixelCount * 4, 0.0f);
      if (format == MTL::PixelFormat::PixelFormatRGBA16Float) {
        for (size_t y = 0; y < capture->height; ++y) {
          const uint8_t *rowBase = static_cast<const uint8_t *>(src) +
                                   y * rowBytes;
          const uint16_t *row = reinterpret_cast<const uint16_t *>(rowBase);
          float *dstRow = dst.data() + y * capture->width * 4;
          for (size_t x = 0; x < capture->width; ++x) {
            for (size_t c = 0; c < 4; ++c) {
              tinyexr::FP16 value{};
              value.u = row[x * 4 + c];
              dstRow[x * 4 + c] = half_to_float(value).f;
            }
          }
        }
        return true;
      }
      if (format == MTL::PixelFormat::PixelFormatRGBA32Float) {
        for (size_t y = 0; y < capture->height; ++y) {
          const uint8_t *rowBase = static_cast<const uint8_t *>(src) +
                                   y * rowBytes;
          const float *row = reinterpret_cast<const float *>(rowBase);
          float *dstRow = dst.data() + y * capture->width * 4;
          std::memcpy(dstRow, row, sizeof(float) * capture->width * 4);
        }
        return true;
      }
      return false;
    };

    std::vector<float> rgba;
    if (!convertBuffer(contents, capture->alignedRowBytes, capture->format,
                       rgba)) {
      std::printf("Unsupported pixel format %u for frame capture.\n",
                 static_cast<unsigned>(capture->format));
      releaseCaptureBuffers();
      continue;
    }
    const std::vector<float> overlayReference =
        capture->containsDebugOverlay ? rgba : std::vector<float>{};

    std::vector<float> albedoData;
    if (capture->albedoBuffer && capture->albedoAlignedRowBytes > 0) {
      if (void *albedoContents = capture->albedoBuffer->contents()) {
        if (!convertBuffer(albedoContents, capture->albedoAlignedRowBytes,
                           capture->albedoFormat, albedoData))
          albedoData.clear();
      }
    }

    std::vector<float> normalData;
    if (capture->normalBuffer && capture->normalAlignedRowBytes > 0) {
      if (void *normalContents = capture->normalBuffer->contents()) {
        if (!convertBuffer(normalContents, capture->normalAlignedRowBytes,
                           capture->normalFormat, normalData))
          normalData.clear();
      }
    }

    bool attemptedOidn = false;
    bool oidnApplied = false;
    if (_disableOfflineOidn) {
      std::printf("Skipping OIDN for frame %llu: deferred denoise mode is "
                  "enabled.\n",
                  static_cast<unsigned long long>(capture->frameIndex));
    } else if (!albedoData.empty() && !normalData.empty()) {
      attemptedOidn = true;
      std::printf("Applying OIDN offline EXR denoiser to frame %llu\n",
                  static_cast<unsigned long long>(capture->frameIndex));
      oidnApplied =
          applyOidnOfflineDenoise(rgba, albedoData, normalData, capture->width,
                                  capture->height, capture->frameIndex);
      if (!oidnApplied) {
        std::printf("OIDN denoise failed for frame %llu; saving original color.\n",
                    static_cast<unsigned long long>(capture->frameIndex));
      }
    } else {
      std::printf("Skipping OIDN for frame %llu: missing albedo/normal AOVs.\n",
                  static_cast<unsigned long long>(capture->frameIndex));
    }
    if (_disableOfflineOidn || !attemptedOidn || !oidnApplied) {
      for (size_t i = 0; i < pixelCount; ++i)
        rgba[i * 4 + 3] = 1.0f;
    }

    if (!overlayReference.empty() && overlayReference.size() == rgba.size()) {
      auto closeToColor = [](float r, float g, float b,
                             float cr, float cg, float cb) {
        constexpr float kPerChannelTol = 0.18f;
        return std::fabs(r - cr) <= kPerChannelTol &&
               std::fabs(g - cg) <= kPerChannelTol &&
               std::fabs(b - cb) <= kPerChannelTol;
      };

      size_t restoredOverlayPixels = 0;
      for (size_t i = 0; i < pixelCount; ++i) {
        size_t o = i * 4;
        float srcR = overlayReference[o + 0];
        float srcG = overlayReference[o + 1];
        float srcB = overlayReference[o + 2];

        bool warmOverlayShape =
            srcR >= 0.82f && srcG >= 0.56f && srcB <= 0.34f &&
            (srcR - srcB) >= 0.42f && (srcG - srcB) >= 0.20f;
        bool frustumColor =
            closeToColor(srcR, srcG, srcB, 1.00f, 0.95f, 0.10f);
        bool trailColor =
            closeToColor(srcR, srcG, srcB, 1.00f, 0.72f, 0.10f);
        if (!(warmOverlayShape && (frustumColor || trailColor)))
          continue;

        rgba[o + 0] = srcR;
        rgba[o + 1] = srcG;
        rgba[o + 2] = srcB;
        rgba[o + 3] = std::max(rgba[o + 3], overlayReference[o + 3]);
        ++restoredOverlayPixels;
      }

      if (restoredOverlayPixels > 0) {
        std::printf("Restored %zu observer overlay pixels after OIDN for frame %llu\n",
                   restoredOverlayPixels,
                   static_cast<unsigned long long>(capture->frameIndex));
      }
    }

    const char *err = nullptr;
    int saveAsFp16 =
        capture->format == MTL::PixelFormat::PixelFormatRGBA16Float ? 1 : 0;
    int ret = SaveEXR(rgba.data(), static_cast<int>(capture->width),
                      static_cast<int>(capture->height), 4, saveAsFp16,
                      capture->filePath.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
      if (err) {
        std::printf("Failed to save EXR frame %llu: %s\n",
                   static_cast<unsigned long long>(capture->frameIndex), err);
        FreeEXRErrorMessage(err);
      } else {
        std::printf("Failed to save EXR frame %llu (error code %d)\n",
                   static_cast<unsigned long long>(capture->frameIndex), ret);
      }
    } else {
      std::printf("Saved EXR frame %llu to %s\n",
                 static_cast<unsigned long long>(capture->frameIndex),
                 capture->filePath.c_str());
    }

    auto saveAuxiliaryExr = [&](const std::vector<float> &data,
                                const std::string &path,
                                MTL::PixelFormat format, const char *label) {
      if (data.empty() || path.empty())
        return;
      std::vector<float> temp(data);
      for (size_t i = 0; i < pixelCount; ++i)
        temp[i * 4 + 3] = 1.0f;
      int auxSaveAsFp16 =
          format == MTL::PixelFormat::PixelFormatRGBA16Float ? 1 : 0;
      const char *auxErr = nullptr;
      int auxRet = SaveEXR(temp.data(), static_cast<int>(capture->width),
                           static_cast<int>(capture->height), 4, auxSaveAsFp16,
                           path.c_str(), &auxErr);
      if (auxRet != TINYEXR_SUCCESS) {
        if (auxErr) {
          std::printf("Failed to save %s for frame %llu: %s\n", label,
                     static_cast<unsigned long long>(capture->frameIndex),
                     auxErr);
          FreeEXRErrorMessage(auxErr);
        } else {
          std::printf("Failed to save %s for frame %llu (error %d)\n", label,
                     static_cast<unsigned long long>(capture->frameIndex),
                     auxRet);
        }
      } else {
        std::printf("Saved %s for frame %llu to %s\n", label,
                   static_cast<unsigned long long>(capture->frameIndex),
                   path.c_str());
      }
    };

    if (!albedoData.empty())
      saveAuxiliaryExr(albedoData, capture->albedoPath, capture->albedoFormat,
                       "albedo");

    if (!normalData.empty()) {
      for (size_t i = 0; i < pixelCount; ++i) {
        float nx = normalData[i * 4 + 0];
        float ny = normalData[i * 4 + 1];
        float nz = normalData[i * 4 + 2];
        float len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 1e-6f) {
          normalData[i * 4 + 0] = nx / len;
          normalData[i * 4 + 1] = ny / len;
          normalData[i * 4 + 2] = nz / len;
        }
      }
      saveAuxiliaryExr(normalData, capture->normalPath, capture->normalFormat,
                       "normal");
    }

      releaseCaptureBuffers();
    }
  }
}


} // namespace NomadPathTracer
