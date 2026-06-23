#include "Renderer.h"

#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstdio>
#include <vector>

#include "../textures/EnvironmentTexture.h"

namespace {

constexpr size_t kTextureResidencyPrimitiveBudgetLocal = 1;
constexpr uint32_t kMaxMaterialTextureSlotsLocal = 64;

size_t alignToLocal(size_t value, size_t alignment) {
  if (alignment == 0)
    return value;
  return (value + alignment - 1) & ~(alignment - 1);
}

size_t bytesPerPixelLocal(MTL::PixelFormat format) {
  switch (format) {
  case MTL::PixelFormat::PixelFormatRGBA32Float:
    return sizeof(float) * 4;
  case MTL::PixelFormat::PixelFormatR32Float:
    return sizeof(float);
  case MTL::PixelFormat::PixelFormatRGBA16Float:
    return sizeof(uint16_t) * 4;
  case MTL::PixelFormat::PixelFormatR16Float:
    return sizeof(uint16_t);
  default:
    return 0;
  }
}

size_t estimateTextureBytesLocal(MTL::PixelFormat format, size_t width,
                                 size_t height, size_t depth,
                                 size_t arrayLength) {
  size_t pixelBytes = bytesPerPixelLocal(format);
  if (pixelBytes == 0 || width == 0 || height == 0)
    return 0;

  size_t clampedDepth = std::max<size_t>(1, depth);
  size_t rowBytes = width * pixelBytes;
  size_t alignedRowBytes = alignToLocal(rowBytes, 256);
  size_t sliceBytes = alignedRowBytes * height * clampedDepth;
  size_t clampedArray = std::max<size_t>(1, arrayLength);
  return sliceBytes * clampedArray;
}

} // namespace

namespace NomadPathTracer {

void Renderer::releaseTextureSlot(ManagedTextureSlot &slot) {
  if (slot.texture) {
    releaseTrackedResource(slot.texture);
    slot.texture = nullptr;
  }
  if (slot.stagingBuffer) {
    releaseTrackedResource(slot.stagingBuffer);
    slot.stagingBuffer = nullptr;
    slot.stagingCapacity = 0;
  }
  slot.stagingValid = false;
  clearTextureHistory(slot);
}

const char *Renderer::textureSlotLabel(const ManagedTextureSlot &slot) const {
  if (&slot == &_colorSlot)
    return "_colorSlot";
  if (&slot == &_albedoSlot)
    return "_albedoSlot";
  if (&slot == &_normalSlot)
    return "_normalSlot";
  if (&slot == &_positionSlot)
    return "_positionSlot";
  if (&slot == &_albedoHistorySlot)
    return "_albedoHistorySlot";
  if (&slot == &_normalHistorySlot)
    return "_normalHistorySlot";
  if (&slot == &_positionHistorySlot)
    return "_positionHistorySlot";
  if (&slot == &_restirData0Slots[0])
    return "_restirData0Slot0";
  if (&slot == &_restirData0Slots[1])
    return "_restirData0Slot1";
  if (&slot == &_restirData1Slots[0])
    return "_restirData1Slot0";
  if (&slot == &_restirData1Slots[1])
    return "_restirData1Slot1";
  if (&slot == &_restirData2Slots[0])
    return "_restirData2Slot0";
  if (&slot == &_restirData2Slots[1])
    return "_restirData2Slot1";
  if (&slot == &_restirSurfacePosSlots[0])
    return "_restirSurfacePosSlot0";
  if (&slot == &_restirSurfacePosSlots[1])
    return "_restirSurfacePosSlot1";
  if (&slot == &_restirSurfaceNormalSlots[0])
    return "_restirSurfaceNormalSlot0";
  if (&slot == &_restirSurfaceNormalSlots[1])
    return "_restirSurfaceNormalSlot1";
  return "<unknown texture slot>";
}

void Renderer::configureTextureSlot(ManagedTextureSlot &slot, NS::UInteger width,
                                    NS::UInteger height,
                                    MTL::PixelFormat format,
                                    MTL::TextureUsage usage) {
  bool descriptorChanged =
      !slot.descriptorValid || slot.width != width || slot.height != height ||
      slot.pixelFormat != format || slot.usage != usage;

  if (!descriptorChanged)
    return;

  releaseTextureSlot(slot);

  slot.width = width;
  slot.height = height;
  slot.pixelFormat = format;
  slot.usage = usage;
  slot.textureType = MTL::TextureType::TextureType2D;
  slot.storageMode = MTL::StorageMode::StorageModePrivate;
  slot.descriptorValid = true;
  slot.stagingValid = false;
}

size_t Renderer::textureByteSize(const ManagedTextureSlot &slot) const {
  if (!slot.descriptorValid || slot.width == 0 || slot.height == 0)
    return 0;

  size_t pixelBytes = bytesPerPixelLocal(slot.pixelFormat);
  if (pixelBytes == 0)
    return 0;

  size_t rowBytes = slot.width * pixelBytes;
  size_t alignedRowBytes = alignToLocal(rowBytes, 256);
  return alignedRowBytes * slot.height;
}

size_t Renderer::textureByteSize(MTL::Texture *texture) const {
  if (!texture)
    return 0;

  size_t pixelBytes = bytesPerPixelLocal(texture->pixelFormat());
  if (pixelBytes == 0)
    return 0;

  size_t rowBytes = texture->width() * pixelBytes;
  size_t alignedRowBytes = alignToLocal(rowBytes, 256);
  return alignedRowBytes * texture->height();
}

void Renderer::clearTextureHistory(ManagedTextureSlot &slot) {
  slot.historyData.clear();
  slot.historyBacking = ManagedTextureSlot::HistoryBacking::None;
  slot.historyIsProxy = false;
  slot.historyWidth = 0;
  slot.historyHeight = 0;
  slot.historyBytesPerRow = 0;
  slot.needsGpuRefresh = false;
}

void Renderer::disableHistoryForMemoryCap() {
  auto disableSlot = [&](ManagedTextureSlot &slot) {
    clearTextureHistory(slot);
    slot.stagingValid = false;
    if (slot.stagingBuffer) {
      releaseTrackedResource(slot.stagingBuffer);
      slot.stagingBuffer = nullptr;
      slot.stagingCapacity = 0;
    }
  };

  disableSlot(_colorSlot);
  disableSlot(_albedoSlot);
  disableSlot(_normalSlot);
  disableSlot(_positionSlot);
  disableSlot(_albedoHistorySlot);
  disableSlot(_normalHistorySlot);
  disableSlot(_positionHistorySlot);
  for (auto &slot : _restirData0Slots)
    disableSlot(slot);
  for (auto &slot : _restirData1Slots)
    disableSlot(slot);
  for (auto &slot : _restirData2Slots)
    disableSlot(slot);
  for (auto &slot : _restirSurfacePosSlots)
    disableSlot(slot);
  for (auto &slot : _restirSurfaceNormalSlots)
    disableSlot(slot);
}

MTL::Texture *Renderer::requestResidentTexture(ManagedTextureSlot &slot,
                                               MTL::CommandBuffer *cmd,
                                               MTL::BlitCommandEncoder *&blit) {
  (void)cmd;
  (void)blit;
  if (slot.texture || !slot.descriptorValid || slot.width == 0 ||
      slot.height == 0) {
    if (slot.texture)
      slot.lastUsedFrame = _renderedFrameCount;
    return slot.texture;
  }

  const char *label = textureSlotLabel(slot);
  size_t requestedBytes = textureByteSize(slot);
  if (!canAllocateTotalGpuMemory(requestedBytes, 0,
                                 GpuMemoryTracker::Category::Textures, label)) {
    std::printf(
        "[TextureResidency] Deferring slot %s: total memory cap would be exceeded.\n",
        label);
    return nullptr;
  }
  MTL::TextureDescriptor *descriptor =
      MTL::TextureDescriptor::alloc()->init();
  descriptor->setTextureType(slot.textureType);
  descriptor->setWidth(slot.width);
  descriptor->setHeight(slot.height);
  descriptor->setPixelFormat(slot.pixelFormat);
  descriptor->setUsage(slot.usage);
  descriptor->setStorageMode(slot.storageMode);

  slot.texture =
      allocateTexture(descriptor, textureCategoryForSlot(slot), label);
  descriptor->release();

  if (!slot.texture) {
    std::printf("[TextureResidency] Failed to restore slot %s: texture allocation returned null.\n",
                label);
    return nullptr;
  }

  std::printf("[TextureResidency] Restored slot %s without history data.\n",
              label);

  slot.lastUsedFrame = _renderedFrameCount;
  return slot.texture;
}

bool Renderer::evictTextureSlot(ManagedTextureSlot &slot,
                                MTL::CommandBuffer *cmd,
                                MTL::BlitCommandEncoder *&blit) {
  (void)cmd;
  (void)blit;
  if (!slot.texture)
    return false;

  const char *label = textureSlotLabel(slot);
  std::printf(
      "[TextureResidency] Evicting slot %s: releasing without history.\n",
      label);
  clearTextureHistory(slot);
  slot.stagingValid = false;
  if (slot.stagingBuffer) {
    releaseTrackedResource(slot.stagingBuffer);
    slot.stagingBuffer = nullptr;
    slot.stagingCapacity = 0;
  }

  if (slot.texture) {
    releaseTrackedResource(slot.texture);
    slot.texture = nullptr;
  }
  return true;
}

void Renderer::updateTextureResidency(MTL::CommandBuffer *cmd) {
  if (!cmd)
    return;

  bool belowBudget =
      _residentPrimitiveCount < kTextureResidencyPrimitiveBudgetLocal;
  double totalMemoryMB = currentGPUMemoryMB();
  double scratchMB = scratchMemoryMB();
  double residencyMB = std::max(0.0, totalMemoryMB - scratchMB);
  bool overMemory = residencyMB > _textureResidencyMemoryCapMB;
  double totalCapMB = effectiveTotalGpuMemoryCapMB();
  bool totalOverCap = totalCapMB > 0.0 &&
                      (totalMemoryMB > totalCapMB ||
                       _pendingTotalMemoryOverageBytes > 0);
  if (!overMemory && totalMemoryMB > _textureResidencyMemoryCapMB &&
      scratchMB > 0.0) {
    std::printf("[TextureResidency] Skipping eviction: total=%.2f MB, scratch=%.2f MB, "
                "residency=%.2f MB (cap=%.2f MB).\n",
                totalMemoryMB, scratchMB, residencyMB, _textureResidencyMemoryCapMB);
  }
  if (!belowBudget && !overMemory && !totalOverCap)
    return;

  if (overMemory) {
    std::printf("[TextureResidency] Triggering eviction: residency=%.2f MB (total=%.2f MB, "
                "scratch=%.2f MB, cap=%.2f MB).\n",
                residencyMB, totalMemoryMB, scratchMB, _textureResidencyMemoryCapMB);
  } else if (totalOverCap) {
    std::printf("[TextureResidency] Triggering eviction: total=%.2f MB "
                "(residency=%.2f MB, scratch=%.2f MB, cap=%.2f MB).\n",
                totalMemoryMB, residencyMB, scratchMB, totalCapMB);
  }

  std::vector<ManagedTextureSlot *> slots = {
      &_colorSlot,        &_albedoSlot,        &_normalSlot,
      &_positionSlot,     &_albedoHistorySlot, &_normalHistorySlot,
      &_positionHistorySlot};
  for (auto &slot : _restirData0Slots)
    slots.push_back(&slot);
  for (auto &slot : _restirData1Slots)
    slots.push_back(&slot);
  for (auto &slot : _restirData2Slots)
    slots.push_back(&slot);
  for (auto &slot : _restirSurfacePosSlots)
    slots.push_back(&slot);
  for (auto &slot : _restirSurfaceNormalSlots)
    slots.push_back(&slot);

  struct ResidencyCandidate {
    ManagedTextureSlot *slot = nullptr;
    double score = 0.0;
    double bytesMB = 0.0;
  };

  auto scoreSlot = [&](ManagedTextureSlot *slot) {
    ResidencyCandidate candidate{};
    candidate.slot = slot;
    size_t bytes = textureByteSize(*slot);
    candidate.bytesMB = static_cast<double>(bytes) / (1024.0 * 1024.0);
    uint64_t frameAge = _renderedFrameCount > slot->lastUsedFrame
                            ? _renderedFrameCount - slot->lastUsedFrame
                            : 0;
    double recency = 1.0 / (1.0 + static_cast<double>(frameAge));
    candidate.score = recency / (1.0 + candidate.bytesMB);
    return candidate;
  };

  std::vector<ResidencyCandidate> candidates;
  candidates.reserve(slots.size());
  for (ManagedTextureSlot *slot : slots) {
    if (!slot->texture)
      continue;
    candidates.push_back(scoreSlot(slot));
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const ResidencyCandidate &a, const ResidencyCandidate &b) {
              if (a.score == b.score)
                return a.bytesMB > b.bytesMB;
              return a.score < b.score;
            });

  bool stopOnMemory = overMemory || totalOverCap;
  bool stopOnBudget = !belowBudget;
  bool needMemoryEviction = overMemory || totalOverCap;
  bool primitiveBudgetSatisfied = belowBudget;
  MTL::BlitCommandEncoder *blit = nullptr;
  for (const ResidencyCandidate &candidate : candidates) {
    bool needMoreEviction =
        (stopOnMemory && needMemoryEviction) ||
        (stopOnBudget && !primitiveBudgetSatisfied);
    if (!needMoreEviction)
      break;
    if (!evictTextureSlot(*candidate.slot, cmd, blit))
      continue;
    residencyMB = std::max(0.0, residencyMB - candidate.bytesMB);
    totalMemoryMB = std::max(0.0, totalMemoryMB - candidate.bytesMB);
    if (_pendingTotalMemoryOverageBytes > 0) {
      size_t bytes = static_cast<size_t>(candidate.bytesMB * 1024.0 * 1024.0);
      _pendingTotalMemoryOverageBytes =
          _pendingTotalMemoryOverageBytes > bytes
              ? _pendingTotalMemoryOverageBytes - bytes
              : 0;
    }
    needMemoryEviction =
        residencyMB > _textureResidencyMemoryCapMB ||
        (totalCapMB > 0.0 && totalMemoryMB > totalCapMB);
    primitiveBudgetSatisfied =
        _residentPrimitiveCount < kTextureResidencyPrimitiveBudgetLocal;
  }
  if (blit)
    blit->endEncoding();
}

void Renderer::clearMaterialTextures() {
  for (MTL::Texture *texture : _materialTextures) {
    if (texture)
      releaseTrackedResource(texture);
  }
  _materialTextures.clear();
}

void Renderer::releaseEnvironmentTexture() {
  if (_environmentTexture) {
    releaseTrackedResource(_environmentTexture);
    _environmentTexture = nullptr;
  }
  _environmentTexturePath.clear();
}

void Renderer::rebuildEnvironmentTexture() {
  std::string desiredPath;
  float desiredBrightness = 1.0f;
  if (_pScene) {
    desiredPath = _pScene->getEnvironmentTexturePath();
    desiredBrightness = _pScene->getEnvironmentBrightness();
  }

  _environmentBrightness = std::max(desiredBrightness, 0.0f);

  if (!_environmentSampler && _pDevice)
    _environmentSampler = CreateEnvironmentSampler(_pDevice);

  if (desiredPath.empty()) {
    releaseEnvironmentTexture();
    return;
  }

  if (!_environmentSampler)
    return;

  if (_environmentTexture && desiredPath == _environmentTexturePath)
    return;

  releaseEnvironmentTexture();

  EnvironmentTextureData data;
  if (!LoadEnvironmentTextureData(desiredPath, data))
    return;

  size_t estimatedBytes = estimateTextureBytesLocal(
      MTL::PixelFormat::PixelFormatRGBA32Float, data.width, data.height, 1, 1);
  if (estimatedBytes > 0) {
    if (!canAllocateTotalGpuMemory(estimatedBytes, 0,
                                   GpuMemoryTracker::Category::Textures,
                                   "EnvironmentTexture")) {
      std::printf(
          "[Renderer] Environment texture allocation denied for %s.\n",
          desiredPath.c_str());
      return;
    }
  }

  MTL::Texture *texture = CreateEnvironmentTexture(_pDevice, data);
  if (!texture) {
    std::printf("[Renderer] Failed to allocate environment texture for %s\n",
                desiredPath.c_str());
    return;
  }

  trackResource(texture, GpuMemoryTracker::Category::Textures);
  _environmentTexture = texture;
  _environmentTexturePath = desiredPath;
}

void Renderer::rebuildMaterialTextures() {
  clearMaterialTextures();

  if (!_pDevice)
    return;

  size_t availableSlots = static_cast<size_t>(kMaxMaterialTextureSlotsLocal);
  size_t requestedTextures = _cachedTextureInfos.size();
  if (requestedTextures > availableSlots) {
    std::printf(
        "[Renderer] Truncating material textures from %zu to %u to fit shader "
        "bind slots.\n",
        requestedTextures, kMaxMaterialTextureSlotsLocal);
  }

  size_t maxTextures = std::min(requestedTextures, availableSlots);
  size_t totalTexels = _cachedTextureData.size();
  _materialTextures.reserve(maxTextures);

  auto createFallbackTexture = [&]() -> MTL::Texture * {
    MTL::TextureDescriptor *descriptor = MTL::TextureDescriptor::alloc()->init();
    descriptor->setTextureType(MTL::TextureType::TextureType2D);
    descriptor->setPixelFormat(MTL::PixelFormat::PixelFormatRGBA32Float);
    descriptor->setWidth(1);
    descriptor->setHeight(1);
    descriptor->setUsage(MTL::TextureUsageShaderRead);
    descriptor->setStorageMode(MTL::StorageMode::StorageModeManaged);

    MTL::Texture *texture =
        allocateTexture(descriptor, GpuMemoryTracker::Category::Textures,
                        "MaterialFallbackTexture");
    descriptor->release();
    if (!texture)
      return nullptr;

    simd::float4 pixel = simd::float4{1.0f, 1.0f, 1.0f, 1.0f};
    NS::UInteger bytesPerRow = sizeof(simd::float4);
    MTL::Region region = MTL::Region::Make2D(0, 0, 1, 1);
    texture->replaceRegion(region, 0, &pixel, bytesPerRow);
    return texture;
  };

  for (size_t texIndex = 0; texIndex < maxTextures; ++texIndex) {
    const TextureInfo &info = _cachedTextureInfos[texIndex];
    size_t texelCount = static_cast<size_t>(info.width) * info.height;
    bool hasValidDimensions = info.width > 0 && info.height > 0;
    bool hasValidRange = info.offset < totalTexels &&
                         texelCount > 0 &&
                         info.offset + texelCount <= totalTexels;

    if (!hasValidDimensions || !hasValidRange) {
      MTL::Texture *fallback = createFallbackTexture();
      _materialTextures.push_back(fallback);
      continue;
    }

    MTL::TextureDescriptor *descriptor = MTL::TextureDescriptor::alloc()->init();
    descriptor->setTextureType(MTL::TextureType::TextureType2D);
    descriptor->setPixelFormat(MTL::PixelFormat::PixelFormatRGBA32Float);
    descriptor->setWidth(info.width);
    descriptor->setHeight(info.height);
    descriptor->setUsage(MTL::TextureUsageShaderRead);
    descriptor->setStorageMode(MTL::StorageMode::StorageModeManaged);

    MTL::Texture *texture =
        allocateTexture(descriptor, GpuMemoryTracker::Category::Textures,
                        "MaterialTexture");
    descriptor->release();

    if (!texture) {
      std::printf(
          "[Renderer] Failed to allocate material texture %zu (%ux%u).\n",
          texIndex, info.width, info.height);
      MTL::Texture *fallback = createFallbackTexture();
      _materialTextures.push_back(fallback);
      continue;
    }

    const simd::float4 *src = _cachedTextureData.data() + info.offset;
    NS::UInteger bytesPerRow =
        static_cast<NS::UInteger>(info.width * sizeof(simd::float4));
    MTL::Region region = MTL::Region::Make2D(0, 0, info.width, info.height);
    texture->replaceRegion(region, 0, src, bytesPerRow);

    _materialTextures.push_back(texture);
  }
}

void Renderer::buildTextures() {
  NS::UInteger width = std::max<NS::UInteger>(
      1, static_cast<NS::UInteger>(Camera::screenSize.x));
  NS::UInteger height = std::max<NS::UInteger>(
      1, static_cast<NS::UInteger>(Camera::screenSize.y));

  MTL::TextureUsage usage = static_cast<MTL::TextureUsage>(
      MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
  MTL::TextureUsage colorUsage = static_cast<MTL::TextureUsage>(
      usage | MTL::TextureUsageRenderTarget);

  configureTextureSlot(_colorSlot, width, height,
                       MTL::PixelFormat::PixelFormatRGBA32Float, colorUsage);
  configureTextureSlot(_albedoSlot, width, height,
                       MTL::PixelFormat::PixelFormatRGBA32Float, usage);
  configureTextureSlot(_normalSlot, width, height,
                       MTL::PixelFormat::PixelFormatRGBA32Float, usage);
  configureTextureSlot(_positionSlot, width, height,
                       MTL::PixelFormat::PixelFormatRGBA32Float, usage);
  configureTextureSlot(_albedoHistorySlot, width, height,
                       MTL::PixelFormat::PixelFormatRGBA32Float, usage);
  configureTextureSlot(_normalHistorySlot, width, height,
                       MTL::PixelFormat::PixelFormatRGBA32Float, usage);
  configureTextureSlot(_positionHistorySlot, width, height,
                       MTL::PixelFormat::PixelFormatRGBA32Float, usage);
  for (size_t i = 0; i < _restirData0Slots.size(); ++i) {
    configureTextureSlot(_restirData0Slots[i], width, height,
                         MTL::PixelFormat::PixelFormatRGBA32Float, usage);
    configureTextureSlot(_restirData1Slots[i], width, height,
                         MTL::PixelFormat::PixelFormatRGBA32Float, usage);
    configureTextureSlot(_restirData2Slots[i], width, height,
                         MTL::PixelFormat::PixelFormatRGBA32Float, usage);
    configureTextureSlot(_restirSurfacePosSlots[i], width, height,
                         MTL::PixelFormat::PixelFormatRGBA32Float, usage);
    configureTextureSlot(_restirSurfaceNormalSlots[i], width, height,
                         MTL::PixelFormat::PixelFormatRGBA32Float, usage);
  }
}

} // namespace NomadPathTracer
