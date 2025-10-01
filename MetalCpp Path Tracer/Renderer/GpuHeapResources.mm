#include "GpuHeapResources.h"

#include <Foundation/Foundation.hpp>
#include <algorithm>

namespace MetalCppPathTracer {

namespace {
constexpr NS::UInteger kDefaultHeapSizeBytes = 4 * 1024 * 1024; // 4 MB
constexpr NS::UInteger kHeapAlignment = 256;

NS::UInteger alignUp(NS::UInteger value, NS::UInteger alignment) {
  if (alignment == 0)
    return value;
  return (value + alignment - 1) & ~(alignment - 1);
}

} // namespace

GpuHeapResources::~GpuHeapResources() { destroy(); }

void GpuHeapResources::initialize(MTL::Device *device, NS::UInteger heapSize,
                                  MTL::StorageMode storageMode,
                                  MTL::HazardTrackingMode hazardMode) {
  if (!device)
    return;

  if (_device && _device != device) {
    destroy();
  }

  if (!_device) {
    _device = device->retain();
  }

  _storageMode = storageMode;
  _hazardMode = hazardMode;
  _defaultHeapSize = heapSize == 0 ? kDefaultHeapSizeBytes : heapSize;

  if (_heap)
    return;

  recreateHeap(_defaultHeapSize);
}

void GpuHeapResources::destroy() {
  releaseBuffer(_vertex);
  releaseBuffer(_index);

  if (_heap) {
    _heap->release();
    _heap = nullptr;
  }

  if (_device) {
    _device->release();
    _device = nullptr;
  }

  _heapSize = 0;
  _defaultHeapSize = 0;
}

MTL::Buffer *GpuHeapResources::ensureOnHeapBuffer(BufferKind kind,
                                                  NS::UInteger requiredBytes,
                                                  MTL::ResourceOptions options,
                                                  MTL::ResourceUsage usage,
                                                  const char *label) {
  if (!_device)
    return nullptr;

  if (!_heap) {
    recreateHeap(std::max(_defaultHeapSize, alignForHeap(requiredBytes)));
  }

  BufferInfo &info = bufferInfo(kind);

  if (info.buffer && requiredBytes <= info.capacity && info.options == options &&
      info.usage == usage) {
    return info.buffer;
  }

  NS::UInteger requiredCapacity = alignForHeap(requiredBytes);
  if (requiredCapacity > _heapSize) {
    NS::UInteger newSize = std::max(requiredCapacity, _heapSize);
    if (newSize == 0)
      newSize = _defaultHeapSize == 0 ? kDefaultHeapSizeBytes : _defaultHeapSize;

    while (newSize < requiredCapacity) {
      newSize *= 2;
    }

    recreateHeap(newSize);
  }

  releaseBuffer(info);

  if (!_heap)
    return nullptr;

  MTL::Buffer *buffer = _heap->newBuffer(requiredCapacity, options);
  if (!buffer)
    return nullptr;

  if (label) {
    using NS::StringEncoding::UTF8StringEncoding;
    buffer->setLabel(NS::String::string(label, UTF8StringEncoding));
  }

  info.buffer = buffer;
  info.capacity = requiredCapacity;
  info.options = options;
  info.usage = usage;

  return buffer;
}

void GpuHeapResources::releaseBuffer(BufferInfo &info) {
  if (info.buffer) {
    info.buffer->release();
    info.buffer = nullptr;
  }
  info.capacity = 0;
  info.options = MTL::ResourceStorageModePrivate;
  info.usage = MTL::ResourceUsage(0);
}

GpuHeapResources::BufferInfo &GpuHeapResources::bufferInfo(BufferKind kind) {
  switch (kind) {
  case BufferKind::Vertex:
    return _vertex;
  case BufferKind::Index:
    return _index;
  }
  return _vertex;
}

NS::UInteger GpuHeapResources::alignForHeap(NS::UInteger size) const {
  if (size == 0)
    return alignUp(_defaultHeapSize == 0 ? kDefaultHeapSizeBytes : _defaultHeapSize,
                   kHeapAlignment);
  return alignUp(size, kHeapAlignment);
}

void GpuHeapResources::recreateHeap(NS::UInteger newSize) {
  releaseBuffer(_vertex);
  releaseBuffer(_index);

  if (_heap) {
    _heap->release();
    _heap = nullptr;
  }

  if (!_device) {
    _heapSize = 0;
    return;
  }

  MTL::HeapDescriptor *desc = MTL::HeapDescriptor::alloc()->init();
  desc->setSize(newSize);
  desc->setStorageMode(_storageMode);
  desc->setHazardTrackingMode(_hazardMode);

  _heap = _device->newHeap(desc);
  desc->release();

  _heapSize = _heap ? _heap->size() : 0;
}

} // namespace MetalCppPathTracer

