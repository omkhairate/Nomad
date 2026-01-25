#include "GpuHeapResources.h"

#include <Foundation/Foundation.hpp>
#include <algorithm>

#include "../Scene/Primitive.h"

namespace NomadPathTracer {

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

void GpuHeapResources::setMemoryTracker(GpuMemoryTracker *tracker,
                                        GpuMemoryTracker::Category category) {
  if (_memoryTracker == tracker && _heapCategory == category)
    return;

  if (_memoryTracker && _heap) {
    _memoryTracker->releaseAllocation(_heap);
  }

  _memoryTracker = tracker;
  _heapCategory = category;

  if (_memoryTracker && _heap) {
    _memoryTracker->recordAllocation(_heapCategory, _heap, _heap->size());
  }
}

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
}

void GpuHeapResources::destroy() {
  releaseBuffer(_vertex);
  releaseBuffer(_index);
  releaseAccelerationStructure();

  if (_heap) {
    if (_memoryTracker)
      _memoryTracker->releaseAllocation(_heap);
    _heap->release();
    _heap = nullptr;
  }

  if (_device) {
    _device->release();
    _device = nullptr;
  }

  _heapSize = 0;
  _defaultHeapSize = 0;
  _accelerationSize = 0;
}

void GpuHeapResources::ensureHeapCapacity(NS::UInteger requiredBytes) {
  if (!_device)
    return;

  if (requiredBytes == 0)
    requiredBytes = alignForHeap(requiredBytes);

  if (_heap && requiredBytes <= _heapSize)
    return;

  NS::UInteger newSize = std::max(requiredBytes, _heapSize);
  if (newSize == 0)
    newSize = _defaultHeapSize == 0 ? kDefaultHeapSizeBytes : _defaultHeapSize;

  while (newSize < requiredBytes) {
    newSize *= 2;
  }

  recreateHeap(newSize);
}

NS::UInteger GpuHeapResources::alignedHeapSize(NS::UInteger size) const {
  return alignForHeap(size);
}

void GpuHeapResources::releaseAllAllocations() {
  releaseBuffer(_vertex);
  releaseBuffer(_index);
  releaseAccelerationStructure();
  tryDestroyHeap();
}

void GpuHeapResources::makeResourcesPurgeable() {
  if (_heap) {
    _heap->setPurgeableState(MTL::PurgeableStateEmpty);
    return;
  }

  if (_vertex.buffer)
    _vertex.buffer->setPurgeableState(MTL::PurgeableStateEmpty);
  if (_index.buffer)
    _index.buffer->setPurgeableState(MTL::PurgeableStateEmpty);
  if (_accelerationStructure)
    _accelerationStructure->setPurgeableState(MTL::PurgeableStateEmpty);
}

MTL::Buffer *GpuHeapResources::ensureOnHeapBuffer(BufferKind kind,
                                                  NS::UInteger requiredBytes,
                                                  MTL::ResourceOptions options,
                                                  MTL::ResourceUsage usage,
                                                  const char *label) {
  if (!_device)
    return nullptr;

  if (!_heap) {
    NS::UInteger initialSize = alignForHeap(requiredBytes);
    if (initialSize == 0) {
      NS::UInteger fallback = (_defaultHeapSize == 0) ? kDefaultHeapSizeBytes
                                                      : _defaultHeapSize;
      initialSize = alignUp(fallback, kHeapAlignment);
    }
    recreateHeap(initialSize);
  }

  BufferInfo &info = bufferInfo(kind);

  if (info.buffer && requiredBytes <= info.capacity && info.options == options &&
      info.usage == usage) {
    return info.buffer;
  }

  NS::UInteger requiredCapacity = alignForHeap(requiredBytes);
  if (requiredCapacity == 0) {
    NS::UInteger fallback = (_defaultHeapSize == 0) ? kDefaultHeapSizeBytes
                                                    : _defaultHeapSize;
    requiredCapacity = alignUp(fallback, kHeapAlignment);
  }
  ensureHeapCapacity(requiredCapacity);

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

MTL::Buffer *GpuHeapResources::ensureVertexBuffer(NS::UInteger requiredBytes,
                                                  const char *label,
                                                  MTL::ResourceOptions options,
                                                  MTL::ResourceUsage usage) {
  return ensureOnHeapBuffer(BufferKind::Vertex, requiredBytes, options, usage,
                            label);
}

MTL::Buffer *GpuHeapResources::ensureIndexBuffer(NS::UInteger requiredBytes,
                                                 const char *label,
                                                 MTL::ResourceOptions options,
                                                 MTL::ResourceUsage usage) {
  return ensureOnHeapBuffer(BufferKind::Index, requiredBytes, options, usage,
                            label);
}

MTL::AccelerationStructure *
GpuHeapResources::ensureAccelerationStructure(NS::UInteger requiredSize,
                                              const char *label) {
  if (!_device)
    return nullptr;

  if (requiredSize == 0) {
    releaseAccelerationStructure();
    _accelerationSize = 0;
    return nullptr;
  }

  MTL::SizeAndAlign alignInfo =
      _device->heapAccelerationStructureSizeAndAlign(requiredSize);
  NS::UInteger alignedSize = alignInfo.size;
  if (alignInfo.align > 0)
    alignedSize = alignUp(alignedSize, alignInfo.align);
  alignedSize = alignForHeap(alignedSize);

  ensureHeapCapacity(alignedSize);

  releaseAccelerationStructure();

  if (!_heap)
    return nullptr;

  MTL::AccelerationStructure *structure =
      _heap->newAccelerationStructure(alignedSize);
  if (!structure)
    return nullptr;

  if (label) {
    using NS::StringEncoding::UTF8StringEncoding;
    structure->setLabel(NS::String::string(label, UTF8StringEncoding));
  }

  _accelerationStructure = structure;
  _accelerationSize = alignedSize;
  return structure;
}

void GpuHeapResources::releaseBuffer(BufferInfo &info) {
  if (info.buffer) {
    info.buffer->release();
    info.buffer = nullptr;
  }
  info.capacity = 0;
  info.options = MTL::ResourceStorageModePrivate;
  info.usage = static_cast<MTL::ResourceUsage>(0);
}

void GpuHeapResources::releaseAccelerationStructure() {
  if (_accelerationStructure) {
    _accelerationStructure->release();
    _accelerationStructure = nullptr;
  }
  _accelerationSize = 0;
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
    return 0;
  return alignUp(size, kHeapAlignment);
}

void GpuHeapResources::recreateHeap(NS::UInteger newSize) {
  releaseBuffer(_vertex);
  releaseBuffer(_index);
  releaseAccelerationStructure();

  if (_heap) {
    if (_memoryTracker)
      _memoryTracker->releaseAllocation(_heap);
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
  if (_memoryTracker && _heap)
    _memoryTracker->recordAllocation(_heapCategory, _heap, _heapSize);
}

void GpuHeapResources::tryDestroyHeap() {
  if (!_heap)
    return;
  if (_vertex.buffer || _index.buffer || _accelerationStructure)
    return;
  if (_memoryTracker)
    _memoryTracker->releaseAllocation(_heap);
  _heap->release();
  _heap = nullptr;
  _heapSize = 0;
}

NS::UInteger GpuHeapResources::primitiveDataSize(size_t primitiveCount) {
  if (primitiveCount == 0)
    return 0;
  size_t float4Count = primitiveCount * kPrimitiveFloat4Count;
  const size_t kFloat4Size = sizeof(float) * 4;
  size_t byteCount = float4Count * kFloat4Size;
  return static_cast<NS::UInteger>(byteCount);
}

} // namespace NomadPathTracer
