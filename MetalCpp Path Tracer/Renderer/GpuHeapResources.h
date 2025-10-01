#pragma once

#include <Metal/Metal.hpp>
#include <cstddef>

namespace MetalCppPathTracer {

class GpuHeapResources {
public:
  enum class BufferKind { Vertex, Index };

  GpuHeapResources() = default;
  ~GpuHeapResources();

  void initialize(MTL::Device *device,
                  NS::UInteger heapSize = 0,
                  MTL::StorageMode storageMode =
                      MTL::StorageMode::StorageModePrivate,
                  MTL::HazardTrackingMode hazardMode =
                      MTL::HazardTrackingModeTracked);
  void destroy();

  MTL::Buffer *ensureOnHeapBuffer(BufferKind kind,
                                  NS::UInteger requiredBytes,
                                  MTL::ResourceOptions options,
                                  MTL::ResourceUsage usage,
                                  const char *label = nullptr);

  MTL::Heap *heap() const { return _heap; }
  MTL::Device *device() const { return _device; }

  MTL::Buffer *vertexBuffer() const { return _vertex.buffer; }
  MTL::Buffer *indexBuffer() const { return _index.buffer; }
  NS::UInteger vertexCapacity() const { return _vertex.capacity; }
  NS::UInteger indexCapacity() const { return _index.capacity; }

private:
  struct BufferInfo {
    MTL::Buffer *buffer = nullptr;
    NS::UInteger capacity = 0;
    MTL::ResourceOptions options = MTL::ResourceStorageModePrivate;
    MTL::ResourceUsage usage = MTL::ResourceUsage(0);
  };

  void releaseBuffer(BufferInfo &info);
  BufferInfo &bufferInfo(BufferKind kind);
  NS::UInteger alignForHeap(NS::UInteger size) const;
  void recreateHeap(NS::UInteger newSize);

  MTL::Device *_device = nullptr;
  MTL::Heap *_heap = nullptr;
  BufferInfo _vertex;
  BufferInfo _index;
  NS::UInteger _heapSize = 0;
  NS::UInteger _defaultHeapSize = 0;
  MTL::StorageMode _storageMode = MTL::StorageMode::StorageModePrivate;
  MTL::HazardTrackingMode _hazardMode = MTL::HazardTrackingModeTracked;
};

} // namespace MetalCppPathTracer

