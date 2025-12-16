#pragma once

#include <Metal/Metal.hpp>
#include <cstddef>

namespace NomadPathTracer {

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

  void ensureHeapCapacity(NS::UInteger requiredBytes);
  NS::UInteger alignedHeapSize(NS::UInteger size) const;

  void releaseAllAllocations();
  void makeResourcesPurgeable();

  MTL::Buffer *ensureOnHeapBuffer(BufferKind kind,
                                  NS::UInteger requiredBytes,
                                  MTL::ResourceOptions options,
                                  MTL::ResourceUsage usage,
                                  const char *label = nullptr);

  MTL::Buffer *ensureVertexBuffer(
      NS::UInteger requiredBytes, const char *label = nullptr,
      MTL::ResourceOptions options = MTL::ResourceStorageModePrivate,
      MTL::ResourceUsage usage =
          static_cast<MTL::ResourceUsage>(MTL::ResourceUsageRead));
  MTL::Buffer *ensureIndexBuffer(
      NS::UInteger requiredBytes, const char *label = nullptr,
      MTL::ResourceOptions options = MTL::ResourceStorageModePrivate,
      MTL::ResourceUsage usage =
          static_cast<MTL::ResourceUsage>(MTL::ResourceUsageRead));
  MTL::AccelerationStructure *ensureAccelerationStructure(
      NS::UInteger requiredSize, const char *label = nullptr);

  MTL::Heap *heap() const { return _heap; }
  MTL::Device *device() const { return _device; }

  MTL::Buffer *vertexBuffer() const { return _vertex.buffer; }
  MTL::Buffer *indexBuffer() const { return _index.buffer; }
  NS::UInteger vertexCapacity() const { return _vertex.capacity; }
  NS::UInteger indexCapacity() const { return _index.capacity; }
  MTL::AccelerationStructure *accelerationStructure() const {
    return _accelerationStructure;
  }
  NS::UInteger accelerationStructureCapacity() const {
    return _accelerationSize;
  }

  static NS::UInteger primitiveDataSize(size_t primitiveCount);

private:
  struct BufferInfo {
    MTL::Buffer *buffer = nullptr;
    NS::UInteger capacity = 0;
    MTL::ResourceOptions options = MTL::ResourceStorageModePrivate;
    MTL::ResourceUsage usage = static_cast<MTL::ResourceUsage>(0);
  };

  void releaseBuffer(BufferInfo &info);
  void releaseAccelerationStructure();
  BufferInfo &bufferInfo(BufferKind kind);
  NS::UInteger alignForHeap(NS::UInteger size) const;
  void recreateHeap(NS::UInteger newSize);
  void tryDestroyHeap();

  MTL::Device *_device = nullptr;
  MTL::Heap *_heap = nullptr;
  BufferInfo _vertex;
  BufferInfo _index;
  MTL::AccelerationStructure *_accelerationStructure = nullptr;
  NS::UInteger _heapSize = 0;
  NS::UInteger _defaultHeapSize = 0;
  NS::UInteger _accelerationSize = 0;
  MTL::StorageMode _storageMode = MTL::StorageMode::StorageModePrivate;
  MTL::HazardTrackingMode _hazardMode = MTL::HazardTrackingModeTracked;
};

} // namespace NomadPathTracer

