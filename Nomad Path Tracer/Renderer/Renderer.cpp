#include "Renderer.h"

#include "Camera.h"
#include "IncrementalUpdateUtils.h"
#include "InputSystem.h"
#include "ParallelFor.h"
#include "Scene.h"
#include "SceneLoader.h"
#include <Metal/MTLArgument.hpp>
#include <Metal/MTLComputePipeline.hpp>
#include <algorithm>
#include <array>
#include <cassert>
#include <CoreFoundation/CoreFoundation.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <dlfcn.h>
#include <dispatch/dispatch.h>
#include <thread>
#include <limits>
#include <functional>
#include <type_traits>
#include <utility>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <future>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <queue>
#include <atomic>
#include <deque>
#include <simd/simd.h>
#include <sstream>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../textures/EnvironmentTexture.h"

namespace {
class TaskLimiter {
public:
  explicit TaskLimiter(size_t maxTasks) : _maxTasks(maxTasks) {}

  void acquire() {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [&] { return _active < _maxTasks; });
    ++_active;
  }

  void release() {
    std::unique_lock<std::mutex> lock(_mutex);
    if (_active > 0) {
      --_active;
      lock.unlock();
      _cv.notify_one();
    }
  }

private:
  size_t _maxTasks = 0;
  size_t _active = 0;
  std::mutex _mutex;
  std::condition_variable _cv;
};

TaskLimiter &sceneBvhTaskLimiter() {
  const size_t hardwareThreads =
      std::max<size_t>(std::thread::hardware_concurrency(), 2);
  constexpr size_t kMinSceneBvhTasks = 4;
  const size_t maxTasks = std::max<size_t>(hardwareThreads, kMinSceneBvhTasks);
  static TaskLimiter limiter(maxTasks);
  return limiter;
}
} // namespace

namespace {
bool parseEnvBool(const char *value) {
  if (!value || !value[0])
    return false;
  if (std::isdigit(static_cast<unsigned char>(value[0])))
    return std::strtoul(value, nullptr, 10) != 0;
  char first =
      static_cast<char>(std::tolower(static_cast<unsigned char>(value[0])));
  return first == 't' || first == 'y' || first == 'o';
}

uint32_t sanitizeDebugAccelerationModeForUniforms() {
  using DebugMode = NomadPathTracer::InputSystem::DebugAccelerationMode;
  DebugMode mode = NomadPathTracer::InputSystem::debugAccelerationMode;
  if (NomadPathTracer::InputSystem::isExposedDebugAccelerationMode(mode)) {
    return NomadPathTracer::InputSystem::rawDebugAccelerationMode(mode);
  }

  static bool warnedAboutNonExposedMode = false;
  if (mode != DebugMode::None && !warnedAboutNonExposedMode) {
    warnedAboutNonExposedMode = true;
    std::printf(
        "[Renderer] Ignoring non-exposed debug acceleration mode %u during "
        "normal rendering.\n",
        NomadPathTracer::InputSystem::rawDebugAccelerationMode(mode));
  }

  NomadPathTracer::InputSystem::setDebugAccelerationMode(DebugMode::None);
  return NomadPathTracer::InputSystem::rawDebugAccelerationMode(
      DebugMode::None);
}

struct TinyJsonValue {
  enum class Type { Null, Bool, Number, String, Array, Object };

  Type type = Type::Null;
  bool boolValue = false;
  double numberValue = 0.0;
  std::string stringValue;
  std::vector<TinyJsonValue> arrayValue;
  std::map<std::string, TinyJsonValue> objectValue;
};

class TinyJsonParser {
public:
  explicit TinyJsonParser(const std::string &text) : _text(text) {}

  bool parse(TinyJsonValue &outValue, std::string &outError) {
    _pos = 0;
    outError.clear();
    skipWhitespace();
    if (!parseValue(outValue, outError))
      return false;
    skipWhitespace();
    if (_pos != _text.size()) {
      outError = "Unexpected trailing JSON content.";
      return false;
    }
    return true;
  }

private:
  bool parseValue(TinyJsonValue &outValue, std::string &outError) {
    skipWhitespace();
    if (_pos >= _text.size()) {
      outError = "Unexpected end of JSON input.";
      return false;
    }

    const char c = _text[_pos];
    if (c == '{')
      return parseObject(outValue, outError);
    if (c == '[')
      return parseArray(outValue, outError);
    if (c == '"')
      return parseStringValue(outValue, outError);
    if (c == '-' || std::isdigit(static_cast<unsigned char>(c)))
      return parseNumber(outValue, outError);
    if (matchLiteral("true")) {
      outValue = TinyJsonValue{};
      outValue.type = TinyJsonValue::Type::Bool;
      outValue.boolValue = true;
      return true;
    }
    if (matchLiteral("false")) {
      outValue = TinyJsonValue{};
      outValue.type = TinyJsonValue::Type::Bool;
      outValue.boolValue = false;
      return true;
    }
    if (matchLiteral("null")) {
      outValue = TinyJsonValue{};
      outValue.type = TinyJsonValue::Type::Null;
      return true;
    }

    outError = "Unexpected token while parsing JSON.";
    return false;
  }

  bool parseObject(TinyJsonValue &outValue, std::string &outError) {
    if (!consume('{', outError))
      return false;

    outValue = TinyJsonValue{};
    outValue.type = TinyJsonValue::Type::Object;
    skipWhitespace();
    if (peek('}')) {
      ++_pos;
      return true;
    }

    while (true) {
      TinyJsonValue keyValue;
      if (!parseStringValue(keyValue, outError))
        return false;
      skipWhitespace();
      if (!consume(':', outError))
        return false;
      TinyJsonValue value;
      if (!parseValue(value, outError))
        return false;
      outValue.objectValue[keyValue.stringValue] = std::move(value);
      skipWhitespace();
      if (peek('}')) {
        ++_pos;
        return true;
      }
      if (!consume(',', outError))
        return false;
      skipWhitespace();
    }
  }

  bool parseArray(TinyJsonValue &outValue, std::string &outError) {
    if (!consume('[', outError))
      return false;

    outValue = TinyJsonValue{};
    outValue.type = TinyJsonValue::Type::Array;
    skipWhitespace();
    if (peek(']')) {
      ++_pos;
      return true;
    }

    while (true) {
      TinyJsonValue value;
      if (!parseValue(value, outError))
        return false;
      outValue.arrayValue.push_back(std::move(value));
      skipWhitespace();
      if (peek(']')) {
        ++_pos;
        return true;
      }
      if (!consume(',', outError))
        return false;
      skipWhitespace();
    }
  }

  bool parseStringValue(TinyJsonValue &outValue, std::string &outError) {
    std::string parsed;
    if (!parseString(parsed, outError))
      return false;
    outValue = TinyJsonValue{};
    outValue.type = TinyJsonValue::Type::String;
    outValue.stringValue = std::move(parsed);
    return true;
  }

  bool parseString(std::string &outString, std::string &outError) {
    if (!consume('"', outError))
      return false;

    outString.clear();
    while (_pos < _text.size()) {
      char c = _text[_pos++];
      if (c == '"')
        return true;
      if (c == '\\') {
        if (_pos >= _text.size()) {
          outError = "Incomplete JSON escape sequence.";
          return false;
        }
        char escaped = _text[_pos++];
        switch (escaped) {
        case '"':
        case '\\':
        case '/':
          outString.push_back(escaped);
          break;
        case 'b':
          outString.push_back('\b');
          break;
        case 'f':
          outString.push_back('\f');
          break;
        case 'n':
          outString.push_back('\n');
          break;
        case 'r':
          outString.push_back('\r');
          break;
        case 't':
          outString.push_back('\t');
          break;
        case 'u':
          if (_pos + 4 > _text.size()) {
            outError = "Incomplete JSON unicode escape.";
            return false;
          }
          outString.push_back('?');
          _pos += 4;
          break;
        default:
          outError = "Unsupported JSON escape sequence.";
          return false;
        }
      } else {
        outString.push_back(c);
      }
    }

    outError = "Unterminated JSON string.";
    return false;
  }

  bool parseNumber(TinyJsonValue &outValue, std::string &outError) {
    const size_t start = _pos;
    if (_text[_pos] == '-')
      ++_pos;

    if (_pos >= _text.size()) {
      outError = "Unexpected end of numeric literal.";
      return false;
    }

    if (_text[_pos] == '0') {
      ++_pos;
    } else {
      if (!std::isdigit(static_cast<unsigned char>(_text[_pos]))) {
        outError = "Invalid numeric literal.";
        return false;
      }
      while (_pos < _text.size() &&
             std::isdigit(static_cast<unsigned char>(_text[_pos]))) {
        ++_pos;
      }
    }

    if (_pos < _text.size() && _text[_pos] == '.') {
      ++_pos;
      if (_pos >= _text.size() ||
          !std::isdigit(static_cast<unsigned char>(_text[_pos]))) {
        outError = "Invalid fractional numeric literal.";
        return false;
      }
      while (_pos < _text.size() &&
             std::isdigit(static_cast<unsigned char>(_text[_pos]))) {
        ++_pos;
      }
    }

    if (_pos < _text.size() && (_text[_pos] == 'e' || _text[_pos] == 'E')) {
      ++_pos;
      if (_pos < _text.size() && (_text[_pos] == '+' || _text[_pos] == '-'))
        ++_pos;
      if (_pos >= _text.size() ||
          !std::isdigit(static_cast<unsigned char>(_text[_pos]))) {
        outError = "Invalid exponent in numeric literal.";
        return false;
      }
      while (_pos < _text.size() &&
             std::isdigit(static_cast<unsigned char>(_text[_pos]))) {
        ++_pos;
      }
    }

    try {
      outValue = TinyJsonValue{};
      outValue.type = TinyJsonValue::Type::Number;
      outValue.numberValue =
          std::stod(_text.substr(start, _pos - start));
      return true;
    } catch (const std::exception &) {
      outError = "Failed to parse numeric literal.";
      return false;
    }
  }

  bool matchLiteral(const char *literal) {
    size_t len = std::strlen(literal);
    if (_text.compare(_pos, len, literal) != 0)
      return false;
    _pos += len;
    return true;
  }

  bool consume(char expected, std::string &outError) {
    skipWhitespace();
    if (_pos >= _text.size() || _text[_pos] != expected) {
      std::ostringstream ss;
      ss << "Expected '" << expected << "' while parsing JSON.";
      outError = ss.str();
      return false;
    }
    ++_pos;
    return true;
  }

  bool peek(char expected) const {
    return _pos < _text.size() && _text[_pos] == expected;
  }

  void skipWhitespace() {
    while (_pos < _text.size() &&
           std::isspace(static_cast<unsigned char>(_text[_pos]))) {
      ++_pos;
    }
  }

  const std::string &_text;
  size_t _pos = 0;
};

constexpr float kIdleVisibleExploreSeed = 1.0f;
constexpr size_t kTotalMemoryCapStallFrames = 8;
constexpr size_t kTotalMemoryEvictionNoProgressLimit = 3;
constexpr size_t kTotalMemoryEvictionBackoffFrames = 2;
}

namespace NomadPathTracer {

ResidentObjectGpuResources::ResidentObjectGpuResources(
    ResidentObjectGpuResources &&other) noexcept {
  std::unique_lock<std::mutex> otherLock;
  if (other.pendingCommandState) {
    otherLock = std::unique_lock<std::mutex>(other.pendingCommandState->mutex);
  }
  resources = std::move(other.resources);
  byteSize = other.byteSize;
  triangleCount = other.triangleCount;
  vertexCount = other.vertexCount;
  vertexBufferOffset = other.vertexBufferOffset;
  indexBufferOffset = other.indexBufferOffset;
  geometryValid = other.geometryValid;
  state = other.state;
  lastStateChange = other.lastStateChange;
  pendingCommandState = std::move(other.pendingCommandState);
}

ResidentObjectGpuResources &
ResidentObjectGpuResources::operator=(ResidentObjectGpuResources &&other) noexcept {
  if (this == &other)
    return *this;

  std::unique_lock<std::mutex> thisLock;
  std::unique_lock<std::mutex> otherLock;
  if (pendingCommandState && other.pendingCommandState) {
    thisLock = std::unique_lock<std::mutex>(pendingCommandState->mutex,
                                            std::defer_lock);
    otherLock = std::unique_lock<std::mutex>(other.pendingCommandState->mutex,
                                             std::defer_lock);
    std::lock(thisLock, otherLock);
  } else if (pendingCommandState) {
    thisLock = std::unique_lock<std::mutex>(pendingCommandState->mutex);
  } else if (other.pendingCommandState) {
    otherLock = std::unique_lock<std::mutex>(other.pendingCommandState->mutex);
  }
  resources = std::move(other.resources);
  byteSize = other.byteSize;
  triangleCount = other.triangleCount;
  vertexCount = other.vertexCount;
  vertexBufferOffset = other.vertexBufferOffset;
  indexBufferOffset = other.indexBufferOffset;
  geometryValid = other.geometryValid;
  state = other.state;
  lastStateChange = other.lastStateChange;
  pendingCommandState = std::move(other.pendingCommandState);
  return *this;
}

void ResidentObjectGpuResources::clearPendingCommand() {
  if (!pendingCommandState)
    return;

  std::lock_guard<std::mutex> lock(pendingCommandState->mutex);
  using Status = MTL::CommandBufferStatus;

  auto isComplete = [](PendingCommand &pending) {
    if (pending.completed || pending.error)
      return true;

    if (!pending.command)
      return true;

    auto status = pending.command->status();
    if (status == Status::CommandBufferStatusCompleted) {
      pending.completed = true;
      return true;
    }
    if (status == Status::CommandBufferStatusError) {
      pending.error = true;
      return true;
    }
    return false;
  };

  auto releaseCommand = [](PendingCommand &pending) {
    if (pending.command) {
      pending.command->release();
      pending.command = nullptr;
    }
  };

  auto &commands = pendingCommandState->commands;
  commands.erase(
      std::remove_if(commands.begin(), commands.end(),
                     [&](PendingCommand &pending) {
                       if (!isComplete(pending))
                         return false;
                       releaseCommand(pending);
                       return true;
                     }),
      commands.end());
}

void ResidentObjectGpuResources::transitionToStreaming(
    MTL::CommandBuffer *pending) {
  clearPendingCommand();
  if (pending) {
    PendingCommand pendingRecord{};
    pendingRecord.command = pending;
    pendingRecord.command->retain();

    std::weak_ptr<PendingCommandState> weakState = pendingCommandState;
    pendingRecord.command->addCompletedHandler(
        [weakState](MTL::CommandBuffer *cmd) {
          auto state = weakState.lock();
          if (!state)
            return;

          std::lock_guard<std::mutex> lock(state->mutex);
          for (auto &pending : state->commands) {
            if (pending.command == cmd) {
              auto status = cmd->status();
              pending.completed =
                  status == MTL::CommandBufferStatusCompleted;
              pending.error = status == MTL::CommandBufferStatusError;
              break;
            }
          }
        });

    if (pendingCommandState) {
      std::lock_guard<std::mutex> lock(pendingCommandState->mutex);
      pendingCommandState->commands.push_back(std::move(pendingRecord));
    }
  }
  state = ResidencyState::Streaming;
  lastStateChange = std::chrono::steady_clock::now();
}

bool ResidentObjectGpuResources::hasPendingCommands() {
  if (!pendingCommandState)
    return false;

  std::lock_guard<std::mutex> lock(pendingCommandState->mutex);
  for (auto &pending : pendingCommandState->commands) {
    if (pending.completed || pending.error)
      continue;

    auto status = pending.command ? pending.command->status()
                                  : MTL::CommandBufferStatusCompleted;
    if (status == MTL::CommandBufferStatusCompleted) {
      pending.completed = true;
      continue;
    }
    if (status == MTL::CommandBufferStatusError) {
      pending.error = true;
      continue;
    }

    return true;
  }
  return false;
}

bool ResidentObjectGpuResources::waitForPendingCommands(
    std::chrono::milliseconds timeout) {
  using namespace std::chrono;
  const auto deadline = steady_clock::now() + timeout;
  while (hasPendingCommands()) {
    if (timeout.count() == 0 || steady_clock::now() >= deadline)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  clearPendingCommand();
  return true;
}

bool ResidentObjectGpuResources::transitionToCold(
    BlasInstanceRecord &instanceRecord) {
  clearPendingCommand();
  if (hasPendingCommands())
    return false;
  resources.makeResourcesPurgeable();
  resources.releaseAllAllocations();
  byteSize = 0;
  triangleCount = 0;
  vertexCount = 0;
  vertexBufferOffset = 0;
  indexBufferOffset = 0;
  geometryValid = false;
  state = ResidencyState::Cold;
  lastStateChange = std::chrono::steady_clock::now();
  instanceRecord.primitiveBase = 0;
  instanceRecord.primitiveIndexBase = 0;
  instanceRecord.blasRootIndex = -1;
  instanceRecord.primitiveCount = 0;
  return true;
}

bool ResidentObjectGpuResources::ensureResident(
    Renderer &renderer, size_t objectIndex, const SceneObject &object,
    BlasInstanceRecord &instanceRecord, bool forceRebuild) {
  renderer.cancelPendingResidentEviction(objectIndex, *this);

  const auto previousState = state;
  const auto previousStateChange = lastStateChange;
  const bool previousGeometryValid = geometryValid;

  if (isResident() && !forceRebuild) {
    lastStateChange = std::chrono::steady_clock::now();
    return true;
  }

  constexpr auto kPendingCommandWait =
      std::chrono::milliseconds(50);
  if (hasPendingCommands()) {
    if (!waitForPendingCommands(kPendingCommandWait)) {
      std::printf(
          "[BLAS] Deferred rebuild for object %zu: pending command buffer "
          "still in flight.\n",
          objectIndex);
      lastStateChange = std::chrono::steady_clock::now();
      return true;
    }
  }

  transitionToStreaming();
  geometryValid = false;
  bool built = renderer.buildObjectBlas(objectIndex, object, *this,
                                        previousState == ResidencyState::Cold);
  if (!built) {
    if (renderer.isAlwaysResidentStrategy()) {
      state = previousState;
      geometryValid = previousGeometryValid;
      lastStateChange = previousStateChange;
      return false;
    }
    renderer.transitionResidentToCold(objectIndex, *this, instanceRecord);
    return false;
  }
  return true;
}

} // namespace NomadPathTracer

using namespace NomadPathTracer;

ParallelForConfig primitivePackingConfig(size_t primitiveCount) {
  // Derived from empirical packing timings: keep tiny scenes on a single core
  // to avoid synchronization overhead, then scale aggressively for larger
  // batches where memory traffic dominates.
  ParallelForConfig config{};
  if (primitiveCount < 2048) {
    config.minChunkSize = 64;
    config.preferredChunkSize = 128;
  } else if (primitiveCount < 32768) {
    config.minChunkSize = 256;
    config.preferredChunkSize = 512;
  } else {
    config.minChunkSize = 512;
    config.preferredChunkSize = 1024;
  }
  return config;
}

ParallelForConfig textureUploadConfig(size_t textureCount, size_t totalTexels) {
  // Texture upload benefits from smaller chunks for texture-heavy scenes with
  // only a few pixels. Larger atlases amortize dispatch overhead with a slightly
  // wider chunk.
  ParallelForConfig config{};
  if (totalTexels < (1u << 16)) {
    config.minChunkSize = 1;
    config.preferredChunkSize = 1;
  } else if (totalTexels < (1u << 20)) {
    config.minChunkSize = 2;
    config.preferredChunkSize = 4;
  } else {
    config.minChunkSize = 4;
    config.preferredChunkSize = 8;
  }

  // Avoid overshooting when scenes only have a handful of textures.
  if (config.preferredChunkSize > textureCount)
    config.preferredChunkSize = textureCount;

  return config;
}

void Renderer::PendingBlasBuild::releaseResources() {
  if (vertexStaging) {
    if (renderer)
      renderer->releaseTrackedResource(vertexStaging);
    else
      vertexStaging->release();
    vertexStaging = nullptr;
  }
  if (indexStaging) {
    if (renderer)
      renderer->releaseTrackedResource(indexStaging);
    else
      indexStaging->release();
    indexStaging = nullptr;
  }
  if (scratchBuffer) {
    if (renderer) {
      renderer->recycleBlasScratchBuffer(scratchBuffer, scratchSize);
    } else {
      scratchBuffer->release();
    }
    scratchBuffer = nullptr;
    scratchSize = 0;
  }
  if (geometryArray) {
    geometryArray->release();
    geometryArray = nullptr;
  }
  if (geometryDesc) {
    geometryDesc->release();
    geometryDesc = nullptr;
  }
  if (accelDesc) {
    accelDesc->release();
    accelDesc = nullptr;
  }
  accelerationStructure = nullptr;
  commandBuffer = nullptr;
}

MTL::Buffer *Renderer::acquireBlasScratchBuffer(NS::UInteger requestedSize,
                                                NS::UInteger &allocatedSize,
                                                bool &reused) {
  allocatedSize = requestedSize;
  reused = false;
  if (!_pDevice || requestedSize == 0)
    return nullptr;

  auto it = _blasScratchPool.lower_bound(requestedSize);
  while (it != _blasScratchPool.end()) {
    auto &bucket = it->second;
    if (!bucket.empty()) {
      auto *buffer = bucket.back();
      bucket.pop_back();
      NS::UInteger bucketSize = it->first;
      if (bucket.empty())
        _blasScratchPool.erase(it);
      allocatedSize = bucketSize;
      reused = true;
      if (_blasScratchPoolAvailableBytes >= bucketSize)
        _blasScratchPoolAvailableBytes -= bucketSize;
      else
        _blasScratchPoolAvailableBytes = 0;
      _blasScratchPoolInUseBytes += bucketSize;
      ++_blasScratchPoolReusedCount;
      std::printf(
          "[BLAS] Reusing scratch buffer (%llu bytes, requested %llu). Pool "
          "in-flight: %zu bytes, available: %zu bytes. (created=%zu, reused=%zu)\n",
          static_cast<unsigned long long>(bucketSize),
          static_cast<unsigned long long>(requestedSize),
          _blasScratchPoolInUseBytes, _blasScratchPoolAvailableBytes,
          _blasScratchPoolCreatedCount, _blasScratchPoolReusedCount);
      updateBlasScratchResidencyBudget();
      return buffer;
    }
    it = _blasScratchPool.erase(it);
  }

  auto *buffer = allocateBuffer(
      requestedSize, MTL::ResourceStorageModePrivate,
      GpuMemoryTracker::Category::Scratch, "BLASScratch");
  if (buffer) {
    allocatedSize = requestedSize;
    _blasScratchPoolInUseBytes += requestedSize;
    ++_blasScratchPoolCreatedCount;
    std::printf(
        "[BLAS] Allocated scratch buffer (%llu bytes). Pool in-flight: %zu "
        "bytes, available: %zu bytes. (created=%zu, reused=%zu)\n",
        static_cast<unsigned long long>(requestedSize),
        _blasScratchPoolInUseBytes, _blasScratchPoolAvailableBytes,
        _blasScratchPoolCreatedCount, _blasScratchPoolReusedCount);
  }
  updateBlasScratchResidencyBudget();
  return buffer;
}

void Renderer::recycleBlasScratchBuffer(MTL::Buffer *buffer, NS::UInteger size) {
  if (!buffer)
    return;

  if (size == 0) {
    releaseTrackedResource(buffer);
    return;
  }

  if (_blasScratchPoolInUseBytes >= size)
    _blasScratchPoolInUseBytes -= size;
  else
    _blasScratchPoolInUseBytes = 0;

  _blasScratchPoolAvailableBytes += size;
  _blasScratchPool[size].push_back(buffer);

  updateBlasScratchResidencyBudget();

  std::printf(
      "[BLAS] Recycled scratch buffer (%llu bytes). Pool in-flight: %zu bytes, "
      "available: %zu bytes. (created=%zu, reused=%zu)\n",
      static_cast<unsigned long long>(size), _blasScratchPoolInUseBytes,
      _blasScratchPoolAvailableBytes, _blasScratchPoolCreatedCount,
      _blasScratchPoolReusedCount);
}

void Renderer::releaseBlasScratchPool() {
  if (!_blasScratchPool.empty() || _blasScratchPoolInUseBytes != 0 ||
      _blasScratchPoolCreatedCount != 0 || _blasScratchPoolReusedCount != 0) {
    std::printf(
        "[BLAS] Releasing scratch pool (available=%zu bytes, in-flight=%zu "
        "bytes, created=%zu, reused=%zu).\n",
        _blasScratchPoolAvailableBytes, _blasScratchPoolInUseBytes,
        _blasScratchPoolCreatedCount, _blasScratchPoolReusedCount);
  }

  for (auto &entry : _blasScratchPool) {
    for (auto *buffer : entry.second) {
      if (buffer)
        releaseTrackedResource(buffer);
    }
  }

  _blasScratchPool.clear();
  _blasScratchPoolAvailableBytes = 0;

  if (_blasScratchPoolInUseBytes != 0) {
    std::printf(
        "[BLAS] Warning: scratch pool destroyed with %zu bytes still in "
        "flight.\n",
        _blasScratchPoolInUseBytes);
    _blasScratchPoolInUseBytes = 0;
  }

  updateBlasScratchResidencyBudget();
}

void Renderer::updateBlasScratchResidencyBudget() {
  _residencyBudget.setBlasScratchBytes(_blasScratchPoolAvailableBytes,
                                       _blasScratchPoolInUseBytes);
}

double Renderer::scratchMemoryMB() const {
  return _residencyBudget.scratchMemoryMB();
}

double Renderer::residencyMemoryMB() const {
  double totalMB = currentGPUMemoryMB();
  return _residencyBudget.residencyMemoryMB(totalMB);
}

double Renderer::residentTextureMemoryMB() const {
  size_t totalBytes = 0;
  auto addSlot = [&](const ManagedTextureSlot &slot) {
    if (!slot.texture)
      return;
    totalBytes += textureByteSize(slot);
  };

  addSlot(_colorSlot);
  addSlot(_albedoSlot);
  addSlot(_normalSlot);
  addSlot(_positionSlot);
  addSlot(_albedoHistorySlot);
  addSlot(_normalHistorySlot);
  addSlot(_positionHistorySlot);
  for (const auto &slot : _restirData0Slots)
    addSlot(slot);
  for (const auto &slot : _restirData1Slots)
    addSlot(slot);
  for (const auto &slot : _restirData2Slots)
    addSlot(slot);
  for (const auto &slot : _restirSurfacePosSlots)
    addSlot(slot);
  for (const auto &slot : _restirSurfaceNormalSlots)
    addSlot(slot);

  for (MTL::Texture *texture : _materialTextures) {
    totalBytes += textureByteSize(texture);
  }
  totalBytes += textureByteSize(_environmentTexture);

  return static_cast<double>(totalBytes) / (1024.0 * 1024.0);
}

double Renderer::restirTextureMemoryMB() const {
  size_t totalBytes = 0;
  auto addSlot = [&](const ManagedTextureSlot &slot) {
    if (!slot.texture)
      return;
    totalBytes += textureByteSize(slot);
  };

  for (const auto &slot : _restirData0Slots)
    addSlot(slot);
  for (const auto &slot : _restirData1Slots)
    addSlot(slot);
  for (const auto &slot : _restirData2Slots)
    addSlot(slot);
  for (const auto &slot : _restirSurfacePosSlots)
    addSlot(slot);
  for (const auto &slot : _restirSurfaceNormalSlots)
    addSlot(slot);

  return static_cast<double>(totalBytes) / (1024.0 * 1024.0);
}

double Renderer::residentGeometryMemoryMB() const {
  size_t totalBytes = residentGeometryMemoryBytes();
  return static_cast<double>(totalBytes) / (1024.0 * 1024.0);
}

double Renderer::strictResidentGeometryMemoryMB() const {
  size_t totalBytes = strictResidentGeometryMemoryBytes();
  return static_cast<double>(totalBytes) / (1024.0 * 1024.0);
}

size_t Renderer::residentGeometryMemoryBytes() const {
  size_t totalBytes = 0;
  for (const auto &resident : _residentObjectGpuResources) {
    if (resident.state != ResidentObjectGpuResources::ResidencyState::Cold)
      totalBytes += resident.byteSize;
  }
  return totalBytes;
}

size_t Renderer::strictResidentGeometryMemoryBytes() const {
  size_t totalBytes = 0;
  for (const auto &resident : _residentObjectGpuResources) {
    if (resident.state == ResidentObjectGpuResources::ResidencyState::Resident)
      totalBytes += resident.byteSize;
  }
  return totalBytes;
}

size_t Renderer::geometryResidencyCapBytes() const {
  double capMB = std::max(0.0, _geometryResidencyMemoryCapMB);
  return static_cast<size_t>(capMB * 1024.0 * 1024.0);
}

double Renderer::effectiveTotalGpuMemoryCapMB() const {
  if (_totalGpuMemoryCapMB > 0.0)
    return _totalGpuMemoryCapMB;
  if (_totalMemoryCapRelaxedMB > 0.0)
    return _totalMemoryCapRelaxedMB;
  return 0.0;
}

size_t Renderer::totalGpuMemoryCapBytes() const {
  double capMB = effectiveTotalGpuMemoryCapMB();
  if (capMB <= 0.0)
    return 0;
  return static_cast<size_t>(capMB * 1024.0 * 1024.0);
}

size_t Renderer::historyFootprintBytes() const {
  return 0;
}

size_t Renderer::essentialGeometryMemoryBytes() const {
  size_t totalBytes = 0;
  for (size_t objectIndex = 0; objectIndex < _residentObjectGpuResources.size();
       ++objectIndex) {
    const auto &resident = _residentObjectGpuResources[objectIndex];
    if (resident.state == ResidentObjectGpuResources::ResidencyState::Cold)
      continue;
    bool essential = true;
    if (objectIndex < _objectActive.size())
      essential = _objectActive[objectIndex];
    if (essential)
      totalBytes += resident.byteSize;
  }
  return totalBytes;
}

size_t Renderer::minimumResidentFootprintBytes() const {
  GpuMemoryTracker::Snapshot snapshot = _gpuMemoryTracker.snapshot();
  size_t mandatoryBytes =
      snapshot.bytes[static_cast<size_t>(
          GpuMemoryTracker::Category::RendererBuffers)] +
      snapshot.bytes[static_cast<size_t>(GpuMemoryTracker::Category::HeapsAS)];
  size_t historyBytes = historyFootprintBytes();
  size_t geometryBytes = essentialGeometryMemoryBytes();
  return mandatoryBytes + historyBytes + geometryBytes;
}

bool Renderer::canAllocateTotalGpuMemory(size_t requestedBytes,
                                         size_t existingBytes,
                                         GpuMemoryTracker::Category category,
                                         const char *context) {
  size_t capBytes = totalGpuMemoryCapBytes();
  if (capBytes == 0 || !_pDevice)
    return true;

  size_t totalBytes = static_cast<size_t>(_pDevice->currentAllocatedSize());
  size_t nextBytes = totalBytes >= existingBytes
                         ? totalBytes - existingBytes + requestedBytes
                         : totalBytes + requestedBytes;
  if (nextBytes <= capBytes)
    return true;

  size_t overageBytes = nextBytes - capBytes;
  const char *label = context ? context : "unspecified";
  bool residencyCategory =
      category == GpuMemoryTracker::Category::Textures ||
      category == GpuMemoryTracker::Category::Geometry ||
      category == GpuMemoryTracker::Category::HeapsAS;
  if (!residencyCategory) {
    ++_totalMemoryCapDeniedCount;
    ++_frameTotalMemoryCapDeniedCount;
    ++_totalMemoryCapNonResidencyDeniedCount;
    ++_frameTotalMemoryCapNonResidencyDeniedCount;
    std::printf(
        "[MemoryBudget] Total memory cap deny (%s): requested %.2f MB "
        "(existing %.2f MB), total %.2f MB cap %.2f MB.\n",
        label,
        static_cast<double>(requestedBytes) / (1024.0 * 1024.0),
        static_cast<double>(existingBytes) / (1024.0 * 1024.0),
        static_cast<double>(totalBytes) / (1024.0 * 1024.0),
        static_cast<double>(capBytes) / (1024.0 * 1024.0));
    return false;
  }

  _pendingTotalMemoryOverageBytes =
      std::max(_pendingTotalMemoryOverageBytes, overageBytes);
  std::printf(
      "[MemoryBudget] Total memory cap exceeded (%s): requested %.2f MB "
      "(existing %.2f MB), total %.2f MB cap %.2f MB. Attempting eviction before allocation.\n",
      label,
      static_cast<double>(requestedBytes) / (1024.0 * 1024.0),
      static_cast<double>(existingBytes) / (1024.0 * 1024.0),
      static_cast<double>(totalBytes) / (1024.0 * 1024.0),
      static_cast<double>(capBytes) / (1024.0 * 1024.0));

  bool evictionAttempted = false;
  if (_renderedFrameCount >= _totalMemoryEvictionBackoffUntilFrame &&
      !_totalMemoryEvictionInProgress && _pCommandQueue &&
      _renderedFrameCount != _lastTotalMemoryCapEvictionFrame) {
    evictionAttempted = true;
    _totalMemoryEvictionInProgress = true;
    _lastTotalMemoryCapEvictionFrame = _renderedFrameCount;
    size_t preEvictionBytes =
        static_cast<size_t>(_pDevice->currentAllocatedSize());
    MTL::CommandBuffer *cmd = _pCommandQueue->commandBuffer();
    if (cmd) {
      updateTextureResidency(cmd);
      updateGeometryResidency(cmd);
      cmd->commit();
      cmd->waitUntilCompleted();
    }
    size_t postEvictionBytes =
        static_cast<size_t>(_pDevice->currentAllocatedSize());
    _totalMemoryEvictionInProgress = false;

    if (postEvictionBytes + (1ull << 20) < preEvictionBytes) {
      _totalMemoryEvictionNoProgressFrames = 0;
    } else {
      ++_totalMemoryEvictionNoProgressFrames;
      _totalMemoryEvictionBackoffUntilFrame =
          _renderedFrameCount + kTotalMemoryEvictionBackoffFrames;
    }
  }

  totalBytes = static_cast<size_t>(_pDevice->currentAllocatedSize());
  nextBytes = totalBytes >= existingBytes
                  ? totalBytes - existingBytes + requestedBytes
                  : totalBytes + requestedBytes;
  if (nextBytes <= capBytes) {
    _pendingTotalMemoryOverageBytes = 0;
    _totalMemoryEvictionNoProgressFrames = 0;
    return true;
  }

  _pendingTotalMemoryOverageBytes =
      std::max(_pendingTotalMemoryOverageBytes, nextBytes - capBytes);

  if (evictionAttempted && _totalMemoryEvictionNoProgressFrames >=
                              kTotalMemoryEvictionNoProgressLimit) {
    double minimumFootprintMB =
        static_cast<double>(minimumResidentFootprintBytes()) / (1024.0 * 1024.0);
    double currentTotalMB = static_cast<double>(totalBytes) / (1024.0 * 1024.0);
    double relaxedCapMB =
        std::max({minimumFootprintMB, currentTotalMB * 1.03, _totalGpuMemoryCapMB});
    if (!_memoryCapFallbackActive || relaxedCapMB > _totalMemoryCapRelaxedMB) {
      _memoryCapFallbackActive = true;
      _totalMemoryCapRelaxedMB = relaxedCapMB;
      _frameTotalMemoryEvictionStall = true;
      disableHistoryForMemoryCap();
      std::printf(
          "[MemoryBudget] Eviction loop guard: no progress for %zu attempts. "
          "Temporarily relaxing total cap to %.2f MB (configured %.2f MB, "
          "footprint %.2f MB).\n",
          _totalMemoryEvictionNoProgressFrames, _totalMemoryCapRelaxedMB,
          _totalGpuMemoryCapMB, minimumFootprintMB);

      capBytes = totalGpuMemoryCapBytes();
      if (capBytes > 0) {
        size_t relaxedNextBytes = totalBytes >= existingBytes
                                      ? totalBytes - existingBytes + requestedBytes
                                      : totalBytes + requestedBytes;
        if (relaxedNextBytes <= capBytes) {
          _pendingTotalMemoryOverageBytes = 0;
          return true;
        }
      }
    }
  }

  ++_totalMemoryCapDeniedCount;
  ++_frameTotalMemoryCapDeniedCount;

  std::printf(
      "[MemoryBudget] Total memory cap deny (%s): requested %.2f MB "
      "(existing %.2f MB), total %.2f MB cap %.2f MB.\n",
      label,
      static_cast<double>(requestedBytes) / (1024.0 * 1024.0),
      static_cast<double>(existingBytes) / (1024.0 * 1024.0),
      static_cast<double>(totalBytes) / (1024.0 * 1024.0),
      static_cast<double>(capBytes) / (1024.0 * 1024.0));
  return false;
}

void Renderer::recordGeometryResidencyHardCapDenied(
    size_t objectIndex, size_t requestedBytes, size_t existingBytes,
    size_t residentBytes, size_t capBytes, const char *context) {
  ++_geometryResidencyHardCapDeniedCount;
  ++_frameGeometryResidencyHardCapDeniedCount;

  std::printf(
      "[GeometryResidency] Hard cap deny for object %zu (%s): requested %.2f MB "
      "(existing %.2f MB), resident %.2f MB, cap %.2f MB.\n",
      objectIndex, context ? context : "unspecified",
      static_cast<double>(requestedBytes) / (1024.0 * 1024.0),
      static_cast<double>(existingBytes) / (1024.0 * 1024.0),
      static_cast<double>(residentBytes) / (1024.0 * 1024.0),
      static_cast<double>(capBytes) / (1024.0 * 1024.0));
}

bool Renderer::checkGeometryResidencyCap(size_t objectIndex,
                                         size_t requestedBytes,
                                         size_t existingBytes,
                                         const char *context) {
  if (_geometryResidencyMemoryCapMB <= 0.0)
    return canAllocateTotalGpuMemory(requestedBytes, existingBytes,
                                     GpuMemoryTracker::Category::Geometry,
                                     context ? context : "geometry");

  size_t capBytes = geometryResidencyCapBytes();
  if (capBytes == 0)
    return canAllocateTotalGpuMemory(requestedBytes, existingBytes,
                                     GpuMemoryTracker::Category::Geometry,
                                     context ? context : "geometry");

  size_t residentBytes = residentGeometryMemoryBytes();
  size_t nextBytes = residentBytes >= existingBytes
                         ? residentBytes - existingBytes + requestedBytes
                         : residentBytes + requestedBytes;

  if (nextBytes <= capBytes)
    return canAllocateTotalGpuMemory(requestedBytes, existingBytes,
                                     GpuMemoryTracker::Category::Geometry,
                                     context ? context : "geometry");

  size_t overageBytes = nextBytes - capBytes;
  _pendingGeometryResidencyOverageBytes =
      std::max(_pendingGeometryResidencyOverageBytes, overageBytes);
  ++_geometryResidencyCapHitCount;
  ++_frameGeometryResidencyCapHitCount;

  recordGeometryResidencyHardCapDenied(objectIndex, requestedBytes,
                                       existingBytes, residentBytes, capBytes,
                                       context);
  std::printf("  [GeometryResidency] Over cap by %.2f MB; deferring allocation "
              "and scheduling eviction.\n",
              static_cast<double>(overageBytes) / (1024.0 * 1024.0));
  return false;
}

struct UniformsData {
  int primitiveIndex;
  simd::float3 cameraPosition;
  simd::float2 screenSize;
  float aperture;
  float focusDistance;

  simd::float3 viewportU;
  simd::float3 viewportV;
  simd::float3 firstPixelPosition;
  simd::float3 rayDx;
  simd::float3 rayDy;

  simd::float3 randomSeed;

  uint64_t primitiveCount;
  uint64_t triangleCount;
  uint64_t totalPrimitiveCount;
  uint64_t tlasNodeCount;
  uint64_t blasNodeCount;
  uint32_t maxRayDepth;
  uint32_t debugAS;
  uint32_t lightCount;
  float lightTotalWeight;
  uint32_t minSamplesPerPixel = 1;
  uint32_t maxSamplesPerPixel = 1;
  uint32_t textureCount = 0;
  uint32_t environmentMapEnabled = 0;
  float environmentMapIntensity = 1.0f;
  float environmentPadding0 = 0.0f;
  float environmentPadding1 = 0.0f;
  uint32_t restirEnabled = 0;
  uint32_t restirCandidateCount = 0;
  uint32_t restirTemporalReuse = 0;
  uint32_t restirPadding0 = 0;
  float cameraMotionMetric = 0.0f;
  simd::float4x4 currentViewProjection{};
  simd::float4x4 prevViewProjection{};
};

void populateViewportUniforms(UniformsData &uData) {
  float aspectRatio = Camera::screenSize.x / Camera::screenSize.y;
  float fovRad = Camera::verticalFov * (M_PI / 180.0f);
  float halfHeight = tanf(fovRad * 0.5f);
  float halfWidth = aspectRatio * halfHeight;

  simd::float3 w = simd::normalize(-Camera::forward);
  simd::float3 u = simd::normalize(simd::cross(Camera::up, w));
  simd::float3 v = simd::cross(w, u);

  simd::float3 viewportU = u * (2.0f * halfWidth);
  simd::float3 viewportV = -v * (2.0f * halfHeight);
  simd::float3 firstPixelPosition =
      Camera::position - w - (viewportU * 0.5f) - (viewportV * 0.5f);

  uData.cameraPosition = Camera::position;
  uData.viewportU = viewportU;
  uData.viewportV = viewportV;
  uData.firstPixelPosition = firstPixelPosition;
  uData.rayDx = viewportU / Camera::screenSize.x;
  uData.rayDy = viewportV / Camera::screenSize.y;
  uData.screenSize = Camera::screenSize;
}

struct TileDispatchRegion {
  uint32_t originX;
  uint32_t originY;
  uint32_t width;
  uint32_t height;
};

constexpr uint32_t kPathTraceTileWidth = 64;
constexpr uint32_t kPathTraceTileHeight = 64;
constexpr size_t kPathTraceMaxTilesPerCommand = 4;
constexpr size_t kPathTraceMinTilesPerCommand = 2;
constexpr double kPathTraceTargetGpuMsPerCommand = 6.0;
constexpr size_t kPathTraceCommandHistorySamples = 30;
constexpr std::chrono::milliseconds kFrameCommandBufferWaitTimeout(4);
constexpr uint32_t kMaxMaterialTextureSlots = 64;
constexpr float kReprojectionNearPlane = 0.1f;
constexpr float kReprojectionFarPlane = 1000.0f;

inline uint32_t bitm_random() {
  static uint32_t current_seed = 92407235;
  const uint32_t state = current_seed * 747796405u + 2891336453u;
  const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state);
  return (current_seed = (word >> 22u) ^ word);
}

inline float randomFloat() {
  return (float)bitm_random() / (float)std::numeric_limits<uint32_t>::max();
}

namespace {

size_t alignTo(size_t value, size_t alignment) {
  if (alignment == 0)
    return value;
  return (value + alignment - 1) & ~(alignment - 1);
}

constexpr size_t kTextureResidencyPrimitiveBudget = 1;
constexpr double kDefaultTextureResidencyMemoryCapMB = 2048.0;
constexpr double kDefaultGeometryResidencyMemoryCapMB = 2048.0;
constexpr double kDefaultTotalGpuMemoryCapMB = 4096.0;
constexpr float kFrustumDebugNear = 0.1f;
constexpr float kFrustumDebugFarMultiplier = 5.0f;
constexpr float kFrustumDebugThicknessWorld = 0.035f;
constexpr size_t kCameraTrailMaxPoints = 4096;
constexpr float kCameraTrailMinStep = 0.05f;

inline float clampUnit(float value) {
  return std::clamp(value, 0.0f, 1.0f);
}

inline simd::float3 lerpFloat3(const simd::float3 &a, const simd::float3 &b,
                               float t) {
  float clamped = clampUnit(t);
  return a + (b - a) * clamped;
}

Material makeResidencyPreviewMaterial(const Material &source, bool active) {
  Material result = source;
  const simd::float3 tint =
      active ? simd::make_float3(0.15f, 0.95f, 0.20f)
             : simd::make_float3(1.0f, 0.18f, 0.12f);
  const float tintStrength = active ? 0.65f : 0.85f;
  result.diffuseColor = lerpFloat3(source.diffuseColor, tint, tintStrength);
  result.specularColor =
      lerpFloat3(source.specularColor, simd::make_float3(0.04f, 0.04f, 0.04f),
                 0.7f);
  result.diffuseTextureIndex = -1;
  result.specularTextureIndex = -1;
  return result;
}

struct OverlayUniforms {
  simd::float4x4 viewProjection;
};

struct OverlayLineVertex {
  simd::float3 position;
  simd::float4 color;
};

constexpr std::array<std::pair<uint32_t, uint32_t>, 12> kFrustumEdges = {
    {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4},
     {0, 4}, {1, 5}, {2, 6}, {3, 7}}};

size_t bytesPerPixel(MTL::PixelFormat format) {
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

size_t estimateTextureBytes(MTL::PixelFormat format, size_t width, size_t height,
                            size_t depth, size_t arrayLength) {
  size_t pixelBytes = bytesPerPixel(format);
  if (pixelBytes == 0 || width == 0 || height == 0)
    return 0;
  size_t clampedDepth = std::max<size_t>(1, depth);
  size_t rowBytes = width * pixelBytes;
  size_t alignedRowBytes = alignTo(rowBytes, 256);
  size_t sliceBytes = alignedRowBytes * height * clampedDepth;
  size_t clampedArray = std::max<size_t>(1, arrayLength);
  return sliceBytes * clampedArray;
}

simd::float4x4 makeViewMatrix(const Camera::State &state) {
  simd::float3 forward = state.forward;
  if (simd::length_squared(forward) < 1e-6f)
    forward = {0.0f, 0.0f, -1.0f};
  forward = simd::normalize(forward);

  simd::float3 up = state.up;
  if (simd::length_squared(up) < 1e-6f)
    up = {0.0f, 1.0f, 0.0f};
  up = simd::normalize(up);

  simd::float3 right = simd::cross(forward, up);
  if (simd::length_squared(right) < 1e-6f)
    right = {1.0f, 0.0f, 0.0f};
  right = simd::normalize(right);
  up = simd::normalize(simd::cross(right, forward));

  simd::float4x4 view;
  view.columns[0] = {right.x, up.x, forward.x, 0.0f};
  view.columns[1] = {right.y, up.y, forward.y, 0.0f};
  view.columns[2] = {right.z, up.z, forward.z, 0.0f};
  view.columns[3] = {-simd::dot(right, state.position),
                     -simd::dot(up, state.position),
                     -simd::dot(forward, state.position), 1.0f};
  return view;
}

simd::float4x4 makePerspectiveMatrix(float verticalFovDegrees, float aspect,
                                     float nearZ, float farZ) {
  float fovRad = verticalFovDegrees * static_cast<float>(M_PI) / 180.0f;
  float yScale = 1.0f / std::tan(fovRad * 0.5f);
  float xScale = yScale / std::max(aspect, 1e-6f);
  float zRange = std::max(farZ - nearZ, 1e-6f);
  float zScale = farZ / zRange;
  float wz = -nearZ * zScale;

  simd::float4x4 proj;
  proj.columns[0] = {xScale, 0.0f, 0.0f, 0.0f};
  proj.columns[1] = {0.0f, yScale, 0.0f, 0.0f};
  proj.columns[2] = {0.0f, 0.0f, zScale, 1.0f};
  proj.columns[3] = {0.0f, 0.0f, wz, 0.0f};
  return proj;
}

float luminance(const simd::float3 &c) {
  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

template <typename Func>
void parallelChunkedAsync(size_t start, size_t end, Func &&func) {
  if (end <= start)
    return;

  auto work = std::forward<Func>(func);
  const size_t total = end - start;
  const unsigned int threadCount =
      std::max(1u, std::thread::hardware_concurrency());
  const size_t smallWorkThreshold =
      static_cast<size_t>(threadCount) * static_cast<size_t>(64);

  if (total <= smallWorkThreshold) {
    work(start, end);
    return;
  }

  const size_t chunkSize = std::max<size_t>(
      1, (total + static_cast<size_t>(threadCount) - 1) /
             static_cast<size_t>(threadCount));

  struct Context {
    size_t start;
    size_t end;
    size_t chunkSize;
    decltype(work) *func;
  } ctx{start, end, chunkSize, &work};

  const size_t chunkCount = (total + chunkSize - 1) / chunkSize;
  dispatch_queue_t queue =
      dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
  dispatch_apply_f(chunkCount, queue, &ctx, [](void *rawCtx, size_t chunkIndex) {
    auto &ctx = *static_cast<Context *>(rawCtx);
    size_t chunkBegin = ctx.start + chunkIndex * ctx.chunkSize;
    size_t chunkEnd = std::min(chunkBegin + ctx.chunkSize, ctx.end);
    if (chunkBegin < ctx.end)
      (*ctx.func)(chunkBegin, chunkEnd);
  });
}

float primitiveArea(const Primitive &p) {
  switch (p.type) {
  case PrimitiveType::Sphere: {
    float r = p.sphere.radius;
    return 4.0f * static_cast<float>(M_PI) * r * r;
  }
  case PrimitiveType::Triangle: {
    simd::float3 e1 = p.triangle.v1 - p.triangle.v0;
    simd::float3 e2 = p.triangle.v2 - p.triangle.v0;
    return 0.5f * simd::length(simd::cross(e1, e2));
  }
  case PrimitiveType::Rectangle: {
    simd::float3 e1 = p.rectangle.u;
    simd::float3 e2 = p.rectangle.v;
    return 4.0f * simd::length(simd::cross(e1, e2));
  }
  }
  return 0.0f;
}

float primitiveImportance(const Primitive &p) {
  float area = std::max(primitiveArea(p), 1e-4f);
  const Material &m = p.material;
  float emissive = m.emissionPower * luminance(m.emissionColor);
  float diffuse = luminance(m.diffuseColor) * m.opacity;
  float specular = luminance(m.specularColor) * m.opacity;
  float transmission = (1.0f - m.opacity) * luminance(m.transmissionColor);
  return area * (emissive + diffuse + specular + transmission);
}

void markBufferModified(MTL::Buffer *buffer, NS::Range range) {
  if (!buffer)
    return;

  if (buffer->storageMode() == MTL::StorageModeManaged)
    buffer->didModifyRange(range);
}

static bool cameraStatesDiffer(const Camera::State &a, const Camera::State &b,
                               float epsilon = 1e-3f) {
  if (simd::length(a.position - b.position) > epsilon)
    return true;
  if (simd::length(a.forward - b.forward) > epsilon)
    return true;
  if (simd::length(a.up - b.up) > epsilon)
    return true;
  if (std::abs(a.verticalFov - b.verticalFov) > epsilon)
    return true;
  if (std::abs(a.focalLength - b.focalLength) > epsilon)
    return true;
  if (std::abs(a.aperture - b.aperture) > epsilon)
    return true;
  if (std::abs(a.focusDistance - b.focusDistance) > epsilon)
    return true;
  return false;
}

static float computeCameraMotionMetric(const Camera::State &current,
                                       const Camera::State &previous) {
  auto normalizeWithFallback = [](const simd::float3 &v) {
    float lenSq = simd::length_squared(v);
    if (lenSq > 1e-6f)
      return simd::normalize(v);
    return simd::make_float3(0.0f, 0.0f, -1.0f);
  };

  simd::float3 currentForward = normalizeWithFallback(current.forward);
  simd::float3 previousForward = normalizeWithFallback(previous.forward);
  simd::float3 currentUp = normalizeWithFallback(current.up);
  simd::float3 previousUp = normalizeWithFallback(previous.up);

  float positionDelta = simd::length(current.position - previous.position);
  float forwardDot = std::clamp(simd::dot(currentForward, previousForward),
                                -1.0f, 1.0f);
  float upDot = std::clamp(simd::dot(currentUp, previousUp), -1.0f, 1.0f);
  float forwardAngle = std::acos(forwardDot);
  float upAngle = std::acos(upDot);
  float fovDelta = std::abs(current.verticalFov - previous.verticalFov) *
                   static_cast<float>(M_PI / 180.0f);

  constexpr float kPositionWeight = 0.25f;
  constexpr float kForwardAngleWeight = 0.5f;
  constexpr float kUpAngleWeight = 0.25f;
  constexpr float kFovWeight = 0.1f;

  float metric = positionDelta * kPositionWeight +
                 forwardAngle * kForwardAngleWeight + upAngle * kUpAngleWeight +
                 fovDelta * kFovWeight;
  return std::clamp(metric, 0.0f, 1.0f);
}

static Camera::State makeCameraState(const simd::float3 &position,
                                     const simd::float3 &lookAt,
                                     const simd::float3 &up,
                                     float verticalFov, float focalLength,
                                     float aperture, float focusDistance,
                                     const simd::float3 &fallbackForward,
                                     const simd::float3 &fallbackUp) {
  auto normalizeWithFallback = [](const simd::float3 &v,
                                  const simd::float3 &fallback) {
    float lenSq = simd::length_squared(v);
    if (lenSq > 1e-6f) {
      return simd::normalize(v);
    }
    float fallbackLenSq = simd::length_squared(fallback);
    if (fallbackLenSq > 1e-6f) {
      return simd::normalize(fallback);
    }
    return simd::make_float3(0.0f, 1.0f, 0.0f);
  };

  Camera::State state{};
  state.position = position;
  simd::float3 direction = lookAt - position;
  state.forward = normalizeWithFallback(direction, fallbackForward);

  simd::float3 desiredUp = normalizeWithFallback(up, fallbackUp);
  float alignment = std::abs(simd::dot(desiredUp, state.forward));
  if (alignment > 0.999f) {
    simd::float3 reference = std::abs(state.forward.y) < 0.999f
                                 ? simd::make_float3(0.0f, 1.0f, 0.0f)
                                 : simd::make_float3(1.0f, 0.0f, 0.0f);
    simd::float3 right = simd::cross(reference, state.forward);
    desiredUp = normalizeWithFallback(simd::cross(state.forward, right),
                                      fallbackUp);
  }
  state.up = desiredUp;
  state.verticalFov = verticalFov;
  state.focalLength = focalLength;
  state.aperture = aperture;
  state.focusDistance = focusDistance;
  return state;
}

} // namespace

void Renderer::ensureBufferCapacity(MTL::Buffer *&buffer, size_t requiredBytes,
                                    size_t &currentCapacity,
                                    bool allowShrink,
                                    MTL::ResourceOptions storageMode,
                                    GpuMemoryTracker::Category category,
                                    const char *label,
                                    const char *resizeContext) {
  if (requiredBytes == 0)
    requiredBytes = 1;

  size_t desiredCapacity = requiredBytes;
  size_t originalCapacity = currentCapacity;
  bool hadBuffer = buffer != nullptr;
  bool shrinking = false;
  bool growing = false;

  if (!allowShrink) {
    if (buffer && requiredBytes <= currentCapacity)
      return;
    desiredCapacity = std::max(requiredBytes, currentCapacity);
    if (buffer && requiredBytes > currentCapacity)
      growing = true;
  } else if (buffer) {
    constexpr double kShrinkRetainRatio = 0.90;
    constexpr double kCompactedGrowthFactor = 1.25;
    size_t shrinkThreshold = static_cast<size_t>(
        std::floor(static_cast<double>(currentCapacity) * kShrinkRetainRatio));
    if (requiredBytes <= currentCapacity && requiredBytes > shrinkThreshold)
      return;
    if (requiredBytes <= shrinkThreshold) {
      shrinking = requiredBytes < currentCapacity;
    } else if (requiredBytes > currentCapacity) {
      size_t growthTarget = static_cast<size_t>(std::ceil(
          static_cast<double>(currentCapacity) * kCompactedGrowthFactor));
      desiredCapacity = std::max(requiredBytes, growthTarget);
      growing = true;
    }
  }

  if (!hadBuffer)
    growing = true;

  if (!canAllocateTotalGpuMemory(desiredCapacity,
                                 hadBuffer ? currentCapacity : 0, category,
                                 resizeContext ? resizeContext
                                               : (label ? label
                                                        : "buffer resize"))) {
    std::printf(
        "[Renderer][Buffer] Allocation denied for %s (requested=%zu bytes, context=%s).\n",
        label ? label : "UnnamedBuffer", desiredCapacity,
        resizeContext ? resizeContext : "capacity");
    return;
  }

  if (buffer) {
    releaseTrackedResource(buffer);
    buffer = nullptr;
    currentCapacity = 0;
  }

  if (allowShrink && desiredCapacity < requiredBytes)
    desiredCapacity = requiredBytes;

  desiredCapacity = std::max(desiredCapacity, size_t(1));
  buffer = allocateBuffer(static_cast<NS::UInteger>(desiredCapacity),
                          storageMode, category, label, true);
  currentCapacity = buffer ? buffer->length() : 0;

  const char *name = label ? label : "UnnamedBuffer";
  const char *context = resizeContext
                            ? resizeContext
                            : (allowShrink ? "shrink-policy" : "capacity");
  if (!buffer) {
    std::printf(
        "[Renderer][Buffer] Failed to allocate %s (requested=%zu bytes, context=%s).\n",
        name, desiredCapacity, context);
    return;
  }

  const char *action = hadBuffer
                           ? (shrinking ? "shrunk" : (growing ? "grown" : "rebound"))
                           : "allocated";
  std::printf(
      "[Renderer][Buffer] %s %s from %zu to %zu bytes (required=%zu, allowShrink=%s, context=%s, activeRatio=%.3f).\n",
      name, action, originalCapacity, currentCapacity, requiredBytes,
      allowShrink ? "true" : "false", context, _lastActivePrimitiveRatio);
}

MTL::Buffer *Renderer::allocateBuffer(NS::UInteger size,
                                      MTL::ResourceOptions storageMode,
                                      GpuMemoryTracker::Category category,
                                      const char *label,
                                      bool skipCapCheck) {
  if (!_pDevice)
    return nullptr;

  if (!skipCapCheck) {
    if (!canAllocateTotalGpuMemory(static_cast<size_t>(size), 0, category,
                                   label ? label : "buffer")) {
      return nullptr;
    }
  }

  MTL::Buffer *buffer = _pDevice->newBuffer(size, storageMode);
  if (!buffer)
    return nullptr;

  if (label) {
    using NS::StringEncoding::UTF8StringEncoding;
    buffer->setLabel(NS::String::string(label, UTF8StringEncoding));
  }

  trackResource(buffer, category);
  return buffer;
}

MTL::Texture *Renderer::allocateTexture(MTL::TextureDescriptor *descriptor,
                                        GpuMemoryTracker::Category category,
                                        const char *label) {
  if (!_pDevice || !descriptor)
    return nullptr;

  size_t estimatedBytes = estimateTextureBytes(
      descriptor->pixelFormat(), static_cast<size_t>(descriptor->width()),
      static_cast<size_t>(descriptor->height()),
      static_cast<size_t>(descriptor->depth()),
      static_cast<size_t>(descriptor->arrayLength()));
  if (estimatedBytes > 0) {
    if (!canAllocateTotalGpuMemory(estimatedBytes, 0, category,
                                   label ? label : "texture")) {
      return nullptr;
    }
  }

  MTL::Texture *texture = _pDevice->newTexture(descriptor);
  if (!texture)
    return nullptr;

  if (label) {
    using NS::StringEncoding::UTF8StringEncoding;
    texture->setLabel(NS::String::string(label, UTF8StringEncoding));
  }

  trackResource(texture, category);
  return texture;
}

void Renderer::trackResource(MTL::Resource *resource,
                             GpuMemoryTracker::Category category) {
  if (!resource)
    return;
  size_t bytes = static_cast<size_t>(resource->allocatedSize());
  _gpuMemoryTracker.trackResource(resource, bytes, category);
}

void Renderer::releaseTrackedResource(MTL::Resource *resource) {
  if (!resource)
    return;
  _gpuMemoryTracker.releaseResource(resource);
  resource->release();
}

GpuMemoryTracker::Category
Renderer::textureCategoryForSlot(const ManagedTextureSlot &slot) const {
  return GpuMemoryTracker::Category::Textures;
}

bool Renderer::isInView(const BoundingSphere &b) {
  simd::float3 toCenter = b.center - Camera::position;
  float dist = simd::length(toCenter);
  if (dist < 1e-5f)
    return true;

  simd::float3 forward = simd::normalize(Camera::forward);
  float forwardLenSq = simd::length_squared(forward);
  if (forwardLenSq < 1e-6f)
    return true;

  simd::float3 dir = toCenter / dist;
  float forwardDot = simd::dot(dir, forward);
  if (forwardDot <= 0.0f)
    return false;

  simd::float3 up = Camera::up;
  float upLenSq = simd::length_squared(up);
  if (upLenSq < 1e-6f) {
    up = {0.0f, 1.0f, 0.0f};
    upLenSq = 1.0f;
  }
  up /= std::sqrt(upLenSq);

  simd::float3 right = simd::cross(forward, up);
  float rightLenSq = simd::length_squared(right);
  if (rightLenSq < 1e-6f) {
    right = {1.0f, 0.0f, 0.0f};
    rightLenSq = 1.0f;
  }
  right /= std::sqrt(rightLenSq);

  up = simd::normalize(simd::cross(right, forward));

  float verticalHalfFov =
      Camera::verticalFov * static_cast<float>(M_PI) / 180.0f * 0.5f;
  if (verticalHalfFov <= 0.0f)
    verticalHalfFov = 1e-3f;

  float aspect = Camera::screenSize.y > 0.0f
                     ? Camera::screenSize.x / Camera::screenSize.y
                     : 1.0f;
  float horizontalHalfFov = std::atan(std::tan(verticalHalfFov) * aspect);

  float radiusAngle = asinf(std::min(b.radius / dist, 1.0f));

  float horizontalAngle =
      std::atan2(std::fabs(simd::dot(dir, right)), forwardDot);
  float verticalAngle =
      std::atan2(std::fabs(simd::dot(dir, up)), forwardDot);

  return horizontalAngle <= horizontalHalfFov + radiusAngle &&
         verticalAngle <= verticalHalfFov + radiusAngle;
}

Renderer::Renderer(MTL::Device *pDevice)
    : _pDevice(pDevice->retain()), _pScene(new Scene()),
      _pendingBlasEvictions(
          [](MTL::CommandBuffer *command) {
            if (command)
              command->retain();
          },
          [](MTL::CommandBuffer *command) {
            if (command)
              command->release();
          }) {
  _pCommandQueue = _pDevice->newCommandQueue();
  _tlasHeap.setMemoryTracker(&_gpuMemoryTracker);
  _dummyBlasResources.setMemoryTracker(&_gpuMemoryTracker);
  _tlasHeap.initialize(_pDevice);
  _dummyBlasResources.initialize(_pDevice);
  _tlasHeap.setHeapShrinkEnabled(_heapShrinkEnabled);
  _dummyBlasResources.setHeapShrinkEnabled(_heapShrinkEnabled);

  Camera::reset();
  _primaryCameraState = Camera::captureState();
  _observerCameraState = _primaryCameraState;
  _residencyPreviewOnly =
      parseEnvBool(std::getenv("MPT_RESIDENCY_PREVIEW_ONLY"));
  _forceObserverCapture = parseEnvBool(std::getenv("MPT_OBSERVER_CAPTURE"));
  _disableOfflineOidn = parseEnvBool(std::getenv("MPT_DISABLE_OFFLINE_OIDN"));
  _neuralFeatureLoggingEnabled =
      parseEnvBool(std::getenv("MPT_LOG_NEURAL_FEATURES"));
  _unifiedNeuralScoreLoggingEnabled =
      parseEnvBool(std::getenv("MPT_LOG_UNIFIED_NEURAL_SCORES"));
  if (const char *clipLengthEnv = std::getenv("MPT_NEURAL_CLIP_LENGTH")) {
    unsigned long parsed = std::strtoul(clipLengthEnv, nullptr, 10);
    if (parsed > 0)
      _neuralClipLength = static_cast<size_t>(parsed);
  }
  if (const char *forcedObjectEnv = std::getenv("MPT_FORCE_OBJECT_OFF")) {
    char *end = nullptr;
    unsigned long long parsed = std::strtoull(forcedObjectEnv, &end, 10);
    if (end != forcedObjectEnv)
      _forcedObjectOffIndex = static_cast<size_t>(parsed);
  }
  if (const char *runsPath = std::getenv("MPT_RUNS_PATH"))
    _runOutputRoot = runsPath;
  if (const char *sceneVariant = std::getenv("MPT_SCENE_VARIANT"))
    _sceneVariantName = sceneVariant;
  if (_forceObserverCapture)
    _frameCaptureSubdirectory = "observer_frames";
  if (_residencyPreviewOnly) {
    std::printf("[Renderer] Residency preview-only mode enabled.\n");
  }
  if (_forceObserverCapture) {
    std::printf("[Renderer] Observer capture mode enabled. Frames will be "
                "saved under Benchmarks/%s.\n",
                _frameCaptureSubdirectory.c_str());
  }
  if (_disableOfflineOidn) {
    std::printf("[Renderer] Offline OIDN is disabled during capture; raw EXRs "
                "will be written for deferred denoising.\n");
  }
  if (_neuralFeatureLoggingEnabled) {
    std::printf("[Renderer] Neural feature logging enabled (clip length %zu).\n",
                _neuralClipLength);
  }
  if (_unifiedNeuralScoreLoggingEnabled) {
    std::printf("[Renderer] UnifiedNeural score logging enabled.\n");
  }
  if (_forcedObjectOffIndex != std::numeric_limits<size_t>::max()) {
    std::printf("[Renderer] Forced object-off ablation enabled for object %zu.\n",
                _forcedObjectOffIndex);
  }

  updateVisibleScene();
  buildShaders();
  buildTextures();
  rebuildEnvironmentTexture();
  _pathTraceTilesPerCommandBudget = kPathTraceMaxTilesPerCommand;

  recalculateViewport();
  initializeBenchmarking();
  initializeNeuralFeatureLogging();
  initializeUnifiedNeuralScoreLogging();
}

Renderer::~Renderer() {
  processPendingCapturedFrames();
  flushNeuralClipFeatures(true);
  std::chrono::steady_clock::time_point frameWaitSnapshot;
  waitForPendingFrameCommands(std::chrono::milliseconds::max(),
                              &frameWaitSnapshot);

  _pendingBlasEvictions.clear();
  assert(_pendingBlasEvictions.empty());

  if (_pSphereBuffer)
    releaseTrackedResource(_pSphereBuffer);
  if (_pSphereMaterialBuffer)
    releaseTrackedResource(_pSphereMaterialBuffer);
  if (_pTriangleVertexBuffer)
    releaseTrackedResource(_pTriangleVertexBuffer);
  if (_pTriangleIndexBuffer)
    releaseTrackedResource(_pTriangleIndexBuffer);
  if (_pUniformsBuffer)
    releaseTrackedResource(_pUniformsBuffer);
  if (_pBVHBuffer)
    releaseTrackedResource(_pBVHBuffer);
  if (_pPrimitiveIndexBuffer)
    releaseTrackedResource(_pPrimitiveIndexBuffer);
  if (_pTLASBuffer)
    releaseTrackedResource(_pTLASBuffer);
  if (_pActiveBuffer)
    releaseTrackedResource(_pActiveBuffer);
  if (_pPrimitiveRemapBuffer)
    releaseTrackedResource(_pPrimitiveRemapBuffer);
  flushRayHitCopy();
  if (_pPrimitiveHitBufferGPU)
    releaseTrackedResource(_pPrimitiveHitBufferGPU);
  if (_pPrimitiveHitReadback)
    releaseTrackedResource(_pPrimitiveHitReadback);
  if (_pRestirStatsBuffer)
    releaseTrackedResource(_pRestirStatsBuffer);
  if (_pRestirStatsReadback)
    releaseTrackedResource(_pRestirStatsReadback);
  if (_pLightIndexBuffer)
    releaseTrackedResource(_pLightIndexBuffer);
  if (_pLightCdfBuffer)
    releaseTrackedResource(_pLightCdfBuffer);
  if (_pLightPdfLookupBuffer)
    releaseTrackedResource(_pLightPdfLookupBuffer);
  if (_pInstanceBuffer)
    releaseTrackedResource(_pInstanceBuffer);
  if (_pGeometryHandleBuffer)
    releaseTrackedResource(_pGeometryHandleBuffer);
  if (_pTlasScratchBuffer)
    releaseTrackedResource(_pTlasScratchBuffer);
  if (_pTlasBuildEvent)
    _pTlasBuildEvent->release();
  if (_pTlasDescriptorStaging)
    releaseTrackedResource(_pTlasDescriptorStaging);
  _pTlasDescriptorStaging = nullptr;
  _tlasDescriptorStagingCapacity = 0;
  _pTlasScratchBuffer = nullptr;
  _tlasScratchCapacity = 0;
  _tlasScratchTracker.reset();
  updateTlasScratchResidentBytes(0);
  _pTlasBuildEvent = nullptr;
  _tlasBuildEventValue = 0;
  _tlasCompletedEventValue.store(0, std::memory_order_relaxed);
  if (_pFrustumVertexBuffer)
    releaseTrackedResource(_pFrustumVertexBuffer);
  releaseEnvironmentTexture();
  if (_environmentSampler)
    _environmentSampler->release();
  _environmentSampler = nullptr;
  clearMaterialTextures();

  _cachedInstanceDescriptors.clear();
  _cachedInstancedAccelerationStructures.clear();
  _pTlasInstanceDescriptorBuffer = nullptr;
  _pTlasStructure = nullptr;
  _pDummyBlas = nullptr;

  _tlasHeap.destroy();
  _dummyBlasResources.destroy();

  releaseTextureSlot(_colorSlot);
  releaseTextureSlot(_albedoSlot);
  releaseTextureSlot(_normalSlot);
  releaseTextureSlot(_positionSlot);
  releaseTextureSlot(_albedoHistorySlot);
  releaseTextureSlot(_normalHistorySlot);
  releaseTextureSlot(_positionHistorySlot);
  for (auto &slot : _restirData0Slots)
    releaseTextureSlot(slot);
  for (auto &slot : _restirData1Slots)
    releaseTextureSlot(slot);
  for (auto &slot : _restirData2Slots)
    releaseTextureSlot(slot);
  for (auto &slot : _restirSurfacePosSlots)
    releaseTextureSlot(slot);
  for (auto &slot : _restirSurfaceNormalSlots)
    releaseTextureSlot(slot);

  if (_pTextureClearBuffer)
    releaseTrackedResource(_pTextureClearBuffer);

  if (_pPathTracePSO)
    _pPathTracePSO->release();
  if (_pAdaptiveSamplingPSO)
    _pAdaptiveSamplingPSO->release();
  if (_pOverlayPSO)
    _pOverlayPSO->release();
  if (_pOverlayCapturePSO)
    _pOverlayCapturePSO->release();

  for (auto &resident : _residentObjectGpuResources) {
    resident.clearPendingCommand();
    resident.resources.destroy();
  }
  _residentObjectGpuResources.clear();

  releaseBlasScratchPool();

  if (_pPSO)
    _pPSO->release();
  if (_pCommandQueue)
    _pCommandQueue->release();
  if (_pDevice)
    _pDevice->release();

  if (_benchmarkStream.is_open())
    _benchmarkStream.close();
  if (_neuralFeatureStream.is_open())
    _neuralFeatureStream.close();
  if (_unifiedNeuralScoreStream.is_open())
    _unifiedNeuralScoreStream.close();

  delete _pScene;
}

void Renderer::setBenchmarkMode(bool enabled) {
  if (enabled == _benchmarkEnabled)
    return;

  _benchmarkEnabled = enabled;
  if (_benchmarkEnabled) {
    _benchmarkFrameCounter = 0;
    _benchmarkHeaderWritten = false;
    _pendingBenchmarkSamples.clear();
    _benchmarkStartTime = std::chrono::steady_clock::now();
    _neuralClipStartFrame = 0;
    _neuralClipAccumulators.clear();
    ensureBenchmarkStream();
    ensureNeuralFeatureStream();
    resetProbabilisticResidencyState();
    if (_benchmarkStream.is_open()) {
      printf("Benchmark logging enabled: %s\n",
             _benchmarkFilePath.empty() ? "<memory>"
                                        : _benchmarkFilePath.c_str());
    }
  } else {
    if (_benchmarkStream.is_open())
      _benchmarkStream.close();
    if (_neuralFeatureLoggingEnabled)
      flushNeuralClipFeatures(true);
    _pendingBenchmarkSamples.clear();
  }
}

bool Renderer::objectForcedOff(size_t objectIndex) const {
  return _forcedObjectOffIndex != std::numeric_limits<size_t>::max() &&
         objectIndex == _forcedObjectOffIndex;
}

void Renderer::initializeNeuralFeatureLogging() {
  if (!_neuralFeatureLoggingEnabled)
    return;
  ensureNeuralFeatureStream();
}

void Renderer::initializeUnifiedNeuralScoreLogging() {
  if (!_unifiedNeuralScoreLoggingEnabled)
    return;
  ensureUnifiedNeuralScoreStream();
}

void Renderer::ensureNeuralFeatureStream() {
  if (!_neuralFeatureLoggingEnabled)
    return;
  if (_neuralFeatureStream.is_open())
    return;

  std::filesystem::path baseDir;
  if (!_runOutputRoot.empty())
    baseDir = _runOutputRoot;
  else
    baseDir = std::filesystem::current_path() / "Benchmarks";

  std::error_code ec;
  std::filesystem::create_directories(baseDir, ec);

  std::filesystem::path outPath = baseDir / "neural_object_features.csv";
  _neuralFeatureFilePath = outPath.string();
  _neuralFeatureStream.open(_neuralFeatureFilePath,
                            std::ios::out | std::ios::trunc);
  if (!_neuralFeatureStream.is_open()) {
    std::printf("Failed to open neural feature log '%s'\n",
                _neuralFeatureFilePath.c_str());
    return;
  }

  _neuralFeatureStream
      << "scene_variant,strategy,strategy_id,clip_id,clip_start_frame,"
         "clip_end_frame,clip_frame_count,forced_object_off,object_id,"
         "primitive_count,visible_frame_fraction,mean_visible_coverage,"
         "max_visible_coverage,mean_distance,min_distance,"
         "mean_hit_probability,mean_object_rayhit_score,total_object_hits,"
         "total_object_rays_tested,toggle_count,active_frame_fraction,"
         "resident_frame_fraction,streaming_frame_fraction,"
         "transport_critical_frame_fraction,mean_object_importance,"
         "mean_estimated_object_bytes,emissive_importance\n";
  _neuralFeatureHeaderWritten = true;
}

void Renderer::ensureUnifiedNeuralScoreStream() {
  if (!_unifiedNeuralScoreLoggingEnabled)
    return;
  if (_unifiedNeuralScoreStream.is_open())
    return;

  std::filesystem::path baseDir;
  if (!_runOutputRoot.empty())
    baseDir = _runOutputRoot;
  else
    baseDir = std::filesystem::current_path() / "Benchmarks";

  std::error_code ec;
  std::filesystem::create_directories(baseDir, ec);

  std::filesystem::path outPath = baseDir / "unified_neural_scores.csv";
  _unifiedNeuralScoreFilePath = outPath.string();
  _unifiedNeuralScoreStream.open(_unifiedNeuralScoreFilePath,
                                 std::ios::out | std::ios::trunc);
  if (!_unifiedNeuralScoreStream.is_open()) {
    std::printf("Failed to open UnifiedNeural score log '%s'\n",
                _unifiedNeuralScoreFilePath.c_str());
    return;
  }

  _unifiedNeuralScoreStream
      << "scene_variant,frame,strategy,strategy_id,object_id,primitive_count,"
         "currently_active,desired_active,residency_state,visible,"
         "adjusted_cost,heuristic_utility,heuristic_norm,neural_prediction,"
         "neural_norm,blend_weight,blended_utility,blended_efficiency\n";
  _unifiedNeuralScoreHeaderWritten = true;
}

void Renderer::accumulateNeuralClipFeatures() {
  if (!_neuralFeatureLoggingEnabled)
    return;
  ensureNeuralFeatureStream();
  if (!_neuralFeatureStream.is_open())
    return;

  updatePrimitiveScreenCoverageForFrame();

  const size_t objectCount = _allSceneObjects.size();
  if (objectCount == 0 || _neuralClipLength == 0)
    return;

  if (_neuralClipAccumulators.size() != objectCount)
    _neuralClipAccumulators.assign(objectCount, NeuralObjectClipAccumulator{});

  if (_benchmarkEnabled && _benchmarkFrameCounter > 0 &&
      ((_benchmarkFrameCounter - 1) % _neuralClipLength == 0)) {
    bool anyInitialized = false;
    for (const auto &acc : _neuralClipAccumulators) {
      if (acc.initialized) {
        anyInitialized = true;
        break;
      }
    }
    if (anyInitialized)
      flushNeuralClipFeatures(false);
    _neuralClipStartFrame = _benchmarkFrameCounter - 1;
  }

  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    NeuralObjectClipAccumulator &acc = _neuralClipAccumulators[objectIndex];
    if (!acc.initialized) {
      acc = NeuralObjectClipAccumulator{};
      acc.initialized = true;
      acc.primitiveCount = objectIndex < _objectPrimitiveCounts.size()
                               ? _objectPrimitiveCounts[objectIndex]
                               : _allSceneObjects[objectIndex].primitiveCount;
      const SceneObject &obj = _allSceneObjects[objectIndex];
      const size_t first = obj.firstPrimitive;
      const size_t last = std::min(first + obj.primitiveCount, _allPrimitives.size());
      double emissive = 0.0;
      for (size_t primIndex = first; primIndex < last; ++primIndex) {
        const Material &m = _allPrimitives[primIndex].material;
        emissive += std::max(
            m.emissionPower * luminance(m.emissionColor) *
                primitiveArea(_allPrimitives[primIndex]),
            0.0f);
      }
      acc.emissiveImportance = emissive;
    }

    const SceneObject &obj = _allSceneObjects[objectIndex];
    bool visible =
        objectIndex < _objectVisible.size() && _objectVisible[objectIndex] != 0u;
    if (!visible && objectIndex < _objectBounds.size())
      visible = isInView(_objectBounds[objectIndex]);
    if (visible)
      acc.visibleFrames += 1;

    double coverage = 0.0;
    const size_t first = obj.firstPrimitive;
    const size_t last =
        std::min(first + obj.primitiveCount, _primitiveScreenCoverage.size());
    for (size_t primIndex = first; primIndex < last; ++primIndex)
      coverage += std::max(_primitiveScreenCoverage[primIndex], 0.0f);
    if (visible)
      acc.visibleCoverageSum += coverage;
    acc.maxVisibleCoverage = std::max(acc.maxVisibleCoverage, coverage);

    if (objectIndex < _objectBounds.size()) {
      simd::float3 delta = _objectBounds[objectIndex].center - Camera::position;
      double distance = static_cast<double>(simd::length(delta));
      acc.distanceSum += distance;
      acc.minDistance = std::min(acc.minDistance, distance);
    }

    if (objectIndex < _objectHitProbability.size())
      acc.hitProbabilitySum +=
          std::clamp(static_cast<double>(_objectHitProbability[objectIndex]), 0.0,
                     1.0);
    if (objectIndex < _objectRayHitScore.size())
      acc.rayHitScoreSum +=
          std::max(static_cast<double>(_objectRayHitScore[objectIndex]), 0.0);
    if (objectIndex < _objectHitLastFrame.size())
      acc.totalHits += _objectHitLastFrame[objectIndex];
    if (objectIndex < _objectRaysTestedLastFrame.size())
      acc.totalRaysTested += _objectRaysTestedLastFrame[objectIndex];
    if (objectIndex < _objectLastToggleFrame.size() &&
        _objectLastToggleFrame[objectIndex] == _renderedFrameCount)
      acc.toggleCount += 1;
    if (objectIndex < _objectActive.size() && _objectActive[objectIndex])
      acc.activeFrames += 1;
    if (objectIndex < _residentObjectGpuResources.size()) {
      const auto &resident = _residentObjectGpuResources[objectIndex];
      if (resident.state == ResidentObjectGpuResources::ResidencyState::Resident)
        acc.residentFrames += 1;
      if (resident.state == ResidentObjectGpuResources::ResidencyState::Streaming)
        acc.streamingFrames += 1;
      acc.estimatedBytesSum += static_cast<double>(resident.byteSize > 0
                                                       ? resident.byteSize
                                                       : resident.resources.currentRequiredBytes());
    }
    double objectImportance = 0.0;
    if (objectIndex < _objectImportance.size())
      objectImportance = std::max(_objectImportance[objectIndex], 0.0f);
    if (!(objectImportance > 0.0)) {
      for (size_t primIndex = first; primIndex < last && primIndex < _primitiveImportance.size();
           ++primIndex) {
        objectImportance += std::max(static_cast<double>(_primitiveImportance[primIndex]), 0.0);
      }
    }
    acc.objectImportanceSum += objectImportance;
    bool transportCritical = false;
    for (size_t primIndex = first; primIndex < last && primIndex < _allPrimitives.size();
         ++primIndex) {
      const Material &m = _allPrimitives[primIndex].material;
      if (m.emissionPower * luminance(m.emissionColor) > 0.0f) {
        transportCritical = true;
        break;
      }
    }
    if (transportCritical)
      acc.transportCriticalFrames += 1;
  }
}

void Renderer::flushNeuralClipFeatures(bool forcePartialClip) {
  if (!_neuralFeatureLoggingEnabled || !_neuralFeatureStream.is_open() ||
      _neuralClipAccumulators.empty())
    return;

  size_t frameCount = _benchmarkFrameCounter > _neuralClipStartFrame
                          ? (_benchmarkFrameCounter - _neuralClipStartFrame)
                          : 0;
  if (!forcePartialClip && frameCount < _neuralClipLength)
    return;
  if (frameCount == 0)
    return;

  auto formatFixedLocal = [](double value, int precision = 6) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;
    return ss.str();
  };
  auto appendCsvEscapedLocal = [](std::ostringstream &ss,
                                  const std::string &value) {
    ss << '"';
    for (char c : value) {
      if (c == '"')
        ss << '"';
      ss << c;
    }
    ss << '"';
  };

  ResidencyStrategy currentStrategy = _frameStrategy;
  if (_pScene)
    currentStrategy = _pScene->getResidencyStrategy();
  const std::string strategyName = residencyStrategyName(currentStrategy);
  const std::string sceneName =
      _sceneVariantName.empty() ? std::string("unknown_scene") : _sceneVariantName;
  const size_t clipEndFrame =
      _benchmarkFrameCounter > 0 ? (_benchmarkFrameCounter - 1) : 0;
  const size_t clipId =
      _neuralClipLength > 0 ? (_neuralClipStartFrame / _neuralClipLength) : 0;

  for (size_t objectIndex = 0; objectIndex < _neuralClipAccumulators.size();
       ++objectIndex) {
    const NeuralObjectClipAccumulator &acc = _neuralClipAccumulators[objectIndex];
    if (!acc.initialized || acc.primitiveCount == 0)
      continue;

    const double invFrames = 1.0 / static_cast<double>(frameCount);
    const double visibleFrameFraction =
        static_cast<double>(acc.visibleFrames) * invFrames;
    const double activeFrameFraction =
        static_cast<double>(acc.activeFrames) * invFrames;
    const double residentFrameFraction =
        static_cast<double>(acc.residentFrames) * invFrames;
    const double streamingFrameFraction =
        static_cast<double>(acc.streamingFrames) * invFrames;
    const double transportCriticalFrameFraction =
        static_cast<double>(acc.transportCriticalFrames) * invFrames;
    const double meanVisibleCoverage =
        acc.visibleFrames > 0
            ? (acc.visibleCoverageSum / static_cast<double>(acc.visibleFrames))
            : 0.0;
    const double meanDistance = acc.distanceSum * invFrames;
    const double minDistance =
        std::isfinite(acc.minDistance) ? acc.minDistance : 0.0;
    const double meanHitProbability = acc.hitProbabilitySum * invFrames;
    const double meanRayHitScore = acc.rayHitScoreSum * invFrames;
    const double meanObjectImportance = acc.objectImportanceSum * invFrames;
    const double meanEstimatedBytes = acc.estimatedBytesSum * invFrames;

    std::ostringstream row;
    appendCsvEscapedLocal(row, sceneName);
    row << ',';
    appendCsvEscapedLocal(row, strategyName);
    row << ',' << static_cast<int>(currentStrategy) << ','
        << clipId << ',' << _neuralClipStartFrame << ',' << clipEndFrame << ','
        << frameCount << ',';
    if (_forcedObjectOffIndex == std::numeric_limits<size_t>::max())
      row << -1;
    else
      row << static_cast<long long>(_forcedObjectOffIndex);
    row << ',' << objectIndex << ',' << acc.primitiveCount << ','
        << formatFixedLocal(visibleFrameFraction) << ','
        << formatFixedLocal(meanVisibleCoverage) << ','
        << formatFixedLocal(acc.maxVisibleCoverage) << ','
        << formatFixedLocal(meanDistance) << ','
        << formatFixedLocal(minDistance) << ','
        << formatFixedLocal(meanHitProbability) << ','
        << formatFixedLocal(meanRayHitScore) << ','
        << acc.totalHits << ',' << acc.totalRaysTested << ','
        << acc.toggleCount << ','
        << formatFixedLocal(activeFrameFraction) << ','
        << formatFixedLocal(residentFrameFraction) << ','
        << formatFixedLocal(streamingFrameFraction) << ','
        << formatFixedLocal(transportCriticalFrameFraction) << ','
        << formatFixedLocal(meanObjectImportance) << ','
        << formatFixedLocal(meanEstimatedBytes) << ','
        << formatFixedLocal(acc.emissiveImportance);
    _neuralFeatureStream << row.str() << '\n';
  }
  _neuralFeatureStream.flush();
  _neuralClipAccumulators.assign(_neuralClipAccumulators.size(),
                                 NeuralObjectClipAccumulator{});
}

void Renderer::setFrameCaptureEnabled(bool enabled) {
  if (_frameCaptureEnabled == enabled)
    return;

  _frameCaptureEnabled = enabled;
  if (_frameCaptureEnabled)
    ensureFrameCaptureDirectory();
  else if (_captureOutputsPending.load(std::memory_order_acquire))
    processPendingCapturedFrames();
}

void Renderer::setFrameCaptureInterval(size_t interval) {
  if (interval == 0)
    interval = 1;
  _frameCaptureInterval = interval;
}

void Renderer::setMaxRayDepth(uint32_t depth) {
  if (depth == 0)
    depth = 1;
  _pScene->maxRayDepth = depth;
}

uint32_t Renderer::maxRayDepth() const { return _pScene ? _pScene->maxRayDepth : 0; }

void Renderer::initializeBenchmarking() {
  const char *primaryEnv = std::getenv("METALPT_BENCH");
  const char *legacyEnv = std::getenv("METALAPT_BENCH");
  const char *probabilityEnv = std::getenv("METALPT_BENCH_LOG_PROBABILITIES");
  const char *heapShrinkEnv = std::getenv("MPT_HEAP_SHRINK");
  const char *env = primaryEnv ? primaryEnv : legacyEnv;
  if (!env)
    env = nullptr;

  if (!primaryEnv && legacyEnv) {
    printf("METALAPT_BENCH detected; please update to METALPT_BENCH for future runs.\n");
  }

  auto parseBoolEnv = [](const char *value) -> bool {
    if (!value)
      return false;
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    return normalized == "1" || normalized == "true" || normalized == "yes" ||
           normalized == "on";
  };

  if (env) {
    bool enable = parseBoolEnv(env);
    if (probabilityEnv) {
      _benchmarkLogProbabilities = parseBoolEnv(probabilityEnv);
    } else {
      _benchmarkLogProbabilities = true;
    }
    setBenchmarkMode(enable);
  }

  if (heapShrinkEnv) {
    applyHeapShrinkSetting(parseBoolEnv(heapShrinkEnv));
  }
}

void Renderer::applyHeapShrinkSetting(bool enabled) {
  _heapShrinkEnabled = enabled;
  _tlasHeap.setHeapShrinkEnabled(enabled);
  _dummyBlasResources.setHeapShrinkEnabled(enabled);
  for (auto &resident : _residentObjectGpuResources) {
    resident.resources.setHeapShrinkEnabled(enabled);
  }
}

void Renderer::updateHeapShrinkCandidates() {
  if (!_heapShrinkEnabled)
    return;
  for (auto &resident : _residentObjectGpuResources) {
    bool hasInFlight = resident.hasPendingCommands();
    NS::UInteger requiredBytes = resident.resources.currentRequiredBytes();
    resident.resources.maybeShrinkHeap(requiredBytes, hasInFlight);
  }
}

void Renderer::ensureBenchmarkStream() {
  if (!_benchmarkEnabled)
    return;
  if (_benchmarkStream.is_open())
    return;

  std::filesystem::path benchmarksDir = std::filesystem::current_path() / "Benchmarks";
  std::error_code ec;
  std::filesystem::create_directories(benchmarksDir, ec);

  auto now = std::chrono::system_clock::now();
  std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
  std::tm timeInfo{};
#if defined(_WIN32)
  localtime_s(&timeInfo, &nowTime);
#else
  localtime_r(&nowTime, &timeInfo);
#endif
  std::ostringstream nameBuilder;
  nameBuilder << "metrics_" << std::put_time(&timeInfo, "%Y%m%d_%H%M%S") << ".csv";
  _benchmarkFilePath = (benchmarksDir / nameBuilder.str()).string();

  _benchmarkStream.open(_benchmarkFilePath, std::ios::out | std::ios::trunc);
  if (_benchmarkStream.is_open()) {
    writeBenchmarkHeader();
  } else {
    printf("Failed to open benchmark log '%s'\n", _benchmarkFilePath.c_str());
  }
}

void Renderer::writeBenchmarkHeader() {
  if (!_benchmarkStream.is_open())
    return;
  if (_benchmarkHeaderWritten)
    return;

  _benchmarkStream
      << "frame,wall_seconds,cpu_ms,gpu_ms,rays_per_second,rays_cast,strategy,"
         "strategy_id,delta_time_seconds,min_samples_per_pixel,max_samples_per_pixel,"
         "primitive_activations,primitive_deactivations,object_activations,"
         "object_deactivations,objects_onload_requested,objects_offload_requested,"
         "onload_requested_mb,offload_requested_mb,blas_build_requests,tlas_rebuilds,"
         "tlas_refits,active_primitives,resident_primitives,total_primitives,"
         "active_triangles,resident_triangles,total_triangles,active_nodes,"
         "resident_nodes,total_nodes,active_objects,resident_objects,"
         "visible_primitives,visible_objects,"
         "primitive_hits_last_frame,primitive_rays_tested_last_frame,"
         "object_hits_last_frame,object_rays_tested_last_frame,"
         "avg_hit_probability,p95_hit_probability,probability_threshold,"
         "probability_target_fraction,probability_visible_floor,"
         "probability_target_primitives,"
         "probability_initial_desired_primitives,"
         "probability_final_desired_primitives,probability_trimmed_primitives,"
         "probability_budget_hit,"
         "camera_motion_metric,"
         "primitive_probabilities,"
         "object_probabilities,probabilistic_toggles,"
         "gpu_memory_mb,scratch_memory_mb,gpu_geometry_mb,gpu_textures_mb,"
         "gpu_restir_mb,gpu_renderer_mb,gpu_heaps_mb,gpu_staging_mb,gpu_other_mb,"
         "resident_geometry_memory_mb,strict_resident_geometry_memory_mb,"
         "resident_texture_memory_mb,residency_memory_mb,"
         "texture_memory_cap_mb,geometry_memory_cap_mb,"
         "total_memory_cap_mb,minimum_resident_footprint_mb,"
         "total_memory_cap_relaxed_mb,over_memory_cap,geometry_over_memory_cap,"
         "total_over_memory_cap,total_memory_overage_warnings,"
         "total_memory_cap_denials,total_memory_cap_non_residency_denials,"
         "total_memory_eviction_stall,geometry_cap_hits,"
         "geometry_hard_cap_denials,residency_compacted,"
         "ray_hit_decay,state_cooldown_frames,lod_toggle_budget,"
         "energy_toggle_budget,screen_toggle_budget,rayhit_toggle_budget,"
         "rayhit_target_fraction,rayhit_min_active,rayhit_rebuild_cooldown,"
         "rayhit_prior_enabled_low_hit_shrink,rayhit_prior_scale_low_hit_shrink,"
         "rayhit_prior_bias_favors_hot,"
         "lod_enter_distance,lod_exit_distance,lod_enter_view_margin,"
         "lod_exit_view_margin,energy_target_fraction,"
         "energy_min_active,energy_visibility_boost,screen_target_fraction,"
         "screen_min_pixels,screen_min_active,screen_min_pixel_skips,"
         "environment_target_fraction,"
         "environment_escape_threshold,env_high_escape,env_low_escape,global_env_escape,"
         "environment_activation_floor,environment_min_active,environment_toggle_budget,"
         "unified_offscreen_decay,unified_offscreen_floor,environment_depth_weights,"
         "environment_depth_radii";
  _benchmarkStream << '\n';
  _benchmarkHeaderWritten = true;
}

static std::string formatFixed(double value, int precision) {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(precision) << value;
  return ss.str();
}

static std::string formatFloatList(const std::vector<float> &values,
                                   int precision = 3) {
  if (values.empty())
    return "";

  std::ostringstream ss;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0)
      ss << '|';
    ss << formatFixed(values[i], precision);
  }
  return ss.str();
}

static void appendCsvEscaped(std::ostringstream &ss, const std::string &value) {
  ss << '"';
  for (char c : value) {
    if (c == '"')
      ss << '"';
    ss << c;
  }
  ss << '"';
}

void Renderer::writeBenchmarkRow(const BenchmarkSample &sample) {
  if (!_benchmarkStream.is_open())
    return;

  auto boolToInt = [](bool v) { return v ? 1 : 0; };

  std::ostringstream row;
  row << sample.frameIndex << ',' << formatFixed(sample.wallSeconds, 6) << ','
      << formatFixed(sample.cpuTimeSeconds * 1000.0, 3) << ','
      << formatFixed(sample.gpuTimeSeconds * 1000.0, 3) << ','
      << formatFixed(sample.raysPerSecond, 2) << ',' << sample.rayCount << ',';
  appendCsvEscaped(row, sample.strategyName);
  row << ',' << static_cast<int>(sample.strategy)
      << ',' << formatFixed(sample.deltaTimeSeconds, 6) << ','
      << sample.minSamplesPerPixel << ',' << sample.maxSamplesPerPixel << ','
      << sample.primitiveActivations << ',' << sample.primitiveDeactivations << ','
      << sample.objectActivations << ',' << sample.objectDeactivations << ','
      << sample.objectsOnloadRequested << ','
      << sample.objectsOffloadRequested << ','
      << formatFixed(sample.onloadRequestedMB, 3) << ','
      << formatFixed(sample.offloadRequestedMB, 3) << ','
      << sample.blasBuildRequests << ','
      << sample.tlasRebuilds << ','
      << sample.tlasRefits << ','
      << sample.activePrimitiveCount << ',' << sample.residentPrimitiveCount << ','
      << sample.totalPrimitiveCount << ',' << sample.activeTriangleCount << ','
      << sample.residentTriangleCount << ',' << sample.totalTriangleCount << ','
      << sample.activeNodeCount << ',' << sample.residentNodeCount << ','
      << sample.totalNodeCount << ',' << sample.activeObjectCount << ','
      << sample.residentObjectCount << ',' << sample.visiblePrimitiveCount << ','
      << sample.visibleObjectCount << ',' << sample.primitiveHitsLastFrame << ','
      << sample.primitiveRaysTestedLastFrame << ',' << sample.objectHitsLastFrame
      << ',' << sample.objectRaysTestedLastFrame << ','
      << formatFixed(sample.avgHitProbability, 6) << ','
      << formatFixed(sample.p95HitProbability, 6) << ','
      << formatFixed(sample.probabilityThreshold, 3) << ','
      << formatFixed(sample.probabilityTargetFraction, 3) << ','
      << formatFixed(sample.probabilityVisibleFloor, 3) << ','
      << sample.probabilityTargetPrimitives << ','
      << sample.probabilityInitialDesiredPrimitives << ','
      << sample.probabilityFinalDesiredPrimitives << ','
      << sample.probabilityTrimmedPrimitives << ','
      << boolToInt(sample.probabilityBudgetHit) << ','
      << formatFixed(sample.cameraMotionMetric, 3) << ',';
  appendCsvEscaped(row, sample.primitiveProbabilities);
  row << ',';
  appendCsvEscaped(row, sample.objectProbabilities);
  row << ','
      << sample.probabilisticToggles << ','
      << formatFixed(sample.gpuMemoryMB, 3) << ','
      << formatFixed(sample.scratchMemoryMB, 3) << ','
      << formatFixed(sample.gpuGeometryMB, 3) << ','
      << formatFixed(sample.gpuTextureMB, 3) << ','
      << formatFixed(sample.gpuRestirMB, 3) << ','
      << formatFixed(sample.gpuRendererMB, 3) << ','
      << formatFixed(sample.gpuHeapsMB, 3) << ','
      << formatFixed(sample.gpuStagingMB, 3) << ','
      << formatFixed(sample.gpuOtherMB, 3) << ','
      << formatFixed(sample.residentGeometryMemoryMB, 3) << ','
      << formatFixed(sample.strictResidentGeometryMemoryMB, 3) << ','
      << formatFixed(sample.residentTextureMemoryMB, 3) << ','
      << formatFixed(sample.residencyMemoryMB, 3) << ','
      << formatFixed(sample.textureMemoryCapMB, 3) << ','
      << formatFixed(sample.geometryMemoryCapMB, 3) << ','
      << formatFixed(sample.totalMemoryCapMB, 3) << ','
      << formatFixed(sample.minimumResidentFootprintMB, 3) << ','
      << formatFixed(sample.totalMemoryCapRelaxedMB, 3) << ','
      << boolToInt(sample.overMemoryCap) << ','
      << boolToInt(sample.geometryOverMemoryCap) << ','
      << boolToInt(sample.totalOverMemoryCap) << ','
      << sample.totalMemoryOverageWarnings << ','
      << sample.totalMemoryCapDeniedCount << ','
      << sample.totalMemoryCapNonResidencyDeniedCount << ','
      << boolToInt(sample.totalMemoryEvictionStall) << ','
      << sample.geometryCapHitCount << ','
      << sample.geometryHardCapDeniedCount << ','
      << boolToInt(sample.residentCompacted) << ','
      << formatFixed(_residencyConfig.rayHitDecay, 3) << ','
      << _residencyConfig.stateCooldownFrames << ','
      << _residencyConfig.lodMaxTogglesPerFrame << ','
      << _residencyConfig.energyMaxTogglesPerFrame << ','
      << _residencyConfig.screenFootprintMaxTogglesPerFrame << ','
      << _residencyConfig.rayHitMaxTogglesPerFrame << ','
      << formatFixed(_residencyConfig.rayHitTargetFraction, 3) << ','
      << _residencyConfig.rayHitMinActivePrimitives << ','
      << _residencyConfig.rayHitRebuildCooldownFrames << ','
      << boolToInt(_residencyConfig.enableRayHitPrior) << ','
      << formatFixed(_residencyConfig.rayHitPriorScale, 3) << ','
      << boolToInt(_residencyConfig.rayHitPriorFavorHighProbability) << ','
      << formatFixed(_residencyConfig.lodEnterDistance, 3) << ','
      << formatFixed(_residencyConfig.lodExitDistance, 3) << ','
      << formatFixed(_residencyConfig.lodEnterViewDegrees, 3) << ','
      << formatFixed(_residencyConfig.lodExitViewDegrees, 3) << ','
      << formatFixed(_residencyConfig.energyTargetFraction, 3) << ','
      << _residencyConfig.energyMinActivePrimitives << ','
      << formatFixed(_residencyConfig.energyVisibilityBoost, 3) << ','
      << formatFixed(_residencyConfig.screenFootprintTargetFraction, 3) << ','
      << formatFixed(_residencyConfig.screenFootprintMinPixelCoverage, 3) << ','
      << _residencyConfig.screenFootprintMinActivePrimitives << ','
      << sample.screenMinPixelCoverageSkips << ','
      << formatFixed(sample.environmentTargetActiveFraction, 3) << ','
      << formatFixed(sample.environmentEscapeThreshold, 3) << ','
      << formatFixed(sample.envHighEscapeThreshold, 3) << ','
      << formatFixed(sample.envLowEscapeThreshold, 3) << ','
      << formatFixed(sample.globalEnvEscape, 6) << ','
      << sample.environmentActivationFloor << ','
      << _residencyConfig.environmentMinActivePrimitives << ','
      << _residencyConfig.environmentMaxTogglesPerFrame << ','
      << formatFixed(_residencyConfig.unifiedOffscreenDecay, 3) << ','
      << formatFixed(_residencyConfig.unifiedOffscreenFloor, 3) << ',';
  appendCsvEscaped(row, sample.environmentDepthWeights);
  row << ',';
  appendCsvEscaped(row, sample.environmentDepthRadii);

  _benchmarkStream << row.str() << '\n';
  _benchmarkStream.flush();
}

std::string Renderer::residencyStrategyName(ResidencyStrategy strategy) const {
  switch (strategy) {
  case ResidencyStrategy::EnergyImportance:
    return "Energy importance";
  case ResidencyStrategy::RayHitBudget:
    return "Ray-hit budget";
  case ResidencyStrategy::ScreenSpaceFootprint:
    return "Screen-space footprint";
  case ResidencyStrategy::Probabilistic:
    return "Probabilistic";
  case ResidencyStrategy::UnifiedScore:
    return "Unified score";
  case ResidencyStrategy::UnifiedNeural:
    return "Unified neural";
  case ResidencyStrategy::PredictiveEnvironment:
    return "Predictive environment";
  case ResidencyStrategy::EnvironmentHit:
    return "Environment hit";
  case ResidencyStrategy::AlwaysResident:
    return "Always resident";
  case ResidencyStrategy::DistanceLOD:
  default:
    return "Distance-based LOD";
  }
}

void Renderer::setDeltaTime(double deltaSeconds) {
  if (deltaSeconds < 0.0)
    deltaSeconds = 0.0;

  _deltaTimeSeconds = deltaSeconds;
  Camera::deltaTime = static_cast<float>(_deltaTimeSeconds);
}

void Renderer::buildShaders() {
  using NS::StringEncoding::UTF8StringEncoding;

  if (_pPSO) {
    _pPSO->release();
    _pPSO = nullptr;
  }
  if (_pOverlayPSO) {
    _pOverlayPSO->release();
    _pOverlayPSO = nullptr;
  }
  if (_pOverlayCapturePSO) {
    _pOverlayCapturePSO->release();
    _pOverlayCapturePSO = nullptr;
  }
  if (_pPathTracePSO) {
    _pPathTracePSO->release();
    _pPathTracePSO = nullptr;
  }
  if (_pAdaptiveSamplingPSO) {
    _pAdaptiveSamplingPSO->release();
    _pAdaptiveSamplingPSO = nullptr;
  }
  NS::Error *pError = nullptr;
  MTL::Library *pLibrary = _pDevice->newDefaultLibrary();

  if (!pLibrary) {
    __builtin_printf("Failed to load Metal library\n");
    assert(false);
  }

  MTL::Function *pVertexFn = pLibrary->newFunction(
      NS::String::string("vertexMain", UTF8StringEncoding));
  MTL::Function *pFragFn = pLibrary->newFunction(
      NS::String::string("presentMain", UTF8StringEncoding));

  MTL::RenderPipelineDescriptor *pDesc =
      MTL::RenderPipelineDescriptor::alloc()->init();
  pDesc->setVertexFunction(pVertexFn);
  pDesc->setFragmentFunction(pFragFn);
  pDesc->colorAttachments()->object(0)->setPixelFormat(
      MTL::PixelFormat::PixelFormatRGBA16Float);

  _pPSO = _pDevice->newRenderPipelineState(pDesc, &pError);
  if (!_pPSO) {
    __builtin_printf("%s\n", pError->localizedDescription()->utf8String());
    assert(false);
  }

  pError = nullptr;
  MTL::Function *pOverlayVertexFn = pLibrary->newFunction(
      NS::String::string("frustumDebugVertexMain", UTF8StringEncoding));
  MTL::Function *pOverlayFragmentFn = pLibrary->newFunction(
      NS::String::string("frustumDebugFragmentMain", UTF8StringEncoding));
  if (pOverlayVertexFn && pOverlayFragmentFn) {
    MTL::RenderPipelineDescriptor *pOverlayDesc =
        MTL::RenderPipelineDescriptor::alloc()->init();
    pOverlayDesc->setVertexFunction(pOverlayVertexFn);
    pOverlayDesc->setFragmentFunction(pOverlayFragmentFn);
    pOverlayDesc->colorAttachments()->object(0)->setPixelFormat(
        MTL::PixelFormat::PixelFormatRGBA16Float);
    pOverlayDesc->setInputPrimitiveTopology(
        MTL::PrimitiveTopologyClass::PrimitiveTopologyClassLine);

    _pOverlayPSO = _pDevice->newRenderPipelineState(pOverlayDesc, &pError);
    if (!_pOverlayPSO && pError) {
      __builtin_printf("%s\n", pError->localizedDescription()->utf8String());
    }
    pOverlayDesc->release();

    pError = nullptr;
    MTL::RenderPipelineDescriptor *pOverlayCaptureDesc =
        MTL::RenderPipelineDescriptor::alloc()->init();
    pOverlayCaptureDesc->setVertexFunction(pOverlayVertexFn);
    pOverlayCaptureDesc->setFragmentFunction(pOverlayFragmentFn);
    pOverlayCaptureDesc->colorAttachments()->object(0)->setPixelFormat(
        MTL::PixelFormat::PixelFormatRGBA32Float);
    pOverlayCaptureDesc->setInputPrimitiveTopology(
        MTL::PrimitiveTopologyClass::PrimitiveTopologyClassLine);

    _pOverlayCapturePSO =
        _pDevice->newRenderPipelineState(pOverlayCaptureDesc, &pError);
    if (!_pOverlayCapturePSO && pError) {
      __builtin_printf("%s\n", pError->localizedDescription()->utf8String());
    }
    pOverlayCaptureDesc->release();
  }
  if (pOverlayVertexFn)
    pOverlayVertexFn->release();
  if (pOverlayFragmentFn)
    pOverlayFragmentFn->release();

  pError = nullptr;
  _useAccelerationStructureBindings = false;
  MTL::Function *pPathTraceFn = pLibrary->newFunction(
      NS::String::string("pathTraceKernel", UTF8StringEncoding));
  if (pPathTraceFn) {
    MTL::AutoreleasedComputePipelineReflection reflection = nullptr;
    _pPathTracePSO = _pDevice->newComputePipelineState(
        pPathTraceFn, MTL::PipelineOptionArgumentInfo, &reflection, &pError);
    if (!_pPathTracePSO) {
      if (pError) {
        __builtin_printf("%s\n", pError->localizedDescription()->utf8String());
      }
      pError = nullptr;
      reflection = nullptr;
      _pPathTracePSO =
          _pDevice->newComputePipelineState(pPathTraceFn, &pError);
      if (!_pPathTracePSO && pError) {
        __builtin_printf("%s\n", pError->localizedDescription()->utf8String());
      }
    }
    if (_pPathTracePSO && reflection) {
      NS::Array *arguments = reflection->arguments();
      if (arguments) {
        for (NS::UInteger i = 0; i < arguments->count(); ++i) {
          auto *argument =
              static_cast<MTL::Argument *>(arguments->object(i));
          if (!argument)
            continue;
          if (argument->type() ==
              MTL::ArgumentType::ArgumentTypeInstanceAccelerationStructure) {
            _useAccelerationStructureBindings = true;
            break;
          }
          if (argument->type() == MTL::ArgumentType::ArgumentTypeBuffer &&
              argument->bufferDataType() ==
                  MTL::DataType::DataTypeInstanceAccelerationStructure) {
            _useAccelerationStructureBindings = true;
            break;
          }
        }
      }
    }
    pPathTraceFn->release();
  }

  pError = nullptr;
  MTL::Function *pAdaptiveFn = pLibrary->newFunction(
      NS::String::string("adaptiveSamplingMain", UTF8StringEncoding));
  if (pAdaptiveFn) {
    _pAdaptiveSamplingPSO =
        _pDevice->newComputePipelineState(pAdaptiveFn, &pError);
    if (!_pAdaptiveSamplingPSO && pError) {
      __builtin_printf("%s\n", pError->localizedDescription()->utf8String());
    }
    pAdaptiveFn->release();
  }

  pVertexFn->release();
  pFragFn->release();
  pDesc->release();
  pLibrary->release();
}

Renderer::SceneAccelerationBuildResult
Renderer::buildSceneAccelerationStructures(size_t primitiveCount,
                                           size_t primitiveHitBytes) {
  SceneAccelerationBuildResult result{};

  MTL::CommandBuffer *clearCmd = nullptr;
  if (_pPrimitiveHitBufferGPU && primitiveHitBytes > 0 && _pCommandQueue) {
    clearCmd = _pCommandQueue->commandBuffer();
    if (clearCmd) {
      MTL::BlitCommandEncoder *blit = clearCmd->blitCommandEncoder();
      if (blit) {
        blit->fillBuffer(_pPrimitiveHitBufferGPU,
                         NS::Range::Make(0, primitiveHitBytes), 0);
        blit->endEncoding();
      }
    }
  }

  std::future<SceneAccelerationBuildResult> resultFuture;
  TaskLimiter *limiter = nullptr;

  if (_pScene) {
    TaskLimiter &limiterRef = sceneBvhTaskLimiter();
    limiterRef.acquire();
    limiter = &limiterRef;

    Scene *scene = _pScene;
    size_t primitiveCountCopy = primitiveCount;

    try {
      resultFuture = std::async(
          std::launch::async,
          [scene, primitiveCountCopy, limiter]() {
            struct LimiterGuard {
              TaskLimiter *limiter;
              ~LimiterGuard() {
                if (limiter)
                  limiter->release();
              }
            } guard{limiter};

            SceneAccelerationBuildResult threadResult{};
            if (scene) {
              scene->buildBVH();
              threadResult.blasNodeCount = scene->getBVHNodeCount();
              size_t tlasCount = 0;
              if (primitiveCountCopy > 0) {
                simd::float4 *tmp = scene->createTLASBuffer(tlasCount);
                if (tmp)
                  delete[] tmp;
              }
              threadResult.tlasNodeCount = tlasCount;
            }
            return threadResult;
          });
    } catch (...) {
      limiterRef.release();
      throw;
    }
  }

  if (clearCmd) {
    clearCmd->commit();
    clearCmd->waitUntilCompleted();
  }

  if (resultFuture.valid()) {
    try {
      result = resultFuture.get();
    } catch (...) {
      throw;
    }
  }

  return result;
}

void Renderer::updateVisibleScene() {
  resetProbabilisticResidencyState();
  if (!SceneLoader::LoadSceneFromXML("scene.xml", _pScene)) {
    std::filesystem::path alt =
        std::filesystem::path(__FILE__).parent_path() / "../scene_bistro_test_v2_distance.xml";
    SceneLoader::LoadSceneFromXML(alt.string(), _pScene);
  }

  Camera::screenSize = _pScene->screenSize;
  _animationFrame = 0;
  _observerActive = false;
  _primaryCameraTrail.clear();

  if (!_pScene->cameraPath.empty()) {
    const auto &k = _pScene->cameraPath.front();
    _primaryCameraState = makeCameraState(
        k.position, k.lookAt, k.up, _primaryCameraState.verticalFov,
        _primaryCameraState.focalLength, k.aperture, k.focusDistance,
        _primaryCameraState.forward, _primaryCameraState.up);
  }

  Camera::applyState(_primaryCameraState);
  _primaryCameraState = Camera::captureState();

  if (_pScene->hasObserverCamera()) {
    const auto &observer = _pScene->getObserverCamera();
    float fov = observer.verticalFov > 0.0f ? observer.verticalFov
                                            : _primaryCameraState.verticalFov;
    _observerCameraState =
        makeCameraState(observer.position, observer.lookAt, observer.up, fov,
                        _primaryCameraState.focalLength, observer.aperture,
                        observer.focusDistance,
                        _primaryCameraState.forward, _primaryCameraState.up);
  } else {
    _observerCameraState = _primaryCameraState;
  }

  Camera::applyState(_primaryCameraState);

  printf("Scene loaded: %zu total primitives (%zu spheres, %zu triangles, %zu "
         "rectangles)\n",
         _pScene->getPrimitiveCount(), _pScene->getSphereCount(),
         _pScene->getTriangleCount(), _pScene->getRectangleCount());

  _residencyConfig = _pScene->getResidencyParameters();
  _residencyConfig.normalizeEnvironmentDepthSettings();
  clearUnifiedNeuralModel();
  if (_pScene->getResidencyStrategy() == ResidencyStrategy::UnifiedNeural) {
    loadUnifiedNeuralModel(_residencyConfig.unifiedNeuralModelPath);
  }
  bool requiresCachedBlas =
      _pScene->getResidencyStrategy() != ResidencyStrategy::AlwaysResident;
  if (_residencyConfig.buildCachedBlas != requiresCachedBlas) {
    _residencyConfig.buildCachedBlas = requiresCachedBlas;
    _pScene->setResidencyParameters(_residencyConfig);
  }
  _textureResidencyMemoryCapMB =
      std::max(_pScene->getTextureResidencyMemoryCapMB(), 0.0);
  if (_textureResidencyMemoryCapMB <= 0.0)
    _textureResidencyMemoryCapMB = kDefaultTextureResidencyMemoryCapMB;
  printf("Texture residency memory cap: %.1f MB\n",
         _textureResidencyMemoryCapMB);
  _geometryResidencyMemoryCapMB =
      std::max(_pScene->getGeometryResidencyMemoryCapMB(), 0.0);
  if (_geometryResidencyMemoryCapMB <= 0.0)
    _geometryResidencyMemoryCapMB = kDefaultGeometryResidencyMemoryCapMB;
  printf("Geometry residency memory cap: %.1f MB\n",
         _geometryResidencyMemoryCapMB);
  _totalGpuMemoryCapMB = std::max(_pScene->getTotalGpuMemoryCapMB(), 0.0);
  if (_totalGpuMemoryCapMB <= 0.0)
    _totalGpuMemoryCapMB = kDefaultTotalGpuMemoryCapMB;
  printf("Total GPU memory hard cap (authoritative budget): %.1f MB\n",
         _totalGpuMemoryCapMB);
  _residentCompacted = _pScene->getStartCompacted();
  _compactionCooldown = 0;

  printf("LOD activation threshold: %.1f, deactivation threshold: %.1f (cooldown "
         "%u frames, toggle budget %zu)\n",
         _residencyConfig.lodEnterDistance, _residencyConfig.lodExitDistance,
         _residencyConfig.stateCooldownFrames,
         _residencyConfig.lodMaxTogglesPerFrame);

  const char *strategyName = nullptr;
  switch (_pScene->getResidencyStrategy()) {
  case ResidencyStrategy::EnergyImportance:
    strategyName = "Energy importance";
    break;
  case ResidencyStrategy::RayHitBudget:
    strategyName = "Ray-hit budget";
    break;
  case ResidencyStrategy::ScreenSpaceFootprint:
    strategyName = "Screen-space footprint";
    break;
  case ResidencyStrategy::UnifiedScore:
    strategyName = "Unified score";
    break;
  case ResidencyStrategy::UnifiedNeural:
    strategyName = "Unified neural";
    break;
  case ResidencyStrategy::PredictiveEnvironment:
    strategyName = "Predictive environment";
    break;
  case ResidencyStrategy::EnvironmentHit:
    strategyName = "Environment hit";
    break;
  case ResidencyStrategy::AlwaysResident:
    strategyName = "Always resident";
    break;
  case ResidencyStrategy::DistanceLOD:
  default:
    strategyName = "Distance-based LOD";
    break;
  }
  printf("Active primitive residency strategy: %s\n", strategyName);

  _alwaysResidentCache.reset();
  _forceAlwaysResidentActivation = true;
  _pendingAlwaysResidentPrewarm =
      _pScene->getResidencyStrategy() == ResidencyStrategy::AlwaysResident ||
      _residencyPreviewOnly;

  ++_cameraVersion;
  _depthWeightCameraVersion = 0;
  _coverageCameraVersion = 0;
  _boundsVersionCounter = 1;

  // Store full primitive list and initialize tracking
  _allPrimitives = _pScene->getPrimitives();
  size_t primCount = _allPrimitives.size();
  _activePrimitive.assign(primCount, false);
  _primitiveCooldown.assign(primCount, 0);
  _primitiveToResidentIndex.assign(primCount, -1);
  _primitiveToObject.assign(primCount, std::numeric_limits<size_t>::max());
  _primitiveBounds.resize(primCount);
  _primitiveBoundsVersion.assign(primCount, _boundsVersionCounter++);
  _primitiveImportance.assign(primCount, 0.0f);
  _objectImportance.clear();
  _objectImportanceHistory.clear();
  _energySortedIndices.clear();
  _primitiveHitScores.assign(primCount, 0.0f);
  _primitiveHitLastFrame.assign(primCount, 0);
  _primitiveRayContributions.assign(primCount, 0.0f);
  _primitiveRaysTestedLastFrame.assign(primCount, 0);
  _primitiveHitAlpha.assign(primCount, 1.0f);
  _primitiveHitBeta.assign(primCount, 1.0f);
  _primitiveHitProbability.assign(primCount, 0.5f);
  _primitiveHitVariance.assign(primCount, 1.0f / 12.0f);
  _primitivePosteriorMass.assign(primCount, 2.0f);
  _primitiveVisible.assign(primCount, 0);
  _rayHitSortedIndices.resize(primCount);
  _probabilitySortedIndices.resize(primCount);
  _primitiveScreenCoverage.assign(primCount, 0.0f);
  _primitiveDistanceFalloffCache.assign(primCount, 0.0f);
  _primitiveCoverageDirty.assign(primCount, 1);
  _primitiveCoverageBoundsVersion.assign(primCount, 0);
  _primitiveCoverageVisibilityKey.assign(primCount, 0xFF);
  _screenCoverageSortedIndices.resize(primCount);
  _totalPrimitiveImportance = 0.0f;

  _residentPrimitives.clear();
  _residentRemap.clear();
  _recentlyActivated.clear();
  _recentlyDeactivated.clear();
  _residentBuffersInitialized = false;
  _cachedPrimitiveData.clear();
  _cachedMaterialData.clear();
  _cachedPrimitiveIndices.clear();
  _cachedBVHNodes.clear();
  _cachedTLASNodes.clear();
  _cachedTriangleVertices.clear();
  _cachedTriangleIndices.clear();
  _cachedLightIndices.clear();
  _cachedLightCdf.clear();
  _cpuActiveMask.clear();

  size_t totalTriangleCount = _pScene->getTriangleCount();
  _maxPrimitiveCount = std::max<size_t>(primCount, 1);
  _maxTriangleVertexCount = std::max<size_t>(totalTriangleCount * 3, 1);
  _maxTriangleIndexCount = std::max<size_t>(totalTriangleCount, 1);
  std::vector<float> importancePartials;
  const unsigned int partialThreadCount =
      std::max(1u, std::thread::hardware_concurrency());
  importancePartials.reserve(partialThreadCount * 2);

  std::mutex partialMutex;
  parallelChunkedAsync(0, primCount, [&](size_t chunkBegin, size_t chunkEnd) {
    float localImportance = 0.0f;
    for (size_t i = chunkBegin; i < chunkEnd; ++i) {
      const Primitive &p = _allPrimitives[i];
      if (p.type == PrimitiveType::Sphere) {
        _primitiveBounds[i] =
            BoundingSphere{p.sphere.center, p.sphere.radius};
      } else if (p.type == PrimitiveType::Triangle) {
        simd::float3 c =
            (p.triangle.v0 + p.triangle.v1 + p.triangle.v2) / 3.0f;
        float r = simd::length(p.triangle.v0 - c);
        r = std::max(r, (float)simd::length(p.triangle.v1 - c));
        r = std::max(r, (float)simd::length(p.triangle.v2 - c));
        _primitiveBounds[i] = {c, r};
      } else {
        float r = simd::length(p.rectangle.u) + simd::length(p.rectangle.v);
        _primitiveBounds[i] = {p.rectangle.center, r};
      }

      _primitiveImportance[i] = primitiveImportance(p);
      if (i < _rayHitSortedIndices.size())
        _rayHitSortedIndices[i] = i;
      if (i < _probabilitySortedIndices.size())
        _probabilitySortedIndices[i] = i;
      if (i < _screenCoverageSortedIndices.size())
        _screenCoverageSortedIndices[i] = i;
      localImportance += std::max(_primitiveImportance[i], 0.0f);
    }

    std::lock_guard<std::mutex> lock(partialMutex);
    importancePartials.push_back(localImportance);
  });

  _totalPrimitiveImportance =
      std::accumulate(importancePartials.begin(), importancePartials.end(),
                      0.0f);

  constexpr size_t kStatsPerPrimitive = 2;
  constexpr size_t kGlobalRayStatsCount = 2;
  size_t hitCount = std::max<size_t>(_maxPrimitiveCount, 1);
  size_t hitBytes =
      hitCount * kStatsPerPrimitive * sizeof(uint32_t) +
      kGlobalRayStatsCount * sizeof(uint32_t);
  flushRayHitCopy();
  ensureBufferCapacity(_pPrimitiveHitBufferGPU, hitBytes,
                       _primitiveHitBufferCapacity, false,
                       MTL::ResourceStorageModePrivate,
                       GpuMemoryTracker::Category::RendererBuffers);
  ensureBufferCapacity(_pPrimitiveHitReadback, hitBytes,
                       _primitiveHitReadbackCapacity, false,
                       MTL::ResourceStorageModeShared,
                       GpuMemoryTracker::Category::Staging);
  if (uint32_t *hitPtr =
          _pPrimitiveHitReadback
              ? static_cast<uint32_t *>(_pPrimitiveHitReadback->contents())
              : nullptr) {
    std::memset(hitPtr, 0, hitBytes);
  }
  SceneAccelerationBuildResult accelerationBuild =
      buildSceneAccelerationStructures(primCount, hitBytes);

  _rayHitRebuildCooldown = 0;

  _maxBlasNodeCount =
      std::max<size_t>(accelerationBuild.blasNodeCount, size_t(1));
  _maxTlasNodeCount =
      std::max<size_t>(accelerationBuild.tlasNodeCount, size_t(1));
  _totalNodeCount = _maxBlasNodeCount + _maxTlasNodeCount;

  _allSceneObjects = _pScene->getObjects();
  size_t objectCount = _allSceneObjects.size();
  _objectBounds.resize(objectCount);
  _objectBoundsVersion.assign(objectCount, _boundsVersionCounter++);
  _objectActive.assign(objectCount, false);
  _objectCooldown.assign(objectCount, 0);
  _objectLastToggleFrame.assign(objectCount, 0);
  _objectImportance.assign(objectCount, 0.0f);
  _energySortedIndices.resize(objectCount);
  std::iota(_energySortedIndices.begin(), _energySortedIndices.end(), size_t(0));
  _residentObjectGpuResources.resize(objectCount);
  _objectHitAlpha.assign(objectCount, 1.0f);
  _objectHitBeta.assign(objectCount, 1.0f);
  _objectHitProbability.assign(objectCount, 0.5f);
  _objectHitVariance.assign(objectCount, 1.0f / 12.0f);
  _objectPosteriorMass.assign(objectCount, 2.0f);
  _objectExplorationScore.assign(objectCount, 0.0f);
  _objectHitLastFrame.assign(objectCount, 0);
  _objectRaysTestedLastFrame.assign(objectCount, 0);
  _objectVisible.assign(objectCount, 0);
  _objectVisibilityEvidence.assign(objectCount, 0.0f);
  _objectProbabilitySortedIndices.resize(objectCount);
  std::iota(_objectProbabilitySortedIndices.begin(),
            _objectProbabilitySortedIndices.end(), size_t(0));
  _objectDepthWeight.assign(objectCount, 1.0f);
  _objectDepthWeightVersion.assign(objectCount, 0);
  _depthWeightCameraVersion = 0;

  _meshGroups.clear();
  _meshGroups.reserve(objectCount);
  _objectPrimitiveCounts.assign(objectCount, 0);
  _objectActivePrimitiveCounts.assign(objectCount, 0);
  _anyMeshGroups = false;
  std::unordered_map<int, size_t> meshGroupLookup;
  meshGroupLookup.reserve(objectCount);

  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    auto &resident = _residentObjectGpuResources[objectIndex];
    resident.clearPendingCommand();
    resident.byteSize = 0;
    resident.state = ResidentObjectGpuResources::ResidencyState::Cold;
    resident.lastStateChange = std::chrono::steady_clock::now();
    resident.resources.setMemoryTracker(&_gpuMemoryTracker);
    resident.resources.initialize(_pDevice);
    resident.resources.setHeapShrinkEnabled(_heapShrinkEnabled);
  }

  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    const SceneObject &obj = _allSceneObjects[objectIndex];
    simd::float3 boundsMin = obj.boundsMin;
    simd::float3 boundsMax = obj.boundsMax;
    simd::float3 center = (boundsMin + boundsMax) * 0.5f;
    float radius = simd::length(boundsMax - center);
    _objectBounds[objectIndex] = {center, radius};

    size_t first = obj.firstPrimitive;
    size_t last = std::min(first + obj.primitiveCount, _primitiveToObject.size());
    size_t declaredCount = obj.primitiveCount;
    size_t actualCount = (last > first) ? (last - first) : 0;
    size_t primitiveCount = declaredCount > 0 ? declaredCount : actualCount;
    _objectPrimitiveCounts[objectIndex] = primitiveCount;

    size_t groupIndex = std::numeric_limits<size_t>::max();
    if (obj.meshGroupId >= 0) {
      _anyMeshGroups = true;
      auto it = meshGroupLookup.find(obj.meshGroupId);
      if (it == meshGroupLookup.end()) {
        groupIndex = _meshGroups.size();
        meshGroupLookup.emplace(obj.meshGroupId, groupIndex);
        _meshGroups.emplace_back();
      } else {
        groupIndex = it->second;
      }
    } else {
      groupIndex = _meshGroups.size();
      _meshGroups.emplace_back();
    }

    if (groupIndex < _meshGroups.size()) {
      auto &group = _meshGroups[groupIndex];
      if (group.objectIndices.empty())
        group.meshGroupId = obj.meshGroupId;
      group.objectIndices.push_back(objectIndex);
      group.primitiveCount += primitiveCount;
    }

    for (size_t prim = first; prim < last && prim < _primitiveToObject.size();
         ++prim)
      _primitiveToObject[prim] = objectIndex;
  }

  updateResidency(true, true);
  rebuildEnvironmentTexture();
}

void Renderer::prewarmAlwaysResidentResources() {
  if (!_pScene)
    return;
  if (!_residencyPreviewOnly &&
      _pScene->getResidencyStrategy() != ResidencyStrategy::AlwaysResident)
    return;

  _pendingAlwaysResidentPrewarm = false;

  updateAlwaysResident(true);
  flushResidencyChanges(true);

  if (!_pPathTracePSO || !_pCommandQueue)
    return;

  MTL::CommandBuffer *cmd = _pCommandQueue->commandBuffer();
  if (!cmd)
    return;

  MTL::BlitCommandEncoder *blit = nullptr;
  auto ensureSlotReady = [&](ManagedTextureSlot &slot) -> MTL::Texture * {
    if (!slot.descriptorValid)
      return nullptr;
    return requestResidentTexture(slot, cmd, blit);
  };

  size_t restirWriteIndex = 0;
  size_t restirReadIndex = 1 % _restirData0Slots.size();
  MTL::Texture *colorTexture = ensureSlotReady(_colorSlot);
  MTL::Texture *albedoTexture = ensureSlotReady(_albedoSlot);
  MTL::Texture *normalTexture = ensureSlotReady(_normalSlot);
  MTL::Texture *positionTexture = ensureSlotReady(_positionSlot);
  MTL::Texture *restirData0Write =
      ensureSlotReady(_restirData0Slots[restirWriteIndex]);
  MTL::Texture *restirData1Write =
      ensureSlotReady(_restirData1Slots[restirWriteIndex]);
  MTL::Texture *restirData2Write =
      ensureSlotReady(_restirData2Slots[restirWriteIndex]);
  MTL::Texture *restirSurfacePosWrite =
      ensureSlotReady(_restirSurfacePosSlots[restirWriteIndex]);
  MTL::Texture *restirSurfaceNormalWrite =
      ensureSlotReady(_restirSurfaceNormalSlots[restirWriteIndex]);
  MTL::Texture *restirData0Read =
      ensureSlotReady(_restirData0Slots[restirReadIndex]);
  MTL::Texture *restirData1Read =
      ensureSlotReady(_restirData1Slots[restirReadIndex]);
  MTL::Texture *restirData2Read =
      ensureSlotReady(_restirData2Slots[restirReadIndex]);
  MTL::Texture *restirSurfacePosRead =
      ensureSlotReady(_restirSurfacePosSlots[restirReadIndex]);
  MTL::Texture *restirSurfaceNormalRead =
      ensureSlotReady(_restirSurfaceNormalSlots[restirReadIndex]);

  if (blit)
    blit->endEncoding();

  bool haveAllTextures =
      colorTexture && albedoTexture && normalTexture && positionTexture &&
      restirData0Write && restirData1Write && restirData2Write &&
      restirSurfacePosWrite && restirSurfaceNormalWrite && restirData0Read &&
      restirData1Read && restirData2Read && restirSurfacePosRead &&
      restirSurfaceNormalRead;

  bool useAccelerationStructureLayout =
      _useAccelerationStructureBindings && _pTlasStructure &&
      _pGeometryHandleBuffer;

  if (!haveAllTextures) {
    trackFrameCommandBuffer(cmd);
    cmd->commit();
    return;
  }

  if (useAccelerationStructureLayout) {
    if (!_pSphereBuffer || !_pSphereMaterialBuffer || !_pUniformsBuffer ||
        !_pActiveBuffer || !_pLightIndexBuffer || !_pLightCdfBuffer ||
        !_pLightPdfLookupBuffer || !_pRestirStatsBuffer ||
        !_pPrimitiveRemapBuffer || !_pPrimitiveHitBufferGPU ||
        !_pInstanceBuffer) {
      trackFrameCommandBuffer(cmd);
      cmd->commit();
      return;
    }
  } else {
    if (!_pBVHBuffer || !_pSphereBuffer || !_pSphereMaterialBuffer ||
        !_pUniformsBuffer || !_pTriangleVertexBuffer ||
        !_pTriangleIndexBuffer || !_pPrimitiveIndexBuffer || !_pTLASBuffer ||
        !_pActiveBuffer || !_pLightIndexBuffer || !_pLightCdfBuffer ||
        !_pLightPdfLookupBuffer || !_pRestirStatsBuffer ||
        !_pPrimitiveRemapBuffer || !_pPrimitiveHitBufferGPU ||
        !_pInstanceBuffer) {
      trackFrameCommandBuffer(cmd);
      cmd->commit();
      return;
    }
  }

  MTL::ComputeCommandEncoder *compute = cmd->computeCommandEncoder();
  if (!compute) {
    trackFrameCommandBuffer(cmd);
    cmd->commit();
    return;
  }

  compute->setComputePipelineState(_pPathTracePSO);

  if (useAccelerationStructureLayout) {
    compute->setAccelerationStructure(_pTlasStructure, 0);
    compute->setBuffer(_pGeometryHandleBuffer, 0, 1);
    compute->setBuffer(_pSphereBuffer, 0, 2);
    compute->setBuffer(_pSphereMaterialBuffer, 0, 3);
    compute->setBuffer(_pUniformsBuffer, 0, 4);
    compute->setBuffer(_pActiveBuffer, 0, 5);
    compute->setBuffer(_pLightIndexBuffer, 0, 6);
    compute->setBuffer(_pLightCdfBuffer, 0, 7);
    compute->setBuffer(_pPrimitiveRemapBuffer, 0, 8);
    compute->setBuffer(_pPrimitiveHitBufferGPU, 0, 9);
    compute->setBuffer(_pInstanceBuffer, 0, 10);
    compute->setBuffer(_pLightPdfLookupBuffer, 0, 14);
    compute->setBuffer(_pRestirStatsBuffer, 0, 15);
    compute->setBuffer(_pPrimitiveHitBufferGPU, 0, 12);
    compute->setBuffer(_pInstanceBuffer, 0, 13);
  } else {
    compute->setBuffer(_pBVHBuffer, 0, 0);
    compute->setBuffer(_pSphereBuffer, 0, 1);
    compute->setBuffer(_pSphereMaterialBuffer, 0, 2);
    compute->setBuffer(_pUniformsBuffer, 0, 3);
    compute->setBuffer(_pTriangleVertexBuffer, 0, 4);
    compute->setBuffer(_pTriangleIndexBuffer, 0, 5);
    compute->setBuffer(_pPrimitiveIndexBuffer, 0, 6);
    compute->setBuffer(_pTLASBuffer, 0, 7);
    compute->setBuffer(_pActiveBuffer, 0, 8);
    compute->setBuffer(_pLightIndexBuffer, 0, 9);
    compute->setBuffer(_pLightCdfBuffer, 0, 10);
    compute->setBuffer(_pPrimitiveRemapBuffer, 0, 11);
    compute->setBuffer(_pPrimitiveHitBufferGPU, 0, 12);
    compute->setBuffer(_pInstanceBuffer, 0, 13);
    compute->setBuffer(_pLightPdfLookupBuffer, 0, 14);
    compute->setBuffer(_pRestirStatsBuffer, 0, 15);
  }

  compute->setTexture(colorTexture, 0);
  compute->setTexture(albedoTexture, 1);
  compute->setTexture(normalTexture, 2);
  compute->setTexture(positionTexture, 3);
  compute->setTexture(restirData0Write, 4);
  compute->setTexture(restirData1Write, 5);
  compute->setTexture(restirData2Write, 6);
  compute->setTexture(restirData0Read, 7);
  compute->setTexture(restirData1Read, 8);
  compute->setTexture(restirData2Read, 9);
  compute->setTexture(restirSurfacePosWrite, 10);
  compute->setTexture(restirSurfaceNormalWrite, 11);
  compute->setTexture(restirSurfacePosRead, 12);
  compute->setTexture(restirSurfaceNormalRead, 13);

  for (uint32_t texIdx = 0; texIdx < kMaxMaterialTextureSlots; ++texIdx) {
    MTL::Texture *materialTex =
        (texIdx < _materialTextures.size()) ? _materialTextures[texIdx]
                                            : nullptr;
    compute->setTexture(materialTex, 14 + texIdx);
  }

  compute->setTexture(_environmentTexture, 14 + kMaxMaterialTextureSlots);
  if (_environmentSampler)
    compute->setSamplerState(_environmentSampler, 0);

  TileDispatchRegion tileParams{};
  tileParams.originX = 0;
  tileParams.originY = 0;
  tileParams.width = 1;
  tileParams.height = 1;
  compute->setBytes(&tileParams, sizeof(TileDispatchRegion), 16);

  NS::UInteger tgWidth =
      std::max<NS::UInteger>(1, _pPathTracePSO->threadExecutionWidth());
  NS::UInteger maxThreads = std::max<NS::UInteger>(
      tgWidth, _pPathTracePSO->maxTotalThreadsPerThreadgroup());
  NS::UInteger tgHeight = std::max<NS::UInteger>(1, maxThreads / tgWidth);
  MTL::Size threadsPerThreadgroup = MTL::Size::Make(tgWidth, tgHeight, 1);

  compute->dispatchThreadgroups(MTL::Size::Make(1, 1, 1),
                                threadsPerThreadgroup);
  compute->endEncoding();

  trackFrameCommandBuffer(cmd);
  cmd->commit();
}


std::array<simd::float3, 8>
Renderer::buildFrustumCorners(const Camera::State &state, float nearDistance,
                              float farDistance) const {
  std::array<simd::float3, 8> corners{};
  if (nearDistance <= 0.0f || farDistance <= nearDistance)
    return corners;

  float aspectRatio = Camera::screenSize.y > 0.0f
                           ? Camera::screenSize.x / Camera::screenSize.y
                           : 1.0f;
  float fovRad = state.verticalFov * static_cast<float>(M_PI) / 180.0f;
  float tanHalfFov = std::tan(fovRad * 0.5f);

  simd::float3 forward = state.forward;
  if (simd::length_squared(forward) < 1e-6f)
    forward = {0.0f, 0.0f, -1.0f};
  forward = simd::normalize(forward);

  simd::float3 up = state.up;
  if (simd::length_squared(up) < 1e-6f)
    up = {0.0f, 1.0f, 0.0f};
  up = simd::normalize(up);

  simd::float3 right = simd::cross(forward, up);
  if (simd::length_squared(right) < 1e-6f)
    right = {1.0f, 0.0f, 0.0f};
  right = simd::normalize(right);
  up = simd::normalize(simd::cross(right, forward));

  auto buildPlane = [&](float distance, float halfWidth, float halfHeight,
                        size_t indexBase) {
    simd::float3 center = state.position + forward * distance;
    simd::float3 offsetRight = right * halfWidth;
    simd::float3 offsetUp = up * halfHeight;
    corners[indexBase + 0] = center + offsetUp - offsetRight;
    corners[indexBase + 1] = center + offsetUp + offsetRight;
    corners[indexBase + 2] = center - offsetUp + offsetRight;
    corners[indexBase + 3] = center - offsetUp - offsetRight;
  };

  float nearHalfHeight = tanHalfFov * nearDistance;
  float nearHalfWidth = nearHalfHeight * aspectRatio;
  float farHalfHeight = tanHalfFov * farDistance;
  float farHalfWidth = farHalfHeight * aspectRatio;

  buildPlane(nearDistance, nearHalfWidth, nearHalfHeight, 0);
  buildPlane(farDistance, farHalfWidth, farHalfHeight, 4);

  return corners;
}

void Renderer::recalculateViewport() {
  UniformsData *uData = (UniformsData *)_pUniformsBuffer->contents();
  if (!uData)
    return;

  populateViewportUniforms(*uData);

  markBufferModified(_pUniformsBuffer, NS::Range::Make(0, sizeof(UniformsData)));

}

bool Renderer::buildObjectBlas(size_t objectIndex, const SceneObject &object,
                               ResidentObjectGpuResources &resident,
                               bool coldOnloadRequest) {
  if (!_pDevice || !_pCommandQueue)
    return false;

  constexpr auto kPendingCommandWait =
      std::chrono::milliseconds(50);
  if (resident.hasPendingCommands()) {
    if (!resident.waitForPendingCommands(kPendingCommandWait)) {
      std::printf(
          "[BLAS] Deferred build for object %zu: waiting for prior command "
          "buffer to complete.\n",
          objectIndex);
      return false;
    }
  }

  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();

  auto cleanupPool = [&]() {
    if (pool) {
      pool->release();
      pool = nullptr;
    }
  };

  size_t first = object.firstPrimitive;
  size_t last = std::min(first + object.primitiveCount, _allPrimitives.size());

  if (last <= first) {
    resident.resources.ensureAccelerationStructure(0, nullptr);
    resident.byteSize = 0;
    resident.triangleCount = 0;
    resident.vertexCount = 0;
    resident.vertexBufferOffset = 0;
    resident.indexBufferOffset = 0;
    resident.geometryValid = false;
    resident.state = ResidentObjectGpuResources::ResidencyState::Resident;
    resident.lastStateChange = std::chrono::steady_clock::now();
    cleanupPool();
    return true;
  }

  const size_t primCount = last - first;
  struct ChunkGeometry {
    size_t offset = 0;
    std::vector<simd::float3> vertices;
    std::vector<uint32_t> indices;
  };

  std::vector<ChunkGeometry> chunks;
  std::mutex chunkMutex;

  parallelChunkedAsync(0, primCount, [&](size_t chunkBegin, size_t chunkEnd) {
    ChunkGeometry local{};
    local.offset = chunkBegin;
    local.vertices.reserve((chunkEnd - chunkBegin) * 3);
    local.indices.reserve((chunkEnd - chunkBegin) * 3);

    for (size_t prim = first + chunkBegin; prim < first + chunkEnd &&
                                        prim < _allPrimitives.size();
         ++prim) {
      const Primitive &p = _allPrimitives[prim];
      if (p.type != PrimitiveType::Triangle)
        continue;

      uint32_t baseIndex = static_cast<uint32_t>(local.vertices.size());
      local.vertices.push_back(p.triangle.v0);
      local.vertices.push_back(p.triangle.v1);
      local.vertices.push_back(p.triangle.v2);
      local.indices.push_back(baseIndex + 0);
      local.indices.push_back(baseIndex + 1);
      local.indices.push_back(baseIndex + 2);
    }

    if (!local.indices.empty()) {
      std::lock_guard<std::mutex> lock(chunkMutex);
      chunks.emplace_back(std::move(local));
    }
  });

  std::sort(chunks.begin(), chunks.end(),
            [](const ChunkGeometry &a, const ChunkGeometry &b) {
              return a.offset < b.offset;
            });

  size_t totalVertices = 0;
  size_t totalIndices = 0;
  for (const auto &chunk : chunks) {
    totalVertices += chunk.vertices.size();
    totalIndices += chunk.indices.size();
  }

  std::vector<simd::float3> vertices;
  std::vector<uint32_t> indices;
  vertices.reserve(totalVertices);
  indices.reserve(totalIndices);

  bool geometryValid = true;
  size_t expectedTriangles = totalIndices / 3;

  for (const auto &chunk : chunks) {
    if (chunk.indices.size() % 3 != 0) {
      geometryValid = false;
      break;
    }
    uint32_t baseOffset = static_cast<uint32_t>(vertices.size());

    uint32_t maxIndex = 0;
    for (uint32_t idx : chunk.indices)
      maxIndex = std::max(maxIndex, idx);

    if (chunk.vertices.empty() && !chunk.indices.empty()) {
      geometryValid = false;
      break;
    }
    if (maxIndex >= chunk.vertices.size()) {
      geometryValid = false;
      break;
    }

    vertices.insert(vertices.end(), chunk.vertices.begin(), chunk.vertices.end());
    indices.reserve(indices.size() + chunk.indices.size());
    for (uint32_t idx : chunk.indices)
      indices.push_back(baseOffset + idx);
  }

  size_t triangleCount = indices.size() / 3;
  if (!geometryValid || triangleCount != expectedTriangles) {
    cleanupPool();
    return false;
  }

  if (triangleCount == 0) {
    resident.resources.ensureAccelerationStructure(0, nullptr);
    resident.byteSize = 0;
    resident.triangleCount = 0;
    resident.vertexCount = 0;
    resident.vertexBufferOffset = 0;
    resident.indexBufferOffset = 0;
    resident.geometryValid = false;
    resident.state = ResidentObjectGpuResources::ResidencyState::Resident;
    resident.lastStateChange = std::chrono::steady_clock::now();
    cleanupPool();
    return true;
  }

  auto buildRequest = std::make_shared<PendingBlasBuild>();
  if (!buildRequest) {
    cleanupPool();
    return false;
  }

  buildRequest->renderer = this;
  buildRequest->resident = &resident;
  buildRequest->objectIndex = objectIndex;
  buildRequest->coldOnloadRequest = coldOnloadRequest;
  buildRequest->vertices = std::move(vertices);
  buildRequest->indices = std::move(indices);
  buildRequest->triangleCount = triangleCount;
  buildRequest->vertexCount = buildRequest->vertices.size();

  enqueueBlasBuild(buildRequest);

  cleanupPool();
  return true;
}

void Renderer::enqueueBlasBuild(
    const std::shared_ptr<PendingBlasBuild> &buildRequest) {
  if (!buildRequest)
    return;

  _pendingBlasBuilds.push_back(buildRequest);
  processBlasBuildQueue();
}

void Renderer::processBlasBuildQueue() {
  if (!_pDevice || !_pCommandQueue)
    return;

  size_t queueIterations = _pendingBlasBuilds.size();
  while (!_pendingBlasBuilds.empty() &&
         _activeBlasBuilds.size() < kMaxBlasBuildsInFlight &&
         queueIterations > 0) {
    auto buildRequest = _pendingBlasBuilds.front();
    _pendingBlasBuilds.pop_front();
    --queueIterations;

    if (buildRequest && buildRequest->resident &&
        buildRequest->resident->hasPendingCommands()) {
      std::printf(
          "[BLAS] Deferred queued build for object %zu: pending command buffer "
          "still in flight.\n",
          buildRequest->objectIndex);
      _pendingBlasBuilds.push_back(buildRequest);
      continue;
    }

    BlasBuildStartResult result = startBlasBuild(buildRequest);
    if (result == BlasBuildStartResult::Started) {
      _activeBlasBuilds.push_back(buildRequest);
      continue;
    }
    if (result == BlasBuildStartResult::DeferredCap) {
      _pendingBlasBuilds.push_back(buildRequest);
      continue;
    }

    if (buildRequest && buildRequest->resident &&
        buildRequest->objectIndex < _instanceRecords.size()) {
      transitionResidentToCold(buildRequest->objectIndex,
                               *buildRequest->resident,
                               _instanceRecords[buildRequest->objectIndex]);
    }
  }
}

Renderer::BlasBuildStartResult Renderer::startBlasBuild(
    const std::shared_ptr<PendingBlasBuild> &buildRequest) {
  if (!buildRequest || !buildRequest->resident || !_pDevice || !_pCommandQueue)
    return BlasBuildStartResult::Failed;

  auto &resident = *buildRequest->resident;

  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  auto cleanupPool = [&]() {
    if (pool) {
      pool->release();
      pool = nullptr;
    }
  };

  const NS::UInteger vertexBytes = static_cast<NS::UInteger>(
      buildRequest->vertices.size() * sizeof(simd::float3));
  const NS::UInteger indexBytes = static_cast<NS::UInteger>(
      buildRequest->indices.size() * sizeof(uint32_t));

  std::string blasLabel = "ObjectBLAS_" + std::to_string(buildRequest->objectIndex);
  std::string vertexLabel =
      "ObjectVertices_" + std::to_string(buildRequest->objectIndex);
  std::string indexLabel =
      "ObjectIndices_" + std::to_string(buildRequest->objectIndex);

  auto geometryDesc = MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()
                          ->init();
  if (!geometryDesc) {
    cleanupPool();
    return BlasBuildStartResult::Failed;
  }
  geometryDesc->setOpaque(true);

  NS::Object *descriptorObjects[] = {geometryDesc};
  auto geometryArray = NS::Array::alloc()->init(descriptorObjects, 1);
  auto accelDesc =
      MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
  if (!geometryArray || !accelDesc) {
    if (geometryArray)
      geometryArray->release();
    geometryDesc->release();
    if (accelDesc)
      accelDesc->release();
    cleanupPool();
    return BlasBuildStartResult::Failed;
  }

  accelDesc->setGeometryDescriptors(geometryArray);

  NS::UInteger alignedVertexSize =
      vertexBytes > 0 ? resident.resources.alignedHeapSize(vertexBytes) : 0;
  NS::UInteger alignedIndexSize =
      indexBytes > 0 ? resident.resources.alignedHeapSize(indexBytes) : 0;

  MTL::Buffer *vertexBuffer = nullptr;
  MTL::Buffer *indexBuffer = nullptr;

  bool capDenied = false;
  auto requestGeometryBuffers = [&]() -> bool {
    NS::UInteger initialHeapBytes = alignedVertexSize + alignedIndexSize;
    if (initialHeapBytes > 0) {
      if (!checkGeometryResidencyCap(buildRequest->objectIndex,
                                     static_cast<size_t>(initialHeapBytes),
                                     resident.byteSize, "initial buffers")) {
        capDenied = true;
        return false;
      }
      resident.resources.ensureHeapCapacity(initialHeapBytes);
    }

    vertexBuffer = resident.resources.ensureVertexBuffer(vertexBytes,
                                                        vertexLabel.c_str());
    indexBuffer = resident.resources.ensureIndexBuffer(indexBytes,
                                                      indexLabel.c_str());
    return vertexBuffer && indexBuffer;
  };

  if (!requestGeometryBuffers()) {
    geometryArray->release();
    geometryDesc->release();
    accelDesc->release();
    cleanupPool();
    return capDenied ? BlasBuildStartResult::DeferredCap
                     : BlasBuildStartResult::Failed;
  }

  auto configureGeometryDescriptor = [&]() {
    geometryDesc->setVertexBuffer(vertexBuffer);
    geometryDesc->setVertexBufferOffset(0);
    geometryDesc->setVertexStride(sizeof(simd::float3));
    geometryDesc->setVertexFormat(
        MTL::AttributeFormat::AttributeFormatFloat3);
    geometryDesc->setIndexBuffer(indexBuffer);
    geometryDesc->setIndexBufferOffset(0);
    geometryDesc->setIndexType(MTL::IndexType::IndexTypeUInt32);
    geometryDesc->setTriangleCount(
        static_cast<NS::UInteger>(buildRequest->triangleCount));
  };

  NS::UInteger totalHeapBytes = 0;
  NS::UInteger alignedAccelerationSize = 0;
  MTL::AccelerationStructureSizes sizes{};
  MTL::SizeAndAlign heapAlign{};

  while (true) {
    configureGeometryDescriptor();

    sizes = _pDevice->accelerationStructureSizes(accelDesc);
    heapAlign = _pDevice->heapAccelerationStructureSizeAndAlign(accelDesc);

    alignedAccelerationSize = static_cast<NS::UInteger>(
        alignTo(static_cast<size_t>(heapAlign.size),
                static_cast<size_t>(heapAlign.align)));
    alignedAccelerationSize =
        resident.resources.alignedHeapSize(alignedAccelerationSize);

    NS::UInteger requiredTotal = alignedAccelerationSize + alignedVertexSize +
                                 alignedIndexSize;

    if (requiredTotal > totalHeapBytes) {
      totalHeapBytes = requiredTotal;
      if (!checkGeometryResidencyCap(buildRequest->objectIndex,
                                     static_cast<size_t>(totalHeapBytes),
                                     resident.byteSize, "heap resize")) {
        geometryArray->release();
        geometryDesc->release();
        accelDesc->release();
        cleanupPool();
        return BlasBuildStartResult::DeferredCap;
      }
      resident.resources.ensureHeapCapacity(totalHeapBytes);

      if (!requestGeometryBuffers()) {
        geometryArray->release();
        geometryDesc->release();
        accelDesc->release();
        cleanupPool();
        return capDenied ? BlasBuildStartResult::DeferredCap
                         : BlasBuildStartResult::Failed;
      }
      continue;
    }

    totalHeapBytes = requiredTotal;
    break;
  }

  configureGeometryDescriptor();

  auto accelerationStructure = resident.resources.ensureAccelerationStructure(
      sizes.accelerationStructureSize, blasLabel.c_str());

  if (!accelerationStructure) {
    geometryArray->release();
    geometryDesc->release();
    accelDesc->release();
    cleanupPool();
    return BlasBuildStartResult::Failed;
  }

  MTL::Buffer *vertexStaging = nullptr;
  MTL::Buffer *indexStaging = nullptr;
  if (vertexBytes > 0) {
    vertexStaging =
        allocateBuffer(vertexBytes, MTL::ResourceStorageModeShared,
                       GpuMemoryTracker::Category::Staging,
                       "BLASVertexStaging");
    if (vertexStaging) {
      std::memcpy(vertexStaging->contents(), buildRequest->vertices.data(),
                  vertexBytes);
      markBufferModified(vertexStaging, NS::Range::Make(0, vertexBytes));
    }
  }
  if (indexBytes > 0) {
    indexStaging =
        allocateBuffer(indexBytes, MTL::ResourceStorageModeShared,
                       GpuMemoryTracker::Category::Staging,
                       "BLASIndexStaging");
    if (indexStaging) {
      std::memcpy(indexStaging->contents(), buildRequest->indices.data(),
                  indexBytes);
      markBufferModified(indexStaging, NS::Range::Make(0, indexBytes));
    }
  }

  if ((vertexBytes > 0 && !vertexStaging) ||
      (indexBytes > 0 && !indexStaging)) {
    if (indexStaging)
      releaseTrackedResource(indexStaging);
    if (vertexStaging)
      releaseTrackedResource(vertexStaging);
    geometryArray->release();
    geometryDesc->release();
    accelDesc->release();
    cleanupPool();
    return BlasBuildStartResult::Failed;
  }

  NS::UInteger requestedScratchSize = sizes.buildScratchBufferSize;
  MTL::Buffer *scratchBuffer = nullptr;
  buildRequest->scratchBuffer = nullptr;
  buildRequest->scratchSize = 0;
  if (requestedScratchSize > 0) {
    bool scratchReused = false;
    NS::UInteger allocatedScratchSize = requestedScratchSize;
    scratchBuffer = acquireBlasScratchBuffer(requestedScratchSize,
                                            allocatedScratchSize,
                                            scratchReused);
    if (!scratchBuffer) {
      if (indexStaging)
        releaseTrackedResource(indexStaging);
      if (vertexStaging)
        releaseTrackedResource(vertexStaging);
      geometryArray->release();
      geometryDesc->release();
      accelDesc->release();
      cleanupPool();
      return BlasBuildStartResult::Failed;
    }

    buildRequest->scratchBuffer = scratchBuffer;
    buildRequest->scratchSize = allocatedScratchSize;
    (void)scratchReused;
  }

  auto commandBuffer = _pCommandQueue->commandBuffer();
  if (!commandBuffer) {
    if (scratchBuffer) {
      recycleBlasScratchBuffer(scratchBuffer, buildRequest->scratchSize);
      buildRequest->scratchBuffer = nullptr;
      buildRequest->scratchSize = 0;
    }
    if (indexStaging)
      releaseTrackedResource(indexStaging);
    if (vertexStaging)
      releaseTrackedResource(vertexStaging);
    geometryArray->release();
    geometryDesc->release();
    accelDesc->release();
    cleanupPool();
    return BlasBuildStartResult::Failed;
  }

  MTL::BlitCommandEncoder *blitEncoder = nullptr;
  auto ensureBlitEncoder = [&]() -> MTL::BlitCommandEncoder * {
    if (!blitEncoder)
      blitEncoder = commandBuffer->blitCommandEncoder();
    return blitEncoder;
  };

  if (vertexStaging && vertexBytes > 0)
    ensureBlitEncoder()->copyFromBuffer(vertexStaging, 0, vertexBuffer, 0,
                                        vertexBytes);
  if (indexStaging && indexBytes > 0)
    ensureBlitEncoder()->copyFromBuffer(indexStaging, 0, indexBuffer, 0,
                                        indexBytes);

  if (blitEncoder)
    blitEncoder->endEncoding();

  auto asEncoder = commandBuffer->accelerationStructureCommandEncoder();
  if (!asEncoder) {
    if (scratchBuffer) {
      recycleBlasScratchBuffer(scratchBuffer, buildRequest->scratchSize);
      buildRequest->scratchBuffer = nullptr;
      buildRequest->scratchSize = 0;
    }
    if (indexStaging)
      releaseTrackedResource(indexStaging);
    if (vertexStaging)
      releaseTrackedResource(vertexStaging);
    geometryArray->release();
    geometryDesc->release();
    accelDesc->release();
    cleanupPool();
    return BlasBuildStartResult::Failed;
  }

  asEncoder->buildAccelerationStructure(accelerationStructure, accelDesc,
                                        scratchBuffer, 0);
  asEncoder->endEncoding();

  buildRequest->geometryDesc = geometryDesc;
  buildRequest->geometryArray = geometryArray;
  buildRequest->accelDesc = accelDesc;
  buildRequest->accelerationStructure = accelerationStructure;
  buildRequest->vertexStaging = vertexStaging;
  buildRequest->indexStaging = indexStaging;
  buildRequest->scratchBuffer = scratchBuffer;
  buildRequest->commandBuffer = commandBuffer;
  buildRequest->totalHeapBytes = totalHeapBytes;

  if (buildRequest->coldOnloadRequest) {
    _frameOnloadRequestedBytes += static_cast<size_t>(buildRequest->totalHeapBytes);
  }

  // Release CPU copies now that staging buffers are populated.
  buildRequest->vertices.clear();
  buildRequest->vertices.shrink_to_fit();
  buildRequest->indices.clear();
  buildRequest->indices.shrink_to_fit();

  resident.transitionToStreaming(commandBuffer);

  auto completion = [this, buildRequest](bool success) {
    this->handleCompletedBlasBuild(buildRequest, success);
  };

  if (!submitAsyncCommandBuffer(commandBuffer, completion)) {
    resident.clearPendingCommand();
    buildRequest->releaseResources();
    cleanupPool();
    return BlasBuildStartResult::Failed;
  }

  cleanupPool();
  return BlasBuildStartResult::Started;
}

void Renderer::handleCompletedBlasBuild(
    const std::shared_ptr<PendingBlasBuild> &buildRequest, bool success) {
  if (!buildRequest || !buildRequest->resident)
    return;

  auto &resident = *buildRequest->resident;

  resident.clearPendingCommand();

  if (success) {
    resident.byteSize = buildRequest->totalHeapBytes;
    resident.triangleCount = buildRequest->triangleCount;
    resident.vertexCount = buildRequest->vertexCount;
    resident.vertexBufferOffset = 0;
    resident.indexBufferOffset = 0;
    resident.geometryValid = buildRequest->triangleCount > 0;
    resident.state = ResidentObjectGpuResources::ResidencyState::Resident;
    resident.lastStateChange = std::chrono::steady_clock::now();
  } else if (buildRequest->objectIndex < _instanceRecords.size()) {
    transitionResidentToCold(buildRequest->objectIndex, resident,
                             _instanceRecords[buildRequest->objectIndex]);
  } else {
    resident.geometryValid = false;
  }

  buildRequest->releaseResources();

  std::printf(
      "[BLAS] Build %s for object %zu complete. Scratch pool in-flight=%zu "
      "bytes, available=%zu bytes (created=%zu, reused=%zu).\n",
      success ? "succeeded" : "failed", buildRequest->objectIndex,
      _blasScratchPoolInUseBytes, _blasScratchPoolAvailableBytes,
      _blasScratchPoolCreatedCount, _blasScratchPoolReusedCount);

  auto it = std::find(_activeBlasBuilds.begin(), _activeBlasBuilds.end(),
                      buildRequest);
  if (it != _activeBlasBuilds.end())
    _activeBlasBuilds.erase(it);

  processBlasBuildQueue();
}

bool Renderer::transitionResidentToCold(size_t objectIndex,
                                        ResidentObjectGpuResources &resident,
                                        BlasInstanceRecord &instanceRecord,
                                        bool removePending) {
  std::chrono::steady_clock::time_point frameWaitSnapshot;
  if (!waitForPendingFrameCommands(kFrameCommandBufferWaitTimeout,
                                   &frameWaitSnapshot))
    return false;

  waitForPendingTlasBuild();

  std::unique_lock<std::mutex> lock(_frameCommandBufferMutex);
  auto newerSubmission = std::find_if(
      _frameCommandBuffers.begin(), _frameCommandBuffers.end(),
      [frameWaitSnapshot](const FrameCommandBufferRecord &record) {
        return record.trackedSince >= frameWaitSnapshot;
      });
  if (newerSubmission != _frameCommandBuffers.end()) {
    lock.unlock();
    return false;
  }

  bool transitioned = resident.transitionToCold(instanceRecord);
  lock.unlock();

  if (!transitioned) {
    if (removePending)
      _pendingBlasEvictions.enqueue(objectIndex, resident);
    return false;
  }

  if (removePending)
    _pendingBlasEvictions.cancel(objectIndex, resident);

  return true;
}

void Renderer::requestResidentEviction(size_t objectIndex,
                                       ResidentObjectGpuResources &resident,
                                       BlasInstanceRecord &instanceRecord) {
  if (objectIndex >= _residentObjectGpuResources.size())
    return;

  bool hasPendingCommand = resident.hasPendingCommands();
  if (!resident.isResident() && !hasPendingCommand) {
    transitionResidentToCold(objectIndex, resident, instanceRecord);
    return;
  }

  resident.transitionToStreaming();
  resident.geometryValid = false;
  _pendingBlasEvictions.enqueue(objectIndex, resident);
}

void Renderer::cancelPendingResidentEviction(size_t objectIndex,
                                             ResidentObjectGpuResources &resident) {
  _pendingBlasEvictions.cancel(objectIndex, resident);
}

void Renderer::handleDeferredBlasEvictions(MTL::CommandBuffer *commandBuffer,
                                           bool success) {
  if (!commandBuffer)
    return;

  std::chrono::steady_clock::time_point frameWaitSnapshot;
  bool ready =
      success && waitForPendingFrameCommands(kFrameCommandBufferWaitTimeout,
                                             &frameWaitSnapshot);

  _pendingBlasEvictions.complete(
      commandBuffer, success && ready,
      [this](ResidentObjectGpuResources &resident, size_t objectIndex) {
        if (objectIndex >= _instanceRecords.size())
          return;
        auto &instanceRecord = _instanceRecords[objectIndex];
        if (!transitionResidentToCold(objectIndex, resident, instanceRecord,
                                      false)) {
          _pendingBlasEvictions.enqueue(objectIndex, resident);
        }
      });
}

bool Renderer::submitAsyncCommandBuffer(
    MTL::CommandBuffer *commandBuffer,
    std::function<void(bool)> completion) {
  if (!commandBuffer)
    return false;

  if (completion) {
    commandBuffer->addCompletedHandler(
        [completion = std::move(completion)](MTL::CommandBuffer *cmd) {
          bool success =
              cmd->status() == MTL::CommandBufferStatusCompleted;
          auto completionCopy = completion;
          dispatch_async(dispatch_get_main_queue(), ^{
            if (completionCopy)
              completionCopy(success);
          });
        });
  }

  commandBuffer->commit();
  return true;
}

bool Renderer::ensureDummyBlas() {
  if (_pDummyBlas)
    return true;

  if (!_pDevice || !_pCommandQueue)
    return false;

  if (_dummyBlasBuildInFlight)
    return false;

  _dummyBlasResources.initialize(_pDevice);
  _dummyBlasResources.setHeapShrinkEnabled(_heapShrinkEnabled);

  MTL::AccelerationStructureTriangleGeometryDescriptor *geometryDesc =
      MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
  geometryDesc->setOpaque(true);
  const NS::UInteger dummyVertexSize =
      static_cast<NS::UInteger>(sizeof(simd::float3) * 3);
  const NS::UInteger dummyIndexSize =
      static_cast<NS::UInteger>(sizeof(uint32_t) * 3);
  MTL::Buffer *dummyVertexBuffer = _dummyBlasResources.ensureVertexBuffer(
      dummyVertexSize, "DummyBLASVertices");
  MTL::Buffer *dummyIndexBuffer = _dummyBlasResources.ensureIndexBuffer(
      dummyIndexSize, "DummyBLASIndices");
  if (!dummyVertexBuffer || !dummyIndexBuffer) {
    geometryDesc->release();
    return false;
  }
  geometryDesc->setVertexStride(sizeof(simd::float3));
  geometryDesc->setVertexFormat(MTL::AttributeFormat::AttributeFormatFloat3);
  geometryDesc->setIndexType(MTL::IndexType::IndexTypeUInt32);
  geometryDesc->setTriangleCount(0);
  geometryDesc->setVertexBuffer(dummyVertexBuffer);
  geometryDesc->setVertexBufferOffset(0);
  geometryDesc->setIndexBuffer(dummyIndexBuffer);
  geometryDesc->setIndexBufferOffset(0);

  struct ZeroRequest {
    MTL::Buffer *target = nullptr;
    MTL::Buffer *staging = nullptr;
    NS::UInteger size = 0;
  };

  auto releaseZeroRequest = [this](ZeroRequest &request) {
    if (request.staging) {
      releaseTrackedResource(request.staging);
      request.staging = nullptr;
    }
    request.target = nullptr;
    request.size = 0;
  };

  ZeroRequest vertexZeroRequest;
  ZeroRequest indexZeroRequest;

  auto prepareZeroRequest = [&](MTL::Buffer *buffer, NS::UInteger size,
                                ZeroRequest &request) -> bool {
    if (!buffer || size == 0)
      return true;

    MTL::StorageMode storageMode = buffer->storageMode();
    if (storageMode == MTL::StorageMode::StorageModeShared ||
        storageMode == MTL::StorageMode::StorageModeManaged) {
      if (void *contents = buffer->contents())
        std::memset(contents, 0, size);
      return true;
    }

    MTL::Buffer *staging =
        allocateBuffer(size, MTL::ResourceStorageModeShared,
                       GpuMemoryTracker::Category::Staging,
                       "DummyBLASZeroStaging");
    if (!staging)
      return false;
    if (void *contents = staging->contents())
      std::memset(contents, 0, size);
    request.target = buffer;
    request.staging = staging;
    request.size = size;
    return true;
  };

  if (!prepareZeroRequest(dummyVertexBuffer, dummyVertexSize,
                          vertexZeroRequest) ||
      !prepareZeroRequest(dummyIndexBuffer, dummyIndexSize, indexZeroRequest)) {
    releaseZeroRequest(vertexZeroRequest);
    releaseZeroRequest(indexZeroRequest);
    geometryDesc->release();
    return false;
  }

  NS::Object *geometryObjects[] = {geometryDesc};
  NS::Array *geometryArray = NS::Array::alloc()->init(geometryObjects, 1);

  MTL::PrimitiveAccelerationStructureDescriptor *accelDesc =
      MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
  accelDesc->setGeometryDescriptors(geometryArray);

  auto releaseDescriptors = [&]() {
    if (geometryArray) {
      geometryArray->release();
      geometryArray = nullptr;
    }
    if (geometryDesc) {
      geometryDesc->release();
      geometryDesc = nullptr;
    }
    if (accelDesc) {
      accelDesc->release();
      accelDesc = nullptr;
    }
  };

  MTL::AccelerationStructureSizes sizes =
      _pDevice->accelerationStructureSizes(accelDesc);

  NS::UInteger alignedAccelerationSize =
      _dummyBlasResources.alignedHeapSize(sizes.accelerationStructureSize);
  NS::UInteger totalRequestedBytes =
      alignedAccelerationSize + dummyVertexSize + dummyIndexSize;
  constexpr NS::UInteger kDummyBlasMemoryBudgetBytes = 1024ull * 1024ull;
  static bool loggedDummyUsage = false;
  if (totalRequestedBytes > kDummyBlasMemoryBudgetBytes) {
    std::printf(
        "Dummy BLAS placeholder exceeded memory budget: %llu bytes used (limit %llu bytes).\n",
        static_cast<unsigned long long>(totalRequestedBytes),
        static_cast<unsigned long long>(kDummyBlasMemoryBudgetBytes));
  } else if (!loggedDummyUsage &&
             totalRequestedBytes > kDummyBlasMemoryBudgetBytes / 2) {
    std::printf(
        "Dummy BLAS placeholder using %llu / %llu bytes of budget.\n",
        static_cast<unsigned long long>(totalRequestedBytes),
        static_cast<unsigned long long>(kDummyBlasMemoryBudgetBytes));
    loggedDummyUsage = true;
  }
  assert(totalRequestedBytes <= kDummyBlasMemoryBudgetBytes &&
         "Dummy BLAS memory usage exceeds placeholder budget");

  MTL::AccelerationStructure *structure =
      _dummyBlasResources.ensureAccelerationStructure(
          sizes.accelerationStructureSize, "DummyBLAS");

  if (!structure) {
    releaseZeroRequest(vertexZeroRequest);
    releaseZeroRequest(indexZeroRequest);
    releaseDescriptors();
    return false;
  }

  NS::UInteger scratchSize = sizes.buildScratchBufferSize;
  MTL::Buffer *scratchBuffer = nullptr;
  if (scratchSize > 0)
    scratchBuffer =
        allocateBuffer(scratchSize, MTL::ResourceStorageModePrivate,
                       GpuMemoryTracker::Category::Scratch,
                       "DummyBLASScratch");

  MTL::CommandBuffer *commandBuffer = _pCommandQueue->commandBuffer();
  if (!commandBuffer) {
    if (scratchBuffer)
      releaseTrackedResource(scratchBuffer);
    releaseZeroRequest(vertexZeroRequest);
    releaseZeroRequest(indexZeroRequest);
    releaseDescriptors();
    return false;
  }

  if (vertexZeroRequest.staging || indexZeroRequest.staging) {
    MTL::BlitCommandEncoder *blitEncoder = commandBuffer->blitCommandEncoder();
    if (!blitEncoder) {
      if (scratchBuffer)
        releaseTrackedResource(scratchBuffer);
      releaseZeroRequest(vertexZeroRequest);
      releaseZeroRequest(indexZeroRequest);
      releaseDescriptors();
      return false;
    }
    if (vertexZeroRequest.staging) {
      blitEncoder->copyFromBuffer(vertexZeroRequest.staging, 0,
                                  vertexZeroRequest.target, 0,
                                  vertexZeroRequest.size);
    }
    if (indexZeroRequest.staging) {
      blitEncoder->copyFromBuffer(indexZeroRequest.staging, 0,
                                  indexZeroRequest.target, 0,
                                  indexZeroRequest.size);
    }
    blitEncoder->endEncoding();
  }

  MTL::AccelerationStructureCommandEncoder *encoder =
      commandBuffer->accelerationStructureCommandEncoder();
  if (!encoder) {
    if (scratchBuffer)
      releaseTrackedResource(scratchBuffer);
    releaseZeroRequest(vertexZeroRequest);
    releaseZeroRequest(indexZeroRequest);
    releaseDescriptors();
    return false;
  }

  encoder->buildAccelerationStructure(structure, accelDesc, scratchBuffer, 0);
  encoder->endEncoding();

  struct DummyBlasBuildContext {
    MTL::Buffer *vertexStaging = nullptr;
    MTL::Buffer *indexStaging = nullptr;
    MTL::Buffer *scratchBuffer = nullptr;
  };

  auto buildContext = std::make_shared<DummyBlasBuildContext>();
  buildContext->vertexStaging = vertexZeroRequest.staging;
  buildContext->indexStaging = indexZeroRequest.staging;
  buildContext->scratchBuffer = scratchBuffer;

  vertexZeroRequest.staging = nullptr;
  indexZeroRequest.staging = nullptr;
  scratchBuffer = nullptr;

  _dummyBlasBuildInFlight = true;

  auto completion = [this, context = buildContext, structure,
                     totalRequestedBytes](bool success) {
    if (context->vertexStaging) {
      releaseTrackedResource(context->vertexStaging);
      context->vertexStaging = nullptr;
    }
    if (context->indexStaging) {
      releaseTrackedResource(context->indexStaging);
      context->indexStaging = nullptr;
    }
    if (context->scratchBuffer) {
      releaseTrackedResource(context->scratchBuffer);
      context->scratchBuffer = nullptr;
    }

    if (success) {
      _pDummyBlas = structure;
      std::printf("Dummy BLAS placeholder ready (%llu bytes).\n",
                  static_cast<unsigned long long>(totalRequestedBytes));
    } else {
      std::printf(
          "Dummy BLAS build failed; releasing placeholder resources.\n");
      _dummyBlasResources.ensureAccelerationStructure(0, nullptr);
    }

    _dummyBlasBuildInFlight = false;
  };

  bool submitted = submitAsyncCommandBuffer(commandBuffer, completion);

  releaseDescriptors();

  if (!submitted) {
    _dummyBlasBuildInFlight = false;
    if (buildContext->vertexStaging) {
      releaseTrackedResource(buildContext->vertexStaging);
      buildContext->vertexStaging = nullptr;
    }
    if (buildContext->indexStaging) {
      releaseTrackedResource(buildContext->indexStaging);
      buildContext->indexStaging = nullptr;
    }
    if (buildContext->scratchBuffer) {
      releaseTrackedResource(buildContext->scratchBuffer);
      buildContext->scratchBuffer = nullptr;
    }
    return false;
  }

  return false;
}

void Renderer::ensureTlasBuildEvent() {
  if (_pTlasBuildEvent || !_pDevice)
    return;

  _pTlasBuildEvent = _pDevice->newSharedEvent();
  if (_pTlasBuildEvent) {
    _pTlasBuildEvent->setLabel(
        NS::String::string("RendererTLASFence", NS::UTF8StringEncoding));
    uint64_t initialValue = _pTlasBuildEvent->signaledValue();
    _tlasCompletedEventValue.store(initialValue, std::memory_order_relaxed);
  }
}

bool Renderer::hasPendingTlasBuild() const {
  if (!_pTlasBuildEvent || _tlasBuildEventValue == 0)
    return false;

  return _tlasCompletedEventValue.load(std::memory_order_acquire) <
         _tlasBuildEventValue;
}

void Renderer::waitForPendingBlasBuilds() {
  constexpr auto kPollInterval = std::chrono::milliseconds(1);
  bool pending = true;
  while (pending) {
    pending = false;
    for (auto &resident : _residentObjectGpuResources) {
      if (!resident.hasPendingCommands())
        continue;
      pending = true;
      resident.waitForPendingCommands(kPollInterval);
    }
  }
}

void Renderer::waitForPendingTlasBuild() {
  if (!hasPendingTlasBuild())
    return;

  uint64_t targetValue = _tlasBuildEventValue;
  uint64_t completedValue =
      _tlasCompletedEventValue.load(std::memory_order_acquire);
  if (completedValue >= targetValue)
    return;

  uint64_t signaled = _pTlasBuildEvent->signaledValue();
  if (signaled >= targetValue) {
    _tlasCompletedEventValue.store(signaled, std::memory_order_release);
    return;
  }

  dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
  if (!semaphore) {
    while (_pTlasBuildEvent->signaledValue() < targetValue)
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    _tlasCompletedEventValue.store(targetValue, std::memory_order_release);
    return;
  }

  MTL::SharedEventListener *listener =
      MTL::SharedEventListener::alloc()->init(
          dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0));

  if (!listener) {
    dispatch_release(semaphore);
    while (_pTlasBuildEvent->signaledValue() < targetValue)
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    _tlasCompletedEventValue.store(targetValue, std::memory_order_release);
    return;
  }

  _pTlasBuildEvent->notifyListener(
      listener, targetValue,
      ^(MTL::SharedEvent *, uint64_t) { dispatch_semaphore_signal(semaphore); });

  dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

  listener->release();
  dispatch_release(semaphore);

  uint64_t latest = _pTlasBuildEvent->signaledValue();
  if (latest < targetValue)
    latest = targetValue;
  _tlasCompletedEventValue.store(latest, std::memory_order_release);

  finalizePendingTlasScratchResize(true);
}

void Renderer::finalizePendingTlasScratchResize(bool force) {
  if (!_tlasScratchTracker.hasPendingResize())
    return;

  bool ready = force;
  if (!ready) {
    uint64_t completed = _tlasCompletedEventValue.load(std::memory_order_acquire);
    ready = _tlasScratchTracker.ready(completed);
  }

  if (!ready)
    return;

  size_t previousCapacity = static_cast<size_t>(_tlasScratchCapacity);
  size_t desiredSize = _tlasScratchTracker.targetRefitSize();
  size_t retiredBytes = _tlasScratchTracker.retiredBytes();

  if (_pTlasScratchBuffer) {
    releaseTrackedResource(_pTlasScratchBuffer);
    _pTlasScratchBuffer = nullptr;
    _tlasScratchCapacity = 0;
  }

  if (desiredSize > 0 && _pDevice) {
    _pTlasScratchBuffer = allocateBuffer(
        static_cast<NS::UInteger>(desiredSize),
        MTL::ResourceStorageModePrivate, GpuMemoryTracker::Category::Scratch,
        "TLASScratch");
    _tlasScratchCapacity =
        _pTlasScratchBuffer ? static_cast<NS::UInteger>(desiredSize)
                            : static_cast<NS::UInteger>(0);
  }

  _tlasScratchTracker.noteAllocation(_tlasScratchCapacity);
  _tlasScratchTracker.finalizeResize();
  updateTlasScratchResidentBytes(_tlasScratchCapacity);

  if (retiredBytes > 0) {
    std::printf("[TLAS] Shrunk scratch buffer from %zu to %zu bytes (retired %zu bytes).\n",
                previousCapacity, static_cast<size_t>(_tlasScratchCapacity), retiredBytes);
  }
}

void Renderer::updateTlasScratchResidentBytes(NS::UInteger bytes) {
  _tlasScratchResidentBytes = static_cast<size_t>(bytes);
  _residencyBudget.setTlasScratchBytes(_tlasScratchResidentBytes);
}

void Renderer::updateTopLevelAccelerationStructure(
    const std::vector<MTL::AccelerationStructureInstanceDescriptor> &descriptors,
    const std::vector<MTL::AccelerationStructure *> &structures) {
  if (!_pDevice || !_pCommandQueue)
    return;

  finalizePendingTlasScratchResize();

  if (!ensureDummyBlas())
    return;

  _tlasHeap.initialize(_pDevice);
  _tlasHeap.setHeapShrinkEnabled(_heapShrinkEnabled);

  std::vector<MTL::AccelerationStructure *> instancedStructures;
  instancedStructures.reserve(structures.size() + 1);
  instancedStructures.push_back(_pDummyBlas);
  for (MTL::AccelerationStructure *structure : structures) {
    instancedStructures.push_back(structure ? structure : _pDummyBlas);
  }

  bool structureListChanged =
      instancedStructures.size() != _cachedInstancedAccelerationStructures.size();
  if (!structureListChanged) {
    for (size_t i = 0; i < instancedStructures.size(); ++i) {
      if (instancedStructures[i] != _cachedInstancedAccelerationStructures[i]) {
        structureListChanged = true;
        break;
      }
    }
  }

  bool descriptorCountChanged =
      descriptors.size() != _cachedInstanceDescriptors.size();
  bool descriptorContentChanged = descriptorCountChanged;
  if (!descriptorContentChanged && !descriptors.empty()) {
    descriptorContentChanged =
        std::memcmp(descriptors.data(), _cachedInstanceDescriptors.data(),
                    descriptors.size() *
                        sizeof(MTL::AccelerationStructureInstanceDescriptor)) != 0;
  }

  bool needsRebuild = structureListChanged || descriptorCountChanged ||
                      (_pTlasStructure == nullptr);
  bool needsDescriptorUpload = descriptorContentChanged || needsRebuild;

  size_t descriptorCount = descriptors.size();
  std::vector<RangeUpdate> descriptorRanges;
  bool uploadFullDescriptors = needsRebuild || descriptorCountChanged;
  if (needsDescriptorUpload && !uploadFullDescriptors && descriptorCount > 0 &&
      !_cachedInstanceDescriptors.empty()) {
    descriptorRanges = computeChangedRanges(
        _cachedInstanceDescriptors.data(), descriptors.data(),
        sizeof(MTL::AccelerationStructureInstanceDescriptor), descriptorCount);
    if (descriptorRanges.empty())
      needsDescriptorUpload = false;
  }

  size_t descriptorBytes =
      descriptorCount * sizeof(MTL::AccelerationStructureInstanceDescriptor);
  size_t uploadBytes =
      std::max<size_t>(descriptorCount, size_t(1)) *
      sizeof(MTL::AccelerationStructureInstanceDescriptor);

  MTL::Buffer *instanceBuffer = _tlasHeap.ensureVertexBuffer(
      static_cast<NS::UInteger>(uploadBytes),
      "TLASInstanceDescriptors", MTL::ResourceStorageModePrivate,
      static_cast<MTL::ResourceUsage>(MTL::ResourceUsageRead));
  if (!instanceBuffer)
    return;

  _pTlasInstanceDescriptorBuffer = instanceBuffer;

  MTL::Buffer *stagingBuffer = nullptr;
  if (needsDescriptorUpload && descriptorBytes > 0) {
    if (!_pTlasDescriptorStaging ||
        _tlasDescriptorStagingCapacity < descriptorBytes) {
      if (_pTlasDescriptorStaging)
        releaseTrackedResource(_pTlasDescriptorStaging);
      _pTlasDescriptorStaging =
          allocateBuffer(descriptorBytes, MTL::ResourceStorageModeShared,
                         GpuMemoryTracker::Category::Staging,
                         "TLASDescriptorStaging");
      _tlasDescriptorStagingCapacity =
          _pTlasDescriptorStaging ? descriptorBytes : 0;
    }

    stagingBuffer = _pTlasDescriptorStaging;
    if (!stagingBuffer)
      return;

    uint8_t *dstBytes = static_cast<uint8_t *>(stagingBuffer->contents());
    const uint8_t *srcBytes =
        reinterpret_cast<const uint8_t *>(descriptors.data());

    if (uploadFullDescriptors || descriptorRanges.empty()) {
      std::memcpy(dstBytes, srcBytes, descriptorBytes);
      descriptorRanges.clear();
      descriptorRanges.push_back({0, descriptorBytes});
    } else {
      for (const RangeUpdate &range : descriptorRanges) {
        if (range.length == 0)
          continue;
        std::memcpy(dstBytes + range.offset, srcBytes + range.offset,
                    range.length);
      }
    }

    markBufferModified(stagingBuffer,
                       NS::Range::Make(0, static_cast<NS::UInteger>(descriptorBytes)));
  } else {
    descriptorRanges.clear();
  }

  MTL::InstanceAccelerationStructureDescriptor *instanceDesc =
      MTL::InstanceAccelerationStructureDescriptor::alloc()->init();
  instanceDesc->setInstanceCount(static_cast<NS::UInteger>(descriptorCount));
  instanceDesc->setInstanceDescriptorBuffer(instanceBuffer);
  instanceDesc->setInstanceDescriptorBufferOffset(0);
  instanceDesc->setInstanceDescriptorStride(
      sizeof(MTL::AccelerationStructureInstanceDescriptor));
  instanceDesc->setInstanceDescriptorType(
      MTL::AccelerationStructureInstanceDescriptorTypeDefault);

  if (!instancedStructures.empty()) {
    std::vector<NS::Object *> nsStructures(instancedStructures.size());
    for (size_t i = 0; i < instancedStructures.size(); ++i)
      nsStructures[i] = instancedStructures[i];
    NS::Array *structuresArray =
        NS::Array::alloc()->init(nsStructures.data(), nsStructures.size());
    instanceDesc->setInstancedAccelerationStructures(structuresArray);
    structuresArray->release();
  }

  MTL::AccelerationStructureSizes sizes =
      _pDevice->accelerationStructureSizes(instanceDesc);

  if (needsRebuild) {
    MTL::AccelerationStructure *structure =
        _tlasHeap.ensureAccelerationStructure(
            sizes.accelerationStructureSize, "SceneTLAS");
    if (!structure) {
      instanceDesc->release();
      return;
    }
    _pTlasStructure = structure;
  }

  NS::UInteger scratchSize = needsRebuild ? sizes.buildScratchBufferSize
                                          : sizes.refitScratchBufferSize;
  size_t buildScratchBytes = static_cast<size_t>(scratchSize);
  size_t refitScratchBytes =
      static_cast<size_t>(sizes.refitScratchBufferSize);
  MTL::Buffer *scratchBuffer = nullptr;
  if (scratchSize > 0) {
    if (!_pTlasScratchBuffer || _tlasScratchCapacity < scratchSize) {
      if (_pTlasScratchBuffer)
        releaseTrackedResource(_pTlasScratchBuffer);
      _pTlasScratchBuffer =
          allocateBuffer(scratchSize, MTL::ResourceStorageModePrivate,
                         GpuMemoryTracker::Category::Scratch, "TLASScratch");
      _tlasScratchCapacity =
          _pTlasScratchBuffer ? scratchSize : static_cast<NS::UInteger>(0);
    }
    scratchBuffer = _pTlasScratchBuffer;
    if (!scratchBuffer) {
      instanceDesc->release();
      return;
    }
  }

  _tlasScratchTracker.noteAllocation(_tlasScratchCapacity);
  updateTlasScratchResidentBytes(_tlasScratchCapacity);

  MTL::CommandBuffer *commandBuffer = _pCommandQueue->commandBuffer();
  if (!commandBuffer) {
    instanceDesc->release();
    return;
  }

  MTL::BlitCommandEncoder *blit = nullptr;
  auto ensureBlitEncoder = [&]() -> MTL::BlitCommandEncoder * {
    if (!blit)
      blit = commandBuffer->blitCommandEncoder();
    return blit;
  };

  if (stagingBuffer && !descriptorRanges.empty()) {
    MTL::BlitCommandEncoder *enc = ensureBlitEncoder();
    if (!enc) {
      instanceDesc->release();
      return;
    }
    for (const RangeUpdate &range : descriptorRanges) {
      if (range.length == 0)
        continue;
      enc->copyFromBuffer(stagingBuffer, static_cast<NS::UInteger>(range.offset),
                          instanceBuffer, static_cast<NS::UInteger>(range.offset),
                          static_cast<NS::UInteger>(range.length));
    }
  } else if (needsDescriptorUpload && descriptorBytes == 0 && instanceBuffer &&
             instanceBuffer->length() > 0) {
    MTL::BlitCommandEncoder *enc = ensureBlitEncoder();
    if (!enc) {
      instanceDesc->release();
      return;
    }
    enc->fillBuffer(instanceBuffer,
                    NS::Range::Make(0, instanceBuffer->length()), 0);
  }

  if (blit)
    blit->endEncoding();

  MTL::AccelerationStructureCommandEncoder *encoder =
      commandBuffer->accelerationStructureCommandEncoder();
  if (!encoder) {
    instanceDesc->release();
    return;
  }

  if (needsRebuild) {
    ++_frameTlasRebuilds;
    encoder->buildAccelerationStructure(_pTlasStructure, instanceDesc,
                                        scratchBuffer, 0);
  } else {
    ++_frameTlasRefits;
    encoder->refitAccelerationStructure(_pTlasStructure, instanceDesc,
                                        _pTlasStructure, scratchBuffer, 0);
  }
  encoder->endEncoding();

  ensureTlasBuildEvent();
  uint64_t signalValue = 0;
  if (_pTlasBuildEvent) {
    signalValue = ++_tlasBuildEventValue;
    commandBuffer->encodeSignalEvent(_pTlasBuildEvent, signalValue);
  }

  if (needsRebuild)
    _tlasScratchTracker.registerRebuild(signalValue, buildScratchBytes,
                                        refitScratchBytes);

  if (signalValue > 0) {
    commandBuffer->addCompletedHandler(
        [this, signalValue](MTL::CommandBuffer *cmd) {
          bool success =
              cmd->status() == MTL::CommandBufferStatusCompleted;
          this->handleDeferredBlasEvictions(cmd, success);
          this->_tlasCompletedEventValue.store(signalValue,
                                               std::memory_order_release);
        });
  } else {
    commandBuffer->addCompletedHandler(
        [this](MTL::CommandBuffer *cmd) {
          bool success =
              cmd->status() == MTL::CommandBufferStatusCompleted;
          this->handleDeferredBlasEvictions(cmd, success);
          this->finalizePendingTlasScratchResize(true);
        });
  }

  _pendingBlasEvictions.assign(commandBuffer);

  commandBuffer->commit();

  instanceDesc->release();

  _cachedInstanceDescriptors = descriptors;
  _cachedInstancedAccelerationStructures = instancedStructures;
}

// Repack GPU-facing caches to match the most recent CPU residency state.
// This routine refreshes the primitive/material buffers, BLAS/TLAS bindings,
// light sampling tables, active masks and any compaction remap tables.  When
// possible it reuses cached data so only modified ranges are uploaded.
void Renderer::rebuildResidentResources(bool forceFullRebuild) {
  size_t totalPrimitiveCount = _allPrimitives.size();
  size_t cachedTotalPrimitiveCount = _cachedTotalPrimitiveCount;

  if (forceFullRebuild) {
    bool startCompacted = _pScene ? _pScene->getStartCompacted() : false;
    _residentCompacted = startCompacted;
    _compactionCooldown = 0;
  }

  const size_t uniformsDataSize = sizeof(UniformsData);
  if (!_pUniformsBuffer) {
    _pUniformsBuffer =
        allocateBuffer(uniformsDataSize, MTL::ResourceStorageModeManaged,
                       GpuMemoryTracker::Category::RendererBuffers,
                       "Uniforms");
    if (_pUniformsBuffer) {
      std::memset(_pUniformsBuffer->contents(), 0, uniformsDataSize);
      markBufferModified(_pUniformsBuffer,
                         NS::Range::Make(0, uniformsDataSize));
    }
  }
  if (_primitiveToResidentIndex.size() < totalPrimitiveCount)
    _primitiveToResidentIndex.resize(totalPrimitiveCount, -1);
  std::fill(_primitiveToResidentIndex.begin(), _primitiveToResidentIndex.end(),
            -1);

  bool sizeChanged = cachedTotalPrimitiveCount != totalPrimitiveCount;
  bool needFullUpload =
      forceFullRebuild || !_residentBuffersInitialized || sizeChanged;

  const auto &sceneObjects = _pScene->getObjects();
  if (_objectResidentState.size() != sceneObjects.size()) {
    _objectResidentState.assign(sceneObjects.size(), false);
    needFullUpload = true;
  }

  if (needFullUpload) {
    _cachedPrimitiveData.assign(totalPrimitiveCount * kPrimitiveFloat4Count,
                                simd::float4{0.0f, 0.0f, 0.0f, 0.0f});
    _cachedMaterialData.assign(totalPrimitiveCount * kMaterialFloat4Count,
                               simd::float4{0.0f, 0.0f, 0.0f, 0.0f});

    parallelFor(totalPrimitiveCount, [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        const Primitive &p = _allPrimitives[i];
        simd::float4 *primBase =
            &_cachedPrimitiveData[kPrimitiveFloat4Count * i];
        simd::float4 *matBase =
            &_cachedMaterialData[kMaterialFloat4Count * i];

        primBase[4] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        primBase[5] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        primBase[6] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};

        switch (p.type) {
        case PrimitiveType::Sphere: {
          primBase[0] =
              simd::make_float4(p.sphere.center, static_cast<float>(p.type));
          primBase[1] = simd::make_float4(
              simd::make_float3(p.sphere.radius, 0.0f, 0.0f), 0.0f);
          primBase[2] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
          primBase[3] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
          primBase[6] = simd::make_float4(0.0f, 0.0f, 1.0f, 0.0f);
          break;
        }
        case PrimitiveType::Rectangle: {
          primBase[0] =
              simd::make_float4(p.rectangle.center, static_cast<float>(p.type));
          primBase[1] = simd::make_float4(p.rectangle.u, 0.0f);
          primBase[2] = simd::make_float4(p.rectangle.v, 0.0f);
          primBase[3] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
          primBase[4] = simd::make_float4(p.rectangle.tangent, 0.0f);
          primBase[5] = simd::make_float4(p.rectangle.bitangent, 0.0f);
          primBase[6] = simd::make_float4(p.rectangle.normal,
                                          p.rectangle.supportsNormalMap ? 1.0f
                                                                        : 0.0f);
          break;
        }
        case PrimitiveType::Triangle: {
          primBase[0] =
              simd::make_float4(p.triangle.v0, static_cast<float>(p.type));
          primBase[1] = simd::make_float4(p.triangle.v1, p.triangle.uv0.x);
          primBase[2] = simd::make_float4(p.triangle.v2, p.triangle.uv0.y);
          primBase[3] = simd::make_float4(p.triangle.uv1.x, p.triangle.uv1.y,
                                          p.triangle.uv2.x, p.triangle.uv2.y);
          primBase[4] = simd::make_float4(p.triangle.tangent, 0.0f);
          primBase[5] = simd::make_float4(p.triangle.bitangent, 0.0f);
          primBase[6] = simd::make_float4(p.triangle.normal, 1.0f);
          break;
        }
        }

        const Material &m = p.material;
        auto packed = encodeMaterial(m);
        for (size_t j = 0; j < kMaterialFloat4Count; ++j) {
          matBase[j] = packed[j];
        }
      }
    }, primitivePackingConfig(totalPrimitiveCount));

    _cachedTextureInfos.clear();
    _cachedTextureData.clear();
    if (_pScene) {
      const auto &sceneTextures = _pScene->getTextures();
      const size_t textureCount = sceneTextures.size();
      std::vector<size_t> textureOffsets(textureCount, 0);
      size_t totalTexelCount = 0;
      for (size_t texIndex = 0; texIndex < textureCount; ++texIndex) {
        textureOffsets[texIndex] = totalTexelCount;
        totalTexelCount += static_cast<size_t>(sceneTextures[texIndex].width) *
                           sceneTextures[texIndex].height;
      }

      _cachedTextureInfos.resize(textureCount);
      _cachedTextureData.resize(totalTexelCount);

      parallelFor(textureCount, [&](size_t begin, size_t end) {
        for (size_t texIndex = begin; texIndex < end; ++texIndex) {
          const auto &tex = sceneTextures[texIndex];
          TextureInfo info{};
          info.offset = static_cast<uint32_t>(textureOffsets[texIndex]);
          info.width = tex.width;
          info.height = tex.height;
          _cachedTextureInfos[texIndex] = info;

          size_t texelCount =
              static_cast<size_t>(tex.width) * static_cast<size_t>(tex.height);
          size_t base = textureOffsets[texIndex];
          simd::float4 *dst = _cachedTextureData.data() + base;

          auto packTexel = [&](size_t texelIndex) {
            size_t idx = texelIndex * 4;
            float r = (idx < tex.pixels.size()) ? tex.pixels[idx + 0] : 0.0f;
            float g =
                (idx + 1 < tex.pixels.size()) ? tex.pixels[idx + 1] : 0.0f;
            float b =
                (idx + 2 < tex.pixels.size()) ? tex.pixels[idx + 2] : 0.0f;
            float a =
                (idx + 3 < tex.pixels.size()) ? tex.pixels[idx + 3] : 1.0f;
            return simd::make_float4(r, g, b, a);
          };

          auto packRange = [&](size_t begin, size_t end) {
            for (size_t t = begin; t < end; ++t) {
              dst[t] = packTexel(t);
            }
          };

          parallelFor(texelCount, packRange,
                      textureUploadConfig(1, texelCount));
        }
      }, textureUploadConfig(textureCount, totalTexelCount));
    }

    rebuildMaterialTextures();
    rebuildEnvironmentTexture();

    _pScene->createTriangleBuffers(_cachedTriangleVertices,
                                   _cachedTriangleIndices);

    const auto &sceneIndices = _pScene->getPrimitiveIndices();
    _cachedPrimitiveIndices.resize(sceneIndices.size());
    for (size_t i = 0; i < sceneIndices.size(); ++i)
      _cachedPrimitiveIndices[i] = static_cast<int>(sceneIndices[i]);

    _cachedBVHNodes.clear();
    _blasNodeCount = _pScene ? _pScene->getBVHNodeCount() : 0;
    if (_blasNodeCount > 0) {
      simd::float4 *bvhRaw = _pScene->createBVHBuffer();
      if (bvhRaw) {
        _cachedBVHNodes.assign(bvhRaw, bvhRaw + _blasNodeCount * 2);
        delete[] bvhRaw;
      }
    }

    _cachedTLASNodes.clear();
    size_t tlasCount = 0;
    if (totalPrimitiveCount > 0) {
      simd::float4 *tlasRaw = _pScene->createTLASBuffer(tlasCount);
      if (tlasRaw) {
        _cachedTLASNodes.assign(tlasRaw, tlasRaw + tlasCount * 2);
        delete[] tlasRaw;
      }
    }
    _tlasNodeCount = tlasCount;
    _cachedTotalPrimitiveCount = totalPrimitiveCount;
  }

  _instanceRecords.resize(sceneObjects.size());
  std::vector<bool> objectShouldBeResident(sceneObjects.size(), false);

  if (_cpuActiveMask.size() < totalPrimitiveCount)
    _cpuActiveMask.resize(totalPrimitiveCount, 0);
  if (needFullUpload)
    std::fill(_cpuActiveMask.begin(), _cpuActiveMask.end(), 0);

  auto deduplicate = [](std::vector<size_t> &values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
  };

  if (needFullUpload) {
    _dirtyResidentObjects.clear();
  } else {
    deduplicate(_recentlyActivated);
    deduplicate(_recentlyDeactivated);
    deduplicate(_dirtyResidentObjects);
    if (!_recentlyActivated.empty() && !_recentlyDeactivated.empty()) {
      for (size_t idx : _recentlyDeactivated) {
        auto it = std::lower_bound(_recentlyActivated.begin(),
                                   _recentlyActivated.end(), idx);
        if (it != _recentlyActivated.end() && *it == idx)
          _recentlyActivated.erase(it);
      }
    }
  }

  _cachedLightIndices.clear();
  _cachedLightCdf.clear();
  float totalLightWeight = 0.0f;
  std::vector<size_t> activeIndices;
  activeIndices.reserve(totalPrimitiveCount);
  size_t activeTriangleCount = 0;

  for (size_t i = 0; i < totalPrimitiveCount; ++i) {
    bool active = i < _activePrimitive.size() && _activePrimitive[i];
    if (active) {
      activeIndices.push_back(i);
      if (_allPrimitives[i].type == PrimitiveType::Triangle)
        ++activeTriangleCount;

      const Material &m = _allPrimitives[i].material;
      float emissionStrength = m.emissionPower * luminance(m.emissionColor);
      if (emissionStrength > 0.0f) {
        float area = primitiveArea(_allPrimitives[i]);
        if (area > 0.0f) {
          float weight = area * emissionStrength;
          if (weight > 0.0f) {
            totalLightWeight += weight;
            _cachedLightIndices.push_back(static_cast<uint32_t>(i));
            _cachedLightCdf.push_back(totalLightWeight);
          }
        }
      }
    }

    if (needFullUpload || _residentCompacted)
      _cpuActiveMask[i] = active ? 1 : 0;
  }

  _activePrimitiveCount = activeIndices.size();
  _activeTriangleCount = activeTriangleCount;
  _lightCount = _cachedLightIndices.size();
  _lightTotalWeight = totalLightWeight;
  _cachedLightPdfLookup.assign(totalPrimitiveCount, 0.0f);
  if (_lightTotalWeight > 0.0f && !_cachedLightCdf.empty()) {
    float prevCdf = 0.0f;
    for (size_t i = 0; i < _cachedLightCdf.size(); ++i) {
      float currentCdf = _cachedLightCdf[i];
      float weight = currentCdf - prevCdf;
      prevCdf = currentCdf;
      if (weight <= 0.0f)
        continue;
      size_t primIndex = _cachedLightIndices[i];
      if (primIndex < _cachedLightPdfLookup.size())
        _cachedLightPdfLookup[primIndex] = weight / _lightTotalWeight;
    }
  }

  float activeRatio = (totalPrimitiveCount > 0)
                          ? static_cast<float>(_activePrimitiveCount) /
                                static_cast<float>(totalPrimitiveCount)
                          : 1.0f;
  activeRatio = std::clamp(activeRatio, 0.0f, 1.0f);
  _lastActivePrimitiveRatio = activeRatio;

  constexpr float kCompactionEnterRatio = 0.45f;
  constexpr float kCompactionExitRatio = 0.7f;
  constexpr uint32_t kCompactionCooldownFrames = 30;

  bool alwaysResident = _frameStrategy == ResidencyStrategy::AlwaysResident;
  bool useCompaction = _residentCompacted;
  if (alwaysResident) {
    useCompaction = false;
  } else if (!_residentCompacted) {
    if (_compactionCooldown == 0 && totalPrimitiveCount > 0 &&
        _activePrimitiveCount < totalPrimitiveCount) {
      float occupancy = static_cast<float>(_activePrimitiveCount) /
                        static_cast<float>(totalPrimitiveCount);
      if (_activePrimitiveCount == 0 || occupancy <= kCompactionEnterRatio)
        useCompaction = true;
    }
  } else {
    float occupancy = (totalPrimitiveCount > 0)
                          ? static_cast<float>(_activePrimitiveCount) /
                                static_cast<float>(totalPrimitiveCount)
                          : 1.0f;
    if (_activePrimitiveCount == 0)
      useCompaction = true;
    else if (_activePrimitiveCount == totalPrimitiveCount ||
             occupancy >= kCompactionExitRatio)
      useCompaction = false;
  }

  if (_residencyPreviewOnly)
    useCompaction = false;

  bool compactionStateChanged = (useCompaction != _residentCompacted);
  if (compactionStateChanged) {
    _residentCompacted = useCompaction;
    _compactionCooldown = kCompactionCooldownFrames;
  }

  float shrinkTarget = std::clamp(_residencyConfig.bufferShrinkActiveRatio, 0.0f, 1.0f);
  bool occupancyShrinkActive = _residencyConfig.enableBufferShrink &&
                               activeRatio <= shrinkTarget &&
                               totalPrimitiveCount > 0;
  if (_residencyPreviewOnly)
    occupancyShrinkActive = false;

  bool allowShrink = useCompaction;
  const char *shrinkContext = nullptr;
  if (allowShrink)
    shrinkContext = occupancyShrinkActive ? "compaction+low-occupancy"
                                          : "compaction";
  else if (occupancyShrinkActive) {
    allowShrink = true;
    shrinkContext = "low-occupancy";
  }

  std::vector<uint32_t> remapUpload;
  std::vector<uint8_t> compactActiveMask;
  std::vector<simd::float4> compactPrimitiveData;
  std::vector<simd::float4> compactMaterialData;
  std::vector<int> compactPrimitiveIndices;
  std::vector<simd::float4> compactBVHNodes;
  std::vector<simd::float3> compactTriangleVertices;
  std::vector<simd::uint3> compactTriangleIndices;

  const std::vector<simd::float4> *primitiveSource = &_cachedPrimitiveData;
  const std::vector<simd::float4> *materialSource = &_cachedMaterialData;
  const std::vector<int> *primitiveIndexSource = &_cachedPrimitiveIndices;
  const std::vector<simd::float4> *bvhSource = &_cachedBVHNodes;
  const std::vector<simd::float4> *tlasSource = &_cachedTLASNodes;
  const std::vector<simd::float3> *triangleVertexSource =
      &_cachedTriangleVertices;
  const std::vector<simd::uint3> *triangleIndexSource =
      &_cachedTriangleIndices;
  std::vector<simd::float4> previewMaterialData;
  if (!useCompaction) {
    remapUpload.resize(totalPrimitiveCount);
    for (size_t i = 0; i < totalPrimitiveCount; ++i) {
      remapUpload[i] = static_cast<uint32_t>(i);
      if (i < _primitiveToResidentIndex.size())
        _primitiveToResidentIndex[i] = static_cast<int32_t>(i);
    }
    _residentPrimitiveCount = totalPrimitiveCount;
    _residentTriangleCount = _cachedTriangleIndices.size();
    _blasNodeCount = _cachedBVHNodes.size() / 2;
    _tlasNodeCount = _cachedTLASNodes.size() / 2;

    if (needFullUpload) {
      for (size_t objectIndex = 0; objectIndex < sceneObjects.size();
           ++objectIndex) {
        const SceneObject &obj = sceneObjects[objectIndex];
        BlasInstanceRecord record{};
        record.primitiveBase = static_cast<uint32_t>(obj.firstPrimitive);
        record.primitiveCount = static_cast<uint32_t>(obj.primitiveCount);
        record.primitiveIndexBase = static_cast<uint32_t>(obj.firstPrimitive);
        bool anyActive = false;
        size_t first = obj.firstPrimitive;
        size_t last = first + obj.primitiveCount;
        for (size_t prim = first;
             prim < last && prim < _activePrimitive.size(); ++prim) {
          if (_activePrimitive[prim]) {
            anyActive = true;
            break;
          }
        }
        record.blasRootIndex = anyActive ? obj.blasRootIndex : -1;
        objectShouldBeResident[objectIndex] = anyActive;
        _instanceRecords[objectIndex] = record;
      }
    } else {
      for (size_t objectIndex = 0; objectIndex < sceneObjects.size();
           ++objectIndex) {
        const SceneObject &obj = sceneObjects[objectIndex];
        BlasInstanceRecord record{};
        record.primitiveBase = static_cast<uint32_t>(obj.firstPrimitive);
        record.primitiveCount = static_cast<uint32_t>(obj.primitiveCount);
        record.primitiveIndexBase = static_cast<uint32_t>(obj.firstPrimitive);
        bool cachedResident =
            (objectIndex < _objectResidentState.size())
                ? _objectResidentState[objectIndex]
                : false;
        record.blasRootIndex = cachedResident ? obj.blasRootIndex : -1;
        _instanceRecords[objectIndex] = record;
        objectShouldBeResident[objectIndex] = cachedResident;
      }

      for (size_t dirtyIndex : _dirtyResidentObjects) {
        if (dirtyIndex >= sceneObjects.size())
          continue;
        const SceneObject &obj = sceneObjects[dirtyIndex];
        bool anyActive =
            (dirtyIndex < _objectActive.size()) ? _objectActive[dirtyIndex]
                                                : false;
        BlasInstanceRecord &record = _instanceRecords[dirtyIndex];
        record.primitiveBase = static_cast<uint32_t>(obj.firstPrimitive);
        record.primitiveCount = static_cast<uint32_t>(obj.primitiveCount);
        record.primitiveIndexBase = static_cast<uint32_t>(obj.firstPrimitive);
        record.blasRootIndex = anyActive ? obj.blasRootIndex : -1;
        objectShouldBeResident[dirtyIndex] = anyActive;
      }
    }
  } else {
    remapUpload.clear();
    compactPrimitiveData.clear();
    compactMaterialData.clear();
    compactPrimitiveIndices.clear();
    compactBVHNodes.clear();
    compactTriangleVertices.clear();
    compactTriangleIndices.clear();
    compactActiveMask.clear();

    compactPrimitiveData.reserve(_activePrimitiveCount *
                                 kPrimitiveFloat4Count);
    compactMaterialData.reserve(_activePrimitiveCount * kMaterialFloat4Count);
    compactPrimitiveIndices.reserve(_activePrimitiveCount);
    compactActiveMask.reserve(_activePrimitiveCount);

    std::vector<int32_t> cachedRemap;
    std::vector<size_t> activeLocalPrims;
    std::vector<Primitive> subset;
    std::vector<int> localPrimitiveIndices;
    std::vector<BVHNode> localNodes;

    size_t blasCacheReused = 0;
    size_t blasCacheFallback = 0;
    size_t blasCacheSkipped = 0;

    for (size_t objectIndex = 0; objectIndex < sceneObjects.size();
         ++objectIndex) {
      const SceneObject &obj = sceneObjects[objectIndex];
      BlasInstanceRecord record{};
      record.blasRootIndex = -1;
      record.primitiveBase = static_cast<uint32_t>(remapUpload.size());
      record.primitiveIndexBase =
          static_cast<uint32_t>(compactPrimitiveIndices.size());

      bool hasCache = !obj.cachedBlasNodes.empty() &&
                      !obj.cachedPrimitiveIndices.empty() &&
                      obj.cachedBlasRootIndex >= 0;

      if (hasCache) {
        size_t remapStart = remapUpload.size();
        size_t compactPrimitiveDataStart = compactPrimitiveData.size();
        size_t compactMaterialDataStart = compactMaterialData.size();
        size_t compactPrimitiveIndicesStart = compactPrimitiveIndices.size();
        size_t compactActiveMaskStart = compactActiveMask.size();
        size_t compactBVHNodesStart = compactBVHNodes.size();
        size_t compactTriangleVerticesStart = compactTriangleVertices.size();
        size_t compactTriangleIndicesStart = compactTriangleIndices.size();

        std::vector<std::pair<size_t, int32_t>> residentIndexEdits;
        residentIndexEdits.reserve(obj.cachedPrimitiveIndices.size());

        cachedRemap.assign(obj.cachedPrimitiveIndices.size(), -1);
        size_t activeCount = 0;
        for (size_t localIdx = 0; localIdx < obj.cachedPrimitiveIndices.size();
             ++localIdx) {
          size_t globalIndex = obj.cachedPrimitiveIndices[localIdx];
          if (globalIndex >= _activePrimitive.size() ||
              !_activePrimitive[globalIndex])
            continue;

          cachedRemap[localIdx] = static_cast<int32_t>(activeCount);
          remapUpload.push_back(static_cast<uint32_t>(globalIndex));
          if (globalIndex < _primitiveToResidentIndex.size()) {
            residentIndexEdits.emplace_back(globalIndex,
                                            _primitiveToResidentIndex[globalIndex]);
            _primitiveToResidentIndex[globalIndex] =
                static_cast<int32_t>(record.primitiveBase + activeCount);
          }

          const Primitive &p = _allPrimitives[globalIndex];
          simd::float4 prim0;
          simd::float4 prim1;
          simd::float4 prim2;
          simd::float4 prim3 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
          simd::float4 prim4 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
          simd::float4 prim5 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
          simd::float4 prim6 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
          switch (p.type) {
          case PrimitiveType::Sphere: {
            prim0 =
                simd::make_float4(p.sphere.center, static_cast<float>(p.type));
            prim1 = simd::make_float4(
                simd::make_float3(p.sphere.radius, 0.0f, 0.0f), 0.0f);
            prim2 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
            prim6 = simd::make_float4(0.0f, 0.0f, 1.0f, 0.0f);
            break;
          }
          case PrimitiveType::Rectangle: {
            prim0 = simd::make_float4(p.rectangle.center,
                                      static_cast<float>(p.type));
            prim1 = simd::make_float4(p.rectangle.u, 0.0f);
            prim2 = simd::make_float4(p.rectangle.v, 0.0f);
            prim4 = simd::make_float4(p.rectangle.tangent, 0.0f);
            prim5 = simd::make_float4(p.rectangle.bitangent, 0.0f);
            prim6 = simd::make_float4(p.rectangle.normal,
                                      p.rectangle.supportsNormalMap ? 1.0f
                                                                    : 0.0f);
            break;
          }
          case PrimitiveType::Triangle: {
            prim0 =
                simd::make_float4(p.triangle.v0, static_cast<float>(p.type));
            prim1 = simd::make_float4(p.triangle.v1, p.triangle.uv0.x);
            prim2 = simd::make_float4(p.triangle.v2, p.triangle.uv0.y);
            prim3 = simd::make_float4(p.triangle.uv1.x, p.triangle.uv1.y,
                                      p.triangle.uv2.x, p.triangle.uv2.y);
            prim4 = simd::make_float4(p.triangle.tangent, 0.0f);
            prim5 = simd::make_float4(p.triangle.bitangent, 0.0f);
            prim6 = simd::make_float4(p.triangle.normal, 1.0f);
            size_t baseVertex = compactTriangleVertices.size();
            compactTriangleVertices.push_back(p.triangle.v0);
            compactTriangleVertices.push_back(p.triangle.v1);
            compactTriangleVertices.push_back(p.triangle.v2);
            compactTriangleIndices.push_back(simd::make_uint3(
                static_cast<uint32_t>(baseVertex),
                static_cast<uint32_t>(baseVertex + 1),
                static_cast<uint32_t>(baseVertex + 2)));
            break;
          }
          }

          compactPrimitiveData.push_back(prim0);
          compactPrimitiveData.push_back(prim1);
          compactPrimitiveData.push_back(prim2);
          compactPrimitiveData.push_back(prim3);
          compactPrimitiveData.push_back(prim4);
          compactPrimitiveData.push_back(prim5);
          compactPrimitiveData.push_back(prim6);

          const Material &m = p.material;
          auto packedMaterial = encodeMaterial(m);
          for (size_t j = 0; j < kMaterialFloat4Count; ++j) {
            compactMaterialData.push_back(packedMaterial[j]);
          }
          compactActiveMask.push_back(1);
          compactPrimitiveIndices.push_back(
              static_cast<int>(record.primitiveBase + activeCount));

          ++activeCount;
        }

        record.primitiveCount = static_cast<uint32_t>(activeCount);
        if (activeCount == 0) {
          for (const auto &edit : residentIndexEdits)
            if (edit.first < _primitiveToResidentIndex.size())
              _primitiveToResidentIndex[edit.first] = edit.second;
          objectShouldBeResident[objectIndex] = false;
          _instanceRecords[objectIndex] = record;
          ++blasCacheSkipped;
          continue;
        }

        size_t nodeBase = compactBVHNodes.size() / 2;

        std::vector<BVHNode> rebuiltNodeStructs;
        rebuiltNodeStructs.reserve(obj.cachedBlasNodes.size());

        std::function<int(int)> rebuildNode;
        rebuildNode = [&](int nodeIdx) -> int {
          if (nodeIdx < 0 ||
              static_cast<size_t>(nodeIdx) >= obj.cachedBlasNodes.size())
            return -1;

          BVHNode adjusted = obj.cachedBlasNodes[nodeIdx];
          if (adjusted.count > 0) {
            int newLeftFirst = -1;
            int newCount = 0;
            int originalCount = adjusted.count;
            int localFirst = adjusted.leftFirst;
            for (int i = 0; i < originalCount; ++i) {
              size_t idx = static_cast<size_t>(localFirst + i);
              if (idx >= cachedRemap.size())
                continue;
              int32_t remapped = cachedRemap[idx];
              if (remapped < 0)
                continue;
              if (newLeftFirst < 0)
                newLeftFirst =
                    static_cast<int>(record.primitiveIndexBase + remapped);
              ++newCount;
            }
            if (newCount == 0)
              return -1;
            if (newLeftFirst < 0)
              newLeftFirst = static_cast<int>(record.primitiveIndexBase);
            adjusted.leftFirst = newLeftFirst;
            adjusted.count = newCount;
          } else {
            int leftChild = rebuildNode(adjusted.leftFirst);
            int rightChild = rebuildNode(-adjusted.count);
            if (leftChild < 0 && rightChild < 0)
              return -1;
            if (leftChild < 0)
              return rightChild;
            if (rightChild < 0)
              return leftChild;
            adjusted.leftFirst = leftChild;
            adjusted.count = -rightChild;
          }

          int newIndex = static_cast<int>(rebuiltNodeStructs.size());
          rebuiltNodeStructs.push_back(adjusted);
          return newIndex;
        };

        int rebuiltRoot = rebuildNode(obj.cachedBlasRootIndex);
        if (rebuiltRoot >= 0) {
          record.blasRootIndex =
              static_cast<int32_t>(nodeBase + rebuiltRoot);
          for (const BVHNode &node : rebuiltNodeStructs) {
            BVHNode adjusted = node;
            if (adjusted.count > 0) {
              // Leaf nodes already reference compact primitive indices.
            } else {
              int leftChild = adjusted.leftFirst + static_cast<int>(nodeBase);
              int rightChild = -adjusted.count + static_cast<int>(nodeBase);
              adjusted.leftFirst = leftChild;
              adjusted.count = -rightChild;
            }
            float leftBits = 0.0f;
            float rightBits = 0.0f;
            std::memcpy(&leftBits, &adjusted.leftFirst, sizeof(int));
            std::memcpy(&rightBits, &adjusted.count, sizeof(int));
            compactBVHNodes.push_back(
                simd::make_float4(adjusted.boundsMin, leftBits));
            compactBVHNodes.push_back(
                simd::make_float4(adjusted.boundsMax, rightBits));
          }

          objectShouldBeResident[objectIndex] = true;
          _instanceRecords[objectIndex] = record;
          ++blasCacheReused;
          continue;
        }

        printf("Cached BLAS leaf remapped to zero primitives for object %zu, rebuilding fallback BVH.\n",
               objectIndex);
        remapUpload.resize(remapStart);
        compactPrimitiveData.resize(compactPrimitiveDataStart);
        compactMaterialData.resize(compactMaterialDataStart);
        compactPrimitiveIndices.resize(compactPrimitiveIndicesStart);
        compactActiveMask.resize(compactActiveMaskStart);
        compactBVHNodes.resize(compactBVHNodesStart);
        compactTriangleVertices.resize(compactTriangleVerticesStart);
        compactTriangleIndices.resize(compactTriangleIndicesStart);

        for (const auto &edit : residentIndexEdits)
          if (edit.first < _primitiveToResidentIndex.size())
            _primitiveToResidentIndex[edit.first] = edit.second;

        record.blasRootIndex = -1;
      }

      activeLocalPrims.reserve(obj.primitiveCount);
      activeLocalPrims.clear();
      size_t first = obj.firstPrimitive;
      size_t last = first + obj.primitiveCount;
      for (size_t prim = first;
           prim < last && prim < _activePrimitive.size(); ++prim) {
        if (_activePrimitive[prim])
          activeLocalPrims.push_back(prim);
      }

      record.primitiveCount = static_cast<uint32_t>(activeLocalPrims.size());
      if (activeLocalPrims.empty()) {
        objectShouldBeResident[objectIndex] = false;
        _instanceRecords[objectIndex] = record;
        continue;
      }

      ++blasCacheFallback;

      subset.clear();
      subset.reserve(activeLocalPrims.size());
      for (size_t idx : activeLocalPrims)
        subset.push_back(_allPrimitives[idx]);

      localPrimitiveIndices.resize(subset.size());
      std::iota(localPrimitiveIndices.begin(), localPrimitiveIndices.end(), 0);

      localNodes.clear();
      size_t localNodeCount = 0;
      simd::float4 *localBVHRaw = _pScene->createBVHBuffer(
          subset, localPrimitiveIndices, localNodeCount, localNodes);
      if (localBVHRaw)
        delete[] localBVHRaw;

      if (localNodeCount == 0 || localNodes.empty()) {
        record.primitiveCount = 0;
        objectShouldBeResident[objectIndex] = false;
        _instanceRecords[objectIndex] = record;
        continue;
      }

      for (size_t local = 0; local < subset.size(); ++local) {
        size_t globalIndex = activeLocalPrims[local];
        remapUpload.push_back(static_cast<uint32_t>(globalIndex));
        if (globalIndex < _primitiveToResidentIndex.size())
          _primitiveToResidentIndex[globalIndex] =
              static_cast<int32_t>(record.primitiveBase + local);

        const Primitive &p = subset[local];
        simd::float4 prim0;
        simd::float4 prim1;
        simd::float4 prim2;
        simd::float4 prim3 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        simd::float4 prim4 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        simd::float4 prim5 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        simd::float4 prim6 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        switch (p.type) {
        case PrimitiveType::Sphere: {
          prim0 = simd::make_float4(p.sphere.center, static_cast<float>(p.type));
          prim1 = simd::make_float4(
              simd::make_float3(p.sphere.radius, 0.0f, 0.0f), 0.0f);
          prim2 = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
          prim6 = simd::make_float4(0.0f, 0.0f, 1.0f, 0.0f);
          break;
        }
        case PrimitiveType::Rectangle: {
          prim0 = simd::make_float4(p.rectangle.center, static_cast<float>(p.type));
          prim1 = simd::make_float4(p.rectangle.u, 0.0f);
          prim2 = simd::make_float4(p.rectangle.v, 0.0f);
          prim4 = simd::make_float4(p.rectangle.tangent, 0.0f);
          prim5 = simd::make_float4(p.rectangle.bitangent, 0.0f);
          prim6 = simd::make_float4(p.rectangle.normal,
                                    p.rectangle.supportsNormalMap ? 1.0f
                                                                  : 0.0f);
          break;
        }
        case PrimitiveType::Triangle: {
          prim0 = simd::make_float4(p.triangle.v0, static_cast<float>(p.type));
          prim1 = simd::make_float4(p.triangle.v1, p.triangle.uv0.x);
          prim2 = simd::make_float4(p.triangle.v2, p.triangle.uv0.y);
          prim3 = simd::make_float4(p.triangle.uv1.x, p.triangle.uv1.y,
                                    p.triangle.uv2.x, p.triangle.uv2.y);
          prim4 = simd::make_float4(p.triangle.tangent, 0.0f);
          prim5 = simd::make_float4(p.triangle.bitangent, 0.0f);
          prim6 = simd::make_float4(p.triangle.normal, 1.0f);
          size_t baseVertex = compactTriangleVertices.size();
          compactTriangleVertices.push_back(p.triangle.v0);
          compactTriangleVertices.push_back(p.triangle.v1);
          compactTriangleVertices.push_back(p.triangle.v2);
          compactTriangleIndices.push_back(simd::make_uint3(
              static_cast<uint32_t>(baseVertex),
              static_cast<uint32_t>(baseVertex + 1),
              static_cast<uint32_t>(baseVertex + 2)));
          break;
        }
        }

        compactPrimitiveData.push_back(prim0);
        compactPrimitiveData.push_back(prim1);
        compactPrimitiveData.push_back(prim2);
        compactPrimitiveData.push_back(prim3);
        compactPrimitiveData.push_back(prim4);
        compactPrimitiveData.push_back(prim5);
        compactPrimitiveData.push_back(prim6);

        const Material &m = p.material;
        auto packedMaterial = encodeMaterial(m);
        for (size_t j = 0; j < kMaterialFloat4Count; ++j) {
          compactMaterialData.push_back(packedMaterial[j]);
        }
        compactActiveMask.push_back(1);
      }

      for (int &idx : localPrimitiveIndices)
        idx = static_cast<int>(record.primitiveBase + idx);
      compactPrimitiveIndices.insert(compactPrimitiveIndices.end(),
                                     localPrimitiveIndices.begin(),
                                     localPrimitiveIndices.end());

      size_t nodeBase = compactBVHNodes.size() / 2;
      for (const BVHNode &node : localNodes) {
        BVHNode adjusted = node;
        if (adjusted.count > 0) {
          adjusted.leftFirst += static_cast<int>(record.primitiveIndexBase);
        } else {
          int leftChild = adjusted.leftFirst + static_cast<int>(nodeBase);
          int rightChild = -adjusted.count + static_cast<int>(nodeBase);
          adjusted.leftFirst = leftChild;
          adjusted.count = -rightChild;
        }
        float leftBits = 0.0f;
        float rightBits = 0.0f;
        std::memcpy(&leftBits, &adjusted.leftFirst, sizeof(int));
        std::memcpy(&rightBits, &adjusted.count, sizeof(int));
        compactBVHNodes.push_back(
            simd::make_float4(adjusted.boundsMin, leftBits));
        compactBVHNodes.push_back(
            simd::make_float4(adjusted.boundsMax, rightBits));
      }

      record.blasRootIndex = static_cast<int32_t>(nodeBase);
      objectShouldBeResident[objectIndex] = record.primitiveCount > 0;
      _instanceRecords[objectIndex] = record;
    }

    printf("BLAS cache stats: reused %zu, rebuilt %zu, skipped %zu\n",
           blasCacheReused, blasCacheFallback, blasCacheSkipped);

    primitiveSource = &compactPrimitiveData;
    materialSource = &compactMaterialData;
    primitiveIndexSource = &compactPrimitiveIndices;
    bvhSource = &compactBVHNodes;
    tlasSource = &_cachedTLASNodes;
    triangleVertexSource = &compactTriangleVertices;
    triangleIndexSource = &compactTriangleIndices;

    _residentPrimitiveCount = remapUpload.size();
    _residentTriangleCount = compactTriangleIndices.size();
    _blasNodeCount = compactBVHNodes.size() / 2;
    _tlasNodeCount = _cachedTLASNodes.size() / 2;

    std::vector<uint32_t> remappedLights;
    remappedLights.reserve(_cachedLightIndices.size());
    for (uint32_t globalIndex : _cachedLightIndices) {
      if (globalIndex < _primitiveToResidentIndex.size()) {
        int32_t local = _primitiveToResidentIndex[globalIndex];
        if (local >= 0)
          remappedLights.push_back(static_cast<uint32_t>(local));
      }
    }
    _cachedLightIndices.swap(remappedLights);
    _lightCount = _cachedLightIndices.size();
  }

  if (_residencyPreviewOnly) {
    for (size_t objectIndex = 0; objectIndex < sceneObjects.size();
         ++objectIndex) {
      const SceneObject &obj = sceneObjects[objectIndex];
      objectShouldBeResident[objectIndex] = obj.primitiveCount > 0;
      BlasInstanceRecord &record = _instanceRecords[objectIndex];
      record.primitiveBase = static_cast<uint32_t>(obj.firstPrimitive);
      record.primitiveCount = static_cast<uint32_t>(obj.primitiveCount);
      record.primitiveIndexBase = static_cast<uint32_t>(obj.firstPrimitive);
      record.blasRootIndex = obj.primitiveCount > 0 ? obj.blasRootIndex : -1;
    }
  }

  if (_residencyPreviewOnly && _forceObserverCapture) {
    previewMaterialData.clear();
    previewMaterialData.reserve(remapUpload.size() * kMaterialFloat4Count);
    for (uint32_t globalIndex : remapUpload) {
      if (globalIndex >= _allPrimitives.size())
        continue;
      bool active = globalIndex < _previewPrimitiveActive.size()
                        ? _previewPrimitiveActive[globalIndex]
                        : (globalIndex < _activePrimitive.size() &&
                           _activePrimitive[globalIndex]);
      Material previewMaterial =
          makeResidencyPreviewMaterial(_allPrimitives[globalIndex].material,
                                       active);
      auto packedMaterial = encodeMaterial(previewMaterial);
      for (size_t j = 0; j < kMaterialFloat4Count; ++j)
        previewMaterialData.push_back(packedMaterial[j]);
    }
    materialSource = &previewMaterialData;
  }

  if (_frameStrategy == ResidencyStrategy::AlwaysResident) {
    for (size_t objectIndex = 0; objectIndex < sceneObjects.size();
         ++objectIndex) {
      const SceneObject &obj = sceneObjects[objectIndex];
      if (obj.primitiveCount > 0) {
        objectShouldBeResident[objectIndex] = true;
      }
    }
  }

  size_t geometryCapBytes = geometryResidencyCapBytes();
  size_t projectedResidentBytes = residentGeometryMemoryBytes();
  bool geometryHardCapEnabled =
      _geometryResidencyMemoryCapMB > 0.0 && geometryCapBytes > 0;
  bool geometryHardCapReached = _pendingGeometryResidencyOverageBytes > 0;
  constexpr size_t kMaxColdObjectOnloadsPerFrame = 32;
  size_t coldObjectOnloadsStarted = 0;

  for (size_t objectIndex = 0; objectIndex < sceneObjects.size(); ++objectIndex) {
    bool shouldBeResident = objectShouldBeResident[objectIndex];
    auto &gpuResident = _residentObjectGpuResources[objectIndex];
    auto &instanceRecord = _instanceRecords[objectIndex];
    ResidentObjectGpuResources::ResidencyState previousState = gpuResident.state;
    size_t previousByteSize = gpuResident.byteSize;

    if (geometryHardCapEnabled && shouldBeResident &&
        gpuResident.state ==
            ResidentObjectGpuResources::ResidencyState::Cold) {
      bool projectedOverCap = geometryHardCapReached ||
                              projectedResidentBytes >= geometryCapBytes ||
                              _pendingGeometryResidencyOverageBytes > 0;
      if (projectedOverCap) {
        recordGeometryResidencyHardCapDenied(objectIndex, 0, 0,
                                             projectedResidentBytes,
                                             geometryCapBytes,
                                             "residency update");
        shouldBeResident = false;
        objectShouldBeResident[objectIndex] = false;
        instanceRecord.blasRootIndex = -1;
        geometryHardCapReached = true;
      }
    }

    if (shouldBeResident) {
      bool deferColdOnload =
          !_residencyPreviewOnly &&
          _frameStrategy != ResidencyStrategy::AlwaysResident &&
          previousState == ResidentObjectGpuResources::ResidencyState::Cold &&
          coldObjectOnloadsStarted >= kMaxColdObjectOnloadsPerFrame;
      if (deferColdOnload)
        continue;

      bool built = gpuResident.ensureResident(
          *this, objectIndex, sceneObjects[objectIndex], instanceRecord,
          needFullUpload);
      if (!built) {
        shouldBeResident = false;
        objectShouldBeResident[objectIndex] = false;
      } else if (previousState == ResidentObjectGpuResources::ResidencyState::Cold &&
                 gpuResident.state != ResidentObjectGpuResources::ResidencyState::Cold) {
        ++coldObjectOnloadsStarted;
        ++_frameObjectsOnloadRequested;
        if (gpuResident.state == ResidentObjectGpuResources::ResidencyState::Streaming)
          ++_frameBlasBuildRequests;
      }
    }

    if (!shouldBeResident && !_residencyPreviewOnly &&
        _frameStrategy != ResidencyStrategy::AlwaysResident) {
      if (previousState != ResidentObjectGpuResources::ResidencyState::Cold) {
        size_t previousRequestedBytes = previousByteSize;
        if (previousRequestedBytes == 0) {
          previousRequestedBytes = static_cast<size_t>(
              gpuResident.resources.currentRequiredBytes());
        }
        ++_frameObjectsOffloadRequested;
        _frameOffloadRequestedBytes += previousRequestedBytes;
      }
      requestResidentEviction(objectIndex, gpuResident, instanceRecord);
    }
  }

  std::vector<MTL::AccelerationStructureInstanceDescriptor> instanceDescriptors(
      sceneObjects.size());
  std::vector<MTL::AccelerationStructure *> instancedStructures(
      sceneObjects.size(), nullptr);
  std::vector<GeometryHandle> geometryHandles(sceneObjects.size() + 1);
  if (!geometryHandles.empty())
    geometryHandles[0] = GeometryHandle{};
  MTL::PackedFloat4x3 identityMatrix;
  identityMatrix[0] = MTL::PackedFloat3(1.0f, 0.0f, 0.0f);
  identityMatrix[1] = MTL::PackedFloat3(0.0f, 1.0f, 0.0f);
  identityMatrix[2] = MTL::PackedFloat3(0.0f, 0.0f, 1.0f);
  identityMatrix[3] = MTL::PackedFloat3(0.0f, 0.0f, 0.0f);

  for (size_t objectIndex = 0; objectIndex < sceneObjects.size();
       ++objectIndex) {
    auto &desc = instanceDescriptors[objectIndex];
    desc.transformationMatrix = identityMatrix;
    desc.options = MTL::AccelerationStructureInstanceOptionNone;
    desc.intersectionFunctionTableOffset = 0;
    desc.accelerationStructureIndex =
        static_cast<uint32_t>(objectIndex + 1);

    bool resident = false;
    if (objectIndex < _residentObjectGpuResources.size()) {
      const auto &gpuResident = _residentObjectGpuResources[objectIndex];
      resident = gpuResident.isResident() && objectShouldBeResident[objectIndex];
      if (resident) {
        MTL::AccelerationStructure *structure =
            gpuResident.resources.accelerationStructure();
        if (structure) {
          instancedStructures[objectIndex] = structure;
        } else {
          resident = false;
        }
      }
    }

    desc.mask = resident ? 0xFFu : 0u;

    GeometryHandle handle{};
    if (resident && objectIndex < _residentObjectGpuResources.size()) {
      const auto &gpuResident = _residentObjectGpuResources[objectIndex];
      if (gpuResident.geometryValid) {
        MTL::Buffer *vertexBuffer = gpuResident.resources.vertexBuffer();
        MTL::Buffer *indexBuffer = gpuResident.resources.indexBuffer();
        if (vertexBuffer && indexBuffer) {
          handle.vertexBufferAddress =
              vertexBuffer->gpuAddress() + gpuResident.vertexBufferOffset;
          handle.indexBufferAddress =
              indexBuffer->gpuAddress() + gpuResident.indexBufferOffset;
          handle.vertexStride = static_cast<uint32_t>(sizeof(simd::float3));
          handle.indexStride = static_cast<uint32_t>(sizeof(uint32_t));
          handle.vertexCount =
              static_cast<uint32_t>(gpuResident.vertexCount);
          handle.indexCount =
              static_cast<uint32_t>(gpuResident.triangleCount * 3);
          handle.instanceSlot = static_cast<uint32_t>(objectIndex + 1);
        }
      }
    }
    geometryHandles[objectIndex + 1] = handle;
  }

  updateTopLevelAccelerationStructure(instanceDescriptors, instancedStructures);

  _residentRemap = remapUpload;

  bool uploadAll = needFullUpload || useCompaction || compactionStateChanged ||
                   (_residencyPreviewOnly && _forceObserverCapture);

  if (uploadAll) {
    size_t primitiveFloat4Count = primitiveSource->size();
    size_t primitiveCount =
        (kPrimitiveFloat4Count > 0)
            ? ((primitiveFloat4Count + kPrimitiveFloat4Count - 1) /
               kPrimitiveFloat4Count)
            : 0;
    size_t primitiveBytes = static_cast<size_t>(
        GpuHeapResources::primitiveDataSize(primitiveCount));
    if (primitiveBytes == 0)
      primitiveBytes = sizeof(simd::float4);
    ensureBufferCapacity(_pSphereBuffer, primitiveBytes, _sphereBufferCapacity,
                         allowShrink, MTL::ResourceStorageModeManaged,
                         GpuMemoryTracker::Category::Geometry, "PrimitiveData",
                         shrinkContext);
    if (_pSphereBuffer) {
      simd::float4 *dst =
          static_cast<simd::float4 *>(_pSphereBuffer->contents());
      size_t modifiedBytes = primitiveBytes;
      if (primitiveFloat4Count > 0)
        std::memcpy(dst, primitiveSource->data(),
                    primitiveFloat4Count * sizeof(simd::float4));
      else
        dst[0] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
      if (primitiveFloat4Count > 0)
        modifiedBytes = primitiveFloat4Count * sizeof(simd::float4);
      markBufferModified(_pSphereBuffer, NS::Range::Make(0, modifiedBytes));
    }

    size_t materialFloat4Count = materialSource->size();
    size_t materialBytes =
        std::max<size_t>(materialFloat4Count, size_t(1)) *
        sizeof(simd::float4);
    ensureBufferCapacity(_pSphereMaterialBuffer, materialBytes,
                         _sphereMaterialBufferCapacity, allowShrink,
                         MTL::ResourceStorageModeManaged,
                         GpuMemoryTracker::Category::Geometry,
                         "PrimitiveMaterials", shrinkContext);
    if (_pSphereMaterialBuffer) {
      simd::float4 *dst = static_cast<simd::float4 *>(
          _pSphereMaterialBuffer->contents());
      if (materialFloat4Count > 0)
        std::memcpy(dst, materialSource->data(),
                    materialFloat4Count * sizeof(simd::float4));
      else {
        dst[0] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        if (materialBytes >= 2 * sizeof(simd::float4))
          dst[1] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
      }
      markBufferModified(_pSphereMaterialBuffer,
                         NS::Range::Make(0, materialBytes));
    }

    size_t primitiveIndexCount = primitiveIndexSource->size();
    size_t primitiveIndexBytes =
        std::max<size_t>(primitiveIndexCount, size_t(1)) * sizeof(int);
    ensureBufferCapacity(_pPrimitiveIndexBuffer, primitiveIndexBytes,
                         _primitiveIndexBufferCapacity, allowShrink,
                         MTL::ResourceStorageModeManaged,
                         GpuMemoryTracker::Category::Geometry,
                         "PrimitiveIndices", shrinkContext);
    if (_pPrimitiveIndexBuffer) {
      int *dst = static_cast<int *>(_pPrimitiveIndexBuffer->contents());
      if (primitiveIndexCount > 0)
        std::memcpy(dst, primitiveIndexSource->data(),
                    primitiveIndexCount * sizeof(int));
      else
        dst[0] = 0;
      markBufferModified(_pPrimitiveIndexBuffer,
                         NS::Range::Make(0, primitiveIndexBytes));
    }

    size_t blasFloat4Count = bvhSource->size();
    size_t bvhBytes =
        std::max<size_t>(blasFloat4Count, size_t(1)) * sizeof(simd::float4);
    ensureBufferCapacity(_pBVHBuffer, bvhBytes, _bvhBufferCapacity, allowShrink,
                         MTL::ResourceStorageModeManaged,
                         GpuMemoryTracker::Category::Geometry, "BLASNodes",
                         shrinkContext);
    if (_pBVHBuffer) {
      simd::float4 *dst =
          static_cast<simd::float4 *>(_pBVHBuffer->contents());
      if (blasFloat4Count > 0)
        std::memcpy(dst, bvhSource->data(),
                    blasFloat4Count * sizeof(simd::float4));
      else {
        dst[0] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        if (bvhBytes >= 2 * sizeof(simd::float4))
          dst[1] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
      }
      markBufferModified(_pBVHBuffer, NS::Range::Make(0, bvhBytes));
    }

    size_t tlasFloat4Count = tlasSource->size();
    size_t tlasBytes =
        std::max<size_t>(tlasFloat4Count, size_t(1)) * sizeof(simd::float4);
    ensureBufferCapacity(_pTLASBuffer, tlasBytes, _tlasBufferCapacity,
                         allowShrink, MTL::ResourceStorageModeManaged,
                         GpuMemoryTracker::Category::Geometry, "TLASNodes",
                         shrinkContext);
    if (_pTLASBuffer) {
      simd::float4 *dst =
          static_cast<simd::float4 *>(_pTLASBuffer->contents());
      if (tlasFloat4Count > 0)
        std::memcpy(dst, tlasSource->data(),
                    tlasFloat4Count * sizeof(simd::float4));
      else {
        dst[0] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
        if (tlasBytes >= 2 * sizeof(simd::float4))
          dst[1] = simd::float4{0.0f, 0.0f, 0.0f, 0.0f};
      }
      markBufferModified(_pTLASBuffer, NS::Range::Make(0, tlasBytes));
    }

    size_t remapCount = std::max<size_t>(_residentRemap.size(), size_t(1));
    ensureBufferCapacity(_pPrimitiveRemapBuffer,
                         remapCount * sizeof(uint32_t),
                         _primitiveRemapBufferCapacity, allowShrink,
                         MTL::ResourceStorageModeManaged,
                         GpuMemoryTracker::Category::Geometry, "PrimitiveRemap",
                         shrinkContext);
    if (_pPrimitiveRemapBuffer) {
      uint32_t *dst = static_cast<uint32_t *>(
          _pPrimitiveRemapBuffer->contents());
      if (_residentRemap.empty()) {
        dst[0] = 0;
      } else {
        std::memcpy(dst, _residentRemap.data(),
                    _residentRemap.size() * sizeof(uint32_t));
        if (_residentRemap.size() < remapCount)
          dst[_residentRemap.size()] = 0;
      }
      markBufferModified(_pPrimitiveRemapBuffer,
                         NS::Range::Make(0, remapCount * sizeof(uint32_t)));
    }

    size_t vertexCount = triangleVertexSource->size();
    size_t vertexBytes =
        std::max<size_t>(vertexCount, size_t(1)) * sizeof(simd::float3);
    ensureBufferCapacity(_pTriangleVertexBuffer, vertexBytes,
                         _triangleVertexBufferCapacity, allowShrink,
                         MTL::ResourceStorageModeManaged,
                         GpuMemoryTracker::Category::Geometry,
                         "TriangleVertices", shrinkContext);
    if (_pTriangleVertexBuffer) {
      simd::float3 *dst = static_cast<simd::float3 *>(
          _pTriangleVertexBuffer->contents());
      if (vertexCount > 0)
        std::memcpy(dst, triangleVertexSource->data(),
                    vertexCount * sizeof(simd::float3));
      else
        dst[0] = simd::float3{0.0f, 0.0f, 0.0f};
      markBufferModified(_pTriangleVertexBuffer,
                         NS::Range::Make(0, vertexBytes));
    }

    size_t indexCount = triangleIndexSource->size();
    size_t indexBytes =
        std::max<size_t>(indexCount, size_t(1)) * sizeof(simd::uint3);
    ensureBufferCapacity(_pTriangleIndexBuffer, indexBytes,
                         _triangleIndexBufferCapacity, allowShrink,
                         MTL::ResourceStorageModeManaged,
                         GpuMemoryTracker::Category::Geometry,
                         "TriangleIndices", shrinkContext);
    if (_pTriangleIndexBuffer) {
      simd::uint3 *dst = static_cast<simd::uint3 *>(
          _pTriangleIndexBuffer->contents());
      if (indexCount > 0)
        std::memcpy(dst, triangleIndexSource->data(),
                    indexCount * sizeof(simd::uint3));
      else
        dst[0] = simd::make_uint3(0, 0, 0);
      markBufferModified(_pTriangleIndexBuffer,
                         NS::Range::Make(0, indexBytes));
    }
    _residentBuffersInitialized = true;
  }

  size_t instanceCount = std::max<size_t>(_instanceRecords.size(), size_t(1));
  size_t instanceBytes = instanceCount * sizeof(BlasInstanceRecord);
  ensureBufferCapacity(_pInstanceBuffer, instanceBytes, _instanceBufferCapacity,
                       allowShrink, MTL::ResourceStorageModeManaged,
                       GpuMemoryTracker::Category::Geometry, "InstanceRecords",
                       shrinkContext);
  if (_pInstanceBuffer) {
    auto *dst = static_cast<BlasInstanceRecord *>(
        _pInstanceBuffer->contents());
    if (_instanceRecords.empty()) {
      dst[0] = BlasInstanceRecord{};
    } else {
      std::memcpy(dst, _instanceRecords.data(),
                  _instanceRecords.size() * sizeof(BlasInstanceRecord));
      if (_instanceRecords.size() < instanceCount)
        dst[_instanceRecords.size()] = BlasInstanceRecord{};
    }
    markBufferModified(_pInstanceBuffer, NS::Range::Make(0, instanceBytes));
  }

  size_t geometryHandleCount = geometryHandles.size();
  size_t geometryHandleBytes =
      std::max<size_t>(geometryHandleCount, size_t(1)) *
      sizeof(GeometryHandle);
  ensureBufferCapacity(_pGeometryHandleBuffer, geometryHandleBytes,
                       _geometryHandleBufferCapacity, allowShrink,
                       MTL::ResourceStorageModeManaged,
                       GpuMemoryTracker::Category::Geometry, "GeometryHandles",
                       shrinkContext);
  if (_pGeometryHandleBuffer) {
    auto *dst = static_cast<GeometryHandle *>(
        _pGeometryHandleBuffer->contents());
    if (!geometryHandles.empty()) {
      std::memcpy(dst, geometryHandles.data(),
                  geometryHandles.size() * sizeof(GeometryHandle));
    } else {
      *dst = GeometryHandle{};
    }
    markBufferModified(_pGeometryHandleBuffer,
                       NS::Range::Make(0, geometryHandleBytes));
  }

  size_t activeMaskCount =
      useCompaction ? _residentPrimitiveCount : totalPrimitiveCount;
  size_t activeBytes =
      std::max<size_t>(activeMaskCount, size_t(1)) * sizeof(uint8_t);
  ensureBufferCapacity(_pActiveBuffer, activeBytes, _activeBufferCapacity,
                       allowShrink, MTL::ResourceStorageModeManaged,
                       GpuMemoryTracker::Category::Geometry, "ActiveMask",
                       shrinkContext);
  if (_pActiveBuffer) {
    uint8_t *activePtr =
        static_cast<uint8_t *>(_pActiveBuffer->contents());
    if (_residencyPreviewOnly) {
      std::memset(activePtr, 1, activeBytes);
      markBufferModified(_pActiveBuffer, NS::Range::Make(0, activeBytes));
    } else if (useCompaction) {
      if (_residentPrimitiveCount > 0) {
        std::memcpy(activePtr, compactActiveMask.data(),
                    _residentPrimitiveCount * sizeof(uint8_t));
      } else {
        activePtr[0] = 0;
      }
      markBufferModified(_pActiveBuffer, NS::Range::Make(0, activeBytes));
    } else if (uploadAll) {
      if (totalPrimitiveCount > 0)
        std::memcpy(activePtr, _cpuActiveMask.data(),
                    totalPrimitiveCount * sizeof(uint8_t));
      else
        activePtr[0] = 0;
      markBufferModified(_pActiveBuffer, NS::Range::Make(0, activeBytes));
    } else {
      auto updateMask = [&](const std::vector<size_t> &indices) {
        for (size_t idx : indices) {
          if (idx >= totalPrimitiveCount)
            continue;
          bool active = idx < _activePrimitive.size() && _activePrimitive[idx];
          uint8_t value = active ? 1 : 0;
          _cpuActiveMask[idx] = value;
          activePtr[idx] = value;
          markBufferModified(_pActiveBuffer, NS::Range::Make(idx, 1));
        }
      };
      updateMask(_recentlyActivated);
      updateMask(_recentlyDeactivated);
    }
  }

  size_t lightIndexBytes =
      std::max<size_t>(_cachedLightIndices.size(), size_t(1)) *
      sizeof(uint32_t);
  ensureBufferCapacity(_pLightIndexBuffer, lightIndexBytes,
                       _lightIndexBufferCapacity, allowShrink,
                       MTL::ResourceStorageModeManaged,
                       GpuMemoryTracker::Category::Geometry, "LightIndices",
                       shrinkContext);
  if (_pLightIndexBuffer) {
    uint32_t *dst = static_cast<uint32_t *>(_pLightIndexBuffer->contents());
    if (_cachedLightIndices.empty())
      dst[0] = 0;
    else
      std::memcpy(dst, _cachedLightIndices.data(),
                  _cachedLightIndices.size() * sizeof(uint32_t));
    markBufferModified(_pLightIndexBuffer,
                       NS::Range::Make(0, lightIndexBytes));
  }

  size_t lightCdfBytes =
      std::max<size_t>(_cachedLightCdf.size(), size_t(1)) * sizeof(float);
  ensureBufferCapacity(_pLightCdfBuffer, lightCdfBytes, _lightCdfBufferCapacity,
                       allowShrink, MTL::ResourceStorageModeManaged,
                       GpuMemoryTracker::Category::Geometry, "LightCdf",
                       shrinkContext);
  if (_pLightCdfBuffer) {
    float *dst = static_cast<float *>(_pLightCdfBuffer->contents());
    if (_cachedLightCdf.empty())
      dst[0] = 0.0f;
    else
      std::memcpy(dst, _cachedLightCdf.data(),
                  _cachedLightCdf.size() * sizeof(float));
    markBufferModified(_pLightCdfBuffer, NS::Range::Make(0, lightCdfBytes));
  }

  size_t lightPdfLookupBytes =
      std::max<size_t>(totalPrimitiveCount, size_t(1)) * sizeof(float);
  ensureBufferCapacity(_pLightPdfLookupBuffer, lightPdfLookupBytes,
                       _lightPdfLookupBufferCapacity, allowShrink,
                       MTL::ResourceStorageModeManaged,
                       GpuMemoryTracker::Category::Geometry, "LightPdfLookup",
                       shrinkContext);
  if (_pLightPdfLookupBuffer) {
    float *dst = static_cast<float *>(_pLightPdfLookupBuffer->contents());
    if (_cachedLightPdfLookup.empty()) {
      dst[0] = 0.0f;
    } else {
      std::memcpy(dst, _cachedLightPdfLookup.data(),
                  _cachedLightPdfLookup.size() * sizeof(float));
      if (_cachedLightPdfLookup.size() <
          (lightPdfLookupBytes / sizeof(float))) {
        dst[_cachedLightPdfLookup.size()] = 0.0f;
      }
    }
    markBufferModified(_pLightPdfLookupBuffer,
                       NS::Range::Make(0, lightPdfLookupBytes));
  }

  recalculateNodeCounters(objectShouldBeResident);

  _objectResidentState = objectShouldBeResident;
  _dirtyResidentObjects.clear();
  _recentlyActivated.clear();
  _recentlyDeactivated.clear();

}

std::vector<bool> Renderer::buildResidentMaskFromGpuResources() const {
  std::vector<bool> residentMask;
  if (_residentObjectGpuResources.empty())
    return residentMask;

  size_t objectCount =
      std::max(_residentObjectGpuResources.size(), _objectResidentState.size());
  residentMask.assign(objectCount, false);
  for (size_t objectIndex = 0; objectIndex < _residentObjectGpuResources.size();
       ++objectIndex) {
    residentMask[objectIndex] =
        _residentObjectGpuResources[objectIndex].isResident();
  }
  return residentMask;
}

void Renderer::recalculateNodeCounters(
    const std::vector<bool> &residentMask) {
  bool hasBlasData =
      _blasNodeCount > 0 && _cachedBVHNodes.size() >= _blasNodeCount * 2;
  bool hasTlasData =
      _tlasNodeCount > 0 && _cachedTLASNodes.size() >= _tlasNodeCount * 2;

  if (_blasNodeCount == 0 && _tlasNodeCount == 0) {
    _residentNodeCount = 0;
    _activeNodeCount = 0;
    return;
  }
  if (isAlwaysResidentStrategy()) {
    _residentNodeCount = _blasNodeCount + _tlasNodeCount;
    _activeNodeCount = _residentNodeCount;
    return;
  }

  const std::vector<SceneObject> *sceneObjects = nullptr;
  if (!_allSceneObjects.empty()) {
    sceneObjects = &_allSceneObjects;
  } else if (_pScene) {
    sceneObjects = &_pScene->getObjects();
  }

  size_t totalPrimitiveCount = _allPrimitives.size();
  size_t residentBlasNodes = 0;
  size_t residentTlasNodes = 0;

  if (_residentCompacted) {
    residentBlasNodes = _blasNodeCount;
    residentTlasNodes = _tlasNodeCount;
  } else {
    if (hasBlasData) {
      std::vector<uint8_t> primitiveResident(totalPrimitiveCount, 0);
      size_t activeLimit =
          std::min(totalPrimitiveCount, _activePrimitive.size());
      for (size_t i = 0; i < activeLimit; ++i) {
        if (_activePrimitive[i])
          primitiveResident[i] = 1;
      }

      if (sceneObjects) {
        for (size_t objectIndex = 0;
             objectIndex < sceneObjects->size(); ++objectIndex) {
          if (objectIndex >= residentMask.size() ||
              !residentMask[objectIndex])
            continue;
          const SceneObject &obj = (*sceneObjects)[objectIndex];
          size_t first = obj.firstPrimitive;
          size_t last = first + obj.primitiveCount;
          for (size_t prim = first;
               prim < last && prim < totalPrimitiveCount; ++prim) {
            primitiveResident[prim] = 1;
          }
        }
      }

      const auto &bvhNodes = _cachedBVHNodes;
      const auto &primitiveIndices = _cachedPrimitiveIndices;
      std::vector<uint8_t> processed(_blasNodeCount, 0);
      std::vector<uint8_t> nodeResident(_blasNodeCount, 0);
      std::vector<size_t> stack;
      stack.reserve(_blasNodeCount);
      stack.push_back(0);

      auto decodeNode = [](const simd::float4 &minVec,
                           const simd::float4 &maxVec, int &leftFirst,
                           int &count) {
        const auto *minWords = reinterpret_cast<const int *>(&minVec);
        const auto *maxWords = reinterpret_cast<const int *>(&maxVec);
        leftFirst = minWords[3];
        count = maxWords[3];
      };

      while (!stack.empty()) {
        size_t nodeIdx = stack.back();
        if (nodeIdx >= _blasNodeCount) {
          stack.pop_back();
          continue;
        }

        if (processed[nodeIdx]) {
          stack.pop_back();
          continue;
        }

        int leftFirst = 0;
        int count = 0;
        decodeNode(bvhNodes[2 * nodeIdx], bvhNodes[2 * nodeIdx + 1], leftFirst,
                   count);

        if (count > 0) {
          bool leafResident = false;
          size_t start = static_cast<size_t>(std::max(leftFirst, 0));
          size_t end = start + static_cast<size_t>(std::max(count, 0));
          for (size_t idx = start; idx < end; ++idx) {
            size_t primitiveIndex =
                (idx < primitiveIndices.size())
                    ? static_cast<size_t>(primitiveIndices[idx])
                    : idx;
            if (primitiveIndex < primitiveResident.size() &&
                primitiveResident[primitiveIndex]) {
              leafResident = true;
              break;
            }
          }
          nodeResident[nodeIdx] = leafResident ? 1 : 0;
          processed[nodeIdx] = 1;
          stack.pop_back();
        } else {
          size_t leftChild =
              static_cast<size_t>(leftFirst >= 0 ? leftFirst : 0);
          size_t rightChild = static_cast<size_t>(-count);
          bool leftDone = leftChild >= _blasNodeCount || processed[leftChild];
          bool rightDone = rightChild >= _blasNodeCount || processed[rightChild];
          if (!leftDone) {
            stack.push_back(leftChild);
            continue;
          }
          if (!rightDone) {
            stack.push_back(rightChild);
            continue;
          }
          bool anyResident =
              (leftChild < _blasNodeCount && nodeResident[leftChild]) ||
              (rightChild < _blasNodeCount && nodeResident[rightChild]);
          nodeResident[nodeIdx] = anyResident ? 1 : 0;
          processed[nodeIdx] = 1;
          stack.pop_back();
        }
      }

      residentBlasNodes =
          std::accumulate(nodeResident.begin(), nodeResident.end(), size_t(0));
    } else {
      residentBlasNodes = _blasNodeCount;
    }

    if (hasTlasData) {
      const auto &tlasNodes = _cachedTLASNodes;
      std::vector<uint8_t> processed(_tlasNodeCount, 0);
      std::vector<uint8_t> nodeResident(_tlasNodeCount, 0);
      std::vector<size_t> stack;
      stack.reserve(_tlasNodeCount);
      stack.push_back(0);

      while (!stack.empty()) {
        size_t nodeIdx = stack.back();
        if (nodeIdx >= _tlasNodeCount) {
          stack.pop_back();
          continue;
        }

        if (processed[nodeIdx]) {
          stack.pop_back();
          continue;
        }

        int leftChild = 0;
        int rightChild = 0;
        const auto *leftWords =
            reinterpret_cast<const int *>(&tlasNodes[2 * nodeIdx]);
        const auto *rightWords =
            reinterpret_cast<const int *>(&tlasNodes[2 * nodeIdx + 1]);
        leftChild = leftWords[3];
        rightChild = rightWords[3];

        if (leftChild < 0) {
          size_t objectIndex = static_cast<size_t>(-leftChild - 1);
          bool resident = objectIndex < residentMask.size() &&
                          residentMask[objectIndex];
          nodeResident[nodeIdx] = resident ? 1 : 0;
          processed[nodeIdx] = 1;
          stack.pop_back();
          continue;
        }

        size_t leftIndex = static_cast<size_t>(leftChild);
        size_t rightIndex = static_cast<size_t>(rightChild);
        bool leftDone = leftIndex >= _tlasNodeCount || processed[leftIndex];
        bool rightDone = rightIndex >= _tlasNodeCount || processed[rightIndex];
        if (!leftDone) {
          stack.push_back(leftIndex);
          continue;
        }
        if (!rightDone) {
          stack.push_back(rightIndex);
          continue;
        }
        bool anyResident =
            (leftIndex < _tlasNodeCount && nodeResident[leftIndex]) ||
            (rightIndex < _tlasNodeCount && nodeResident[rightIndex]);
        nodeResident[nodeIdx] = anyResident ? 1 : 0;
        processed[nodeIdx] = 1;
        stack.pop_back();
      }

      residentTlasNodes =
          std::accumulate(nodeResident.begin(), nodeResident.end(), size_t(0));
    } else {
      residentTlasNodes = _tlasNodeCount;
    }
  }

  _residentNodeCount = residentBlasNodes + residentTlasNodes;

  size_t activeBlasNodes = 0;
  if (_residentCompacted) {
    activeBlasNodes = _blasNodeCount;
  } else if (hasBlasData) {
    const auto &bvhNodes = _cachedBVHNodes;
    const auto &primitiveIndices = _cachedPrimitiveIndices;
    std::vector<uint8_t> processed(_blasNodeCount, 0);
    std::vector<uint8_t> nodeActive(_blasNodeCount, 0);
    std::vector<size_t> stack;
    stack.reserve(_blasNodeCount);
    stack.push_back(0);

    auto decodeNode = [](const simd::float4 &minVec,
                         const simd::float4 &maxVec, int &leftFirst,
                         int &count) {
      const auto *minWords = reinterpret_cast<const int *>(&minVec);
      const auto *maxWords = reinterpret_cast<const int *>(&maxVec);
      leftFirst = minWords[3];
      count = maxWords[3];
    };

    while (!stack.empty()) {
      size_t nodeIdx = stack.back();
      if (nodeIdx >= _blasNodeCount) {
        stack.pop_back();
        continue;
      }

      if (processed[nodeIdx]) {
        stack.pop_back();
        continue;
      }

      int leftFirst = 0;
      int count = 0;
      decodeNode(bvhNodes[2 * nodeIdx], bvhNodes[2 * nodeIdx + 1], leftFirst,
                 count);

      if (count > 0) {
        bool leafActive = false;
        size_t start = static_cast<size_t>(std::max(leftFirst, 0));
        size_t end = start + static_cast<size_t>(std::max(count, 0));
        for (size_t idx = start; idx < end; ++idx) {
          size_t primitiveIndex =
              (idx < primitiveIndices.size())
                  ? static_cast<size_t>(primitiveIndices[idx])
                  : idx;
          if (primitiveIndex < _activePrimitive.size() &&
              _activePrimitive[primitiveIndex]) {
            leafActive = true;
            break;
          }
        }
        nodeActive[nodeIdx] = leafActive ? 1 : 0;
        processed[nodeIdx] = 1;
        stack.pop_back();
      } else {
        size_t leftChild = static_cast<size_t>(leftFirst >= 0 ? leftFirst : 0);
        size_t rightChild = static_cast<size_t>(-count);
        bool leftDone = leftChild >= _blasNodeCount || processed[leftChild];
        bool rightDone = rightChild >= _blasNodeCount || processed[rightChild];
        if (!leftDone) {
          stack.push_back(leftChild);
          continue;
        }
        if (!rightDone) {
          stack.push_back(rightChild);
          continue;
        }
        bool anyActive =
            (leftChild < _blasNodeCount && nodeActive[leftChild]) ||
            (rightChild < _blasNodeCount && nodeActive[rightChild]);
        nodeActive[nodeIdx] = anyActive ? 1 : 0;
        processed[nodeIdx] = 1;
        stack.pop_back();
      }
    }

    activeBlasNodes =
        std::accumulate(nodeActive.begin(), nodeActive.end(), size_t(0));
  }

  size_t activeTlasNodes = 0;
  if (hasTlasData) {
    const auto &tlasNodes = _cachedTLASNodes;
    std::vector<uint8_t> processed(_tlasNodeCount, 0);
    std::vector<uint8_t> nodeActive(_tlasNodeCount, 0);
    std::vector<size_t> stack;
    stack.reserve(_tlasNodeCount);
    stack.push_back(0);

    while (!stack.empty()) {
      size_t nodeIdx = stack.back();
      if (nodeIdx >= _tlasNodeCount) {
        stack.pop_back();
        continue;
      }

      if (processed[nodeIdx]) {
        stack.pop_back();
        continue;
      }

      int leftChild = 0;
      int rightChild = 0;
      const auto *leftWords =
          reinterpret_cast<const int *>(&tlasNodes[2 * nodeIdx]);
      const auto *rightWords =
          reinterpret_cast<const int *>(&tlasNodes[2 * nodeIdx + 1]);
      leftChild = leftWords[3];
      rightChild = rightWords[3];

      if (leftChild < 0) {
        size_t objectIndex = static_cast<size_t>(-leftChild - 1);
        bool active = objectIndex < _objectActive.size() &&
                      _objectActive[objectIndex];
        nodeActive[nodeIdx] = active ? 1 : 0;
        processed[nodeIdx] = 1;
        stack.pop_back();
        continue;
      }

      size_t leftIndex = static_cast<size_t>(leftChild);
      size_t rightIndex = static_cast<size_t>(rightChild);
      bool leftDone = leftIndex >= _tlasNodeCount || processed[leftIndex];
      bool rightDone = rightIndex >= _tlasNodeCount || processed[rightIndex];
      if (!leftDone) {
        stack.push_back(leftIndex);
        continue;
      }
      if (!rightDone) {
        stack.push_back(rightIndex);
        continue;
      }
      bool anyActive =
          (leftIndex < _tlasNodeCount && nodeActive[leftIndex]) ||
          (rightIndex < _tlasNodeCount && nodeActive[rightIndex]);
      nodeActive[nodeIdx] = anyActive ? 1 : 0;
      processed[nodeIdx] = 1;
      stack.pop_back();
    }

    activeTlasNodes =
        std::accumulate(nodeActive.begin(), nodeActive.end(), size_t(0));
  } else if (_residentCompacted) {
    activeTlasNodes = _tlasNodeCount;
  }

  _activeNodeCount = activeBlasNodes + activeTlasNodes;
}
void Renderer::updateGeometryResidency(MTL::CommandBuffer *cmd) {
  if (!cmd)
    return;
  double totalCapMB = effectiveTotalGpuMemoryCapMB();
  size_t totalCapBytes = totalGpuMemoryCapBytes();
  size_t totalBytes =
      _pDevice ? static_cast<size_t>(_pDevice->currentAllocatedSize()) : 0;
  size_t totalOverageBytes = 0;
  if (totalCapBytes > 0) {
    if (totalBytes > totalCapBytes)
      totalOverageBytes = totalBytes - totalCapBytes;
    if (_pendingTotalMemoryOverageBytes > totalOverageBytes)
      totalOverageBytes = _pendingTotalMemoryOverageBytes;
  }
  if (_geometryResidencyMemoryCapMB <= 0.0 && totalOverageBytes == 0)
    return;

  size_t capBytes = geometryResidencyCapBytes();
  size_t residentBytes = residentGeometryMemoryBytes();
  size_t targetBytes = residentBytes;
  if (_pendingGeometryResidencyOverageBytes > 0) {
    if (capBytes > _pendingGeometryResidencyOverageBytes)
      targetBytes = capBytes - _pendingGeometryResidencyOverageBytes;
    else
      targetBytes = 0;
  } else if (_geometryResidencyMemoryCapMB > 0.0) {
    targetBytes = capBytes;
  }
  if (totalOverageBytes > 0) {
    size_t totalTarget =
        residentBytes > totalOverageBytes ? residentBytes - totalOverageBytes : 0;
    targetBytes = std::min(targetBytes, totalTarget);
  }

  if (residentBytes <= targetBytes) {
    if (_pendingGeometryResidencyOverageBytes > 0)
      _pendingGeometryResidencyOverageBytes = 0;
    if (totalOverageBytes == 0)
      _pendingTotalMemoryOverageBytes = 0;
    return;
  }

  double residentMB =
      static_cast<double>(residentBytes) / (1024.0 * 1024.0);
  double capMB = static_cast<double>(capBytes) / (1024.0 * 1024.0);
  double targetMB = static_cast<double>(targetBytes) / (1024.0 * 1024.0);

  if (totalOverageBytes > 0) {
    std::printf(
        "[GeometryResidency] Total memory eviction: total=%.2f MB cap=%.2f MB, "
        "residency=%.2f MB target=%.2f MB.\n",
        static_cast<double>(totalBytes) / (1024.0 * 1024.0), totalCapMB,
        residentMB, targetMB);
  } else if (_pendingGeometryResidencyOverageBytes > 0) {
    std::printf(
        "[GeometryResidency] Over budget eviction: residency=%.2f MB (cap=%.2f "
        "MB, target=%.2f MB) to satisfy pending allocation.\n",
        residentMB, capMB, targetMB);
  } else {
    std::printf("[GeometryResidency] Over budget eviction: residency=%.2f MB "
                "(cap=%.2f MB).\n",
                residentMB, capMB);
  }

  struct GeometryCandidate {
    size_t objectIndex = 0;
    double score = 0.0;
    double bytesMB = 0.0;
  };

  std::vector<GeometryCandidate> candidates;
  candidates.reserve(_residentObjectGpuResources.size());
  for (size_t objectIndex = 0; objectIndex < _residentObjectGpuResources.size();
       ++objectIndex) {
    const auto &resident = _residentObjectGpuResources[objectIndex];
    if (!resident.isResident() || !resident.geometryValid)
      continue;
    if (objectIndex >= _instanceRecords.size())
      continue;

    bool visible = false;
    if (objectIndex < _objectBounds.size())
      visible = isInView(_objectBounds[objectIndex]);
    if (objectIndex < _objectVisible.size())
      visible = visible || (_objectVisible[objectIndex] != 0u);
    if (visible)
      continue;

    double bytesMB =
        static_cast<double>(resident.byteSize) / (1024.0 * 1024.0);
    if (bytesMB <= 0.0)
      continue;

    uint64_t lastHitFrame = 0;
    if (objectIndex < _objectHitLastFrame.size())
      lastHitFrame = _objectHitLastFrame[objectIndex];
    uint64_t frameAge =
        _renderedFrameCount > lastHitFrame ? _renderedFrameCount - lastHitFrame
                                           : 0;
    double recency = 1.0 / (1.0 + static_cast<double>(frameAge));
    bool active =
        objectIndex < _objectActive.size() && _objectActive[objectIndex];
    double score = recency / (1.0 + bytesMB);
    if (active)
      score += 1.0;

    candidates.push_back({objectIndex, score, bytesMB});
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const GeometryCandidate &a, const GeometryCandidate &b) {
              if (a.score == b.score)
                return a.bytesMB > b.bytesMB;
              return a.score < b.score;
            });

  for (const auto &candidate : candidates) {
    if (residentBytes <= targetBytes)
      break;
    if (candidate.objectIndex >= _residentObjectGpuResources.size() ||
        candidate.objectIndex >= _instanceRecords.size())
      continue;
    auto &resident = _residentObjectGpuResources[candidate.objectIndex];
    requestResidentEviction(candidate.objectIndex, resident,
                            _instanceRecords[candidate.objectIndex]);
    double candidateBytes =
        candidate.bytesMB * (1024.0 * 1024.0);
    residentBytes =
        residentBytes > static_cast<size_t>(candidateBytes)
            ? residentBytes - static_cast<size_t>(candidateBytes)
            : 0;
    if (_pendingTotalMemoryOverageBytes > 0) {
      size_t bytes = static_cast<size_t>(candidateBytes);
      _pendingTotalMemoryOverageBytes =
          _pendingTotalMemoryOverageBytes > bytes
              ? _pendingTotalMemoryOverageBytes - bytes
              : 0;
    }
  }

  if (residentBytes <= targetBytes)
    _pendingGeometryResidencyOverageBytes = 0;
}

void Renderer::updateAdaptiveSamplingMaps(MTL::CommandBuffer *pCmd) {
  (void)pCmd;
}

bool Renderer::updateCameraStates() {
  const auto &path = _pScene->cameraPath;
  bool hadObserver = _observerActive;
  Camera::State previousViewState =
      hadObserver ? _observerCameraState : _primaryCameraState;
  const bool freezeAlwaysResidentAnimation =
      _pScene &&
      _pScene->getResidencyStrategy() == ResidencyStrategy::AlwaysResident &&
      _renderedFrameCount == 0;

  bool toggled = false;
  if (_forceObserverCapture) {
    _observerActive = true;
    InputSystem::observerToggleRequest = false;
  } else if (InputSystem::observerToggleRequest) {
    _observerActive = !_observerActive;
    InputSystem::observerToggleRequest = false;
    toggled = true;
  }

  double originalDelta = _deltaTimeSeconds;

  if (!path.empty()) {
    Camera::State newState = _primaryCameraState;
    if (_animationFrame <= path.front().frame) {
      const auto &k = path.front();
      newState = makeCameraState(k.position, k.lookAt, k.up,
                                 _primaryCameraState.verticalFov,
                                 _primaryCameraState.focalLength, k.aperture,
                                 k.focusDistance,
                                 _primaryCameraState.forward,
                                 _primaryCameraState.up);
    } else if (_animationFrame >= path.back().frame) {
      const auto &k = path.back();
      newState = makeCameraState(k.position, k.lookAt, k.up,
                                 _primaryCameraState.verticalFov,
                                 _primaryCameraState.focalLength, k.aperture,
                                 k.focusDistance,
                                 _primaryCameraState.forward,
                                 _primaryCameraState.up);
    } else {
      for (size_t i = 0; i + 1 < path.size(); ++i) {
        const auto &k0 = path[i];
        const auto &k1 = path[i + 1];
        if (_animationFrame >= k0.frame && _animationFrame <= k1.frame) {
          float t =
              float(_animationFrame - k0.frame) / float(k1.frame - k0.frame);
          simd::float3 position =
              k0.position + t * (k1.position - k0.position);
          simd::float3 look = k0.lookAt + t * (k1.lookAt - k0.lookAt);
          simd::float3 up = k0.up + t * (k1.up - k0.up);
          newState = makeCameraState(position, look, up,
                                     _primaryCameraState.verticalFov,
                                     _primaryCameraState.focalLength,
                                     k0.aperture + t * (k1.aperture - k0.aperture),
                                     k0.focusDistance +
                                         t * (k1.focusDistance - k0.focusDistance),
                                     _primaryCameraState.forward,
                                     _primaryCameraState.up);
          break;
        }
      }
    }

    _primaryCameraState = newState;

    _deltaTimeSeconds = 0.0;
    Camera::deltaTime = 0.0f;

    Camera::applyState(_observerCameraState);
    Camera::deltaTime = static_cast<float>(originalDelta);
    if (Camera::transformWithInputs())
      _observerCameraState = Camera::captureState();

    if (!freezeAlwaysResidentAnimation)
      _animationFrame++;
  } else {
    Camera::State *target =
        _observerActive ? &_observerCameraState : &_primaryCameraState;
    Camera::applyState(*target);
    Camera::deltaTime = static_cast<float>(_deltaTimeSeconds);
    if (Camera::transformWithInputs())
      *target = Camera::captureState();
  }

  const Camera::State &activeView =
      _observerActive ? _observerCameraState : _primaryCameraState;
  Camera::applyState(activeView);
  Camera::deltaTime = _observerActive ? 0.0f
                                      : static_cast<float>(_deltaTimeSeconds);

  bool viewChanged = toggled ||
                     cameraStatesDiffer(activeView, previousViewState) ||
                     hadObserver != _observerActive;
  if (viewChanged) {
    ++_cameraVersion;
    _depthWeightCameraVersion = 0;
    recalculateViewport();
  }

  return viewChanged;
}

void Renderer::updateUniforms(bool cameraChanged) {
  UniformsData nextUniforms{};

  const Camera::State &activeView =
      _observerActive ? _observerCameraState : _primaryCameraState;
  float motionMetric = 0.0f;
  if (_lastUniformCameraStateValid) {
    motionMetric = computeCameraMotionMetric(activeView, _lastUniformCameraState);
  }

  populateViewportUniforms(nextUniforms);
  nextUniforms.primitiveIndex = 0;
  nextUniforms.randomSeed = {randomFloat(), randomFloat(), randomFloat()};

  uint32_t minSamples = std::min(_minSamplesPerPixel, _maxSamplesPerPixel);
  uint32_t maxSamples = std::max(_minSamplesPerPixel, _maxSamplesPerPixel);
  minSamples = std::max<uint32_t>(minSamples, 1);
  maxSamples = std::max(maxSamples, minSamples);

  nextUniforms.minSamplesPerPixel = minSamples;
  nextUniforms.maxSamplesPerPixel = maxSamples;
  size_t boundTextureCount = std::min(
      _materialTextures.size(), static_cast<size_t>(kMaxMaterialTextureSlots));
  nextUniforms.textureCount = static_cast<uint32_t>(boundTextureCount);
  nextUniforms.aperture = activeView.aperture;
  nextUniforms.focusDistance = activeView.focusDistance;
  nextUniforms.environmentMapEnabled =
      (_environmentTexture && _environmentSampler) ? 1u : 0u;
  nextUniforms.environmentMapIntensity = _environmentBrightness;
  nextUniforms.environmentPadding0 = 0.0f;
  nextUniforms.environmentPadding1 = 0.0f;
  nextUniforms.restirEnabled =
      (_pScene && _pScene->getRestirEnabled()) ? 1u : 0u;
  nextUniforms.restirCandidateCount = 8u;
  const bool restirReuseAllowed =
      (_renderedFrameCount > 0) &&
      (_renderedFrameCount >= _restirReuseDisableUntilFrame);
  nextUniforms.restirTemporalReuse =
      (nextUniforms.restirEnabled && restirReuseAllowed) ? 1u : 0u;
  nextUniforms.restirPadding0 = 0u;
  float aspect = Camera::screenSize.y > 0.0f
                     ? Camera::screenSize.x / Camera::screenSize.y
                     : 1.0f;
  nextUniforms.currentViewProjection =
      simd_mul(makePerspectiveMatrix(activeView.verticalFov, aspect,
                                     kReprojectionNearPlane,
                                     kReprojectionFarPlane),
               makeViewMatrix(activeView));
  if (_lastUniformCameraStateValid) {
    nextUniforms.prevViewProjection =
        simd_mul(makePerspectiveMatrix(_lastUniformCameraState.verticalFov,
                                       aspect, kReprojectionNearPlane,
                                       kReprojectionFarPlane),
                 makeViewMatrix(_lastUniformCameraState));
  } else {
    nextUniforms.prevViewProjection =
        simd_mul(makePerspectiveMatrix(activeView.verticalFov, aspect,
                                       kReprojectionNearPlane,
                                       kReprojectionFarPlane),
                 makeViewMatrix(activeView));
  }
  nextUniforms.cameraMotionMetric =
      _lastUniformCameraStateValid ? motionMetric : 0.0f;
  _frameCameraMotionMetric = nextUniforms.cameraMotionMetric;
  _lastUniformCameraState = activeView;
  _lastUniformCameraStateValid = true;

  uint64_t residentPrimitiveCount = _residentPrimitiveCount;
  uint64_t residentTriangleCount = _residentTriangleCount;

  if (_useAccelerationStructureBindings && _pTlasStructure &&
      _pGeometryHandleBuffer) {
    residentPrimitiveCount = 0;
    residentTriangleCount = 0;
    size_t sharedCount =
        std::min(_residentObjectGpuResources.size(), _instanceRecords.size());
    for (size_t i = 0; i < sharedCount; ++i) {
      const auto &resident = _residentObjectGpuResources[i];
      if (!resident.isResident() || !resident.geometryValid)
        continue;
      residentPrimitiveCount += _instanceRecords[i].primitiveCount;
      residentTriangleCount += resident.triangleCount;
    }
  }

  uint64_t totalPrimitiveCount = 0;
  for (const auto &record : _instanceRecords)
    totalPrimitiveCount += record.primitiveCount;
  if (totalPrimitiveCount == 0)
    totalPrimitiveCount = _allPrimitives.size();

  nextUniforms.primitiveCount = residentPrimitiveCount;
  nextUniforms.triangleCount = residentTriangleCount;
  nextUniforms.totalPrimitiveCount = totalPrimitiveCount;
  nextUniforms.tlasNodeCount = _tlasNodeCount;
  nextUniforms.blasNodeCount = _blasNodeCount;
  nextUniforms.maxRayDepth = _pScene->maxRayDepth;
  nextUniforms.debugAS = sanitizeDebugAccelerationModeForUniforms();
  nextUniforms.lightCount = static_cast<uint32_t>(_lightCount);
  nextUniforms.lightTotalWeight = _lightTotalWeight;

  UniformsData &u = *((UniformsData *)_pUniformsBuffer->contents());
  u = nextUniforms;

  markBufferModified(_pUniformsBuffer, NS::Range::Make(0, sizeof(UniformsData)));
}

void Renderer::draw(MTK::View *pView) {
  _didRenderFrame = false;
  processRayHitCounters();
  bool cameraChanged = updateCameraStates();
  Camera::State viewCamera =
      _observerActive ? _observerCameraState : _primaryCameraState;

  Camera::applyState(_primaryCameraState);
  updateResidency();

  if (_frameStrategy == ResidencyStrategy::AlwaysResident &&
      !alwaysResidentResidencyReady()) {
    ++_alwaysResidentWarmupWaitCount;
    if (_alwaysResidentWarmupWaitCount == 1 ||
        (_alwaysResidentWarmupWaitCount % 60) == 0) {
      std::printf(
          "[AlwaysResident] Waiting for full residency warmup before first "
          "measured frame (pending=%zu active=%zu).\n",
          _pendingBlasBuilds.size(), _activeBlasBuilds.size());
    }
    return;
  }
  _alwaysResidentWarmupWaitCount = 0;

  Camera::applyState(viewCamera);
  if (_captureOutputsPending.load(std::memory_order_acquire))
    processPendingCapturedFrames();

  Camera::deltaTime =
      _observerActive ? 0.0f : static_cast<float>(_deltaTimeSeconds);
  updateUniforms(cameraChanged);
  double configuredTotalCapMB = _totalGpuMemoryCapMB;
  double minimumFootprintMB =
      static_cast<double>(minimumResidentFootprintBytes()) / (1024.0 * 1024.0);
  _frameMinimumResidentFootprintMB = minimumFootprintMB;
  bool capBelowFootprint =
      configuredTotalCapMB > 0.0 && configuredTotalCapMB < minimumFootprintMB;
  if (capBelowFootprint) {
    ++_totalMemoryBelowFootprintFrames;
    _frameTotalMemoryEvictionStall = true;
    if (_totalMemoryBelowFootprintFrames == kTotalMemoryCapStallFrames) {
      _memoryCapFallbackActive = true;
      _totalMemoryCapRelaxedMB = minimumFootprintMB;
      disableHistoryForMemoryCap();
      std::printf(
          "[MemoryBudget] Warning: total cap %.2f MB below minimum footprint "
          "%.2f MB for %zu frames; disabling history and relaxing cap to %.2f MB.\n",
          configuredTotalCapMB, minimumFootprintMB,
          kTotalMemoryCapStallFrames, _totalMemoryCapRelaxedMB);
    }
  } else {
    _totalMemoryBelowFootprintFrames = 0;
    if (_memoryCapFallbackActive &&
        configuredTotalCapMB >= minimumFootprintMB) {
      _memoryCapFallbackActive = false;
      _totalMemoryCapRelaxedMB = 0.0;
      _totalMemoryEvictionNoProgressFrames = 0;
      _totalMemoryEvictionBackoffUntilFrame = 0;
      std::printf(
          "[MemoryBudget] Total cap %.2f MB now meets minimum footprint %.2f MB; "
          "clearing fallback.\n",
          configuredTotalCapMB, minimumFootprintMB);
    }
  }
  if (_memoryCapFallbackActive)
    _frameTotalMemoryEvictionStall = true;
  beginFrameMetrics();

  uint64_t frameIndex = _renderedFrameCount;
  bool captureThisFrame = _frameCaptureEnabled && _frameCaptureInterval > 0 &&
                          (frameIndex % _frameCaptureInterval == 0);
  size_t restirWriteIndex = frameIndex % _restirData0Slots.size();
  size_t restirReadIndex = (frameIndex + 1) % _restirData0Slots.size();

  NS::AutoreleasePool *pPool = NS::AutoreleasePool::alloc()->init();

  auto captureList =
      std::make_shared<std::vector<std::shared_ptr<FrameCaptureRequest>>>();

  MTL::CommandBuffer *prepCmd = _pCommandQueue->commandBuffer();
  if (!prepCmd) {
    if (_benchmarkEnabled && !_pendingBenchmarkSamples.empty())
      _pendingBenchmarkSamples.pop_back();
    completeFrameMetrics(nullptr);
    pPool->release();
    return;
  }

  const size_t restirStatsBytes = sizeof(RestirStatsHost);
  ensureBufferCapacity(_pRestirStatsBuffer, restirStatsBytes,
                       _restirStatsBufferCapacity, false,
                       MTL::ResourceStorageModePrivate,
                       GpuMemoryTracker::Category::RendererBuffers,
                       "ReSTIR Stats Buffer",
                       "ReSTIR stats buffer allocation");
  ensureBufferCapacity(_pRestirStatsReadback, restirStatsBytes,
                       _restirStatsReadbackCapacity, false,
                       MTL::ResourceStorageModeShared,
                       GpuMemoryTracker::Category::Staging,
                       "ReSTIR Stats Readback",
                       "ReSTIR stats readback allocation");

  bool belowBudget =
      _residentPrimitiveCount < kTextureResidencyPrimitiveBudget;
  double geometryMemory = residentGeometryMemoryMB();
  bool geometryOverCap = geometryMemory > _geometryResidencyMemoryCapMB;
  double contentMemory = residencyMemoryMB();
  bool overCap = contentMemory > _textureResidencyMemoryCapMB;
  double totalMemoryMB = currentGPUMemoryMB();
  double effectiveTotalCapMB = effectiveTotalGpuMemoryCapMB();

  bool totalOverCap = effectiveTotalCapMB > 0.0 &&
                      (totalMemoryMB > effectiveTotalCapMB ||
                       _pendingTotalMemoryOverageBytes > 0);
  if (totalOverCap && _pendingTotalMemoryOverageBytes > 0)
    _frameTotalMemoryEvictionStall = true;

  if (totalOverCap) {
    ++_frameTotalMemoryOverageWarnings;
    std::printf(
        "[MemoryBudget] Total GPU memory over hard cap: total=%.3f MB "
        "cap=%.3f MB (residency=%.3f MB, scratch=%.3f MB)\n",
        totalMemoryMB, effectiveTotalCapMB, contentMemory, scratchMemoryMB());
  }

  if (belowBudget || overCap || totalOverCap)
    updateTextureResidency(prepCmd);

  if (geometryOverCap || _pendingGeometryResidencyOverageBytes > 0 ||
      totalOverCap)
    updateGeometryResidency(prepCmd);

  bool allowResidency = !belowBudget && !overCap && !totalOverCap;

  MTL::BlitCommandEncoder *restoreBlit = nullptr;
  MTL::Texture *colorTexture = _colorSlot.texture;
  MTL::Texture *albedoTexture = _albedoSlot.texture;
  MTL::Texture *normalTexture = _normalSlot.texture;
  MTL::Texture *positionTexture = _positionSlot.texture;
  MTL::Texture *restirData0Write = _restirData0Slots[restirWriteIndex].texture;
  MTL::Texture *restirData1Write = _restirData1Slots[restirWriteIndex].texture;
  MTL::Texture *restirData2Write = _restirData2Slots[restirWriteIndex].texture;
  MTL::Texture *restirSurfacePosWrite =
      _restirSurfacePosSlots[restirWriteIndex].texture;
  MTL::Texture *restirSurfaceNormalWrite =
      _restirSurfaceNormalSlots[restirWriteIndex].texture;
  MTL::Texture *restirData0Read = _restirData0Slots[restirReadIndex].texture;
  MTL::Texture *restirData1Read = _restirData1Slots[restirReadIndex].texture;
  MTL::Texture *restirData2Read = _restirData2Slots[restirReadIndex].texture;
  MTL::Texture *restirSurfacePosRead =
      _restirSurfacePosSlots[restirReadIndex].texture;
  MTL::Texture *restirSurfaceNormalRead =
      _restirSurfaceNormalSlots[restirReadIndex].texture;
  if (allowResidency) {
    colorTexture = requestResidentTexture(_colorSlot, prepCmd, restoreBlit);
    albedoTexture = requestResidentTexture(_albedoSlot, prepCmd, restoreBlit);
    normalTexture = requestResidentTexture(_normalSlot, prepCmd, restoreBlit);
    positionTexture =
        requestResidentTexture(_positionSlot, prepCmd, restoreBlit);
    restirData0Write = requestResidentTexture(_restirData0Slots[restirWriteIndex],
                                              prepCmd, restoreBlit);
    restirData1Write = requestResidentTexture(_restirData1Slots[restirWriteIndex],
                                              prepCmd, restoreBlit);
    restirData2Write = requestResidentTexture(_restirData2Slots[restirWriteIndex],
                                              prepCmd, restoreBlit);
    restirSurfacePosWrite =
        requestResidentTexture(_restirSurfacePosSlots[restirWriteIndex],
                               prepCmd, restoreBlit);
    restirSurfaceNormalWrite =
        requestResidentTexture(_restirSurfaceNormalSlots[restirWriteIndex],
                               prepCmd, restoreBlit);
    restirData0Read = requestResidentTexture(_restirData0Slots[restirReadIndex],
                                             prepCmd, restoreBlit);
    restirData1Read = requestResidentTexture(_restirData1Slots[restirReadIndex],
                                             prepCmd, restoreBlit);
    restirData2Read = requestResidentTexture(_restirData2Slots[restirReadIndex],
                                             prepCmd, restoreBlit);
    restirSurfacePosRead =
        requestResidentTexture(_restirSurfacePosSlots[restirReadIndex],
                               prepCmd, restoreBlit);
    restirSurfaceNormalRead =
        requestResidentTexture(_restirSurfaceNormalSlots[restirReadIndex],
                               prepCmd, restoreBlit);
  }

  auto ensureSlotReady = [&](ManagedTextureSlot &slot, MTL::Texture *&handle) {
    if (!handle &&
        slot.historyBacking != ManagedTextureSlot::HistoryBacking::None) {
      handle = requestResidentTexture(slot, prepCmd, restoreBlit);
    } else if (handle && slot.needsGpuRefresh &&
               slot.historyBacking != ManagedTextureSlot::HistoryBacking::None) {
      requestResidentTexture(slot, prepCmd, restoreBlit);
    }
  };

  ensureSlotReady(_colorSlot, colorTexture);
  ensureSlotReady(_albedoSlot, albedoTexture);
  ensureSlotReady(_normalSlot, normalTexture);
  ensureSlotReady(_positionSlot, positionTexture);
  ensureSlotReady(_restirData0Slots[restirWriteIndex], restirData0Write);
  ensureSlotReady(_restirData1Slots[restirWriteIndex], restirData1Write);
  ensureSlotReady(_restirData2Slots[restirWriteIndex], restirData2Write);
  ensureSlotReady(_restirSurfacePosSlots[restirWriteIndex],
                  restirSurfacePosWrite);
  ensureSlotReady(_restirSurfaceNormalSlots[restirWriteIndex],
                  restirSurfaceNormalWrite);
  ensureSlotReady(_restirData0Slots[restirReadIndex], restirData0Read);
  ensureSlotReady(_restirData1Slots[restirReadIndex], restirData1Read);
  ensureSlotReady(_restirData2Slots[restirReadIndex], restirData2Read);
  ensureSlotReady(_restirSurfacePosSlots[restirReadIndex], restirSurfacePosRead);
  ensureSlotReady(_restirSurfaceNormalSlots[restirReadIndex],
                  restirSurfaceNormalRead);

  if (_pRestirStatsBuffer) {
    if (!restoreBlit)
      restoreBlit = prepCmd->blitCommandEncoder();
    if (restoreBlit) {
      size_t bytes = _pRestirStatsBuffer->length();
      if (bytes > 0) {
        restoreBlit->fillBuffer(_pRestirStatsBuffer,
                                NS::Range::Make(0, bytes), 0);
      }
    }
  }

  auto markSlotUsed = [&](ManagedTextureSlot &slot, MTL::Texture *handle) {
    if (handle)
      slot.lastUsedFrame = _renderedFrameCount;
  };

  markSlotUsed(_colorSlot, colorTexture);
  markSlotUsed(_albedoSlot, albedoTexture);
  markSlotUsed(_normalSlot, normalTexture);
  markSlotUsed(_positionSlot, positionTexture);
  markSlotUsed(_restirData0Slots[restirWriteIndex], restirData0Write);
  markSlotUsed(_restirData1Slots[restirWriteIndex], restirData1Write);
  markSlotUsed(_restirData2Slots[restirWriteIndex], restirData2Write);
  markSlotUsed(_restirSurfacePosSlots[restirWriteIndex], restirSurfacePosWrite);
  markSlotUsed(_restirSurfaceNormalSlots[restirWriteIndex],
               restirSurfaceNormalWrite);
  markSlotUsed(_restirData0Slots[restirReadIndex], restirData0Read);
  markSlotUsed(_restirData1Slots[restirReadIndex], restirData1Read);
  markSlotUsed(_restirData2Slots[restirReadIndex], restirData2Read);
  markSlotUsed(_restirSurfacePosSlots[restirReadIndex], restirSurfacePosRead);
  markSlotUsed(_restirSurfaceNormalSlots[restirReadIndex],
               restirSurfaceNormalRead);

  if (restoreBlit)
    restoreBlit->endEncoding();

  bool haveAllTextures =
      colorTexture && albedoTexture && normalTexture && positionTexture &&
      restirData0Write && restirData1Write && restirData2Write &&
      restirSurfacePosWrite && restirSurfaceNormalWrite && restirData0Read &&
      restirData1Read && restirData2Read && restirSurfacePosRead &&
      restirSurfaceNormalRead;

  prepCmd->commit();

  if (!haveAllTextures) {
    MTL::CommandBuffer *presentCmd = _pCommandQueue->commandBuffer();
    if (presentCmd) {
      presentCmd->addCompletedHandler(
          [this, captureList](MTL::CommandBuffer *cmd) {
            this->completeFrameMetrics(cmd);
            for (const auto &capture : *captureList)
              this->finalizeFrameCapture(capture);
          });
      MTL::Drawable *drawable = pView->currentDrawable();
      if (drawable)
        presentCmd->presentDrawable(drawable);
      trackFrameCommandBuffer(presentCmd);
      presentCmd->commit();
    } else {
      completeFrameMetrics(nullptr);
    }
    ++_renderedFrameCount;
    pPool->release();
    return;
  }

  uint32_t minSamples = 1;
  uint32_t maxSamples = 1;
  std::vector<TileDispatchRegion> tiles;
  NS::UInteger generatedTileWidth = 0;
  NS::UInteger generatedTileHeight = 0;
  if (_pPathTracePSO && colorTexture) {
    NS::UInteger width = colorTexture->width();
    NS::UInteger height = colorTexture->height();
    if (width > 0 && height > 0) {
      NS::UInteger tileWidth =
          std::max<NS::UInteger>(kPathTraceTileWidth, 1);
      NS::UInteger tileHeight =
          std::max<NS::UInteger>(kPathTraceTileHeight, 1);

      minSamples = std::min(_minSamplesPerPixel, _maxSamplesPerPixel);
      maxSamples = std::max(_minSamplesPerPixel, _maxSamplesPerPixel);
      minSamples = std::max<uint32_t>(minSamples, 1);
      maxSamples = std::max<uint32_t>(maxSamples, minSamples);

      if (maxSamples > 1) {
        double sampleScale =
            std::max<double>(1.0, std::sqrt(static_cast<double>(maxSamples)));
        double scaledWidth =
            std::floor(static_cast<double>(tileWidth) / sampleScale);
        double scaledHeight =
            std::floor(static_cast<double>(tileHeight) / sampleScale);

        tileWidth = std::max<NS::UInteger>(
            1, static_cast<NS::UInteger>(std::max(1.0, scaledWidth)));
        tileHeight = std::max<NS::UInteger>(
            1, static_cast<NS::UInteger>(std::max(1.0, scaledHeight)));
      }
      generatedTileWidth = tileWidth;
      generatedTileHeight = tileHeight;
      for (NS::UInteger y = 0; y < height; y += tileHeight) {
        NS::UInteger actualHeight = std::min(tileHeight, height - y);
        for (NS::UInteger x = 0; x < width; x += tileWidth) {
          NS::UInteger actualWidth = std::min(tileWidth, width - x);
          TileDispatchRegion region{};
          region.originX = static_cast<uint32_t>(x);
          region.originY = static_cast<uint32_t>(y);
          region.width = static_cast<uint32_t>(actualWidth);
          region.height = static_cast<uint32_t>(actualHeight);
          tiles.push_back(region);
        }
      }
    }
  }

  if (_pPathTracePSO && !tiles.empty()) {
    if (_useAccelerationStructureBindings && _pTlasStructure &&
        _pGeometryHandleBuffer) {
      waitForPendingBlasBuilds();
      waitForPendingTlasBuild();
    }

    NS::UInteger tgWidth =
        std::max<NS::UInteger>(1, _pPathTracePSO->threadExecutionWidth());
    NS::UInteger maxThreads = std::max<NS::UInteger>(
        tgWidth, _pPathTracePSO->maxTotalThreadsPerThreadgroup());
    NS::UInteger tgHeight =
        std::max<NS::UInteger>(1, maxThreads / tgWidth);
    MTL::Size threadsPerThreadgroup =
        MTL::Size::Make(tgWidth, tgHeight, 1);

    auto parseMaxTileWorkEnv = []() -> size_t {
      const char *env = std::getenv("MPT_MAX_TILE_WORK");
      if (!env)
        return 0;
      char *end = nullptr;
      unsigned long long parsed = std::strtoull(env, &end, 10);
      if (end == env)
        return 0;
      return static_cast<size_t>(parsed);
    };

    auto parseMaxTilesPerCommandEnv = []() -> size_t {
      const char *env = std::getenv("MPT_MAX_TILES_PER_COMMAND");
      if (!env)
        return 0;
      char *end = nullptr;
      unsigned long long parsed = std::strtoull(env, &end, 10);
      if (end == env)
        return 0;
      return static_cast<size_t>(parsed);
    };

    auto parseAlwaysResidentHalfTileWorkEnv = []() -> bool {
      const char *env = std::getenv("MPT_ALWAYS_RESIDENT_TILE_WORK_HALF");
      if (!env)
        return false;
      char *end = nullptr;
      unsigned long parsed = std::strtoul(env, &end, 10);
      if (end == env)
        return false;
      return parsed > 0;
    };

    size_t tileIndex = 0;
    const size_t effectiveMaxSamples = std::max<size_t>(maxSamples, 1);
    const size_t envTileWork = parseMaxTileWorkEnv();
    const bool envTileWorkValid = envTileWork > 0;
    size_t tileWorkBudget = envTileWorkValid ? envTileWork
                                             : kDefaultMaxTileSampleWorkPerCommand;
    const bool alwaysResidentHalfTileWork = parseAlwaysResidentHalfTileWorkEnv();
    const size_t envMaxTilesPerCommand = parseMaxTilesPerCommandEnv();
    size_t maxTilesPerCommand = envMaxTilesPerCommand > 0
                                    ? envMaxTilesPerCommand
                                    : kPathTraceMaxTilesPerCommand;
    size_t adaptiveTilesPerCommand = updatePathTraceTilesPerCommandBudget();
    maxTilesPerCommand =
        std::min<size_t>(maxTilesPerCommand, adaptiveTilesPerCommand);
    maxTilesPerCommand = std::clamp(maxTilesPerCommand,
                                    kPathTraceMinTilesPerCommand,
                                    kPathTraceMaxTilesPerCommand);

    if (!envTileWorkValid && _pScene) {
      if (_pScene->hasCustomMaxTileSampleWorkPerCommand()) {
        tileWorkBudget = _pScene->getMaxTileSampleWorkPerCommand();
      } else if (_frameStrategy == ResidencyStrategy::AlwaysResident &&
                 alwaysResidentHalfTileWork) {
        tileWorkBudget = std::max<size_t>(kDefaultMaxTileSampleWorkPerCommand / 2, 1);
      }
    }

    size_t tileBaseWidth =
        std::max<size_t>(static_cast<size_t>(generatedTileWidth), 1);
    size_t tileBaseHeight =
        std::max<size_t>(static_cast<size_t>(generatedTileHeight), 1);
    size_t baseTileWork =
        std::max<size_t>(tileBaseWidth * tileBaseHeight, 1);
    baseTileWork = std::max<size_t>(baseTileWork * effectiveMaxSamples, 1);
    size_t maxWorkPerCommand = std::max(tileWorkBudget, baseTileWork);

    while (tileIndex < tiles.size()) {
      size_t batchStart = tileIndex;
      size_t batchWork = 0;

      while (tileIndex < tiles.size()) {
        const TileDispatchRegion &tile = tiles[tileIndex];
        size_t tileWork = static_cast<size_t>(tile.width) * tile.height;
        tileWork = std::max<size_t>(tileWork, 1);
        tileWork *= effectiveMaxSamples;

        if (tileIndex > batchStart &&
            (batchWork + tileWork > maxWorkPerCommand ||
             (tileIndex - batchStart) >= maxTilesPerCommand))
          break;

        batchWork += tileWork;
        ++tileIndex;
      }

      size_t batchEnd = std::max<size_t>(batchStart + 1, tileIndex);
      size_t tilesInCommand = batchEnd - batchStart;

      MTL::CommandBuffer *computeCmd = _pCommandQueue->commandBuffer();
      if (!computeCmd)
        break;

      MTL::ComputeCommandEncoder *pCompute = computeCmd->computeCommandEncoder();
      if (!pCompute) {
        trackFrameCommandBuffer(computeCmd);
        computeCmd->commit();
        break;
      }

      pCompute->setComputePipelineState(_pPathTracePSO);

      bool useAccelerationStructureLayout =
          _useAccelerationStructureBindings && _pTlasStructure &&
          _pGeometryHandleBuffer;

      if (useAccelerationStructureLayout) {
        pCompute->setAccelerationStructure(_pTlasStructure, 0);
        pCompute->setBuffer(_pGeometryHandleBuffer, 0, 1);
        pCompute->setBuffer(_pSphereBuffer, 0, 2);
        pCompute->setBuffer(_pSphereMaterialBuffer, 0, 3);
        pCompute->setBuffer(_pUniformsBuffer, 0, 4);
        pCompute->setBuffer(_pActiveBuffer, 0, 5);
        pCompute->setBuffer(_pLightIndexBuffer, 0, 6);
        pCompute->setBuffer(_pLightCdfBuffer, 0, 7);
        pCompute->setBuffer(_pPrimitiveRemapBuffer, 0, 8);
        pCompute->setBuffer(_pPrimitiveHitBufferGPU, 0, 9);
        pCompute->setBuffer(_pInstanceBuffer, 0, 10);
        pCompute->setBuffer(_pLightPdfLookupBuffer, 0, 14);
        pCompute->setBuffer(_pRestirStatsBuffer, 0, 15);
        pCompute->setBuffer(_pPrimitiveHitBufferGPU, 0, 12);
        pCompute->setBuffer(_pInstanceBuffer, 0, 13);
      } else {
        pCompute->setBuffer(_pBVHBuffer, 0, 0);
        pCompute->setBuffer(_pSphereBuffer, 0, 1);
        pCompute->setBuffer(_pSphereMaterialBuffer, 0, 2);
        pCompute->setBuffer(_pUniformsBuffer, 0, 3);
        pCompute->setBuffer(_pTriangleVertexBuffer, 0, 4);
        pCompute->setBuffer(_pTriangleIndexBuffer, 0, 5);
        pCompute->setBuffer(_pPrimitiveIndexBuffer, 0, 6);
        pCompute->setBuffer(_pTLASBuffer, 0, 7);
        pCompute->setBuffer(_pActiveBuffer, 0, 8);
        pCompute->setBuffer(_pLightIndexBuffer, 0, 9);
        pCompute->setBuffer(_pLightCdfBuffer, 0, 10);
        pCompute->setBuffer(_pPrimitiveRemapBuffer, 0, 11);
        pCompute->setBuffer(_pPrimitiveHitBufferGPU, 0, 12);
        pCompute->setBuffer(_pInstanceBuffer, 0, 13);
        pCompute->setBuffer(_pLightPdfLookupBuffer, 0, 14);
        pCompute->setBuffer(_pRestirStatsBuffer, 0, 15);
      }

      pCompute->setTexture(colorTexture, 0);
      pCompute->setTexture(albedoTexture, 1);
      pCompute->setTexture(normalTexture, 2);
      pCompute->setTexture(positionTexture, 3);
      pCompute->setTexture(restirData0Write, 4);
      pCompute->setTexture(restirData1Write, 5);
      pCompute->setTexture(restirData2Write, 6);
      pCompute->setTexture(restirData0Read, 7);
      pCompute->setTexture(restirData1Read, 8);
      pCompute->setTexture(restirData2Read, 9);
      pCompute->setTexture(restirSurfacePosWrite, 10);
      pCompute->setTexture(restirSurfaceNormalWrite, 11);
      pCompute->setTexture(restirSurfacePosRead, 12);
      pCompute->setTexture(restirSurfaceNormalRead, 13);

      for (uint32_t texIdx = 0; texIdx < kMaxMaterialTextureSlots; ++texIdx) {
        MTL::Texture *materialTex =
            (texIdx < _materialTextures.size()) ? _materialTextures[texIdx]
                                                : nullptr;
        pCompute->setTexture(materialTex, 14 + texIdx);
      }

      pCompute->setTexture(_environmentTexture,
                            14 + kMaxMaterialTextureSlots);
      if (_environmentSampler)
        pCompute->setSamplerState(_environmentSampler, 0);

      for (size_t batchIdx = batchStart; batchIdx < batchEnd; ++batchIdx) {
        const TileDispatchRegion &tile = tiles[batchIdx];
        TileDispatchRegion tileParams = tile;
        pCompute->setBytes(&tileParams, sizeof(TileDispatchRegion), 16);

        NS::UInteger localWidth = static_cast<NS::UInteger>(tile.width);
        NS::UInteger localHeight = static_cast<NS::UInteger>(tile.height);
        MTL::Size threadgroups = MTL::Size::Make(
            (localWidth + threadsPerThreadgroup.width - 1) /
                threadsPerThreadgroup.width,
            (localHeight + threadsPerThreadgroup.height - 1) /
                threadsPerThreadgroup.height,
            1);

        pCompute->dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
      }

      pCompute->endEncoding();
      computeCmd->addCompletedHandler(
          [this, tilesInCommand](MTL::CommandBuffer *cmd) {
            if (!cmd ||
                cmd->status() != MTL::CommandBufferStatusCompleted)
              return;
            double gpuMs =
                (cmd->GPUEndTime() - cmd->GPUStartTime()) * 1000.0;
            this->recordPathTraceCommandTime(gpuMs, tilesInCommand);
          });
      trackFrameCommandBuffer(computeCmd);
      computeCmd->commit();
    }
  }

  MTL::CommandBuffer *presentCmd = _pCommandQueue->commandBuffer();
  if (!presentCmd) {
    completeFrameMetrics(nullptr);
    ++_renderedFrameCount;
    pPool->release();
    return;
  }

  presentCmd->addCompletedHandler([this, captureList](MTL::CommandBuffer *cmd) {
    this->completeFrameMetrics(cmd);
    for (const auto &capture : *captureList)
      this->finalizeFrameCapture(capture);
  });

  MTL::RenderPassDescriptor *pRpd = pView->currentRenderPassDescriptor();
  MTL::RenderCommandEncoder *pEnc = presentCmd->renderCommandEncoder(pRpd);
  std::vector<OverlayLineVertex> overlayVertices;
  OverlayUniforms overlayUniforms{};
  bool overlayReady = false;

  pEnc->setRenderPipelineState(_pPSO);
  pEnc->setFragmentTexture(colorTexture, 0);

  pEnc->drawPrimitives(MTL::PrimitiveType::PrimitiveTypeTriangle,
                       NS::UInteger(0), NS::UInteger(6));

  if (_pOverlayPSO && _observerActive) {
    overlayVertices.reserve(kFrustumEdges.size() * 2 + kCameraTrailMaxPoints * 2);

    float nearDistance = kFrustumDebugNear;
    float baseDistance = std::max(_primaryCameraState.focalLength, 1.0f);
    float farDistance =
        std::max(baseDistance * kFrustumDebugFarMultiplier, nearDistance * 2.0f);

    auto corners =
        buildFrustumCorners(_primaryCameraState, nearDistance, farDistance);
    const simd::float4 frustumColor = {1.0f, 0.95f, 0.10f, 1.0f};
    const std::array<simd::float3, 5> frustumOffsets = {
        simd::make_float3(0.0f, 0.0f, 0.0f),
        simd::make_float3(kFrustumDebugThicknessWorld, 0.0f, 0.0f),
        simd::make_float3(-kFrustumDebugThicknessWorld, 0.0f, 0.0f),
        simd::make_float3(0.0f, kFrustumDebugThicknessWorld, 0.0f),
        simd::make_float3(0.0f, -kFrustumDebugThicknessWorld, 0.0f)};
    for (const auto &edge : kFrustumEdges) {
      size_t firstIndex = static_cast<size_t>(edge.first);
      size_t secondIndex = static_cast<size_t>(edge.second);
      if (firstIndex >= corners.size() || secondIndex >= corners.size())
        continue;
      for (const simd::float3 &offset : frustumOffsets) {
        overlayVertices.push_back({corners[firstIndex] + offset, frustumColor});
        overlayVertices.push_back({corners[secondIndex] + offset, frustumColor});
      }
    }

    if (_residencyPreviewOnly && _forceObserverCapture) {
      const simd::float3 cameraPosition = _primaryCameraState.position;
      const float minStepSq = kCameraTrailMinStep * kCameraTrailMinStep;
      bool appendPoint = _primaryCameraTrail.empty();
      if (!appendPoint) {
        simd::float3 delta = cameraPosition - _primaryCameraTrail.back();
        appendPoint = simd::length_squared(delta) >= minStepSq;
      }
      if (appendPoint) {
        _primaryCameraTrail.push_back(cameraPosition);
        if (_primaryCameraTrail.size() > kCameraTrailMaxPoints) {
          _primaryCameraTrail.erase(_primaryCameraTrail.begin(),
                                    _primaryCameraTrail.begin() +
                                        (_primaryCameraTrail.size() -
                                         kCameraTrailMaxPoints));
        }
      }

      const simd::float4 trailColor = {1.0f, 0.72f, 0.10f, 1.0f};
      for (size_t i = 1; i < _primaryCameraTrail.size(); ++i) {
        overlayVertices.push_back({_primaryCameraTrail[i - 1], trailColor});
        overlayVertices.push_back({_primaryCameraTrail[i], trailColor});
      }
    }

    size_t vertexCount = overlayVertices.size();
    size_t requiredBytes = vertexCount * sizeof(OverlayLineVertex);
    ensureBufferCapacity(_pFrustumVertexBuffer, requiredBytes,
                         _frustumVertexCapacity, false,
                         MTL::ResourceStorageModeShared,
                         GpuMemoryTracker::Category::RendererBuffers);

    if (_pFrustumVertexBuffer && requiredBytes > 0) {
      void *dst = _pFrustumVertexBuffer->contents();
      if (dst)
        std::memcpy(dst, overlayVertices.data(), requiredBytes);
      if (_pFrustumVertexBuffer->storageMode() == MTL::StorageModeManaged)
        markBufferModified(_pFrustumVertexBuffer,
                           NS::Range::Make(0, requiredBytes));

      float aspect = Camera::screenSize.y > 0.0f
                         ? Camera::screenSize.x / Camera::screenSize.y
                         : 1.0f;
      constexpr float kViewNear = 0.1f;
      constexpr float kViewFar = 1000.0f;
      overlayUniforms.viewProjection =
          simd_mul(makePerspectiveMatrix(viewCamera.verticalFov, aspect,
                                         kViewNear, kViewFar),
                   makeViewMatrix(viewCamera));
      overlayReady = vertexCount > 0;

      pEnc->setRenderPipelineState(_pOverlayPSO);
      pEnc->setVertexBuffer(_pFrustumVertexBuffer, 0, 0);
      pEnc->setVertexBytes(&overlayUniforms, sizeof(overlayUniforms), 1);
      pEnc->drawPrimitives(MTL::PrimitiveTypeLine, NS::UInteger(0),
                           NS::UInteger(vertexCount));

      pEnc->setRenderPipelineState(_pPSO);
      pEnc->setFragmentTexture(colorTexture, 0);
    }
  }

  pEnc->endEncoding();

  if (captureThisFrame && overlayReady && _pOverlayCapturePSO &&
      _pFrustumVertexBuffer &&
      colorTexture) {
    MTL::RenderPassDescriptor *captureOverlayRpd =
        MTL::RenderPassDescriptor::renderPassDescriptor();
    if (captureOverlayRpd) {
      MTL::RenderPassColorAttachmentDescriptor *colorAttachment =
          captureOverlayRpd->colorAttachments()->object(0);
      if (colorAttachment) {
        colorAttachment->setTexture(colorTexture);
        colorAttachment->setLoadAction(MTL::LoadActionLoad);
        colorAttachment->setStoreAction(MTL::StoreActionStore);
      }

      MTL::RenderCommandEncoder *captureOverlayEnc =
          presentCmd->renderCommandEncoder(captureOverlayRpd);
      if (captureOverlayEnc) {
        captureOverlayEnc->setRenderPipelineState(_pOverlayCapturePSO);
        captureOverlayEnc->setVertexBuffer(_pFrustumVertexBuffer, 0, 0);
        captureOverlayEnc->setVertexBytes(&overlayUniforms, sizeof(overlayUniforms),
                                          1);
        captureOverlayEnc->drawPrimitives(
            MTL::PrimitiveTypeLine, NS::UInteger(0),
            NS::UInteger(overlayVertices.size()));
        captureOverlayEnc->endEncoding();
      }
    }
  }

  MTL::BlitCommandEncoder *pBlit = nullptr;
  bool performedRayHitReadback = false;

  if (captureThisFrame && colorTexture) {
    if (auto capture = encodeFrameCapture(colorTexture, albedoTexture, normalTexture,
                                          frameIndex, presentCmd, pBlit)) {
      capture->containsDebugOverlay = overlayReady && _observerActive;
      captureList->push_back(capture);
    }
  }

  bool canScheduleRayHitReadback = false;
  if (_pPrimitiveHitBufferGPU && _pPrimitiveHitReadback) {
    if (!_lastRayHitCommandBuffer) {
      canScheduleRayHitReadback = true;
    } else if (rayHitCopyReady()) {
      flushRayHitCopy();
      canScheduleRayHitReadback = (_lastRayHitCommandBuffer == nullptr);
    }
  }

  if (canScheduleRayHitReadback) {
    if (!pBlit)
      pBlit = presentCmd->blitCommandEncoder();
    if (pBlit) {
      size_t bytes =
          std::min(_pPrimitiveHitBufferGPU->length(),
                   _pPrimitiveHitReadback->length());
      if (bytes > 0) {
        pBlit->copyFromBuffer(_pPrimitiveHitBufferGPU, 0,
                              _pPrimitiveHitReadback, 0, bytes);
        pBlit->fillBuffer(_pPrimitiveHitBufferGPU,
                          NS::Range::Make(0, bytes), 0);
        performedRayHitReadback = true;
      }
    }
  }
  if (_pRestirStatsBuffer && _pRestirStatsReadback) {
    if (!pBlit)
      pBlit = presentCmd->blitCommandEncoder();
    if (pBlit) {
      size_t bytes = std::min(_pRestirStatsBuffer->length(),
                              _pRestirStatsReadback->length());
      if (bytes > 0) {
        pBlit->copyFromBuffer(_pRestirStatsBuffer, 0, _pRestirStatsReadback, 0,
                              bytes);
      }
    }
  }
  if (pBlit)
    pBlit->endEncoding();

  MTL::Drawable *drawable = pView->currentDrawable();
  if (drawable)
    presentCmd->presentDrawable(drawable);
  trackFrameCommandBuffer(presentCmd);
  presentCmd->commit();
  if (!_frameCaptureEnabled &&
      _captureOutputsPending.load(std::memory_order_acquire))
    processPendingCapturedFrames();

  if (performedRayHitReadback) {
    if (_lastRayHitCommandBuffer)
      _lastRayHitCommandBuffer->release();
    _lastRayHitCommandBuffer = presentCmd;
    _lastRayHitCommandBuffer->retain();
    _rayHitCopyError = false;
  }

  _didRenderFrame = true;
  ++_renderedFrameCount;
  pPool->release();

}

// Propagate any pending primitive/object toggles to GPU memory.  The method
// skips expensive repacks if nothing changed, otherwise it forwards to
// rebuildResidentResources to patch only the dirty ranges (or rebuild
// everything when forced).
void Renderer::drawableSizeWillChange(MTK::View *pView, CGSize size) {
  Camera::screenSize = {(float)size.width, (float)size.height};

  buildTextures();
  recalculateViewport();
}

bool Renderer::hasKeyframes() const { return !_pScene->cameraPath.empty(); }

bool Renderer::setPrimitiveActive(size_t index, bool active) {
  if (index >= _activePrimitive.size())
    return false;
  if (_activePrimitive[index] == active)
    return false;
  _activePrimitive[index] = active;
  if (active)
    ++_framePrimitiveActivations;
  else
    ++_framePrimitiveDeactivations;
  if (!active)
    _alwaysResidentCache.markDirty();
  auto &cancelList = active ? _recentlyDeactivated : _recentlyActivated;
  cancelList.erase(
      std::remove(cancelList.begin(), cancelList.end(), index),
      cancelList.end());
  if (active)
    _recentlyActivated.push_back(index);
  else
    _recentlyDeactivated.push_back(index);
  if (index < _primitiveCooldown.size())
    _primitiveCooldown[index] = _residencyConfig.stateCooldownFrames;

  if (index < _primitiveToObject.size()) {
    size_t objectIndex = _primitiveToObject[index];
    if (objectIndex < _allSceneObjects.size()) {
      if (objectIndex >= _objectActive.size())
        _objectActive.resize(objectIndex + 1, false);
      if (objectIndex >= _objectCooldown.size())
        _objectCooldown.resize(objectIndex + 1, 0);
      if (objectIndex >= _objectActivePrimitiveCounts.size())
        _objectActivePrimitiveCounts.resize(objectIndex + 1, 0);

      size_t &activeCount = _objectActivePrimitiveCounts[objectIndex];
      if (active)
        ++activeCount;
      else if (activeCount > 0)
        --activeCount;

      bool newState = activeCount > 0;
      bool fullyInactive = activeCount == 0;
      bool prevState = _objectActive[objectIndex];
      _objectActive[objectIndex] = newState;
      if (prevState != newState || fullyInactive)
        _objectCooldown[objectIndex] = _residencyConfig.stateCooldownFrames;

      _dirtyResidentObjects.push_back(objectIndex);
    }
  }
  return true;
}

void Renderer::dumpAccelerationStructure(const std::string &path) {
  std::filesystem::create_directories(
      std::filesystem::path(path).parent_path());
  std::ofstream out(path);
  if (!out.is_open())
    return;

  out << "{\n";

  size_t tlasCount = 0;
  simd::float4 *tlasData = _pScene->createTLASBuffer(tlasCount);
  out << "  \"tlas\": [\n";
  for (size_t i = 0; i < tlasCount; ++i) {
    simd::float4 bmin = tlasData[2 * i];
    simd::float4 bmax = tlasData[2 * i + 1];
    int first = reinterpret_cast<const int *>(&bmin)[3];
    int second = reinterpret_cast<const int *>(&bmax)[3];
    bool isLeaf = first < 0;
    out << "    {\"index\":" << i << ",\"leaf\":"
        << (isLeaf ? "true" : "false") << ",\"min\":[" << bmin.x << ","
        << bmin.y << "," << bmin.z << "],\"max\":[" << bmax.x << ","
        << bmax.y << "," << bmax.z << "]";
    if (isLeaf) {
      int objectIndex = -(first + 1);
      int blasRoot = -1;
      if (objectIndex >= 0 &&
          static_cast<size_t>(objectIndex) < _instanceRecords.size())
        blasRoot = _instanceRecords[objectIndex].blasRootIndex;
      out << ",\"object\":" << objectIndex << ",\"instance\":"
          << second << ",\"blasRoot\":" << blasRoot;
    } else {
      out << ",\"left\":" << first << ",\"right\":" << second;
    }
    out << "}";
    if (i + 1 < tlasCount)
      out << ",\n";
    else
      out << "\n";
  }
  out << "  ],\n";
  delete[] tlasData;

  const auto &nodes = _pScene->getBVHNodes();
  out << "  \"blas\": [\n";
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto &n = nodes[i];
    out << "    {\"index\":" << i << ",\"min\":[" << n.boundsMin.x << ","
        << n.boundsMin.y << "," << n.boundsMin.z << "],\"max\":["
        << n.boundsMax.x << "," << n.boundsMax.y << "," << n.boundsMax.z
        << "],\"leftFirst\":" << n.leftFirst << ",\"count\":" << n.count << "}";
    if (i + 1 < nodes.size())
      out << ",\n";
    else
      out << "\n";
  }
  out << "  ],\n";

  out << "  \"primitives\": [\n";
  size_t primCount = std::min(_allPrimitives.size(), _activePrimitive.size());
  for (size_t i = 0; i < primCount; ++i) {
    out << "    {\"index\":" << i
        << ",\"active\":" << (_activePrimitive[i] ? "true" : "false")
        << "}";
    if (i + 1 < primCount)
      out << ",\n";
    else
      out << "\n";
  }
  out << "  ]\n";

  out << "}\n";
}

double Renderer::currentGPUMemoryMB() const {
  return static_cast<double>(_pDevice->currentAllocatedSize()) /
         (1024.0 * 1024.0);
}

bool Renderer::rayHitCopyReady() const {
  if (!_lastRayHitCommandBuffer)
    return true;

  auto status = _lastRayHitCommandBuffer->status();
  return status == MTL::CommandBufferStatus::CommandBufferStatusCompleted ||
         status == MTL::CommandBufferStatus::CommandBufferStatusError;
}

bool Renderer::flushRayHitCopy() {
  if (!_lastRayHitCommandBuffer)
    return true;

  auto status = _lastRayHitCommandBuffer->status();

  const ResidencyStrategy strategy =
      _pScene ? _pScene->getResidencyStrategy()
              : ResidencyStrategy::DistanceLOD;
  const bool strategyUsesHits =
      strategy == ResidencyStrategy::RayHitBudget ||
      strategy == ResidencyStrategy::Probabilistic ||
      strategy == ResidencyStrategy::EnvironmentHit ||
      strategy == ResidencyStrategy::PredictiveEnvironment ||
      strategy == ResidencyStrategy::UnifiedScore ||
      strategy == ResidencyStrategy::UnifiedNeural;

  if ((status == MTL::CommandBufferStatus::CommandBufferStatusCommitted ||
       status == MTL::CommandBufferStatus::CommandBufferStatusScheduled ||
       status == MTL::CommandBufferStatus::CommandBufferStatusEnqueued ||
       status == MTL::CommandBufferStatus::CommandBufferStatusNotEnqueued) &&
      (strategyUsesHits || _benchmarkEnabled)) {
    _lastRayHitCommandBuffer->waitUntilCompleted();
    status = _lastRayHitCommandBuffer->status();
  }

  switch (status) {
  case MTL::CommandBufferStatus::CommandBufferStatusCompleted:
    _rayHitCopyError = false;
    break;
  case MTL::CommandBufferStatus::CommandBufferStatusError:
    _rayHitCopyError = true;
    break;
  case MTL::CommandBufferStatus::CommandBufferStatusCommitted:
  case MTL::CommandBufferStatus::CommandBufferStatusScheduled:
  case MTL::CommandBufferStatus::CommandBufferStatusEnqueued:
  case MTL::CommandBufferStatus::CommandBufferStatusNotEnqueued:
  default:
    return false;
  }

  _lastRayHitCommandBuffer->release();
  _lastRayHitCommandBuffer = nullptr;
  return true;
}

void Renderer::processRayHitCounters() {
  if (!flushRayHitCopy())
    return;

  if (!_pPrimitiveHitReadback) {
    _primitiveHitScores.clear();
    _primitiveHitLastFrame.clear();
    _primitiveRayContributions.clear();
    _primitiveRaysTestedLastFrame.clear();
    _primitiveHitAlpha.clear();
    _primitiveHitBeta.clear();
    _primitiveHitProbability.clear();
    _primitiveHitVariance.clear();
    _primitivePosteriorMass.clear();
    _primitiveExplorationScore.clear();
    _probabilitySortedIndices.clear();
    _objectHitAlpha.clear();
    _objectHitBeta.clear();
    _objectHitProbability.clear();
    _objectHitVariance.clear();
    _objectPosteriorMass.clear();
    _objectExplorationScore.clear();
    _objectRayHitScore.clear();
    _objectHitLastFrame.clear();
    _objectRaysTestedLastFrame.clear();
    _objectVisible.clear();
    _objectVisibilityEvidence.clear();
    _objectProbabilitySortedIndices.clear();
    return;
  }

  size_t bufferLength = _pPrimitiveHitReadback->length();
  uint32_t *hitPtr =
      static_cast<uint32_t *>(_pPrimitiveHitReadback->contents());
  if (!hitPtr) {
    _primitiveHitScores.clear();
    _primitiveHitLastFrame.clear();
    _primitiveRayContributions.clear();
    _primitiveRaysTestedLastFrame.clear();
    _primitiveHitAlpha.clear();
    _primitiveHitBeta.clear();
    _primitiveHitProbability.clear();
    _primitiveHitVariance.clear();
    _primitivePosteriorMass.clear();
    _primitiveExplorationScore.clear();
    _probabilitySortedIndices.clear();
    _objectHitAlpha.clear();
    _objectHitBeta.clear();
    _objectHitProbability.clear();
    _objectHitVariance.clear();
    _objectPosteriorMass.clear();
    _objectExplorationScore.clear();
    _objectRayHitScore.clear();
    _objectHitLastFrame.clear();
    _objectRaysTestedLastFrame.clear();
    _objectVisible.clear();
    _objectVisibilityEvidence.clear();
    _objectProbabilitySortedIndices.clear();
    return;
  }

  if (_rayHitCopyError) {
    std::memset(hitPtr, 0, bufferLength);
    _primitiveHitScores.clear();
    _primitiveHitLastFrame.clear();
    _primitiveRayContributions.clear();
    _primitiveRaysTestedLastFrame.clear();
    _primitiveHitAlpha.clear();
    _primitiveHitBeta.clear();
    _primitiveHitProbability.clear();
    _primitiveHitVariance.clear();
    _primitivePosteriorMass.clear();
    _primitiveExplorationScore.clear();
    _probabilitySortedIndices.clear();
    _rayHitCopyError = false;
    _objectHitAlpha.clear();
    _objectHitBeta.clear();
    _objectHitProbability.clear();
    _objectHitVariance.clear();
    _objectPosteriorMass.clear();
    _objectExplorationScore.clear();
    _objectRayHitScore.clear();
    _objectHitLastFrame.clear();
    _objectRaysTestedLastFrame.clear();
    _objectVisible.clear();
    _objectVisibilityEvidence.clear();
    _objectProbabilitySortedIndices.clear();
    return;
  }

  ResidencyStrategy strategy =
      _pScene ? _pScene->getResidencyStrategy()
              : ResidencyStrategy::DistanceLOD;
  bool strategyUsesHits =
      strategy == ResidencyStrategy::RayHitBudget ||
      strategy == ResidencyStrategy::Probabilistic ||
      strategy == ResidencyStrategy::EnvironmentHit ||
      strategy == ResidencyStrategy::PredictiveEnvironment ||
      strategy == ResidencyStrategy::UnifiedScore ||
      strategy == ResidencyStrategy::UnifiedNeural ||
      _benchmarkEnabled || _neuralFeatureLoggingEnabled;
  if (!strategyUsesHits) {
    std::memset(hitPtr, 0, bufferLength);
    _primitiveHitScores.clear();
    _primitiveHitLastFrame.clear();
    _primitiveRayContributions.clear();
    _primitiveRaysTestedLastFrame.clear();
    _primitiveHitAlpha.clear();
    _primitiveHitBeta.clear();
    _primitiveHitProbability.clear();
    _primitiveHitVariance.clear();
    _primitivePosteriorMass.clear();
    _primitiveExplorationScore.clear();
    _probabilitySortedIndices.clear();
    _objectHitAlpha.clear();
    _objectHitBeta.clear();
    _objectHitProbability.clear();
    _objectHitVariance.clear();
    _objectPosteriorMass.clear();
    _objectExplorationScore.clear();
    _objectHitLastFrame.clear();
    _objectRaysTestedLastFrame.clear();
    _objectVisible.clear();
    _objectVisibilityEvidence.clear();
    _objectProbabilitySortedIndices.clear();
    return;
  }

  size_t totalPrimitiveCount = _allPrimitives.size();
  if (totalPrimitiveCount == 0)
    return;

  constexpr size_t kStatsPerPrimitive = 2;
  constexpr size_t kGlobalRayStatsCount = 2;
  size_t bufferCount = bufferLength / sizeof(uint32_t);
  size_t availableForPrimitives =
      (bufferCount >= kGlobalRayStatsCount)
          ? (bufferCount - kGlobalRayStatsCount)
          : size_t(0);
  size_t count =
      std::min(totalPrimitiveCount, availableForPrimitives / kStatsPerPrimitive);
  size_t globalStatsBase = count * kStatsPerPrimitive;
  uint32_t envHitCount = 0;
  uint32_t totalRayCount = 0;
  if (bufferCount >= globalStatsBase + kGlobalRayStatsCount) {
    envHitCount = hitPtr[globalStatsBase + 0];
    totalRayCount = hitPtr[globalStatsBase + 1];
  }
  if (count == 0 && totalRayCount == 0)
    return;

  if (_primitiveHitScores.size() < totalPrimitiveCount)
    _primitiveHitScores.resize(totalPrimitiveCount, 0.0f);
  if (_primitiveHitLastFrame.size() < totalPrimitiveCount)
    _primitiveHitLastFrame.resize(totalPrimitiveCount, 0);
  if (_primitiveRayContributions.size() < totalPrimitiveCount)
    _primitiveRayContributions.resize(totalPrimitiveCount, 0.0f);
  if (_primitiveRaysTestedLastFrame.size() < totalPrimitiveCount)
    _primitiveRaysTestedLastFrame.resize(totalPrimitiveCount, 0);
  if (_primitiveHitAlpha.size() < totalPrimitiveCount)
    _primitiveHitAlpha.resize(totalPrimitiveCount, 1.0f);
  if (_primitiveHitBeta.size() < totalPrimitiveCount)
    _primitiveHitBeta.resize(totalPrimitiveCount, 1.0f);
  if (_primitiveHitProbability.size() < totalPrimitiveCount)
    _primitiveHitProbability.resize(totalPrimitiveCount, 0.5f);
  if (_primitiveHitVariance.size() < totalPrimitiveCount)
    _primitiveHitVariance.resize(totalPrimitiveCount, 1.0f / 12.0f);
  if (_primitivePosteriorMass.size() < totalPrimitiveCount)
    _primitivePosteriorMass.resize(totalPrimitiveCount, 2.0f);
  if (_primitiveExplorationScore.size() < totalPrimitiveCount)
    _primitiveExplorationScore.resize(totalPrimitiveCount, 0.0f);
  if (_primitiveIdleFrames.size() < totalPrimitiveCount)
    _primitiveIdleFrames.resize(totalPrimitiveCount, 0u);
  if (_probabilitySortedIndices.size() != totalPrimitiveCount) {
    _probabilitySortedIndices.resize(totalPrimitiveCount);
    std::iota(_probabilitySortedIndices.begin(), _probabilitySortedIndices.end(),
              size_t(0));
  }
  float rayHitDecay = _residencyConfig.rayHitDecay;
  float probabilityDecay = _residencyConfig.probabilityDecay;
  float probabilityThreshold = _residencyConfig.probabilityThreshold;
  constexpr float kMinPosteriorMass = 1.0e-3f;
  const float configuredWindow = _residencyConfig.probabilityEvidenceWindow;
  const float maxPosteriorMass =
      (configuredWindow > 0.0f && std::isfinite(configuredWindow))
          ? std::max(configuredWindow, kMinPosteriorMass)
          : std::numeric_limits<float>::max();
  const bool clampPosteriorMass =
      maxPosteriorMass < std::numeric_limits<float>::max();
  const uint32_t idleCooldownFrames =
      _residencyConfig.probabilityIdleCooldownFrames;
  const float idleGraceDecay =
      std::min(probabilityDecay,
               std::clamp(_residencyConfig.probabilityIdleDecay, 0.0f, 1.0f));
  auto renormalizePosterior = [&](float &alpha, float &beta) {
    float sum = alpha + beta;
    if (!(sum > 0.0f)) {
      alpha = beta = kMinPosteriorMass * 0.5f;
      sum = kMinPosteriorMass;
    } else if (sum < kMinPosteriorMass) {
      float scale = kMinPosteriorMass / sum;
      alpha *= scale;
      beta *= scale;
      sum = kMinPosteriorMass;
    }
    if (clampPosteriorMass && sum > maxPosteriorMass) {
      float scale = maxPosteriorMass / sum;
      alpha *= scale;
      beta *= scale;
      sum = maxPosteriorMass;
    }
    return sum;
  };
  parallelChunkedAsync(0, count, [&](size_t chunkStart, size_t chunkEnd) {
    for (size_t i = chunkStart; i < chunkEnd; ++i) {
      size_t base = i * kStatsPerPrimitive;
      uint32_t hits = hitPtr[base + 0];
      uint32_t raysTested = hitPtr[base + 1];
      _primitiveHitLastFrame[i] = hits;
      _primitiveRaysTestedLastFrame[i] = raysTested;
      _primitiveHitScores[i] = _primitiveHitScores[i] * rayHitDecay +
                               static_cast<float>(hits);
      _primitiveRayContributions[i] =
          _primitiveRayContributions[i] * rayHitDecay +
          static_cast<float>(raysTested);

      float success = static_cast<float>(hits);
      float failure =
          std::max(static_cast<float>(raysTested) - success, 0.0f);
      float alpha = _primitiveHitAlpha[i] * probabilityDecay + success;
      float beta = _primitiveHitBeta[i] * probabilityDecay + failure;
      float sum = alpha + beta;
      float probability = (sum > 0.0f) ? (alpha / sum) : 0.5f;
      bool wasVisible =
          (i < _primitiveVisible.size()) ? (_primitiveVisible[i] != 0) : false;
      bool wasActive =
          (i < _activePrimitive.size()) ? (_activePrimitive[i] != 0) : false;
      uint32_t idleFrames =
          (i < _primitiveIdleFrames.size()) ? _primitiveIdleFrames[i] : 0u;
      bool idleGraceActive =
          (idleCooldownFrames > 0) && (wasVisible || wasActive) &&
          (idleFrames < idleCooldownFrames);
      if (raysTested == 0) {
        // When no rays were fired this frame we still decay the posterior so
        // idle primitives drift toward deactivation. Idle primitives that are
        // still visible or were recently active are given a grace period
        // before applying the full cooling decay. During the grace period the
        // decay is either skipped entirely or uses a milder factor to avoid
        // immediately dropping the exploration probability.
        float coolingFactor = probabilityDecay;
        bool applyCooling = true;
        if (idleGraceActive) {
          if (idleGraceDecay >= 0.999f)
            applyCooling = false;
          else
            coolingFactor = idleGraceDecay;
        }
        if (applyCooling) {
          float cooledProbability =
              std::clamp(probability * coolingFactor, 0.0f, 1.0f);
          float cooledMass = std::max(sum, kMinPosteriorMass);
          if (clampPosteriorMass && cooledMass > maxPosteriorMass)
            cooledMass = maxPosteriorMass;
          alpha = cooledProbability * cooledMass;
          beta = std::max(cooledMass - alpha, 0.0f);
        }
      }
      sum = renormalizePosterior(alpha, beta);
      probability = (sum > 0.0f) ? (alpha / sum) : 0.5f;
      float clampedProbability = std::clamp(probability, 0.0f, 1.0f);
      float variance = 0.0f;
      if (sum > 1.0f)
        variance = (alpha * beta) / ((sum * sum) * (sum + 1.0f));
      _primitiveHitAlpha[i] = alpha;
      _primitiveHitBeta[i] = beta;
      _primitiveHitProbability[i] = clampedProbability;
      _primitiveHitVariance[i] = std::max(variance, 0.0f);
      _primitivePosteriorMass[i] = sum;

      float exploration = _primitiveExplorationScore[i] * probabilityDecay;
      if (raysTested > 0)
        _primitiveIdleFrames[i] = 0u;
      else if (i < _primitiveIdleFrames.size()) {
        uint32_t next = idleFrames;
        if (next < std::numeric_limits<uint32_t>::max())
          ++next;
        _primitiveIdleFrames[i] = next;
      }
      if (raysTested > 0) {
        if (clampedProbability < probabilityThreshold)
          exploration += static_cast<float>(raysTested);
        else
          exploration *= rayHitDecay;
      } else {
        exploration *= rayHitDecay;
        if (wasVisible || wasActive)
          exploration = std::max(exploration, kIdleVisibleExploreSeed);
      }
      _primitiveExplorationScore[i] = exploration;
      hitPtr[base + 0] = 0;
      hitPtr[base + 1] = 0;
    }
  });

  if (count < totalPrimitiveCount) {
    parallelChunkedAsync(count, totalPrimitiveCount,
                         [&](size_t chunkStart, size_t chunkEnd) {
                           for (size_t i = chunkStart; i < chunkEnd; ++i) {
                             _primitiveHitScores[i] *= rayHitDecay;
                             _primitiveRayContributions[i] *= rayHitDecay;
                             _primitiveHitLastFrame[i] = 0;
                             _primitiveRaysTestedLastFrame[i] = 0;

                             float alpha = _primitiveHitAlpha[i] * probabilityDecay;
                             float beta = _primitiveHitBeta[i] * probabilityDecay;
                             float sum = alpha + beta;
                             float probability =
                                 (sum > 0.0f) ? (alpha / sum) : 0.5f;
                             float cooledProbability =
                                 std::clamp(probability * probabilityDecay, 0.0f, 1.0f);
                             float cooledMass = std::max(sum, kMinPosteriorMass);
                             if (clampPosteriorMass && cooledMass > maxPosteriorMass)
                               cooledMass = maxPosteriorMass;
                             alpha = cooledProbability * cooledMass;
                             beta = std::max(cooledMass - alpha, 0.0f);
                             float updatedSum = renormalizePosterior(alpha, beta);
                             float updatedProbability =
                                 (updatedSum > 0.0f) ? (alpha / updatedSum) : 0.5f;
                             float variance = 0.0f;
                             if (updatedSum > 1.0f)
                               variance =
                                   (alpha * beta) /
                                   ((updatedSum * updatedSum) *
                                    (updatedSum + 1.0f));
                             _primitiveHitAlpha[i] = alpha;
                             _primitiveHitBeta[i] = beta;
                             _primitiveHitProbability[i] =
                                 std::clamp(updatedProbability, 0.0f, 1.0f);
                             _primitiveHitVariance[i] = std::max(variance, 0.0f);
                             _primitivePosteriorMass[i] = updatedSum;

                             float exploration =
                                 _primitiveExplorationScore[i] * probabilityDecay;
                             exploration *= rayHitDecay;
                             bool wasVisible =
                                 (i < _primitiveVisible.size())
                                     ? (_primitiveVisible[i] != 0)
                                     : false;
                             if (wasVisible)
                               exploration =
                                   std::max(exploration, kIdleVisibleExploreSeed);
                           _primitiveExplorationScore[i] = exploration;
                         }
                       });
  }

  if (bufferCount >= globalStatsBase + kGlobalRayStatsCount) {
    hitPtr[globalStatsBase + 0] = 0;
    hitPtr[globalStatsBase + 1] = 0;
  }

  _lastFrameGlobalEnvEscape =
      (totalRayCount > 0)
          ? static_cast<float>(envHitCount) /
                static_cast<float>(std::max(totalRayCount, 1u))
          : 0.0f;

  size_t objectCount = _allSceneObjects.size();
  if (objectCount == 0) {
    _objectHitAlpha.clear();
    _objectHitBeta.clear();
    _objectHitProbability.clear();
    _objectHitVariance.clear();
    _objectPosteriorMass.clear();
    _objectExplorationScore.clear();
    _objectRayHitScore.clear();
    _objectHitLastFrame.clear();
    _objectRaysTestedLastFrame.clear();
    _objectVisible.clear();
    _objectProbabilitySortedIndices.clear();
    return;
  }

  if (_objectHitAlpha.size() < objectCount)
    _objectHitAlpha.resize(objectCount, 1.0f);
  if (_objectHitBeta.size() < objectCount)
    _objectHitBeta.resize(objectCount, 1.0f);
  if (_objectHitProbability.size() < objectCount)
    _objectHitProbability.resize(objectCount, 0.5f);
  if (_objectHitVariance.size() < objectCount)
    _objectHitVariance.resize(objectCount, 1.0f / 12.0f);
  if (_objectPosteriorMass.size() < objectCount)
    _objectPosteriorMass.resize(objectCount, 2.0f);
  if (_objectExplorationScore.size() < objectCount)
    _objectExplorationScore.resize(objectCount, 0.0f);
  if (_objectRayHitScore.size() < objectCount)
    _objectRayHitScore.resize(objectCount, 0.0f);
  if (_objectHitLastFrame.size() < objectCount)
    _objectHitLastFrame.resize(objectCount, 0u);
  if (_objectRaysTestedLastFrame.size() < objectCount)
    _objectRaysTestedLastFrame.resize(objectCount, 0u);
  if (_objectIdleFrames.size() < objectCount)
    _objectIdleFrames.resize(objectCount, 0u);
  if (_objectVisible.size() < objectCount)
    _objectVisible.resize(objectCount, 0u);
  if (_objectProbabilitySortedIndices.size() != objectCount) {
    _objectProbabilitySortedIndices.resize(objectCount);
    std::iota(_objectProbabilitySortedIndices.begin(),
              _objectProbabilitySortedIndices.end(), size_t(0));
  }

  std::vector<uint64_t> objectHits(objectCount, 0);
  std::vector<uint64_t> objectRays(objectCount, 0);
  std::vector<uint8_t> objectActiveFlags(objectCount, 0);
  size_t processedPrimitiveCount = std::min(count, totalPrimitiveCount);
  for (size_t i = 0; i < processedPrimitiveCount; ++i) {
    size_t objectIndex =
        (i < _primitiveToObject.size()) ? _primitiveToObject[i] : SIZE_MAX;
    if (objectIndex >= objectCount)
      continue;
    objectHits[objectIndex] += _primitiveHitLastFrame[i];
    objectRays[objectIndex] += _primitiveRaysTestedLastFrame[i];
  }
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    if (objectIndex >= _allSceneObjects.size())
      break;
    const SceneObject &object = _allSceneObjects[objectIndex];
    size_t start = object.firstPrimitive;
    size_t end = std::min(start + object.primitiveCount, _activePrimitive.size());
    bool active = false;
    for (size_t prim = start; prim < end; ++prim) {
      if (_activePrimitive[prim]) {
        active = true;
        break;
      }
    }
    objectActiveFlags[objectIndex] = active ? 1u : 0u;
  }

  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    float success = static_cast<float>(objectHits[objectIndex]);
    float raysTested = static_cast<float>(objectRays[objectIndex]);
    _objectHitLastFrame[objectIndex] = static_cast<uint32_t>(success);
    _objectRaysTestedLastFrame[objectIndex] =
        static_cast<uint32_t>(raysTested);
    float decayedHits =
        (_objectRayHitScore[objectIndex] * rayHitDecay) + success;
    _objectRayHitScore[objectIndex] = decayedHits;

    float failure = std::max(raysTested - success, 0.0f);
    float alpha = _objectHitAlpha[objectIndex] * probabilityDecay + success;
    float beta = _objectHitBeta[objectIndex] * probabilityDecay + failure;
    float sum = alpha + beta;
    float probability = (sum > 0.0f) ? (alpha / sum) : 0.5f;
    uint32_t idleFrames =
        (objectIndex < _objectIdleFrames.size()) ? _objectIdleFrames[objectIndex]
                                                 : 0u;
    bool hasActivePrimitive =
        (objectIndex < objectActiveFlags.size()) &&
        (objectActiveFlags[objectIndex] != 0);
    bool visible =
        (objectIndex < _objectBounds.size()) ? isInView(_objectBounds[objectIndex])
                                             : false;
    if (objectIndex < _objectVisible.size())
      _objectVisible[objectIndex] = visible ? 1 : 0;
    if (raysTested == 0.0f) {
      bool idleGraceActive =
          (idleCooldownFrames > 0) && (visible || hasActivePrimitive) &&
          (idleFrames < idleCooldownFrames);
      float coolingFactor = probabilityDecay;
      bool applyCooling = true;
      if (idleGraceActive) {
        if (idleGraceDecay >= 0.999f)
          applyCooling = false;
        else
          coolingFactor = idleGraceDecay;
      }
      if (applyCooling) {
        float cooledProbability =
            std::clamp(probability * coolingFactor, 0.0f, 1.0f);
        float cooledMass = std::max(sum, kMinPosteriorMass);
        if (clampPosteriorMass && cooledMass > maxPosteriorMass)
          cooledMass = maxPosteriorMass;
        alpha = cooledProbability * cooledMass;
        beta = std::max(cooledMass - alpha, 0.0f);
      }
    }

    sum = renormalizePosterior(alpha, beta);
    probability = (sum > 0.0f) ? (alpha / sum) : 0.5f;
    float clampedProbability = std::clamp(probability, 0.0f, 1.0f);
    float variance = 0.0f;
    if (sum > 1.0f)
      variance = (alpha * beta) / ((sum * sum) * (sum + 1.0f));
    _objectHitAlpha[objectIndex] = alpha;
    _objectHitBeta[objectIndex] = beta;
    _objectHitProbability[objectIndex] = clampedProbability;
    _objectHitVariance[objectIndex] = std::max(variance, 0.0f);
    _objectPosteriorMass[objectIndex] = sum;

    float exploration = _objectExplorationScore[objectIndex] * probabilityDecay;
    if (raysTested > 0.0f) {
      _objectIdleFrames[objectIndex] = 0u;
      if (clampedProbability < probabilityThreshold)
        exploration += raysTested;
      else
        exploration *= rayHitDecay;
    } else {
      uint32_t next = idleFrames;
      if (next < std::numeric_limits<uint32_t>::max())
        ++next;
      _objectIdleFrames[objectIndex] = next;
      exploration *= rayHitDecay;
      if (visible || hasActivePrimitive)
        exploration = std::max(exploration, kIdleVisibleExploreSeed);
    }
    _objectExplorationScore[objectIndex] = exploration;
  }
}

void Renderer::trackFrameCommandBuffer(MTL::CommandBuffer *commandBuffer) {
  if (!commandBuffer)
    return;

  commandBuffer->retain();

  FrameCommandBufferRecord record;
  record.buffer = commandBuffer;
  record.trackedSince = std::chrono::steady_clock::now();

  {
    std::lock_guard<std::mutex> lock(_frameCommandBufferMutex);
    _frameCommandBuffers.push_back(record);
  }

  commandBuffer->addCompletedHandler([this](MTL::CommandBuffer *completed) {
    bool release = false;
    {
      std::lock_guard<std::mutex> lock(_frameCommandBufferMutex);
      auto it = std::find_if(
          _frameCommandBuffers.begin(), _frameCommandBuffers.end(),
          [completed](const FrameCommandBufferRecord &record) {
            return record.buffer == completed;
          });
      if (it != _frameCommandBuffers.end()) {
        _frameCommandBuffers.erase(it);
        release = true;
      }
    }
    if (release)
      completed->release();
  });
}

void Renderer::recordPathTraceCommandTime(double gpuMs, size_t tileCount) {
  if (gpuMs <= 0.0 || tileCount == 0)
    return;

  std::lock_guard<std::mutex> lock(_pathTraceBudgetMutex);
  const double msPerTile = gpuMs / static_cast<double>(tileCount);
  if (!std::isfinite(msPerTile) || msPerTile <= 0.0)
    return;

  _pathTraceGpuMsPerTileHistory.push_back(msPerTile);
  if (_pathTraceGpuMsPerTileHistory.size() > kPathTraceCommandHistorySamples) {
    _pathTraceGpuMsPerTileHistory.pop_front();
  }
}

size_t Renderer::updatePathTraceTilesPerCommandBudget() {
  std::lock_guard<std::mutex> lock(_pathTraceBudgetMutex);
  size_t budget = _pathTraceTilesPerCommandBudget;
  bool timeoutHit = _pathTraceCommandTimeout;
  _pathTraceCommandTimeout = false;

  if (timeoutHit) {
    budget = std::max(kPathTraceMinTilesPerCommand, budget / 2);
  }

  if (!_pathTraceGpuMsPerTileHistory.empty()) {
    double totalMs = std::accumulate(
        _pathTraceGpuMsPerTileHistory.begin(),
        _pathTraceGpuMsPerTileHistory.end(), 0.0);
    double avgMsPerTile =
        totalMs / static_cast<double>(_pathTraceGpuMsPerTileHistory.size());
    if (avgMsPerTile > 0.0) {
      double targetTiles =
          std::floor(kPathTraceTargetGpuMsPerCommand / avgMsPerTile);
      size_t desired = static_cast<size_t>(std::max(1.0, targetTiles));
      desired = std::clamp(desired, kPathTraceMinTilesPerCommand,
                           kPathTraceMaxTilesPerCommand);
      if (desired < budget) {
        budget = desired;
      } else if (!timeoutHit && desired > budget) {
        budget = std::min(budget + 1, desired);
      }
    }
  }

  budget = std::clamp(budget, kPathTraceMinTilesPerCommand,
                      kPathTraceMaxTilesPerCommand);
  _pathTraceTilesPerCommandBudget = budget;
  return budget;
}

bool Renderer::waitForPendingFrameCommands(
    std::chrono::milliseconds timeout,
    std::chrono::steady_clock::time_point *waitSnapshot) {
  std::vector<FrameCommandBufferRecord> pending;
  std::vector<MTL::CommandBuffer *> snapshotBuffers;
  std::chrono::steady_clock::time_point waitStart;
  {
    std::lock_guard<std::mutex> lock(_frameCommandBufferMutex);
    pending.reserve(_frameCommandBuffers.size());
    snapshotBuffers.reserve(_frameCommandBuffers.size());
    for (const auto &record : _frameCommandBuffers) {
      if (record.buffer)
        record.buffer->retain();
      pending.push_back(record);
      if (record.buffer)
        snapshotBuffers.push_back(record.buffer);
    }
    waitStart = std::chrono::steady_clock::now();
    if (waitSnapshot)
      *waitSnapshot = waitStart;
  }

  const bool infiniteTimeout = timeout == std::chrono::milliseconds::max();
  bool allComplete = true;

  for (auto &record : pending) {
    auto *buffer = record.buffer;
    if (!buffer)
      continue;

    const auto startTime = record.trackedSince;
    bool completed = false;

    while (true) {
      auto status = buffer->status();
      if (status == MTL::CommandBufferStatusCompleted ||
          status == MTL::CommandBufferStatusError) {
        completed = true;
        break;
      }

      if (!infiniteTimeout) {
        auto now = std::chrono::steady_clock::now();
        if (now - startTime >= timeout)
          break;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (!completed)
      allComplete = false;
  }

  std::unordered_set<MTL::CommandBuffer *> snapshotSet(snapshotBuffers.begin(),
                                                       snapshotBuffers.end());
  bool newCommandTracked = false;
  {
    std::lock_guard<std::mutex> lock(_frameCommandBufferMutex);
    for (const auto &record : _frameCommandBuffers) {
      if (!record.buffer)
        continue;
      bool presentInSnapshot = snapshotSet.find(record.buffer) != snapshotSet.end();
      if (!presentInSnapshot || record.trackedSince >= waitStart) {
        newCommandTracked = true;
        break;
      }
    }
  }

  for (auto &record : pending) {
    if (record.buffer)
      record.buffer->release();
  }

  if (!infiniteTimeout && !allComplete) {
    std::lock_guard<std::mutex> lock(_pathTraceBudgetMutex);
    _pathTraceCommandTimeout = true;
  }

  if (newCommandTracked)
    return false;

  return allComplete;
}

void Renderer::beginFrameMetrics() {
  _cpuStart = std::chrono::high_resolution_clock::now();
  _lastRayCount = static_cast<size_t>(Camera::screenSize.x * Camera::screenSize.y);

  if (_benchmarkEnabled) {
    ensureBenchmarkStream();
    BenchmarkSample sample;
    sample.frameIndex = _benchmarkFrameCounter++;
    sample.rayCount = _lastRayCount;
    sample.primitiveActivations = _framePrimitiveActivations;
    sample.primitiveDeactivations = _framePrimitiveDeactivations;
    sample.objectActivations = _frameObjectActivations;
    sample.objectDeactivations = _frameObjectDeactivations;
    sample.objectsOnloadRequested = _frameObjectsOnloadRequested;
    sample.objectsOffloadRequested = _frameObjectsOffloadRequested;
    sample.onloadRequestedMB =
        static_cast<double>(_frameOnloadRequestedBytes) / (1024.0 * 1024.0);
    sample.offloadRequestedMB =
        static_cast<double>(_frameOffloadRequestedBytes) / (1024.0 * 1024.0);
    sample.blasBuildRequests = _frameBlasBuildRequests;
    sample.tlasRebuilds = _frameTlasRebuilds;
    sample.tlasRefits = _frameTlasRefits;
    sample.activePrimitiveCount = _activePrimitiveCount;
    sample.residentPrimitiveCount = _residentPrimitiveCount;
    sample.totalPrimitiveCount = _allPrimitives.size();
    sample.activeTriangleCount = _activeTriangleCount;
    sample.residentTriangleCount = _residentTriangleCount;
    sample.totalTriangleCount = _pScene ? _pScene->getTriangleCount() : 0;
    sample.activeNodeCount = _activeNodeCount;
    sample.residentNodeCount = _residentNodeCount;
    sample.totalNodeCount = _totalNodeCount;
    size_t activeObjects = 0;
    for (bool active : _objectActive)
      if (active)
        ++activeObjects;
    sample.activeObjectCount = activeObjects;
    size_t residentObjects = 0;
    for (const auto &resident : _residentObjectGpuResources)
      if (resident.isResident())
        ++residentObjects;
    sample.residentObjectCount = residentObjects;
    sample.visiblePrimitiveCount = static_cast<size_t>(std::count_if(
        _primitiveVisible.begin(), _primitiveVisible.end(),
        [](uint8_t visible) { return visible != 0; }));
    sample.visibleObjectCount = static_cast<size_t>(std::count_if(
        _objectVisible.begin(), _objectVisible.end(),
        [](uint8_t visible) { return visible != 0; }));
    sample.primitiveHitsLastFrame = std::accumulate(
        _primitiveHitLastFrame.begin(), _primitiveHitLastFrame.end(), size_t(0));
    sample.primitiveRaysTestedLastFrame =
        std::accumulate(_primitiveRaysTestedLastFrame.begin(),
                        _primitiveRaysTestedLastFrame.end(), size_t(0));
    sample.objectHitsLastFrame = std::accumulate(
        _objectHitLastFrame.begin(), _objectHitLastFrame.end(), size_t(0));
    sample.objectRaysTestedLastFrame =
        std::accumulate(_objectRaysTestedLastFrame.begin(),
                        _objectRaysTestedLastFrame.end(), size_t(0));
    sample.gpuMemoryMB = currentGPUMemoryMB();
    sample.scratchMemoryMB = scratchMemoryMB();
    sample.textureMemoryCapMB = _textureResidencyMemoryCapMB;
    sample.geometryMemoryCapMB = _geometryResidencyMemoryCapMB;
    sample.totalMemoryCapMB = effectiveTotalGpuMemoryCapMB();
    sample.minimumResidentFootprintMB = _frameMinimumResidentFootprintMB;
    sample.totalMemoryCapRelaxedMB = _totalMemoryCapRelaxedMB;
    double geometryResidentMB = residentGeometryMemoryMB();
    double strictGeometryResidentMB = strictResidentGeometryMemoryMB();
    double textureResidentMB = residentTextureMemoryMB();
    sample.residentGeometryMemoryMB = geometryResidentMB;
    sample.strictResidentGeometryMemoryMB = strictGeometryResidentMB;
    sample.residentTextureMemoryMB = textureResidentMB;
    sample.residencyMemoryMB =
        std::max(0.0, geometryResidentMB + textureResidentMB);
    sample.avgHitProbability = 0.0;
    sample.p95HitProbability = 0.0;
    sample.probabilityThreshold = _residencyConfig.probabilityThreshold;
    sample.probabilityTargetFraction = _residencyConfig.probabilityTargetFraction;
    sample.probabilityVisibleFloor = _residencyConfig.probabilityVisibleFloor;
    sample.cameraMotionMetric = _frameCameraMotionMetric;
    sample.environmentTargetActiveFraction =
        _residencyConfig.environmentTargetActiveFraction;
    sample.environmentEscapeThreshold =
        _residencyConfig.environmentEscapeThreshold;
    sample.envHighEscapeThreshold = _residencyConfig.envHighEscapeThreshold;
    sample.envLowEscapeThreshold = _residencyConfig.envLowEscapeThreshold;
    sample.globalEnvEscape = _lastFrameGlobalEnvEscape;
    sample.environmentActivationFloor = _frameEnvironmentActivationFloor;
    sample.environmentDepthWeights =
        formatFloatList(_residencyConfig.environmentDepthWeights);
    sample.environmentDepthRadii =
        formatFloatList(_residencyConfig.environmentDepthRadii);
    if (_benchmarkLogProbabilities) {
      if (!_primitiveHitProbability.empty()) {
        size_t primitiveCount =
            std::min(_primitiveHitProbability.size(), _allPrimitives.size());
        std::vector<float> validProbabilities;
        validProbabilities.reserve(primitiveCount);
        double probabilitySum = 0.0;
        std::ostringstream primitiveStream;
        bool firstPrimitive = true;

        std::vector<double> objectProbabilitySums(_allSceneObjects.size(), 0.0);
        std::vector<size_t> objectProbabilityCounts(_allSceneObjects.size(), 0);

        for (size_t index = 0; index < primitiveCount; ++index) {
          float probability = _primitiveHitProbability[index];
          bool probabilityFinite = std::isfinite(probability);
          float sanitized = probabilityFinite
                                 ? std::clamp(probability, 0.0f, 1.0f)
                                 : 0.0f;

          if (!firstPrimitive)
            primitiveStream << ';';
          primitiveStream << index << ':'
                          << formatFixed(static_cast<double>(sanitized), 6);
          firstPrimitive = false;

          if (probabilityFinite) {
            probabilitySum += sanitized;
            validProbabilities.push_back(sanitized);
          }

          if (index < _primitiveToObject.size()) {
            size_t objectIndex = _primitiveToObject[index];
            if (objectIndex < objectProbabilitySums.size()) {
              objectProbabilitySums[objectIndex] +=
                  static_cast<double>(sanitized);
              objectProbabilityCounts[objectIndex] += 1;
            }
          }
        }

        sample.primitiveProbabilities = primitiveStream.str();

        if (!_allSceneObjects.empty()) {
          std::ostringstream objectStream;
          bool firstObject = true;
          for (size_t objectIndex = 0; objectIndex < _allSceneObjects.size();
               ++objectIndex) {
            double avgProbability = 0.0;
            if (objectIndex < objectProbabilitySums.size() &&
                objectProbabilityCounts[objectIndex] > 0) {
              avgProbability = objectProbabilitySums[objectIndex] /
                                static_cast<double>(
                                    objectProbabilityCounts[objectIndex]);
            }
            if (!firstObject)
              objectStream << ';';
            objectStream << objectIndex << ':'
                         << formatFixed(avgProbability, 6);
            firstObject = false;
          }
          sample.objectProbabilities = objectStream.str();
        }

        if (!validProbabilities.empty()) {
          sample.avgHitProbability =
              probabilitySum / static_cast<double>(validProbabilities.size());
          size_t percentileIndex = static_cast<size_t>(std::floor(
              0.95 * static_cast<double>(validProbabilities.size() - 1)));
          percentileIndex = std::min(percentileIndex,
                                      validProbabilities.size() - 1);
          std::nth_element(validProbabilities.begin(),
                           validProbabilities.begin() + percentileIndex,
                           validProbabilities.end());
          sample.p95HitProbability = validProbabilities[percentileIndex];
        }
      } else if (!_allSceneObjects.empty()) {
        std::ostringstream objectStream;
        bool firstObject = true;
        for (size_t objectIndex = 0; objectIndex < _allSceneObjects.size();
             ++objectIndex) {
          if (!firstObject)
            objectStream << ';';
          objectStream << objectIndex << ':' << formatFixed(0.0, 6);
          firstObject = false;
        }
        sample.objectProbabilities = objectStream.str();
      }
    }
    sample.probabilisticToggles = _frameProbabilisticToggles;
    sample.probabilityTargetPrimitives = _frameProbabilityTargetPrimitives;
    sample.probabilityInitialDesiredPrimitives =
        _frameProbabilityInitialDesiredPrimitives;
    sample.probabilityFinalDesiredPrimitives =
        _frameProbabilityFinalDesiredPrimitives;
    sample.probabilityTrimmedPrimitives = _frameProbabilityTrimmedPrimitives;
    sample.probabilityBudgetHit = _frameProbabilityBudgetHit;
    sample.screenMinPixelCoverageSkips = _frameScreenMinPixelCoverageSkips;
    sample.deltaTimeSeconds = _deltaTimeSeconds;
    sample.wallSeconds = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - _benchmarkStartTime)
                            .count();
    ResidencyStrategy currentStrategy = _frameStrategy;
    if (_pScene)
      currentStrategy = _pScene->getResidencyStrategy();
    _frameStrategy = currentStrategy;
    sample.strategy = currentStrategy;
    sample.strategyName = residencyStrategyName(currentStrategy);
    sample.minSamplesPerPixel = _minSamplesPerPixel;
    sample.maxSamplesPerPixel = _maxSamplesPerPixel;
    sample.residentCompacted = _residentCompacted;
    sample.overMemoryCap =
        sample.residencyMemoryMB > _textureResidencyMemoryCapMB;
    sample.geometryOverMemoryCap =
        geometryResidentMB > _geometryResidencyMemoryCapMB;
    sample.totalOverMemoryCap =
        effectiveTotalGpuMemoryCapMB() > 0.0 &&
        sample.gpuMemoryMB > effectiveTotalGpuMemoryCapMB();
    sample.geometryCapHitCount = _frameGeometryResidencyCapHitCount;
    sample.geometryHardCapDeniedCount =
        _frameGeometryResidencyHardCapDeniedCount;
    sample.totalMemoryOverageWarnings = _frameTotalMemoryOverageWarnings;
    sample.totalMemoryCapDeniedCount = _frameTotalMemoryCapDeniedCount;
    sample.totalMemoryCapNonResidencyDeniedCount =
        _frameTotalMemoryCapNonResidencyDeniedCount;
    sample.totalMemoryEvictionStall = _frameTotalMemoryEvictionStall;
    _pendingBenchmarkSamples.push_back(std::move(sample));
  }

  if (_neuralFeatureLoggingEnabled && _benchmarkEnabled)
    accumulateNeuralClipFeatures();
}

void Renderer::completeFrameMetrics(MTL::CommandBuffer *pCmd) {
  auto cpuEnd = std::chrono::high_resolution_clock::now();
  _lastCPUTime = std::chrono::duration<double>(cpuEnd - _cpuStart).count();
  if (pCmd) {
    _lastGPUTime = pCmd->GPUEndTime() - pCmd->GPUStartTime();
  } else {
    _lastGPUTime = 0.0;
  }
  if (_lastGPUTime > 0.0) {
    _lastRaysPerSecond = static_cast<double>(_lastRayCount) / _lastGPUTime;
  } else {
    _lastRaysPerSecond = 0.0;
  }
  _lastRestirReuseCandidates = 0;
  _lastRestirReuseAccepted = 0;
  if (pCmd && _pRestirStatsReadback) {
    RestirStatsHost *stats =
        static_cast<RestirStatsHost *>(_pRestirStatsReadback->contents());
    if (stats) {
      _lastRestirReuseCandidates = stats->reuseCandidates;
      _lastRestirReuseAccepted = stats->reuseAccepted;
    }
  }
  bool canRecalculateNodes =
      (_blasNodeCount > 0 && _cachedBVHNodes.size() >= _blasNodeCount * 2) ||
      (_tlasNodeCount > 0 && _cachedTLASNodes.size() >= _tlasNodeCount * 2) ||
      (_residentCompacted && (_blasNodeCount > 0 || _tlasNodeCount > 0));
  if (canRecalculateNodes) {
    if (_residentObjectGpuResources.empty()) {
      recalculateNodeCounters(_objectResidentState);
    } else {
      auto gpuResidentMask = buildResidentMaskFromGpuResources();
      recalculateNodeCounters(gpuResidentMask);
    }
  }

  size_t offloaded = _totalNodeCount > _residentNodeCount ?
                         _totalNodeCount - _residentNodeCount :
                         0;
  double geometryResidentMB = residentGeometryMemoryMB();
  size_t totalAllocatedBytes =
      static_cast<size_t>(_pDevice->currentAllocatedSize());
  GpuMemoryTracker::Snapshot memSnapshot = _gpuMemoryTracker.snapshot();
  size_t otherBytes = totalAllocatedBytes > memSnapshot.totalTracked
                          ? totalAllocatedBytes - memSnapshot.totalTracked
                          : 0;
  auto bytesToMB = [](size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
  };
  double totalMemoryMB = bytesToMB(totalAllocatedBytes);
  double scratchMB = bytesToMB(memSnapshot.bytes[static_cast<size_t>(
      GpuMemoryTracker::Category::Scratch)]);
  double geometryMB = bytesToMB(memSnapshot.bytes[static_cast<size_t>(
      GpuMemoryTracker::Category::Geometry)]);
  double textureMB = bytesToMB(memSnapshot.bytes[static_cast<size_t>(
      GpuMemoryTracker::Category::Textures)]);
  double restirMB = restirTextureMemoryMB();
  if (!_pScene || !_pScene->getRestirEnabled())
    restirMB = 0.0;
  double rendererMB = bytesToMB(memSnapshot.bytes[static_cast<size_t>(
      GpuMemoryTracker::Category::RendererBuffers)]);
  double heapsMB = bytesToMB(memSnapshot.bytes[static_cast<size_t>(
      GpuMemoryTracker::Category::HeapsAS)]);
  double stagingMB = bytesToMB(memSnapshot.bytes[static_cast<size_t>(
      GpuMemoryTracker::Category::Staging)]);
  double otherMB = bytesToMB(otherBytes);
  printf("Active nodes: %zu Resident nodes: %zu Offloaded nodes: %zu CPU: %.3f ms GPU: %.3f ms Rays/s: %.2f Resident geometry: %.3f MB\n",
         _activeNodeCount, _residentNodeCount, offloaded,
         _lastCPUTime * 1000.0, _lastGPUTime * 1000.0, _lastRaysPerSecond,
         geometryResidentMB);
  printf(
      "GPU mem (MB) total=%.3f scratch=%.3f geometry=%.3f textures=%.3f restir=%.3f renderer=%.3f heaps=%.3f staging=%.3f other=%.3f\n",
      totalMemoryMB, scratchMB, geometryMB, textureMB, restirMB, rendererMB,
      heapsMB, stagingMB, otherMB);
  if (_frameTotalMemoryCapNonResidencyDeniedCount > 0) {
    printf(
        "[MemoryBudget] Total GPU memory cap blocked non-residency allocations: %zu\n",
        _frameTotalMemoryCapNonResidencyDeniedCount);
  }
  double restirReuseRate =
      (_lastRayCount > 0)
          ? static_cast<double>(_lastRestirReuseCandidates) /
                static_cast<double>(_lastRayCount)
          : 0.0;
  double restirAcceptRate =
      (_lastRestirReuseCandidates > 0)
          ? static_cast<double>(_lastRestirReuseAccepted) /
                static_cast<double>(_lastRestirReuseCandidates)
          : 0.0;
  printf(
      "ReSTIR temporal reuse stats: restirReuseCandidates=%u "
      "restirReuseAccepted=%u restirReuseRate=%.6f restirAcceptRate=%.6f "
      "(reuseCandidates=temporal surface valid attempts; reuseAccepted=temporal "
      "candidate built and fed into restirUpdateReservoir; "
      "reuseRate=reuseCandidates/primaryRays; acceptRate=reuseAccepted/"
      "reuseCandidates)\n",
      _lastRestirReuseCandidates, _lastRestirReuseAccepted, restirReuseRate,
      restirAcceptRate);
  if (isAlwaysResidentStrategy() && offloaded > 0 &&
      _forcedObjectOffIndex == std::numeric_limits<size_t>::max()) {
    printf("Always-resident strategy reported %zu offloaded nodes.\n",
           offloaded);
    assert(offloaded == 0 &&
           "Always-resident strategy should not report offloaded nodes");
  }

  if (_benchmarkEnabled && !_pendingBenchmarkSamples.empty()) {
    BenchmarkSample sample = std::move(_pendingBenchmarkSamples.front());
    _pendingBenchmarkSamples.pop_front();
    sample.cameraMotionMetric = _frameCameraMotionMetric;
    sample.cpuTimeSeconds = _lastCPUTime;
    sample.gpuTimeSeconds = _lastGPUTime;
    sample.raysPerSecond = _lastRaysPerSecond;
    sample.rayCount = _lastRayCount;
    sample.gpuMemoryMB = totalMemoryMB;
    sample.scratchMemoryMB = scratchMB;
    sample.gpuGeometryMB = geometryMB;
    sample.gpuTextureMB = textureMB;
    sample.gpuRestirMB = restirMB;
    sample.gpuRendererMB = rendererMB;
    sample.gpuHeapsMB = heapsMB;
    sample.gpuStagingMB = stagingMB;
    sample.gpuOtherMB = otherMB;
    sample.wallSeconds = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - _benchmarkStartTime)
                            .count();
    writeBenchmarkRow(sample);
  }
}

double Renderer::lastCPUTime() const { return _lastCPUTime; }
double Renderer::lastGPUTime() const { return _lastGPUTime; }
double Renderer::lastRaysPerSecond() const { return _lastRaysPerSecond; }
size_t Renderer::activeNodeCount() const { return _activeNodeCount; }
size_t Renderer::residentNodeCount() const { return _residentNodeCount; }
size_t Renderer::totalNodeCount() const { return _totalNodeCount; }
bool Renderer::isAlwaysResidentStrategy() const {
  return _frameStrategy == ResidencyStrategy::AlwaysResident;
}
