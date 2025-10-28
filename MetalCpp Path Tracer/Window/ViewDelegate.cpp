#include "ViewDelegate.h"
#include "ControllerView.hpp"
#include <AppKit/AppKit.hpp>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>

using namespace MetalCppPathTracer;

ViewDelegate::ViewDelegate(MTL::Device *pDevice)
    : MTK::ViewDelegate(), _pRenderer(new Renderer(pDevice)),
      _lastTime(std::chrono::steady_clock::now()) {
  if (const char *env = std::getenv("MPT_MAX_FRAMES"))
    _maxFrames = std::strtoul(env, nullptr, 10);
  if (const char *runs = std::getenv("MPT_RUNS_PATH")) {
    std::filesystem::path base(runs);
    std::filesystem::create_directories(base);

    _dumpPath = (base / "as").string();
    std::filesystem::create_directories(_dumpPath);

    std::filesystem::path perf = base / "perf.csv";
    _perfLog.open(perf);
    if (_perfLog.is_open())
      _perfLog <<
          "frame,fps,cpu_ms,command_submit_ms,gpu_ms,rays_per_second,active_nodes,resident_nodes,offloaded_nodes,total_nodes\n";

    std::filesystem::path gpu = base / "gpu_mem.csv";
    _gpuMemLog.open(gpu);
    if (_gpuMemLog.is_open())
      _gpuMemLog << "frame,gpu_memory_mb\n";
  }

  auto parseBoolEnv = [](const char *value) {
    if (!value || !value[0])
      return false;
    if (std::isdigit(static_cast<unsigned char>(value[0])))
      return std::strtoul(value, nullptr, 10) != 0;
    char first = static_cast<char>(
        std::tolower(static_cast<unsigned char>(value[0])));
    return first == 't' || first == 'y';
  };

  bool captureEnabled = parseBoolEnv(std::getenv("MPT_CAPTURE_EXR"));

  size_t captureInterval = 4;
  if (const char *intervalEnv = std::getenv("MPT_CAPTURE_INTERVAL")) {
    unsigned long parsed = std::strtoul(intervalEnv, nullptr, 10);
    if (parsed > 0)
      captureInterval = parsed;
  }

  _pRenderer->setFrameCaptureEnabled(captureEnabled);
  _pRenderer->setFrameCaptureInterval(captureInterval);
}

ViewDelegate::~ViewDelegate() {
  if (_gpuMemLog.is_open())
    _gpuMemLog.close();
  if (_perfLog.is_open())
    _perfLog.close();
  delete _pRenderer;
}

void ViewDelegate::drawInMTKView(MTK::View *pView) {
  auto current = std::chrono::steady_clock::now();
  double deltaSeconds =
      std::chrono::duration<double>(current - _lastTime).count();
  if (deltaSeconds <= 0.0)
    deltaSeconds = std::numeric_limits<double>::min();
  double fps = 1.0 / deltaSeconds;
  _lastTime = current;
  updateFPS(fps);
  updateMemoryUsage(_pRenderer->currentGPUMemoryMB());
  _pRenderer->setDeltaTime(deltaSeconds);
  _pRenderer->draw(pView);
  if (_perfLog.is_open()) {
    double cpu_ms = _pRenderer->lastCPUTime() * 1000.0;
    double submit_ms = _pRenderer->lastCommandSubmissionTime() * 1000.0;
    double gpu_ms = _pRenderer->lastGPUTime() * 1000.0;
    double rays = _pRenderer->lastRaysPerSecond();
    size_t active = _pRenderer->activeNodeCount();
    size_t resident = _pRenderer->residentNodeCount();
    size_t total = _pRenderer->totalNodeCount();
    size_t offloaded = total > resident ? total - resident : 0;
    _perfLog << _frameCount << "," << fps << "," << cpu_ms << ","
             << submit_ms << "," << gpu_ms << "," << rays << "," << active
             << "," << resident
             << "," << offloaded << "," << total << "\n";
    _perfLog.flush();
  }
  if (_gpuMemLog.is_open()) {
    _gpuMemLog << _frameCount << "," << _pRenderer->currentGPUMemoryMB()
               << "\n";
    _gpuMemLog.flush();
  }
  if (!_dumpPath.empty()) {
    char file[256];
    std::snprintf(file, sizeof(file), "%s/frame_%04zu.json", _dumpPath.c_str(),
                  _frameCount);
    _pRenderer->dumpAccelerationStructure(file);
  }
  ++_frameCount;
  if (_maxFrames > 0 && _pRenderer->hasKeyframes() && _frameCount >= _maxFrames)
    NS::Application::sharedApplication()->terminate(nullptr);
}

void ViewDelegate::drawableSizeWillChange(MTK::View *pView, CGSize size) {
  _pRenderer->drawableSizeWillChange(pView, size);
}
