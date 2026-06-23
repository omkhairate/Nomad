#include "Renderer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <dispatch/dispatch.h>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr float kIdleVisibleExploreSeed = 1.0f;
constexpr std::chrono::milliseconds kFrameCommandBufferWaitTimeout(4);

inline float clampUnit(float value) {
  return std::clamp(value, 0.0f, 1.0f);
}

inline float lerpFloat(float a, float b, float t) {
  return a + (b - a) * clampUnit(t);
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

float primitiveArea(const NomadPathTracer::Primitive &p) {
  switch (p.type) {
  case NomadPathTracer::PrimitiveType::Sphere: {
    float r = p.sphere.radius;
    return 4.0f * static_cast<float>(M_PI) * r * r;
  }
  case NomadPathTracer::PrimitiveType::Triangle: {
    simd::float3 e1 = p.triangle.v1 - p.triangle.v0;
    simd::float3 e2 = p.triangle.v2 - p.triangle.v0;
    return 0.5f * simd::length(simd::cross(e1, e2));
  }
  case NomadPathTracer::PrimitiveType::Rectangle: {
    simd::float3 e1 = p.rectangle.u;
    simd::float3 e2 = p.rectangle.v;
    return 4.0f * simd::length(simd::cross(e1, e2));
  }
  }
  return 0.0f;
}

float primitiveImportance(const NomadPathTracer::Primitive &p) {
  float area = std::max(primitiveArea(p), 1e-4f);
  const NomadPathTracer::Material &m = p.material;
  float emissive = m.emissionPower * luminance(m.emissionColor);
  float diffuse = luminance(m.diffuseColor) * m.opacity;
  float specular = luminance(m.specularColor) * m.opacity;
  float transmission = (1.0f - m.opacity) * luminance(m.transmissionColor);
  return area * (emissive + diffuse + specular + transmission);
}

float sanitizeSortValue(float value) {
  if (std::isnan(value))
    return -std::numeric_limits<float>::max();

  const float finiteMax = std::numeric_limits<float>::max();
  if (value >= finiteMax)
    return finiteMax;
  if (value <= -finiteMax)
    return -finiteMax;


  return value;
}

std::string formatFixed(double value, int precision = 6) {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(precision) << value;
  return ss.str();
}

void appendCsvEscaped(std::ostringstream &ss, const std::string &value) {
  ss << '"';
  for (char c : value) {
    if (c == '"')
      ss << '"';
    ss << c;
  }
  ss << '"';
}

} // namespace

namespace NomadPathTracer {

void Renderer::resetProbabilisticResidencyState() {
  std::fill(_primitiveHitScores.begin(), _primitiveHitScores.end(), 0.0f);
  std::fill(_primitiveHitAlpha.begin(), _primitiveHitAlpha.end(), 1.0f);
  std::fill(_primitiveHitBeta.begin(), _primitiveHitBeta.end(), 1.0f);
  std::fill(_primitiveHitProbability.begin(), _primitiveHitProbability.end(),
            0.5f);
  std::fill(_primitiveHitVariance.begin(), _primitiveHitVariance.end(),
            1.0f / 12.0f);
  std::fill(_primitivePosteriorMass.begin(), _primitivePosteriorMass.end(),
            2.0f);
  std::fill(_primitiveExplorationScore.begin(),
            _primitiveExplorationScore.end(), 0.0f);

  std::fill(_objectHitAlpha.begin(), _objectHitAlpha.end(), 1.0f);
  std::fill(_objectHitBeta.begin(), _objectHitBeta.end(), 1.0f);
  std::fill(_objectHitProbability.begin(), _objectHitProbability.end(), 0.5f);
  std::fill(_objectHitVariance.begin(), _objectHitVariance.end(), 1.0f / 12.0f);
  std::fill(_objectPosteriorMass.begin(), _objectPosteriorMass.end(), 2.0f);
  std::fill(_objectExplorationScore.begin(), _objectExplorationScore.end(),
            0.0f);

  std::fill(_primitiveCooldown.begin(), _primitiveCooldown.end(), 0u);
  std::fill(_objectCooldown.begin(), _objectCooldown.end(), 0u);
  std::fill(_objectLastToggleFrame.begin(), _objectLastToggleFrame.end(), 0u);
  _rayHitRebuildCooldown = 0;
}

void Renderer::updateResidency(bool forceAllToggles, bool forceFullRebuild) {
  if (!_pScene)
    return;

  _framePrimitiveActivations = 0;
  _framePrimitiveDeactivations = 0;
  _frameObjectActivations = 0;
  _frameObjectDeactivations = 0;
  _frameObjectsOnloadRequested = 0;
  _frameObjectsOffloadRequested = 0;
  _frameOnloadRequestedBytes = 0;
  _frameOffloadRequestedBytes = 0;
  _frameBlasBuildRequests = 0;
  _frameTlasRebuilds = 0;
  _frameTlasRefits = 0;
  _frameProbabilisticToggles = 0;
  _frameScreenMinPixelCoverageSkips = 0;
  _frameEnvironmentActivationFloor = 0;
  _frameGeometryResidencyCapHitCount = 0;
  _frameGeometryResidencyHardCapDeniedCount = 0;
  _frameTotalMemoryOverageWarnings = 0;
  _frameTotalMemoryCapDeniedCount = 0;
  _frameTotalMemoryCapNonResidencyDeniedCount = 0;
  _frameTotalMemoryEvictionStall = false;
  _frameMinimumResidentFootprintMB = 0.0;
  _frameTransportCriticalResidencyChanged = false;
  ResidencyStrategy strategy = _pScene->getResidencyStrategy();
  if (strategy != _lastResidencyStrategy) {
    if (strategy == ResidencyStrategy::AlwaysResident) {
      _alwaysResidentCache.markDirty();
      _forceAlwaysResidentActivation = true;
      _pendingAlwaysResidentPrewarm = true;
      _alwaysResidentWarmupWaitCount = 0;
    } else {
      _alwaysResidentWarmupWaitCount = 0;
    }
    _lastResidencyStrategy = strategy;
  }
  _frameStrategy = strategy;

  for (uint32_t &cooldown : _primitiveCooldown)
    if (cooldown > 0)
      --cooldown;
  for (uint32_t &cooldown : _objectCooldown)
    if (cooldown > 0)
      --cooldown;
  if (_compactionCooldown > 0)
    --_compactionCooldown;

  bool changed = false;
  switch (strategy) {
  case ResidencyStrategy::EnergyImportance:
    changed = updateEnergyImportance(forceAllToggles);
    break;
  case ResidencyStrategy::UnifiedScore:
  case ResidencyStrategy::UnifiedNeural:
    changed = updateUnifiedResidency(forceAllToggles);
    break;
  case ResidencyStrategy::RayHitBudget:
    changed = updateRayHitBudget(forceAllToggles);
    break;
  case ResidencyStrategy::Probabilistic:
    changed = updateProbabilisticResidency(forceAllToggles);
    break;
  case ResidencyStrategy::ScreenSpaceFootprint:
    changed = updateScreenSpaceFootprint(forceAllToggles);
    break;
  case ResidencyStrategy::PredictiveEnvironment:
    changed = updatePredictiveResidency(forceAllToggles);
    break;
  case ResidencyStrategy::EnvironmentHit:
    changed = updateEnvironmentHitResidency(forceAllToggles);
    break;
  case ResidencyStrategy::AlwaysResident:
    changed = updateAlwaysResident(forceAllToggles);
    break;
  case ResidencyStrategy::DistanceLOD:
  default:
    changed = updateLODByDistance(forceAllToggles);
    break;
  }

  const bool requiresVisibleGuard =
      strategy != ResidencyStrategy::DistanceLOD &&
      strategy != ResidencyStrategy::RayHitBudget &&
      strategy != ResidencyStrategy::AlwaysResident;
  if (requiresVisibleGuard &&
      enforceTransportCriticalObjectResidency(forceAllToggles))
    changed = true;
  if (requiresVisibleGuard && enforceEmissiveObjectResidency(forceAllToggles))
    changed = true;
  if (requiresVisibleGuard && enforceVisibleObjectResidency(forceAllToggles))
    changed = true;

  if (requiresVisibleGuard) {
    const size_t primitiveChurn =
        _framePrimitiveActivations + _framePrimitiveDeactivations;
    const size_t objectChurn = _frameObjectActivations + _frameObjectDeactivations;
    const size_t primitiveCount = _activePrimitive.size();
    const size_t churnThreshold =
        std::max<size_t>(24, static_cast<size_t>(std::ceil(
                                 static_cast<double>(primitiveCount) * 0.03)));
    if (_frameTransportCriticalResidencyChanged || primitiveChurn >= churnThreshold ||
        objectChurn >= 6) {
      _restirReuseDisableUntilFrame =
          std::max(_restirReuseDisableUntilFrame, _renderedFrameCount + 3);
    }
  }

  if (_residencyPreviewOnly) {
    _previewPrimitiveActive = _activePrimitive;
    _previewObjectActive = _objectActive;
  }

  if (changed || forceFullRebuild)
    flushResidencyChanges(forceFullRebuild);

  if (_pendingAlwaysResidentPrewarm &&
      (strategy == ResidencyStrategy::AlwaysResident || _residencyPreviewOnly))
    prewarmAlwaysResidentResources();

  updateHeapShrinkCandidates();
}

bool Renderer::enforceEmissiveObjectResidency(bool forceAllToggles) {
  const size_t objectCount = _allSceneObjects.size();
  if (objectCount == 0 || _activePrimitive.empty())
    return false;

  size_t baseBudget = 0;
  switch (_frameStrategy) {
  case ResidencyStrategy::EnergyImportance:
  case ResidencyStrategy::UnifiedScore:
  case ResidencyStrategy::UnifiedNeural:
    baseBudget = _residencyConfig.energyMaxTogglesPerFrame;
    break;
  case ResidencyStrategy::ScreenSpaceFootprint:
    baseBudget = _residencyConfig.screenFootprintMaxTogglesPerFrame;
    break;
  case ResidencyStrategy::Probabilistic:
    baseBudget = _residencyConfig.probabilityMaxTogglesPerFrame;
    break;
  case ResidencyStrategy::PredictiveEnvironment:
  case ResidencyStrategy::EnvironmentHit:
    baseBudget = _residencyConfig.environmentMaxTogglesPerFrame;
    break;
  default:
    baseBudget = _residencyConfig.energyMaxTogglesPerFrame;
    break;
  }

  size_t emissiveToggleBudget = std::max<size_t>(32, baseBudget * 4);
  if (forceAllToggles)
    emissiveToggleBudget = std::numeric_limits<size_t>::max();

  bool changed = false;
  size_t usedBudget = 0;

  auto promoteEmissivePass = [&](bool visibleOnly) {
    for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
      const SceneObject &obj = _allSceneObjects[objectIndex];
      if (obj.primitiveCount == 0)
        continue;

      const size_t first = obj.firstPrimitive;
      const size_t last =
          std::min(first + obj.primitiveCount, _activePrimitive.size());
      bool hasEmissivePrimitive = false;
      bool missingEmissivePrimitive = false;
      for (size_t primIndex = first; primIndex < last; ++primIndex) {
        if (primIndex >= _allPrimitives.size())
          break;
        const Material &m = _allPrimitives[primIndex].material;
        const float emissionStrength =
            m.emissionPower * luminance(m.emissionColor);
        if (emissionStrength <= 0.0f)
          continue;
        hasEmissivePrimitive = true;
        if (!_activePrimitive[primIndex])
          missingEmissivePrimitive = true;
      }

      if (!hasEmissivePrimitive || !missingEmissivePrimitive)
        continue;

      bool visible = false;
      if (objectIndex < _objectBounds.size())
        visible = isInView(_objectBounds[objectIndex]);
      if (objectIndex < _objectVisible.size())
        _objectVisible[objectIndex] = visible ? 1u : 0u;
      if (visibleOnly && !visible)
        continue;

      if (!forceAllToggles && usedBudget >= emissiveToggleBudget)
        return;

      size_t toggled = setObjectActive(objectIndex, true);
      if (toggled > 0) {
        changed = true;
        _frameTransportCriticalResidencyChanged = true;
        if (!forceAllToggles)
          usedBudget = std::min(usedBudget + toggled, emissiveToggleBudget);
      }
    }
  };

  promoteEmissivePass(true);
  if (usedBudget < emissiveToggleBudget)
    promoteEmissivePass(false);

  return changed;
}

bool Renderer::enforceTransportCriticalObjectResidency(bool forceAllToggles) {
  const size_t objectCount = _allSceneObjects.size();
  if (objectCount == 0 || _activePrimitive.empty())
    return false;

  size_t baseBudget = 0;
  switch (_frameStrategy) {
  case ResidencyStrategy::EnergyImportance:
  case ResidencyStrategy::UnifiedScore:
  case ResidencyStrategy::UnifiedNeural:
    baseBudget = _residencyConfig.energyMaxTogglesPerFrame;
    break;
  case ResidencyStrategy::ScreenSpaceFootprint:
    baseBudget = _residencyConfig.screenFootprintMaxTogglesPerFrame;
    break;
  case ResidencyStrategy::Probabilistic:
    baseBudget = _residencyConfig.probabilityMaxTogglesPerFrame;
    break;
  case ResidencyStrategy::PredictiveEnvironment:
  case ResidencyStrategy::EnvironmentHit:
    baseBudget = _residencyConfig.environmentMaxTogglesPerFrame;
    break;
  default:
    baseBudget = _residencyConfig.energyMaxTogglesPerFrame;
    break;
  }

  size_t toggleBudget = std::max<size_t>(12, baseBudget);
  if (forceAllToggles)
    toggleBudget = std::numeric_limits<size_t>::max();

  struct Candidate {
    size_t objectIndex = 0;
    float score = 0.0f;
    bool visible = false;
    bool emissive = false;
    bool stronglyNeeded = false;
  };

  std::vector<float> supportImportance(objectCount, 0.0f);
  std::vector<float> emissiveImportance(objectCount, 0.0f);
  std::vector<float> hitScore(objectCount, 0.0f);
  std::vector<float> exploration(objectCount, 0.0f);
  std::vector<float> nearWeight(objectCount, 0.0f);
  std::vector<float> visibilityCoverage(objectCount, 0.0f);
  std::vector<uint8_t> visibleMask(objectCount, 0u);

  float maxSupportImportance = 0.0f;
  float maxEmissiveImportance = 0.0f;
  float maxHitScore = 0.0f;
  float maxExploration = 0.0f;

  float screenArea = Camera::screenSize.x * Camera::screenSize.y;
  if (!(screenArea > 0.0f))
    screenArea = 1.0f;
  const simd::float3 forward = simd::normalize(Camera::forward);
  const float tanHalfFov =
      std::max(std::tan(Camera::verticalFov * static_cast<float>(M_PI) /
                            180.0f * 0.5f),
               1.0e-3f);

  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    const SceneObject &obj = _allSceneObjects[objectIndex];
    if (obj.primitiveCount == 0)
      continue;

    const size_t first = obj.firstPrimitive;
    const size_t last =
        std::min({first + obj.primitiveCount, _allPrimitives.size(),
                  _primitiveImportance.size()});
    for (size_t primIndex = first; primIndex < last; ++primIndex) {
      const Primitive &p = _allPrimitives[primIndex];
      const Material &m = p.material;
      supportImportance[objectIndex] +=
          std::max(_primitiveImportance[primIndex], 0.0f);
      emissiveImportance[objectIndex] +=
          std::max(m.emissionPower * luminance(m.emissionColor) * primitiveArea(p),
                   0.0f);
    }

    hitScore[objectIndex] = (objectIndex < _objectRayHitScore.size())
                                ? std::max(_objectRayHitScore[objectIndex], 0.0f)
                                : 0.0f;
    exploration[objectIndex] = (objectIndex < _objectExplorationScore.size())
                                   ? std::max(_objectExplorationScore[objectIndex],
                                              0.0f)
                                   : 0.0f;
    maxSupportImportance =
        std::max(maxSupportImportance, supportImportance[objectIndex]);
    maxEmissiveImportance =
        std::max(maxEmissiveImportance, emissiveImportance[objectIndex]);
    maxHitScore = std::max(maxHitScore, hitScore[objectIndex]);
    maxExploration = std::max(maxExploration, exploration[objectIndex]);

    if (objectIndex < _objectBounds.size()) {
      const BoundingSphere &bounds = _objectBounds[objectIndex];
      const simd::float3 toCenter = bounds.center - Camera::position;
      const float depth = std::max(simd::dot(toCenter, forward), 0.0f);
      const float distance = std::max(simd::length(toCenter), 1.0e-3f);
      visibleMask[objectIndex] = isInView(bounds) ? 1u : 0u;
      if (visibleMask[objectIndex] != 0 && depth > 1.0e-3f) {
        float radiusPixels =
            (bounds.radius / depth) / tanHalfFov * (Camera::screenSize.y * 0.5f);
        radiusPixels = std::max(radiusPixels, 0.0f);
        visibilityCoverage[objectIndex] =
            std::clamp((static_cast<float>(M_PI) * radiusPixels * radiusPixels) /
                           screenArea,
                       0.0f, 1.0f);
      }
      nearWeight[objectIndex] =
          std::clamp((bounds.radius * 8.0f) / (distance + bounds.radius * 8.0f),
                     0.0f, 1.0f);
    }
    if (objectIndex < _objectVisible.size())
      _objectVisible[objectIndex] = visibleMask[objectIndex];
  }

  auto normalizeByMax = [](float value, float maximum) {
    if (!(maximum > 0.0f))
      return 0.0f;
    return std::clamp(value / maximum, 0.0f, 1.0f);
  };

  std::vector<Candidate> candidates;
  candidates.reserve(objectCount);
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    const SceneObject &obj = _allSceneObjects[objectIndex];
    if (obj.primitiveCount == 0)
      continue;

    const bool active =
        (objectIndex < _objectActive.size()) ? _objectActive[objectIndex] : false;
    if (active)
      continue;

    const float hitProbability =
        (objectIndex < _objectHitProbability.size())
            ? std::clamp(_objectHitProbability[objectIndex], 0.0f, 1.0f)
            : 0.0f;
    const float supportNorm =
        normalizeByMax(supportImportance[objectIndex], maxSupportImportance);
    const float emissiveNorm =
        normalizeByMax(emissiveImportance[objectIndex], maxEmissiveImportance);
    const float hitScoreNorm = normalizeByMax(hitScore[objectIndex], maxHitScore);
    const float explorationNorm =
        normalizeByMax(exploration[objectIndex], maxExploration);
    const float coverage = visibilityCoverage[objectIndex];
    const float near = nearWeight[objectIndex];
    const bool visible = visibleMask[objectIndex] != 0;
    const bool emissive = emissiveImportance[objectIndex] > 0.0f;

    const float score = 2.6f * coverage + 1.8f * hitProbability +
                        1.25f * hitScoreNorm + 1.1f * near +
                        0.9f * supportNorm + 0.35f * explorationNorm +
                        (emissive ? 2.0f + 2.0f * emissiveNorm : 0.0f) +
                        (visible ? 1.0f : 0.0f);

    const bool stronglyNeeded =
        emissive || (visible && (coverage >= 0.0025f || hitProbability >= 0.10f ||
                                 supportNorm >= 0.25f)) ||
        (hitProbability >= 0.28f && near >= 0.45f) ||
        (hitScoreNorm >= 0.35f && near >= 0.40f) ||
        (supportNorm >= 0.55f && near >= 0.60f);

    if (!stronglyNeeded && score < 1.75f)
      continue;

    candidates.push_back({objectIndex, score, visible, emissive, stronglyNeeded});
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &a, const Candidate &b) {
              if (a.stronglyNeeded != b.stronglyNeeded)
                return a.stronglyNeeded > b.stronglyNeeded;
              if (a.emissive != b.emissive)
                return a.emissive > b.emissive;
              if (a.visible != b.visible)
                return a.visible > b.visible;
              if (a.score == b.score)
                return a.objectIndex < b.objectIndex;
              return a.score > b.score;
            });

  bool changed = false;
  size_t usedBudget = 0;
  for (const Candidate &candidate : candidates) {
    if (!forceAllToggles) {
      if (candidate.objectIndex < _objectCooldown.size() &&
          _objectCooldown[candidate.objectIndex] > 0)
        continue;
      if (usedBudget >= toggleBudget)
        break;
    }

    size_t toggled = setObjectActive(candidate.objectIndex, true);
    if (toggled > 0) {
      changed = true;
      _frameTransportCriticalResidencyChanged = true;
      if (!forceAllToggles)
        usedBudget = std::min(usedBudget + toggled, toggleBudget);
    }
  }

  return changed;
}

bool Renderer::enforceVisibleObjectResidency(bool forceAllToggles) {
  const size_t objectCount = _allSceneObjects.size();
  if (objectCount == 0)
    return false;

  size_t baseBudget = 0;
  switch (_frameStrategy) {
  case ResidencyStrategy::EnergyImportance:
  case ResidencyStrategy::UnifiedScore:
  case ResidencyStrategy::UnifiedNeural:
    baseBudget = _residencyConfig.energyMaxTogglesPerFrame;
    break;
  case ResidencyStrategy::ScreenSpaceFootprint:
    baseBudget = _residencyConfig.screenFootprintMaxTogglesPerFrame;
    break;
  case ResidencyStrategy::Probabilistic:
    baseBudget = _residencyConfig.probabilityMaxTogglesPerFrame;
    break;
  case ResidencyStrategy::PredictiveEnvironment:
  case ResidencyStrategy::EnvironmentHit:
    baseBudget = _residencyConfig.environmentMaxTogglesPerFrame;
    break;
  default:
    baseBudget = _residencyConfig.energyMaxTogglesPerFrame;
    break;
  }

  size_t visibleToggleBudget = std::max<size_t>(8, baseBudget / 2);
  if (forceAllToggles)
    visibleToggleBudget = std::numeric_limits<size_t>::max();

  bool changed = false;
  size_t usedBudget = 0;

  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    if (objectIndex >= _allSceneObjects.size())
      break;
    const SceneObject &obj = _allSceneObjects[objectIndex];
    if (obj.primitiveCount == 0)
      continue;

    bool currentlyActive =
        (objectIndex < _objectActive.size()) ? _objectActive[objectIndex] : false;
    if (currentlyActive)
      continue;

    bool visible = false;
    if (objectIndex < _objectBounds.size())
      visible = isInView(_objectBounds[objectIndex]);
    if (objectIndex < _objectVisible.size())
      _objectVisible[objectIndex] = visible ? 1u : 0u;
    if (!visible)
      continue;

    if (!forceAllToggles) {
      if (objectIndex < _objectCooldown.size() && _objectCooldown[objectIndex] > 0)
        continue;
      if (usedBudget >= visibleToggleBudget)
        break;
    }

    size_t toggled = setObjectActive(objectIndex, true);
    if (toggled > 0) {
      changed = true;
      if (!forceAllToggles)
        usedBudget = std::min(usedBudget + toggled, visibleToggleBudget);
    }
  }

  return changed;
}

bool Renderer::updateAlwaysResident(bool forceAllToggles) {
  size_t primitiveCount = _activePrimitive.size();
  size_t objectCount = _allSceneObjects.size();
  bool hasRecentChanges = !_recentlyActivated.empty() || !_recentlyDeactivated.empty();
  bool forceActivation = forceAllToggles || _forceAlwaysResidentActivation;
  const bool hasForcedObjectOff =
      _forcedObjectOffIndex != std::numeric_limits<size_t>::max();

  bool changed = false;

  if (forceActivation)
    _recentlyDeactivated.clear();

  if (forceActivation ||
      _alwaysResidentCache.needsUpdate(forceActivation, primitiveCount, objectCount,
                                       hasRecentChanges)) {
    for (size_t objectIndex = 0; objectIndex < _allSceneObjects.size();
         ++objectIndex) {
      size_t toggled = setObjectActive(objectIndex, true);
      if (toggled > 0)
        changed = true;
    }

    for (size_t objectIndex = 0; objectIndex < _allSceneObjects.size();
         ++objectIndex) {
      const SceneObject &obj = _allSceneObjects[objectIndex];
      const bool keepActive = !objectForcedOff(objectIndex);
      size_t first = obj.firstPrimitive;
      size_t last = first + obj.primitiveCount;
      for (size_t primIndex = first;
           primIndex < last && primIndex < _activePrimitive.size(); ++primIndex) {
        if (setPrimitiveActive(primIndex, keepActive))
          changed = true;
      }
    }

    _alwaysResidentCache.markUpdated(primitiveCount, objectCount);
    _forceAlwaysResidentActivation = false;
  }

  bool anyInactivePrimitive =
      std::any_of(_activePrimitive.begin(), _activePrimitive.end(),
                  [](bool active) { return !active; });
  bool anyInactiveObject = false;
  for (size_t objectIndex = 0; objectIndex < _allSceneObjects.size(); ++objectIndex) {
    if (objectForcedOff(objectIndex))
      continue;
    const SceneObject &obj = _allSceneObjects[objectIndex];
    if (obj.primitiveCount == 0)
      continue;
    bool active =
        (objectIndex < _objectActive.size()) ? _objectActive[objectIndex] : false;
    if (!active) {
      anyInactiveObject = true;
      break;
    }
  }

  if ((anyInactivePrimitive || anyInactiveObject || !_recentlyDeactivated.empty()) &&
      !hasForcedObjectOff) {
    printf("Always-resident strategy detected attempted eviction (%zu primitives pending).\n",
           _recentlyDeactivated.size());
  }

  assert((!anyInactivePrimitive || hasForcedObjectOff) &&
         "Always-resident strategy should keep all primitives active");
  assert((!anyInactiveObject || hasForcedObjectOff) &&
         "Always-resident strategy should keep populated objects active");
  assert((_recentlyDeactivated.empty() || hasForcedObjectOff) &&
         "Always-resident strategy should not queue primitive deactivations");

  return changed;
}

bool Renderer::alwaysResidentResidencyReady() {
  if (_pendingAlwaysResidentPrewarm)
    return false;
  if (!_pendingBlasBuilds.empty() || !_activeBlasBuilds.empty())
    return false;

  for (size_t objectIndex = 0; objectIndex < _allSceneObjects.size();
       ++objectIndex) {
    if (objectForcedOff(objectIndex))
      continue;
    const SceneObject &object = _allSceneObjects[objectIndex];
    if (object.primitiveCount == 0)
      continue;
    if (objectIndex >= _residentObjectGpuResources.size())
      return false;

    auto &resident = _residentObjectGpuResources[objectIndex];
    resident.clearPendingCommand();
    if (resident.hasPendingCommands())
      return false;
    if (!resident.isResident())
      return false;
    if (resident.triangleCount > 0 && !resident.geometryValid)
      return false;
  }

  return true;
}

bool Renderer::updateLODByDistance(bool forceAllToggles) {
  // Use hysteresis so objects do not flicker when hovering near the activation
  // boundary. Inactive objects only become active once the camera is closer
  // than LOD_ENTER_DISTANCE, while active objects stay active until the camera
  // has moved beyond LOD_EXIT_DISTANCE. Objects fully behind the camera are
  // treated as infinitely far away and immediately culled regardless of
  // distance thresholds. Additionally, objects must satisfy angular frustum
  // checks with separate entry/exit margins so camera rotations do not
  // immediately thrash residency state.

  size_t toggles = 0;
  bool changed = false;

  const float cameraMotion =
      clampUnit(static_cast<float>(std::min(_frameCameraMotionMetric, 1.0)));
  const double totalCapMB = effectiveTotalGpuMemoryCapMB();
  const double gpuMB = currentGPUMemoryMB();
  const float memoryPressure =
      (totalCapMB > 0.0)
          ? clampUnit(static_cast<float>((gpuMB / totalCapMB - 0.80) / 0.20))
          : 0.0f;
  const size_t baseLodToggleBudget =
      std::max<size_t>(_residencyConfig.lodMaxTogglesPerFrame, 1);
  const size_t adaptiveLodToggleBudget = std::max<size_t>(
      1, static_cast<size_t>(std::llround(
             static_cast<double>(baseLodToggleBudget) *
             (1.0 + 0.40 * static_cast<double>(cameraMotion) +
              0.60 * static_cast<double>(memoryPressure)))));

  const size_t objectCount = _allSceneObjects.size();
  std::vector<float> objectDistances(objectCount,
                                     std::numeric_limits<float>::max());
  std::vector<bool> objectBehind(objectCount, false);
  std::vector<bool> objectViewEnter(objectCount, true);
  std::vector<bool> objectViewExit(objectCount, true);
  std::vector<size_t> sortedIndices(objectCount);
  simd::float3 forward = Camera::forward;
  float forwardLenSq = simd::length_squared(forward);
  bool forwardValid = forwardLenSq >= 1e-6f;
  if (forwardValid)
    forward /= std::sqrt(forwardLenSq);
  simd::float3 up = Camera::up;
  float upLenSq = simd::length_squared(up);
  if (upLenSq < 1e-6f) {
    up = {0.0f, 1.0f, 0.0f};
    upLenSq = 1.0f;
  }
  up /= std::sqrt(upLenSq);

  simd::float3 right = simd::make_float3(1.0f, 0.0f, 0.0f);
  if (forwardValid) {
    right = simd::cross(forward, up);
    float rightLenSq = simd::length_squared(right);
    if (rightLenSq < 1e-6f) {
      right = {1.0f, 0.0f, 0.0f};
    } else {
      right /= std::sqrt(rightLenSq);
    }
  }

  if (forwardValid)
    up = simd::normalize(simd::cross(right, forward));

  float verticalHalfFov =
      Camera::verticalFov * static_cast<float>(M_PI) / 180.0f * 0.5f;
  if (verticalHalfFov <= 0.0f)
    verticalHalfFov = 1e-3f;
  float aspect = Camera::screenSize.y > 0.0f
                     ? Camera::screenSize.x / Camera::screenSize.y
                     : 1.0f;
  float horizontalHalfFov = std::atan(std::tan(verticalHalfFov) * aspect);

  constexpr float kDegToRad = static_cast<float>(M_PI) / 180.0f;
  float enterMargin =
      std::max(_residencyConfig.lodEnterViewDegrees, 0.0f) * kDegToRad;
  float exitMargin =
      std::max(_residencyConfig.lodExitViewDegrees, 0.0f) * kDegToRad;
  exitMargin = std::max(exitMargin, enterMargin);
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    const BoundingSphere &sphere =
        (objectIndex < _objectBounds.size())
            ? _objectBounds[objectIndex]
            : BoundingSphere{simd::make_float3(0.0f, 0.0f, 0.0f), 0.0f};
    simd::float3 toCenter = sphere.center - Camera::position;
    float distanceToCenter = simd::length(toCenter);
    float dist = distanceToCenter - sphere.radius;
    float forwardDepth = forwardValid ? simd::dot(toCenter, forward) : 0.0f;
    bool behind = forwardValid && forwardDepth + sphere.radius <= 0.0f;
    objectBehind[objectIndex] = behind;
    objectDistances[objectIndex] =
        behind ? std::numeric_limits<float>::max() : std::max(dist, 0.0f);
    sortedIndices[objectIndex] = objectIndex;

    if (!forwardValid) {
      objectViewEnter[objectIndex] = true;
      objectViewExit[objectIndex] = true;
      continue;
    }

    if (distanceToCenter <= 1e-5f || behind) {
      objectViewEnter[objectIndex] = !behind;
      objectViewExit[objectIndex] = !behind;
      continue;
    }

    float depth = std::max(forwardDepth, 1e-5f);
    float horiz = simd::dot(toCenter, right);
    float vert = simd::dot(toCenter, up);
    float horizontalAngle = std::atan2(std::fabs(horiz), depth);
    float verticalAngle = std::atan2(std::fabs(vert), depth);
    float radiusAngle = asinf(std::min(sphere.radius / distanceToCenter, 1.0f));

    float enterHorizontalLimit = horizontalHalfFov + radiusAngle + enterMargin;
    float exitHorizontalLimit = horizontalHalfFov + radiusAngle + exitMargin;
    float enterVerticalLimit = verticalHalfFov + radiusAngle + enterMargin;
    float exitVerticalLimit = verticalHalfFov + radiusAngle + exitMargin;

    objectViewEnter[objectIndex] =
        horizontalAngle <= enterHorizontalLimit &&
        verticalAngle <= enterVerticalLimit;
    objectViewExit[objectIndex] =
        horizontalAngle <= exitHorizontalLimit &&
        verticalAngle <= exitVerticalLimit;
  }

  std::sort(sortedIndices.begin(), sortedIndices.end(),
            [&objectDistances](size_t a, size_t b) {
              return objectDistances[a] < objectDistances[b];
            });

  for (size_t orderIndex = 0; orderIndex < objectCount; ++orderIndex) {
    size_t objectIndex = sortedIndices[orderIndex];
    float dist = (objectIndex < objectDistances.size())
                     ? objectDistances[objectIndex]
                     : 0.0f;

    bool currentlyActive =
        objectIndex < _objectActive.size() && _objectActive[objectIndex];
    bool behind = objectIndex < objectBehind.size() && objectBehind[objectIndex];
    bool viewAllowed = currentlyActive
                           ? (objectIndex < objectViewExit.size() &&
                              objectViewExit[objectIndex])
                           : (objectIndex < objectViewEnter.size() &&
                              objectViewEnter[objectIndex]);
    float enterThreshold = _residencyConfig.lodEnterDistance;
    float exitThreshold = _residencyConfig.lodExitDistance;
    if (_residencyConfig.enableDistanceEnvPrior) {
      float hitProbability =
          (objectIndex < _objectHitProbability.size())
              ? std::clamp(_objectHitProbability[objectIndex], 0.0f, 1.0f)
              : 0.0f;
      float priorScale = std::max(_residencyConfig.distancePriorScale, 0.0f);
      float bias = _residencyConfig.distancePriorFavorHighProbability ? -1.0f : 1.0f;
      float priorWeight = 1.0f + (hitProbability - 0.5f) * priorScale * bias;
      float maxPrior = _residencyConfig.distancePriorFavorHighProbability ? 4.0f : 1.0f;
      priorWeight = std::clamp(priorWeight, 0.25f, maxPrior);
      enterThreshold *= priorWeight;
      exitThreshold *= priorWeight;
    }
    bool shouldBeActive = viewAllowed && !behind &&
                          (currentlyActive ? (dist <= exitThreshold)
                                           : (dist < enterThreshold));
    bool canToggle =
        forceAllToggles || objectIndex >= _objectCooldown.size() ||
        _objectCooldown[objectIndex] == 0;
    if (!canToggle || shouldBeActive == currentlyActive)
      continue;

    const SceneObject &obj = _allSceneObjects[objectIndex];
    size_t togglesNeeded = 0;
    size_t first = obj.firstPrimitive;
    size_t last = first + obj.primitiveCount;
    for (size_t prim = first; prim < last && prim < _activePrimitive.size();
         ++prim) {
      if (_activePrimitive[prim] != shouldBeActive)
        ++togglesNeeded;
    }

    if (togglesNeeded == 0)
      continue;

    size_t toggleCost = forceAllToggles ? togglesNeeded : size_t(1);
    if (!forceAllToggles && toggles + toggleCost > adaptiveLodToggleBudget)
      continue;

    size_t toggled = setObjectActive(objectIndex, shouldBeActive);
    if (toggled > 0) {
      toggles += toggleCost;
      changed = true;
    }
  }

  size_t activeCount = 0;
  for (bool active : _activePrimitive)
    if (active)
      ++activeCount;

  if (activeCount == 0 && !_activePrimitive.empty()) {
    // Ensure at least one primitive remains visible to avoid a blank scene
    if (setPrimitiveActive(0, true))
      changed = true;
  }

  return changed;
}

size_t Renderer::setObjectActive(size_t objectIndex, bool active) {
  if (objectIndex >= _allSceneObjects.size())
    return 0;
  if (objectForcedOff(objectIndex))
    active = false;

  if (objectIndex >= _objectActive.size())
    _objectActive.resize(objectIndex + 1, false);
  if (objectIndex >= _objectCooldown.size())
    _objectCooldown.resize(objectIndex + 1, 0);
  if (objectIndex >= _objectLastToggleFrame.size())
    _objectLastToggleFrame.resize(objectIndex + 1, 0);
  if (objectIndex >= _objectActivePrimitiveCounts.size())
    _objectActivePrimitiveCounts.resize(objectIndex + 1, 0);
  if (objectIndex >= _desiredObjectState.size())
    _desiredObjectState.resize(objectIndex + 1, 0);
  if (objectIndex >= _pendingDesiredObjects.size())
    _pendingDesiredObjects.resize(objectIndex + 1, 0);
  if (objectIndex >= _desiredObjectPromotionFrame.size())
    _desiredObjectPromotionFrame.resize(objectIndex + 1, 0);
  if (objectIndex >= _desiredObjectDemotionFrame.size())
    _desiredObjectDemotionFrame.resize(objectIndex + 1, 0);

  bool prevState = _objectActive[objectIndex];

  const SceneObject &obj = _allSceneObjects[objectIndex];
  size_t toggled = 0;
  size_t first = obj.firstPrimitive;
  size_t last = first + obj.primitiveCount;
  for (size_t prim = first; prim < last && prim < _activePrimitive.size(); ++prim)
    if (setPrimitiveActive(prim, active))
      ++toggled;

  size_t activeCount =
      objectIndex < _objectActivePrimitiveCounts.size()
          ? _objectActivePrimitiveCounts[objectIndex]
          : size_t(0);
  bool newState = activeCount > 0;
  bool fullyInactive = activeCount == 0;

  _objectActive[objectIndex] = newState;
  if (objectIndex < _desiredObjectState.size()) {
    _desiredObjectState[objectIndex] = newState ? 1 : 0;
    if (newState && objectIndex < _desiredObjectPromotionFrame.size())
      _desiredObjectPromotionFrame[objectIndex] = _renderedFrameCount;
    if (!newState && objectIndex < _desiredObjectDemotionFrame.size())
      _desiredObjectDemotionFrame[objectIndex] = _renderedFrameCount;
  }
  if ((!newState || toggled > 0) && objectIndex < _pendingDesiredObjects.size())
    _pendingDesiredObjects[objectIndex] = 0;

  if (prevState != newState) {
    _objectLastToggleFrame[objectIndex] = _renderedFrameCount;
    if (newState)
      ++_frameObjectActivations;
    else
      ++_frameObjectDeactivations;
  }

  if (toggled > 0 || prevState != newState || fullyInactive)
    _objectCooldown[objectIndex] = _residencyConfig.stateCooldownFrames;

  if (toggled > 0 || prevState != newState || fullyInactive)
    _dirtyResidentObjects.push_back(objectIndex);

  return toggled;
}

std::vector<float> Renderer::computeUnifiedImportance(float &outTotalScore) {
  const size_t primCount = _activePrimitive.size();
  std::vector<float> unifiedScores(primCount, 0.0f);
  outTotalScore = 0.0f;
  if (primCount == 0)
    return unifiedScores;

  if (_primitiveScreenCoverage.size() != primCount)
    _primitiveScreenCoverage.assign(primCount, 0.0f);
  if (_primitiveDistanceFalloffCache.size() != primCount)
    _primitiveDistanceFalloffCache.assign(primCount, 0.0f);
  if (_primitiveCoverageDirty.size() != primCount)
    _primitiveCoverageDirty.assign(primCount, 1);
  if (_primitiveCoverageBoundsVersion.size() != primCount)
    _primitiveCoverageBoundsVersion.assign(primCount, 0);
  if (_primitiveCoverageVisibilityKey.size() != primCount)
    _primitiveCoverageVisibilityKey.assign(primCount, 0xFF);
  if (_primitiveUnifiedPrevVisible.size() != primCount)
    _primitiveUnifiedPrevVisible.assign(primCount, 0);
  if (_primitiveUnifiedLastVisibleHit.size() != primCount)
    _primitiveUnifiedLastVisibleHit.assign(primCount, 0.0f);
  if (_primitiveUnifiedLastVisibleEnergy.size() != primCount)
    _primitiveUnifiedLastVisibleEnergy.assign(primCount, 0.0f);

  float screenArea = Camera::screenSize.x * Camera::screenSize.y;
  if (screenArea <= 0.0f)
    screenArea = 1.0f;
  float halfFov = Camera::verticalFov * static_cast<float>(M_PI) / 180.0f * 0.5f;
  float tanHalfFov = std::tan(halfFov);
  if (tanHalfFov <= 0.0f)
    tanHalfFov = 1e-3f;

  simd::float3 forward = simd::normalize(Camera::forward);
  simd::float3 up = simd::normalize(Camera::up);
  simd::float3 right = simd::cross(forward, up);
  float rightLenSq = simd::length_squared(right);
  if (rightLenSq < 1e-6f)
    right = {1.0f, 0.0f, 0.0f};
  else
    right /= std::sqrt(rightLenSq);

  float aspect = Camera::screenSize.y > 0.0f
                     ? Camera::screenSize.x / Camera::screenSize.y
                     : 1.0f;
  float horizontalHalfFov = std::atan(tanHalfFov * aspect);

  bool cameraDirty = _coverageCameraVersion != _cameraVersion;
  if (cameraDirty) {
    _coverageCameraVersion = _cameraVersion;
    std::fill(_primitiveCoverageDirty.begin(), _primitiveCoverageDirty.end(), 1);

    // When the camera changes, decay cached hit information so the unified
    // score does not overly favor primitives that were visible from a previous
    // viewpoint.
    const float cameraHitDecay =
        std::clamp(_residencyConfig.cameraHitDecay, 0.0f, 1.0f);
    for (float &hit : _primitiveHitScores)
      hit *= cameraHitDecay;
  }

  auto boundsVersionForPrimitive = [&](size_t primIndex) -> uint64_t {
    if (primIndex < _primitiveBoundsVersion.size())
      return _primitiveBoundsVersion[primIndex];
    size_t objectIndex = primIndex < _primitiveToObject.size()
                             ? _primitiveToObject[primIndex]
                             : std::numeric_limits<size_t>::max();
    if (objectIndex < _objectBoundsVersion.size())
      return _objectBoundsVersion[objectIndex];
    return 0;
  };

  auto coverageForSphere = [&](const BoundingSphere &b, bool &visible) -> float {
    visible = isInView(b);
    if (!visible)
      return 0.0f;
    simd::float3 toCenter = b.center - Camera::position;
    float depth = simd::dot(toCenter, forward);
    if (depth <= 1e-3f)
      return 0.0f;
    float dist = simd::length(toCenter);
    float cosAngle = depth / std::max(dist, 1e-3f);
    float horiz = simd::dot(toCenter, right);
    float horizAngle = std::atan2(std::fabs(horiz), depth);
    if (horizAngle > horizontalHalfFov + 0.1f)
      return 0.0f;
    float radiusPixels = (b.radius / depth) / tanHalfFov *
                         (Camera::screenSize.y * 0.5f);
    radiusPixels = std::max(radiusPixels, 0.0f);
    float area = static_cast<float>(M_PI) * radiusPixels * radiusPixels;
    float angleFactor = std::max(cosAngle, 0.0f);
    return std::min(area * angleFactor, screenArea);
  };

  auto distanceFalloff = [&](const BoundingSphere &sphere,
                             bool sphereVisible) -> float {
    if (!sphereVisible && !isInView(sphere))
      return 0.0f;

    simd::float3 toCenter = sphere.center - Camera::position;
    float depth = simd::dot(toCenter, forward);
    if (depth <= 1e-3f)
      return 0.0f;

    float dist = simd::length(toCenter);
    float cosAngle = depth / std::max(dist, 1e-3f);
    float horiz = simd::dot(toCenter, right);
    float horizAngle = std::atan2(std::fabs(horiz), depth);
    if (horizAngle > horizontalHalfFov + 0.1f)
      return 0.0f;

    float angleFactor = std::max(cosAngle, 0.0f);
    return angleFactor / (1.0f + depth);
  };

  if (_primitiveHitScoresSnapshot.size() < primCount)
    _primitiveHitScoresSnapshot.resize(primCount, 0.0f);
  size_t copyCount = std::min(_primitiveHitScores.size(), primCount);
  std::copy_n(_primitiveHitScores.begin(), copyCount,
              _primitiveHitScoresSnapshot.begin());
  if (copyCount < primCount) {
    std::fill(_primitiveHitScoresSnapshot.begin() + copyCount,
              _primitiveHitScoresSnapshot.begin() + primCount, 0.0f);
  }

  const float alpha = _residencyConfig.unifiedEnergyWeight;
  const float beta = _residencyConfig.unifiedHitWeight;
  const float gamma = _residencyConfig.unifiedCoverageWeight;
  const float delta = _residencyConfig.unifiedDistanceWeight;
  const float offscreenDecay =
      std::clamp(_residencyConfig.unifiedOffscreenDecay, 0.0f, 1.0f);
  const float offscreenFloor =
      std::max(_residencyConfig.unifiedOffscreenFloor, 0.0f);
  const bool normalizeUnified = _residencyConfig.unifiedNormalize;
  constexpr float normalizationEpsilon = 1e-6f;

  std::vector<float> adjustedEnergy(primCount, 0.0f);
  std::vector<float> adjustedHit(primCount, 0.0f);
  std::vector<float> adjustedCoverage(primCount, 0.0f);
  std::vector<float> adjustedDistance(primCount, 0.0f);
  std::vector<uint8_t> skipScore(primCount, 0);

  float minEnergy = std::numeric_limits<float>::max();
  float maxEnergy = std::numeric_limits<float>::lowest();
  float minHit = std::numeric_limits<float>::max();
  float maxHit = std::numeric_limits<float>::lowest();
  float minCoverage = std::numeric_limits<float>::max();
  float maxCoverage = std::numeric_limits<float>::lowest();
  float minDistance = std::numeric_limits<float>::max();
  float maxDistance = std::numeric_limits<float>::lowest();

  auto updateMinMax = [](float value, float &minValue, float &maxValue) {
    minValue = std::min(minValue, value);
    maxValue = std::max(maxValue, value);
  };

  for (size_t i = 0; i < primCount; ++i) {
    uint64_t boundsVersion = boundsVersionForPrimitive(i);
    if (i < _primitiveCoverageBoundsVersion.size() &&
        _primitiveCoverageBoundsVersion[i] != boundsVersion)
      _primitiveCoverageDirty[i] = 1;

    uint8_t activeKey =
        (i < _activePrimitive.size() && _activePrimitive[i]) ? 0x2 : 0x0;
    bool dirty =
        (i < _primitiveCoverageDirty.size()) ? (_primitiveCoverageDirty[i] != 0)
                                             : true;
    if (i >= _primitiveCoverageVisibilityKey.size() ||
        ((_primitiveCoverageVisibilityKey[i] & 0x2) != activeKey))
      dirty = true;

    BoundingSphere chosenSphere{};
    bool haveSphere = false;
    if (i < _primitiveBounds.size()) {
      chosenSphere = _primitiveBounds[i];
      haveSphere = true;
    } else {
      size_t objectIndex = (i < _primitiveToObject.size())
                               ? _primitiveToObject[i]
                               : std::numeric_limits<size_t>::max();
      if (objectIndex < _objectBounds.size()) {
        chosenSphere = _objectBounds[objectIndex];
        haveSphere = true;
      }
    }

    bool visible = false;
    float coverage = 0.0f;
    float distanceScore = 0.0f;
    if (dirty) {
      if (i < _primitiveBounds.size())
        coverage = coverageForSphere(_primitiveBounds[i], visible);

      distanceScore = haveSphere ? distanceFalloff(chosenSphere, visible) : 0.0f;

      _primitiveScreenCoverage[i] = coverage;
      _primitiveDistanceFalloffCache[i] = distanceScore;
      if (i < _primitiveCoverageBoundsVersion.size())
        _primitiveCoverageBoundsVersion[i] = boundsVersion;
      if (i < _primitiveCoverageVisibilityKey.size())
        _primitiveCoverageVisibilityKey[i] = (visible ? 1 : 0) | activeKey;
      if (i < _primitiveCoverageDirty.size())
        _primitiveCoverageDirty[i] = 0;
    } else {
      coverage = _primitiveScreenCoverage[i];
      distanceScore = _primitiveDistanceFalloffCache[i];
      visible = (i < _primitiveCoverageVisibilityKey.size())
                    ? ((_primitiveCoverageVisibilityKey[i] & 0x1) != 0)
                    : false;
    }

    float energy = (i < _primitiveImportance.size()) ? _primitiveImportance[i]
                                                     : 0.0f;
    float hit = (i < _primitiveHitScoresSnapshot.size())
                    ? _primitiveHitScoresSnapshot[i]
                    : 0.0f;

    float lastVisibleHit =
        (i < _primitiveUnifiedLastVisibleHit.size())
            ? _primitiveUnifiedLastVisibleHit[i]
            : 0.0f;
    float lastVisibleEnergy =
        (i < _primitiveUnifiedLastVisibleEnergy.size())
            ? _primitiveUnifiedLastVisibleEnergy[i]
            : 0.0f;

    bool wasVisible = (i < _primitiveUnifiedPrevVisible.size())
                          ? (_primitiveUnifiedPrevVisible[i] != 0)
                          : false;

    if (hit == 0.0f && energy == 0.0f && coverage == 0.0f &&
        distanceScore == 0.0f && !visible) {
      _primitiveUnifiedPrevVisible[i] = visible ? 1 : 0;
      adjustedEnergy[i] = energy;
      adjustedHit[i] = hit;
      adjustedCoverage[i] = coverage;
      adjustedDistance[i] = distanceScore;
      skipScore[i] = 1;
      updateMinMax(adjustedEnergy[i], minEnergy, maxEnergy);
      updateMinMax(adjustedHit[i], minHit, maxHit);
      updateMinMax(adjustedCoverage[i], minCoverage, maxCoverage);
      updateMinMax(adjustedDistance[i], minDistance, maxDistance);
      continue;
    }

    bool offscreenNoFalloff =
        !visible && coverage == 0.0f && distanceScore == 0.0f;
    if (offscreenNoFalloff) {
      hit *= offscreenDecay;
      energy *= offscreenDecay;
      hit = std::max(hit, offscreenFloor);
      energy = std::max(energy, offscreenFloor);
    }

    bool becameVisible = visible && !wasVisible;
    if (becameVisible) {
      hit = std::max(hit, lastVisibleHit);
      energy = std::max(energy, lastVisibleEnergy);
      hit *= _residencyConfig.unifiedReentryBoost;
      energy *= _residencyConfig.unifiedReentryBoost;
    }

    if (visible && i < _primitiveUnifiedLastVisibleHit.size()) {
      _primitiveUnifiedLastVisibleHit[i] = hit;
      _primitiveUnifiedLastVisibleEnergy[i] = energy;
    }

    if (i < _primitiveUnifiedPrevVisible.size())
      _primitiveUnifiedPrevVisible[i] = visible ? 1 : 0;

    adjustedEnergy[i] = energy;
    adjustedHit[i] = hit;
    adjustedCoverage[i] = coverage;
    adjustedDistance[i] = distanceScore;
    updateMinMax(adjustedEnergy[i], minEnergy, maxEnergy);
    updateMinMax(adjustedHit[i], minHit, maxHit);
    updateMinMax(adjustedCoverage[i], minCoverage, maxCoverage);
    updateMinMax(adjustedDistance[i], minDistance, maxDistance);
  }

  auto normalizeValue = [&](float value, float minValue, float maxValue) {
    float range = maxValue - minValue;
    if (range <= normalizationEpsilon)
      return 0.0f;
    return (value - minValue) / (range + normalizationEpsilon);
  };

  for (size_t i = 0; i < primCount; ++i) {
    if (skipScore[i] != 0)
      continue;

    float energy = adjustedEnergy[i];
    float hit = adjustedHit[i];
    float coverage = adjustedCoverage[i];
    float distanceScore = adjustedDistance[i];

    if (normalizeUnified) {
      energy = normalizeValue(energy, minEnergy, maxEnergy);
      hit = normalizeValue(hit, minHit, maxHit);
      coverage = normalizeValue(coverage, minCoverage, maxCoverage);
      distanceScore = normalizeValue(distanceScore, minDistance, maxDistance);
    }

    float score = alpha * energy + beta * hit + gamma * coverage +
                  delta * distanceScore;
    unifiedScores[i] = score;
    outTotalScore += std::max(score, 0.0f);
  }

  return unifiedScores;
}

bool Renderer::updateEnergyImportance(bool forceAllToggles) {
  if (_activePrimitive.empty())
    return false;

  const float cameraMotion =
      clampUnit(static_cast<float>(std::min(_frameCameraMotionMetric, 1.0)));
  const double totalCapMB = effectiveTotalGpuMemoryCapMB();
  const double gpuMB = currentGPUMemoryMB();
  const float memoryPressure =
      (totalCapMB > 0.0)
          ? clampUnit(static_cast<float>((gpuMB / totalCapMB - 0.78) / 0.22))
          : 0.0f;
  const float adaptiveEnergyTargetFraction =
      std::clamp(_residencyConfig.energyTargetFraction +
                     0.08f * cameraMotion - 0.12f * memoryPressure,
                 0.65f, 0.97f);
  const size_t baseEnergyToggleBudget =
      std::max<size_t>(_residencyConfig.energyMaxTogglesPerFrame, 1);
  const size_t adaptiveEnergyToggleBudget = std::max<size_t>(
      1, static_cast<size_t>(std::llround(
             static_cast<double>(baseEnergyToggleBudget) *
             (1.0 + 0.55 * static_cast<double>(cameraMotion) +
              0.35 * static_cast<double>(memoryPressure)))));

  float recomputedTotalImportance = 0.0f;
  for (float importance : _primitiveImportance)
    recomputedTotalImportance += std::max(importance, 0.0f);
  _totalPrimitiveImportance = recomputedTotalImportance;

  const size_t objectCount = _allSceneObjects.size();
  if (objectCount == 0) {
    // Fall back to primitive-level logic if no objects are available.
    const size_t primCount = _activePrimitive.size();
    std::vector<int8_t> desiredState(primCount, -1);
    std::vector<size_t> sortedIndices(primCount);
    std::iota(sortedIndices.begin(), sortedIndices.end(), size_t(0));
    size_t minActive =
        std::min(primCount, _residencyConfig.energyMinActivePrimitives);
    size_t targetActive = static_cast<size_t>(std::ceil(
        static_cast<float>(primCount) * adaptiveEnergyTargetFraction));
    size_t sortCount =
        std::min(primCount, std::max(minActive, targetActive));
    if (sortCount > 0) {
      auto comparator = [this](size_t a, size_t b) {
        float scoreA = sanitizeSortValue(_primitiveImportance[a]);
        float scoreB = sanitizeSortValue(_primitiveImportance[b]);
        if (scoreA == scoreB)
          return a < b;
        return scoreA > scoreB;
      };
      std::partial_sort(sortedIndices.begin(),
                        sortedIndices.begin() + sortCount,
                        sortedIndices.end(), comparator);
    }

    if (_totalPrimitiveImportance <= 0.0f) {
      size_t enabledPrimitives = 0;
      for (size_t i = 0; i < sortCount; ++i) {
        size_t index = sortedIndices[i];
        desiredState[index] = 1;
        ++enabledPrimitives;
        if (enabledPrimitives >= minActive)
          break;
      }
    } else {
      float cumulative = 0.0f;
      float targetImportance =
          _totalPrimitiveImportance * adaptiveEnergyTargetFraction;
      size_t enabled = 0;
      for (size_t i = 0; i < sortCount; ++i) {
        size_t index = sortedIndices[i];
        desiredState[index] = 1;
        cumulative += std::max(_primitiveImportance[index], 0.0f);
        ++enabled;
        if (enabled >= minActive && cumulative >= targetImportance)
          break;
      }

      for (size_t i = 0; i < minActive && i < sortCount; ++i)
        desiredState[sortedIndices[i]] = 1;
    }

    bool changed = false;
    size_t toggles = 0;
    for (size_t i = 0; i < sortCount; ++i) {
      size_t primIndex = sortedIndices[i];
      int8_t desired = desiredState[primIndex];
      if (desired < 0)
        continue;
      bool shouldBeActive = desired != 0;
      if (shouldBeActive == _activePrimitive[primIndex])
        continue;
      if (!forceAllToggles) {
        if (primIndex < _primitiveCooldown.size() &&
            _primitiveCooldown[primIndex] > 0)
          continue;
        if (toggles >= adaptiveEnergyToggleBudget)
          break;
      }
      if (setPrimitiveActive(primIndex, shouldBeActive)) {
        changed = true;
        ++toggles;
      }
    }

    size_t activeCount = 0;
    for (bool active : _activePrimitive)
      if (active)
        ++activeCount;

    if (activeCount == 0 && !_activePrimitive.empty()) {
      size_t fallback = !sortedIndices.empty() ? sortedIndices.front() : size_t(0);
      if (setPrimitiveActive(fallback, true))
        changed = true;
    }

    return changed;
  }

  if (_objectImportance.size() != objectCount)
    _objectImportance.assign(objectCount, 0.0f);
  else
    std::fill(_objectImportance.begin(), _objectImportance.end(), 0.0f);

  bool resetImportanceHistory = false;
  if (_objectImportanceHistory.size() != objectCount) {
    _objectImportanceHistory.assign(objectCount, 0.0f);
    resetImportanceHistory = true;
  }

  if (_energySortedIndices.size() != objectCount)
    _energySortedIndices.resize(objectCount);
  std::iota(_energySortedIndices.begin(), _energySortedIndices.end(),
            size_t(0));

  float screenArea = Camera::screenSize.x * Camera::screenSize.y;
  if (screenArea <= 0.0f)
    screenArea = 1.0f;

  float visibilityBoost = std::max(_residencyConfig.energyVisibilityBoost, 1.0f);
  const bool applyVisibilityBoost = visibilityBoost > 1.0001f;

  if (applyVisibilityBoost)
    updatePrimitiveScreenCoverageForFrame();

  simd::float3 forward = simd::normalize(Camera::forward);
  simd::float3 up = simd::normalize(Camera::up);
  simd::float3 right = simd::cross(forward, up);
  float rightLenSq = simd::length_squared(right);
  if (rightLenSq < 1e-6f)
    right = {1.0f, 0.0f, 0.0f};
  else
    right /= std::sqrt(rightLenSq);

  float halfFov = Camera::verticalFov * static_cast<float>(M_PI) / 180.0f * 0.5f;
  float tanHalfFov = std::tan(halfFov);
  if (tanHalfFov <= 0.0f)
    tanHalfFov = 1e-3f;

  float aspect = Camera::screenSize.y > 0.0f
                     ? Camera::screenSize.x / Camera::screenSize.y
                     : 1.0f;
  float horizontalHalfFov = std::atan(tanHalfFov * aspect);

  auto coverageForSphere = [&](const BoundingSphere &b) -> float {
    if (!applyVisibilityBoost)
      return 0.0f;
    if (!isInView(b))
      return 0.0f;
    simd::float3 toCenter = b.center - Camera::position;
    float depth = simd::dot(toCenter, forward);
    if (depth <= 1e-3f)
      return 0.0f;
    float dist = simd::length(toCenter);
    float cosAngle = depth / std::max(dist, 1e-3f);
    float horiz = simd::dot(toCenter, right);
    float horizAngle = std::atan2(std::fabs(horiz), depth);
    if (horizAngle > horizontalHalfFov + 0.1f)
      return 0.0f;
    float radiusPixels = (b.radius / depth) / tanHalfFov *
                         (Camera::screenSize.y * 0.5f);
    radiusPixels = std::max(radiusPixels, 0.0f);
    float area = static_cast<float>(M_PI) * radiusPixels * radiusPixels;
    float angleFactor = std::max(cosAngle, 0.0f);
    return std::min(area * angleFactor, screenArea);
  };

  const auto &objectPrimitiveCounts = _objectPrimitiveCounts;
  bool anyMeshGroups = _anyMeshGroups;
  std::vector<float> objectTransportBoost(objectCount, 1.0f);
  std::vector<float> objectEmissiveImportance(objectCount, 0.0f);
  float maxObjectEmissiveImportance = 0.0f;
  float maxObjectRayHitScore = 0.0f;
  float boostedTotalImportance = 0.0f;
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    const SceneObject &obj = _allSceneObjects[objectIndex];
    size_t first = obj.firstPrimitive;
    size_t last = first + obj.primitiveCount;
    float totalImportance = 0.0f;
    for (size_t prim = first; prim < last && prim < _primitiveImportance.size();
         ++prim) {
      totalImportance += std::max(_primitiveImportance[prim], 0.0f);
      const Primitive &p = _allPrimitives[prim];
      const Material &m = p.material;
      objectEmissiveImportance[objectIndex] +=
          std::max(m.emissionPower * luminance(m.emissionColor) * primitiveArea(p),
                   0.0f);
    }
    maxObjectEmissiveImportance = std::max(maxObjectEmissiveImportance,
                                           objectEmissiveImportance[objectIndex]);
    float rayHitScore = (objectIndex < _objectRayHitScore.size())
                            ? std::max(_objectRayHitScore[objectIndex], 0.0f)
                            : 0.0f;
    maxObjectRayHitScore = std::max(maxObjectRayHitScore, rayHitScore);
    if (applyVisibilityBoost) {
      float sphereCoverage = 0.0f;
      bool visible = false;
      if (objectIndex < _objectBounds.size()) {
        sphereCoverage = coverageForSphere(_objectBounds[objectIndex]);
        visible = isInView(_objectBounds[objectIndex]);
      }
      if (objectIndex < _objectVisible.size())
        _objectVisible[objectIndex] = visible ? 1u : 0u;
      float coverageFraction =
          std::clamp(sphereCoverage / screenArea, 0.0f, 1.0f);
      float multiplier =
          1.0f + (visibilityBoost - 1.0f) * coverageFraction;
      _objectImportance[objectIndex] = totalImportance * multiplier;
    } else {
      _objectImportance[objectIndex] = totalImportance;
    }

    if (applyVisibilityBoost)
      boostedTotalImportance += std::max(_objectImportance[objectIndex], 0.0f);
  }

  auto normalizeByMax = [](float value, float maximum) {
    if (!(maximum > 0.0f))
      return 0.0f;
    return std::clamp(value / maximum, 0.0f, 1.0f);
  };

  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    float hitProbability =
        (objectIndex < _objectHitProbability.size())
            ? std::clamp(_objectHitProbability[objectIndex], 0.0f, 1.0f)
            : 0.0f;
    float rayHitNorm = normalizeByMax(
        (objectIndex < _objectRayHitScore.size())
            ? std::max(_objectRayHitScore[objectIndex], 0.0f)
            : 0.0f,
        maxObjectRayHitScore);
    float emissiveNorm =
        normalizeByMax(objectEmissiveImportance[objectIndex],
                       maxObjectEmissiveImportance);
    float depthWeight = (objectIndex < _objectDepthWeight.size())
                            ? std::max(_objectDepthWeight[objectIndex], 0.0f)
                            : 1.0f;
    float visibleWeight =
        (objectIndex < _objectVisible.size() && _objectVisible[objectIndex] != 0u)
            ? 1.0f
            : 0.0f;
    float transportBoost = 1.0f + 0.90f * hitProbability + 0.55f * rayHitNorm +
                           0.40f * depthWeight + 0.45f * visibleWeight +
                           0.90f * emissiveNorm;
    objectTransportBoost[objectIndex] = transportBoost;
    _objectImportance[objectIndex] *= transportBoost;
  }

  std::vector<float> energyImportance(objectCount, 0.0f);
  float historyWeight = std::clamp(_residencyConfig.energyImportanceSmoothing,
                                   0.0f, 0.999f);
  float currentWeight = 1.0f - historyWeight;
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    float current = _objectImportance[objectIndex];
    float smoothed = current;
    if (historyWeight > 0.0f) {
      float previous = resetImportanceHistory
                           ? current
                           : _objectImportanceHistory[objectIndex];
      smoothed = previous * historyWeight + current * currentWeight;
    }
    _objectImportanceHistory[objectIndex] = smoothed;
    energyImportance[objectIndex] = smoothed;
  }

  if (applyVisibilityBoost) {
    boostedTotalImportance = 0.0f;
    for (float importance : energyImportance)
      boostedTotalImportance += std::max(importance, 0.0f);
  }

  float targetImportanceBase = applyVisibilityBoost ? boostedTotalImportance
                                                    : _totalPrimitiveImportance;

  const bool useAverageImportance =
      _residencyConfig.energyUseAverageImportance && !anyMeshGroups;
  std::vector<float> energySelectionImportance(objectCount, 0.0f);
  float totalEnergySelectionImportance = 0.0f;
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    size_t count = objectIndex < objectPrimitiveCounts.size()
                       ? objectPrimitiveCounts[objectIndex]
                       : 0;
    float importance = energyImportance[objectIndex];
    if (useAverageImportance) {
      if (count == 0)
        continue;
      importance /= static_cast<float>(count);
    }
    energySelectionImportance[objectIndex] = importance;
    totalEnergySelectionImportance += std::max(importance, 0.0f);
  }

  std::sort(_energySortedIndices.begin(), _energySortedIndices.end(),
            [&](size_t a, size_t b) {
              float scoreA = (a < energySelectionImportance.size())
                                 ? sanitizeSortValue(energySelectionImportance[a])
                                 : 0.0f;
              float scoreB = (b < energySelectionImportance.size())
                                 ? sanitizeSortValue(energySelectionImportance[b])
                                 : 0.0f;
              if (scoreA == scoreB)
                return a < b;
              return scoreA > scoreB;
            });

  const size_t primCount = _activePrimitive.size();
  const size_t minActivePrimitives =
      std::min(primCount, _residencyConfig.energyMinActivePrimitives);

  if (_objectLastToggleFrame.size() < objectCount)
    _objectLastToggleFrame.resize(objectCount, 0);

  auto withinObjectStateCooldown = [&](size_t objectIndex) {
    if (forceAllToggles)
      return false;
    uint64_t minDuration =
        static_cast<uint64_t>(_residencyConfig.stateCooldownFrames);
    if (minDuration == 0)
      return false;
    if (objectIndex >= _objectLastToggleFrame.size())
      return false;
    uint64_t lastFrame = _objectLastToggleFrame[objectIndex];
    if (lastFrame == 0)
      return false;
    return (_renderedFrameCount >= lastFrame) &&
           (_renderedFrameCount - lastFrame < minDuration);
  };

  if (!anyMeshGroups) {
    std::vector<bool> desiredObjectState(objectCount, false);
    size_t primitivesEnabled = 0;
    float targetImportanceBaseSelection =
        totalEnergySelectionImportance > 0.0f
            ? totalEnergySelectionImportance
            : targetImportanceBase;

    if (_totalPrimitiveImportance <= 0.0f) {
      for (size_t idx : _energySortedIndices) {
        if (idx >= objectPrimitiveCounts.size())
          continue;
        size_t count = objectPrimitiveCounts[idx];
        if (count == 0)
          continue;
        desiredObjectState[idx] = true;
        primitivesEnabled += count;
        if (primitivesEnabled >= minActivePrimitives)
          break;
      }
    } else {
      float cumulativeImportance = 0.0f;
      float targetImportance =
          targetImportanceBaseSelection * adaptiveEnergyTargetFraction;
      for (size_t idx : _energySortedIndices) {
        if (idx >= objectPrimitiveCounts.size())
          continue;
        size_t count = objectPrimitiveCounts[idx];
        float importance = (idx < energySelectionImportance.size())
                               ? energySelectionImportance[idx]
                               : 0.0f;
        if (count == 0 && importance <= 0.0f)
          continue;
        desiredObjectState[idx] = true;
        primitivesEnabled += count;
        cumulativeImportance += std::max(importance, 0.0f);
        if (primitivesEnabled >= minActivePrimitives &&
            cumulativeImportance >= targetImportance)
          break;
      }
    }

    if (primitivesEnabled < minActivePrimitives) {
      for (size_t idx : _energySortedIndices) {
        if (idx >= objectPrimitiveCounts.size())
          continue;
        if (desiredObjectState[idx])
          continue;
        size_t count = objectPrimitiveCounts[idx];
        if (count == 0)
          continue;
        desiredObjectState[idx] = true;
        primitivesEnabled += count;
        if (primitivesEnabled >= minActivePrimitives)
          break;
      }
    }

    bool changed = false;
    size_t toggledPrimitiveCount = 0;
    auto attemptToggleObject = [&](size_t objectIndex,
                                   bool shouldBeActive) -> size_t {
      if (objectIndex >= _allSceneObjects.size())
        return 0;
      bool currentlyActive =
          objectIndex < _objectActive.size() && _objectActive[objectIndex];
      if (shouldBeActive == currentlyActive)
        return 0;

      if (!forceAllToggles) {
        if (objectIndex < _objectCooldown.size() &&
            _objectCooldown[objectIndex] > 0)
          return 0;
        if (withinObjectStateCooldown(objectIndex))
          return 0;
      }

      const SceneObject &obj = _allSceneObjects[objectIndex];
      size_t first = obj.firstPrimitive;
      size_t last = first + obj.primitiveCount;
      size_t togglesNeeded = 0;
      bool canToggle = true;
      for (size_t prim = first; prim < last && prim < _activePrimitive.size();
           ++prim) {
        if (_activePrimitive[prim] == shouldBeActive)
          continue;
        if (!forceAllToggles && prim < _primitiveCooldown.size() &&
            _primitiveCooldown[prim] > 0) {
          canToggle = false;
          break;
        }
        ++togglesNeeded;
      }

      if (!canToggle || togglesNeeded == 0)
        return 0;

      if (!forceAllToggles &&
          toggledPrimitiveCount >= adaptiveEnergyToggleBudget)
        return 0;

      size_t toggled = setObjectActive(objectIndex, shouldBeActive);
      if (toggled > 0 && !forceAllToggles) {
        toggledPrimitiveCount =
            std::min(toggledPrimitiveCount + toggled,
                     adaptiveEnergyToggleBudget);
      }
      return toggled;
    };

    for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
      bool shouldBeActive = desiredObjectState[objectIndex];
      size_t toggled = attemptToggleObject(objectIndex, shouldBeActive);
      if (toggled > 0) {
        changed = true;
      }
    }

    bool anyActiveObject = false;
    for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
      if (objectIndex >= _objectActive.size() ||
          objectIndex >= objectPrimitiveCounts.size())
        continue;
      if (objectPrimitiveCounts[objectIndex] == 0)
        continue;
      if (_objectActive[objectIndex]) {
        anyActiveObject = true;
        break;
      }
    }

    if (!anyActiveObject) {
      for (size_t idx : _energySortedIndices) {
        if (idx >= objectPrimitiveCounts.size())
          continue;
        if (objectPrimitiveCounts[idx] == 0)
          continue;
        if (attemptToggleObject(idx, true) > 0) {
          changed = true;
          anyActiveObject = true;
        }
        if (anyActiveObject)
          break;
      }
    }

    if (!anyActiveObject && !_activePrimitive.empty()) {
      size_t fallbackObject = std::numeric_limits<size_t>::max();
      for (size_t idx : _energySortedIndices) {
        if (idx >= objectPrimitiveCounts.size())
          continue;
        if (objectPrimitiveCounts[idx] == 0)
          continue;
        fallbackObject = idx;
        break;
      }

      if (fallbackObject < _allSceneObjects.size()) {
        if (attemptToggleObject(fallbackObject, true) > 0)
          changed = true;
      } else {
        if (setPrimitiveActive(0, true))
          changed = true;
      }
    }

    return changed;
  }

  struct MeshGroupAggregate {
    const MeshGroupInfo *info = nullptr;
    float importance = 0.0f;
    size_t primitiveCount = 0;
  };

  std::vector<MeshGroupAggregate> meshGroups;
  meshGroups.reserve(_meshGroups.size());
  for (const auto &info : _meshGroups) {
    MeshGroupAggregate aggregate;
    aggregate.info = &info;
    size_t fallbackPrimitiveCount = 0;
    for (size_t objectIndex : info.objectIndices) {
      if (objectIndex < energyImportance.size())
        aggregate.importance += energyImportance[objectIndex];
      if (objectIndex < objectPrimitiveCounts.size())
        fallbackPrimitiveCount += objectPrimitiveCounts[objectIndex];
    }
    if (info.primitiveCount > 0)
      aggregate.primitiveCount = info.primitiveCount;
    else
      aggregate.primitiveCount = fallbackPrimitiveCount;
    meshGroups.push_back(std::move(aggregate));
  }

  std::vector<float> meshGroupAverageImportance(meshGroups.size(), 0.0f);
  for (size_t groupIndex = 0; groupIndex < meshGroups.size(); ++groupIndex) {
    const MeshGroupAggregate &group = meshGroups[groupIndex];
    if (group.primitiveCount > 0) {
      meshGroupAverageImportance[groupIndex] =
          group.importance /
          static_cast<float>(group.primitiveCount);
    }
  }

  std::vector<size_t> meshSortedIndices(meshGroups.size());
  std::iota(meshSortedIndices.begin(), meshSortedIndices.end(), size_t(0));
  std::sort(meshSortedIndices.begin(), meshSortedIndices.end(),
            [&meshGroups](size_t a, size_t b) {
              float scoreA = (a < meshGroups.size())
                                 ? sanitizeSortValue(meshGroups[a].importance)
                                 : 0.0f;
              float scoreB = (b < meshGroups.size())
                                 ? sanitizeSortValue(meshGroups[b].importance)
                                 : 0.0f;
              if (scoreA == scoreB)
                return a < b;
              return scoreA > scoreB;
            });

  std::vector<bool> desiredGroupState(meshGroups.size(), false);
  size_t primitivesEnabled = 0;

  bool meshTargetSatisfied = false;
  float meshLastPrimaryAverage = 0.0f;
  float meshPrevPrimaryAverage = std::numeric_limits<float>::quiet_NaN();
  bool meshHasPrimaryAverage = false;
  size_t meshLastPrimarySortedPos = std::numeric_limits<size_t>::max();

  if (_totalPrimitiveImportance <= 0.0f) {
    for (size_t idx : meshSortedIndices) {
      if (idx >= meshGroups.size())
        continue;
      const MeshGroupAggregate &group = meshGroups[idx];
      if (group.primitiveCount == 0)
        continue;
      desiredGroupState[idx] = true;
      primitivesEnabled += group.primitiveCount;
      if (primitivesEnabled >= minActivePrimitives)
        break;
    }
  } else {
    float cumulativeImportance = 0.0f;
    float targetImportance =
        targetImportanceBase * adaptiveEnergyTargetFraction;
    for (size_t sortedPos = 0; sortedPos < meshSortedIndices.size();
         ++sortedPos) {
      size_t idx = meshSortedIndices[sortedPos];
      if (idx >= meshGroups.size())
        continue;
      const MeshGroupAggregate &group = meshGroups[idx];
      if (group.primitiveCount == 0 && group.importance <= 0.0f)
        continue;
      desiredGroupState[idx] = true;
      primitivesEnabled += group.primitiveCount;
      cumulativeImportance += std::max(group.importance, 0.0f);
      if (meshHasPrimaryAverage)
        meshPrevPrimaryAverage = meshLastPrimaryAverage;
      meshLastPrimaryAverage =
          (idx < meshGroupAverageImportance.size())
              ? meshGroupAverageImportance[idx]
              : 0.0f;
      meshHasPrimaryAverage = true;
      meshLastPrimarySortedPos = sortedPos;
      if (primitivesEnabled >= minActivePrimitives &&
          cumulativeImportance >= targetImportance) {
        meshTargetSatisfied = true;
        break;
      }
    }
  }

  if (primitivesEnabled < minActivePrimitives) {
    for (size_t idx : meshSortedIndices) {
      if (idx >= meshGroups.size())
        continue;
      const MeshGroupAggregate &group = meshGroups[idx];
      if (group.primitiveCount == 0)
        continue;
      if (desiredGroupState[idx])
        continue;
      desiredGroupState[idx] = true;
      primitivesEnabled += group.primitiveCount;
      if (primitivesEnabled >= minActivePrimitives)
        break;
    }
  }

  if (meshTargetSatisfied && meshHasPrimaryAverage &&
      meshLastPrimaryAverage > 0.0f) {
    float prevDiff = std::numeric_limits<float>::infinity();
    if (!std::isnan(meshPrevPrimaryAverage))
      prevDiff =
          std::fabs(meshLastPrimaryAverage - meshPrevPrimaryAverage);

    float nextDiff = std::numeric_limits<float>::infinity();
    size_t nextSortedPos = meshLastPrimarySortedPos + 1;
    if (nextSortedPos < meshSortedIndices.size()) {
      size_t nextIdx = meshSortedIndices[nextSortedPos];
      if (nextIdx < meshGroupAverageImportance.size()) {
        float nextAverage = meshGroupAverageImportance[nextIdx];
        if (std::isfinite(nextAverage))
          nextDiff = std::fabs(meshLastPrimaryAverage - nextAverage);
      }
    }

    float minNeighborDiff = std::min(prevDiff, nextDiff);
    float allowedRatio = 0.0f;
    if (std::isfinite(minNeighborDiff) && minNeighborDiff > 0.0f)
      allowedRatio = minNeighborDiff /
                     std::max(meshLastPrimaryAverage, std::numeric_limits<float>::min());

    const float kRelativeFallback = 0.01f;
    allowedRatio = std::max(allowedRatio, kRelativeFallback);

    float allowedDelta = meshLastPrimaryAverage * allowedRatio;
    const float kMaxAllowedFraction = 0.25f;
    if (std::isfinite(allowedDelta) && meshLastPrimaryAverage > 0.0f) {
      float maxAllowedDelta = meshLastPrimaryAverage * kMaxAllowedFraction;
      if (std::isfinite(maxAllowedDelta))
        allowedDelta = std::min(allowedDelta, maxAllowedDelta);
    }

    if (allowedDelta > 0.0f) {
      float epsilon = std::max(1e-5f, meshLastPrimaryAverage * 1e-3f);
      for (size_t groupIndex = 0; groupIndex < meshGroups.size();
           ++groupIndex) {
        if (desiredGroupState[groupIndex])
          continue;
        const MeshGroupAggregate &group = meshGroups[groupIndex];
        if (group.primitiveCount == 0)
          continue;
        float average = meshGroupAverageImportance[groupIndex];
        if (average <= 0.0f)
          continue;
        float difference = meshLastPrimaryAverage - average;
        if (difference <= allowedDelta + epsilon)
          desiredGroupState[groupIndex] = true;
      }
    }
  }

  std::vector<bool> desiredObjectState(objectCount, false);
  for (size_t groupIndex = 0; groupIndex < meshGroups.size(); ++groupIndex) {
    if (!desiredGroupState[groupIndex])
      continue;
    const MeshGroupAggregate &group = meshGroups[groupIndex];
    const auto *objectIndices =
        group.info ? &group.info->objectIndices : nullptr;
    if (!objectIndices)
      continue;
    for (size_t objectIndex : *objectIndices) {
      if (objectIndex < desiredObjectState.size())
        desiredObjectState[objectIndex] = true;
    }
  }

  bool changed = false;
  size_t toggledPrimitiveCount = 0;
  auto attemptToggleObject = [&](size_t objectIndex, bool shouldBeActive) {
    if (objectIndex >= _allSceneObjects.size())
      return size_t(0);
    bool currentlyActive =
        (objectIndex < _objectActive.size()) ? _objectActive[objectIndex] : false;
    if (currentlyActive == shouldBeActive)
      return size_t(0);

    if (!forceAllToggles) {
      if (objectIndex < _objectCooldown.size() &&
          _objectCooldown[objectIndex] > 0)
        return size_t(0);
      if (withinObjectStateCooldown(objectIndex))
        return size_t(0);
    }

    const SceneObject &obj = _allSceneObjects[objectIndex];
    size_t first = obj.firstPrimitive;
    size_t last = first + obj.primitiveCount;
    size_t togglesNeeded = 0;
    bool canToggle = true;
    for (size_t prim = first; prim < last && prim < _activePrimitive.size();
         ++prim) {
      if (_activePrimitive[prim] == shouldBeActive)
        continue;
      if (!forceAllToggles && prim < _primitiveCooldown.size() &&
          _primitiveCooldown[prim] > 0) {
        canToggle = false;
        break;
      }
      ++togglesNeeded;
    }

    if (!canToggle || togglesNeeded == 0)
      return size_t(0);
    if (!forceAllToggles &&
        toggledPrimitiveCount >= adaptiveEnergyToggleBudget)
      return size_t(0);

    size_t toggled = setObjectActive(objectIndex, shouldBeActive);
    if (toggled > 0 && !forceAllToggles) {
      toggledPrimitiveCount =
          std::min(toggledPrimitiveCount + toggled,
                   adaptiveEnergyToggleBudget);
    }
    return toggled;
  };

  for (size_t groupIndex = 0; groupIndex < meshGroups.size(); ++groupIndex) {
    const MeshGroupAggregate &group = meshGroups[groupIndex];
    bool groupShouldBeActive = desiredGroupState[groupIndex];
    size_t groupToggleCount = 0;
    bool groupNeedsToggle = false;
    bool groupCanToggle = true;
    const auto *objectIndices = group.info ? &group.info->objectIndices : nullptr;

    if (!objectIndices)
      continue;

    for (size_t objectIndex : *objectIndices) {
      bool shouldBeActive =
          (objectIndex < desiredObjectState.size())
              ? desiredObjectState[objectIndex]
              : groupShouldBeActive;
      bool currentlyActive =
          (objectIndex < _objectActive.size()) ? _objectActive[objectIndex]
                                               : false;
      if (shouldBeActive == currentlyActive)
        continue;
      if (!forceAllToggles) {
        if (objectIndex < _objectCooldown.size() &&
            _objectCooldown[objectIndex] > 0) {
          groupCanToggle = false;
          break;
        }
        if (withinObjectStateCooldown(objectIndex)) {
          groupCanToggle = false;
          break;
        }
      }
      groupNeedsToggle = true;

      const SceneObject &obj = _allSceneObjects[objectIndex];
      size_t first = obj.firstPrimitive;
      size_t last = first + obj.primitiveCount;
      for (size_t prim = first; prim < last && prim < _activePrimitive.size();
           ++prim) {
        if (_activePrimitive[prim] == shouldBeActive)
          continue;
        if (!forceAllToggles && prim < _primitiveCooldown.size() &&
            _primitiveCooldown[prim] > 0) {
          groupCanToggle = false;
          break;
        }
        ++groupToggleCount;
      }

      if (!groupCanToggle)
        break;
    }

    if (!groupNeedsToggle || !groupCanToggle)
      continue;

    if (!forceAllToggles &&
        toggledPrimitiveCount >= adaptiveEnergyToggleBudget)
      continue;

    size_t toggledThisGroup = 0;
    for (size_t objectIndex : *objectIndices) {
      bool shouldBeActive =
          (objectIndex < desiredObjectState.size())
              ? desiredObjectState[objectIndex]
              : groupShouldBeActive;
      toggledThisGroup += attemptToggleObject(objectIndex, shouldBeActive);
    }

    if (toggledThisGroup > 0) {
      changed = true;
    }
  }

  bool anyActiveObject = false;
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    if (objectIndex >= _objectActive.size() ||
        objectIndex >= objectPrimitiveCounts.size())
      continue;
    if (objectPrimitiveCounts[objectIndex] == 0)
      continue;
    if (_objectActive[objectIndex]) {
      anyActiveObject = true;
      break;
    }
  }

  if (!anyActiveObject) {
    for (size_t idx : meshSortedIndices) {
      if (idx >= meshGroups.size())
        continue;
      const MeshGroupAggregate &group = meshGroups[idx];
      if (group.primitiveCount == 0)
        continue;
      const auto *objectIndices = group.info ? &group.info->objectIndices : nullptr;
      if (!objectIndices)
        continue;
      size_t toggled = 0;
      for (size_t objectIndex : *objectIndices)
        toggled += attemptToggleObject(objectIndex, true);
      if (toggled > 0) {
        changed = true;
        anyActiveObject = true;
      }
      if (anyActiveObject)
        break;
    }
  }

  if (!anyActiveObject && !_activePrimitive.empty()) {
    size_t fallbackGroup = std::numeric_limits<size_t>::max();
    for (size_t idx : meshSortedIndices) {
      if (idx >= meshGroups.size())
        continue;
      if (meshGroups[idx].primitiveCount == 0)
        continue;
      fallbackGroup = idx;
      break;
    }

    if (fallbackGroup < meshGroups.size()) {
      size_t toggled = 0;
      const auto *objectIndices =
          meshGroups[fallbackGroup].info
              ? &meshGroups[fallbackGroup].info->objectIndices
              : nullptr;
      if (objectIndices) {
        for (size_t objectIndex : *objectIndices)
          toggled += attemptToggleObject(objectIndex, true);
      }
      if (toggled > 0)
        changed = true;
    } else {
      if (setPrimitiveActive(0, true))
        changed = true;
    }
  }

  return changed;
}

bool Renderer::updatePrimitiveScreenCoverageForFrame() {
  if (_activePrimitive.empty())
    return false;

  const size_t primCount = _activePrimitive.size();
  if (_coverageUpdatedFrame == _renderedFrameCount &&
      _primitiveScreenCoverage.size() == primCount)
    return false;

  if (_primitiveScreenCoverage.size() != primCount)
    _primitiveScreenCoverage.assign(primCount, 0.0f);

  float screenArea = Camera::screenSize.x * Camera::screenSize.y;
  if (screenArea <= 0.0f)
    screenArea = 1.0f;
  float halfFov = Camera::verticalFov * static_cast<float>(M_PI) / 180.0f * 0.5f;
  float tanHalfFov = std::tan(halfFov);
  if (tanHalfFov <= 0.0f)
    tanHalfFov = 1e-3f;

  simd::float3 forward = simd::normalize(Camera::forward);
  simd::float3 up = simd::normalize(Camera::up);
  simd::float3 right = simd::cross(forward, up);
  float rightLenSq = simd::length_squared(right);
  if (rightLenSq < 1e-6f)
    right = {1.0f, 0.0f, 0.0f};
  else
    right /= std::sqrt(rightLenSq);

  float aspect = Camera::screenSize.y > 0.0f
                     ? Camera::screenSize.x / Camera::screenSize.y
                     : 1.0f;
  float horizontalHalfFov = std::atan(tanHalfFov * aspect);

  const auto coverageForSphere =
      [this, screenArea, forward, right, horizontalHalfFov,
       tanHalfFov](const BoundingSphere &b) -> float {
    if (!isInView(b))
      return 0.0f;

    simd::float3 toCenter = b.center - Camera::position;
    float depth = simd::dot(toCenter, forward);
    if (depth <= 1e-3f)
      return 0.0f;

    float dist = simd::length(toCenter);
    float cosAngle = depth / std::max(dist, 1e-3f);
    float horiz = simd::dot(toCenter, right);
    float horizAngle = std::atan2(std::fabs(horiz), depth);
    if (horizAngle > horizontalHalfFov + 0.1f)
      return 0.0f;

    float radiusPixels =
        (b.radius / depth) / tanHalfFov * (Camera::screenSize.y * 0.5f);
    radiusPixels = std::max(radiusPixels, 0.0f);
    float area = static_cast<float>(M_PI) * radiusPixels * radiusPixels;
    float angleFactor = std::max(cosAngle, 0.0f);
    return std::min(area * angleFactor, screenArea);
  };

  const size_t objectCount = _allSceneObjects.size();
  std::vector<float> objectFallbackCoverage(objectCount, 0.0f);
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    if (objectIndex < _objectBounds.size())
      objectFallbackCoverage[objectIndex] =
          coverageForSphere(_objectBounds[objectIndex]);
  }

  parallelChunkedAsync(0, primCount,
                       [this, &objectFallbackCoverage,
                        &coverageForSphere](size_t chunkBegin, size_t chunkEnd) {
                         for (size_t i = chunkBegin; i < chunkEnd; ++i) {
                           float coverage = 0.0f;
                           if (i < _primitiveBounds.size()) {
                             coverage = coverageForSphere(_primitiveBounds[i]);
                           } else {
                             size_t objectIndex =
                                 (i < _primitiveToObject.size())
                                     ? _primitiveToObject[i]
                                     : std::numeric_limits<size_t>::max();
                             if (objectIndex < objectFallbackCoverage.size())
                               coverage = objectFallbackCoverage[objectIndex];
                           }
                           _primitiveScreenCoverage[i] = coverage;
                         }
                       });

  _coverageUpdatedFrame = _renderedFrameCount;
  return true;
}

bool Renderer::updateUnifiedResidency(bool forceAllToggles) {
  if (_activePrimitive.empty())
    return false;

  float totalUnifiedScore = 0.0f;
  std::vector<float> unifiedScores = computeUnifiedImportance(totalUnifiedScore);
  if (unifiedScores.empty())
    return false;

  const float cameraMotion =
      clampUnit(static_cast<float>(std::min(_frameCameraMotionMetric, 1.0)));
  const double totalCapMB = effectiveTotalGpuMemoryCapMB();
  const double gpuMB = currentGPUMemoryMB();
  const float memoryPressure =
      (totalCapMB > 0.0)
          ? clampUnit(static_cast<float>((gpuMB / totalCapMB - 0.79) / 0.21))
          : 0.0f;
  const float adaptiveUnifiedTargetFraction =
      std::clamp(_residencyConfig.energyTargetFraction + 0.06f * cameraMotion -
                     0.12f * memoryPressure,
                 0.55f, 0.96f);
  const size_t baseUnifiedToggleBudget =
      std::max<size_t>(_residencyConfig.energyMaxTogglesPerFrame, 1);
  const size_t adaptiveUnifiedToggleBudget = std::max<size_t>(
      1, static_cast<size_t>(std::llround(
             static_cast<double>(baseUnifiedToggleBudget) *
             (1.0 + 0.45 * static_cast<double>(cameraMotion) +
              0.40 * static_cast<double>(memoryPressure)))));

  const size_t primCount = _activePrimitive.size();
  const size_t minActivePrimitives =
      std::min(primCount, _residencyConfig.energyMinActivePrimitives);
  size_t targetResidentPrimitives = static_cast<size_t>(std::ceil(
      static_cast<double>(primCount) *
      static_cast<double>(adaptiveUnifiedTargetFraction)));
  targetResidentPrimitives =
      std::max(targetResidentPrimitives, minActivePrimitives);
  targetResidentPrimitives = std::min(targetResidentPrimitives, primCount);

  const size_t objectCount = _allSceneObjects.size();
  if (objectCount == 0) {
    std::vector<size_t> sortedIndices(primCount);
    std::iota(sortedIndices.begin(), sortedIndices.end(), size_t(0));
    std::sort(sortedIndices.begin(), sortedIndices.end(),
              [&unifiedScores](size_t a, size_t b) {
                float scoreA = (a < unifiedScores.size())
                                   ? sanitizeSortValue(unifiedScores[a])
                                   : 0.0f;
                float scoreB = (b < unifiedScores.size())
                                   ? sanitizeSortValue(unifiedScores[b])
                                   : 0.0f;
                if (scoreA == scoreB)
                  return a < b;
                return scoreA > scoreB;
              });

    std::vector<bool> desired(primCount, false);
    for (size_t i = 0; i < targetResidentPrimitives && i < sortedIndices.size();
         ++i)
      desired[sortedIndices[i]] = true;

    bool changed = false;
    size_t toggles = 0;
    for (size_t primIndex = 0; primIndex < primCount; ++primIndex) {
      bool shouldBeActive = desired[primIndex];
      if (_activePrimitive[primIndex] == shouldBeActive)
        continue;
      if (!forceAllToggles) {
        if (primIndex < _primitiveCooldown.size() &&
            _primitiveCooldown[primIndex] > 0)
          continue;
        if (toggles >= adaptiveUnifiedToggleBudget)
          break;
      }
      if (setPrimitiveActive(primIndex, shouldBeActive)) {
        changed = true;
        ++toggles;
      }
    }
    return changed;
  }

  if (_objectLastToggleFrame.size() < objectCount)
    _objectLastToggleFrame.resize(objectCount, 0);

  auto primitiveCountForObject = [&](size_t objectIndex) -> size_t {
    if (objectIndex < _objectPrimitiveCounts.size() &&
        _objectPrimitiveCounts[objectIndex] > 0)
      return _objectPrimitiveCounts[objectIndex];
    if (objectIndex < _allSceneObjects.size())
      return _allSceneObjects[objectIndex].primitiveCount;
    return 0;
  };

  auto withinObjectStateCooldown = [&](size_t objectIndex) {
    if (forceAllToggles)
      return false;
    uint64_t minDuration =
        static_cast<uint64_t>(_residencyConfig.stateCooldownFrames);
    if (minDuration == 0)
      return false;
    if (objectIndex >= _objectLastToggleFrame.size())
      return false;
    uint64_t lastFrame = _objectLastToggleFrame[objectIndex];
    if (lastFrame == 0)
      return false;
    return (_renderedFrameCount >= lastFrame) &&
           (_renderedFrameCount - lastFrame < minDuration);
  };

  struct UnifiedCandidate {
    size_t objectIndex = 0;
    float utility = 0.0f;
    float heuristicUtility = 0.0f;
    float heuristicNorm = 0.0f;
    float neuralImpact = 0.0f;
    float neuralNorm = 0.0f;
    float teacherRaw = 0.0f;
    float teacherNorm = 0.0f;
    float teacherMapped = 0.0f;
    float adjustedCost = 1.0f;
    float efficiency = 0.0f;
    size_t primitiveCount = 0;
  };

  std::vector<UnifiedCandidate> candidates;
  candidates.reserve(objectCount);
  const bool wantUnifiedNeural =
      _frameStrategy == ResidencyStrategy::UnifiedNeural &&
      _unifiedNeuralModel.valid;
  const bool residualNeuralMode =
      wantUnifiedNeural && _unifiedNeuralModel.residualTeacher.enabled;
  bool useUnifiedNeural = wantUnifiedNeural;
  std::vector<float> unifiedNeuralFeatures;
  UnifiedNeuralRuntimeSignals unifiedNeuralSignals;

  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    const SceneObject &obj = _allSceneObjects[objectIndex];
    const size_t primitiveCount = primitiveCountForObject(objectIndex);
    if (primitiveCount == 0)
      continue;

    const size_t first = obj.firstPrimitive;
    const size_t last = std::min(first + obj.primitiveCount, unifiedScores.size());
    float utility = 0.0f;
    for (size_t prim = first; prim < last; ++prim)
      utility += std::max(unifiedScores[prim], 0.0f);

    bool currentlyActive =
        objectIndex < _objectActive.size() && _objectActive[objectIndex];
    float adjustedCost = static_cast<float>(std::max<size_t>(primitiveCount, 1));
    if (objectIndex < _residentObjectGpuResources.size()) {
      const auto state = _residentObjectGpuResources[objectIndex].state;
      if (state == ResidentObjectGpuResources::ResidencyState::Cold)
        adjustedCost *= 1.20f;
      else if (state == ResidentObjectGpuResources::ResidencyState::Streaming)
        adjustedCost *= 1.10f;
    }
    if (currentlyActive)
      adjustedCost *= 0.92f;

    float efficiency = utility / std::max(adjustedCost, 1.0e-5f);
    float neuralImpact = 0.0f;
    float teacherRaw = 0.0f;
    if (useUnifiedNeural) {
      if (!computeUnifiedNeuralRuntimeSignals(objectIndex, primitiveCount,
                                             unifiedNeuralSignals) ||
          !buildUnifiedNeuralFeatureVector(objectIndex, primitiveCount,
                                           unifiedNeuralFeatures)) {
        useUnifiedNeural = false;
      } else {
        neuralImpact = predictUnifiedNeuralImpact(unifiedNeuralFeatures);
        if (residualNeuralMode)
          teacherRaw = computeUnifiedNeuralTeacherRawScore(unifiedNeuralSignals);
      }
    }
    candidates.push_back({objectIndex, utility, utility, 0.0f, neuralImpact,
                          0.0f, teacherRaw, 0.0f, 0.0f,
                          adjustedCost, efficiency, primitiveCount});
  }

  if (wantUnifiedNeural && !useUnifiedNeural) {
    std::printf(
        "[Renderer] UnifiedNeural runtime features were unavailable; falling "
        "back to heuristic unified scoring for this scene.\n");
  }

  if (useUnifiedNeural && !candidates.empty()) {
    std::string unifiedNeuralMode = _residencyConfig.unifiedNeuralMode;
    std::transform(unifiedNeuralMode.begin(), unifiedNeuralMode.end(),
                   unifiedNeuralMode.begin(), [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    const bool neuralOnlyMode =
        unifiedNeuralMode == "neural_only" || unifiedNeuralMode == "neural-only" ||
        unifiedNeuralMode == "neuralonly" || unifiedNeuralMode == "neural";
    const float blendWeight = neuralOnlyMode
                                  ? 1.0f
                                  : std::clamp(
                                        _residencyConfig.unifiedNeuralBlendWeight,
                                        0.0f, 1.0f);
    float minHeuristic = std::numeric_limits<float>::max();
    float maxHeuristic = std::numeric_limits<float>::lowest();
    float minNeural = std::numeric_limits<float>::max();
    float maxNeural = std::numeric_limits<float>::lowest();
    float minTeacher = std::numeric_limits<float>::max();
    float maxTeacher = std::numeric_limits<float>::lowest();
    for (const UnifiedCandidate &candidate : candidates) {
      minHeuristic = std::min(minHeuristic, candidate.heuristicUtility);
      maxHeuristic = std::max(maxHeuristic, candidate.heuristicUtility);
      minTeacher = std::min(minTeacher, candidate.teacherRaw);
      maxTeacher = std::max(maxTeacher, candidate.teacherRaw);
    }

    auto normalizeValue = [](float value, float minValue, float maxValue) {
      const float range = maxValue - minValue;
      if (!(range > 1.0e-6f))
        return value > 0.0f ? 1.0f : 0.0f;
      return std::clamp((value - minValue) / range, 0.0f, 1.0f);
    };

    for (UnifiedCandidate &candidate : candidates) {
      candidate.heuristicNorm =
          normalizeValue(candidate.heuristicUtility, minHeuristic, maxHeuristic);
      candidate.teacherNorm =
          normalizeValue(candidate.teacherRaw, minTeacher, maxTeacher);
      if (residualNeuralMode) {
        candidate.teacherMapped =
            candidate.teacherNorm * _unifiedNeuralModel.residualTeacher.scale +
            _unifiedNeuralModel.residualTeacher.bias;
        candidate.neuralImpact = candidate.teacherMapped + candidate.neuralImpact;
      }
      minNeural = std::min(minNeural, candidate.neuralImpact);
      maxNeural = std::max(maxNeural, candidate.neuralImpact);
    }

    for (UnifiedCandidate &candidate : candidates) {
      const float neuralNorm =
          normalizeValue(candidate.neuralImpact, minNeural, maxNeural);
      candidate.neuralNorm = neuralNorm;
      candidate.utility =
          neuralOnlyMode
              ? neuralNorm
              : lerpFloat(candidate.heuristicNorm, neuralNorm, blendWeight);
      candidate.efficiency =
          candidate.utility / std::max(candidate.adjustedCost, 1.0e-5f);
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const UnifiedCandidate &a, const UnifiedCandidate &b) {
              float effA = sanitizeSortValue(a.efficiency);
              float effB = sanitizeSortValue(b.efficiency);
              if (effA != effB)
                return effA > effB;
              float utilityA = sanitizeSortValue(a.utility);
              float utilityB = sanitizeSortValue(b.utility);
              if (utilityA != utilityB)
                return utilityA > utilityB;
              return a.objectIndex < b.objectIndex;
            });

  std::vector<bool> desiredObjectState(objectCount, false);
  size_t selectedPrimitiveCount = 0;
  for (const UnifiedCandidate &candidate : candidates) {
    if (selectedPrimitiveCount >= targetResidentPrimitives)
      break;
    desiredObjectState[candidate.objectIndex] = true;
    selectedPrimitiveCount += candidate.primitiveCount;
  }

  if (selectedPrimitiveCount < minActivePrimitives) {
    for (const UnifiedCandidate &candidate : candidates) {
      if (desiredObjectState[candidate.objectIndex])
        continue;
      desiredObjectState[candidate.objectIndex] = true;
      selectedPrimitiveCount += candidate.primitiveCount;
      if (selectedPrimitiveCount >= minActivePrimitives)
        break;
    }
  }

  if (_unifiedNeuralScoreLoggingEnabled) {
    ensureUnifiedNeuralScoreStream();
    if (_unifiedNeuralScoreStream.is_open()) {
      auto residencyStateName = [&](size_t objectIndex) -> const char * {
        if (objectIndex >= _residentObjectGpuResources.size())
          return "unknown";
        switch (_residentObjectGpuResources[objectIndex].state) {
        case ResidentObjectGpuResources::ResidencyState::Cold:
          return "cold";
        case ResidentObjectGpuResources::ResidencyState::Streaming:
          return "streaming";
        case ResidentObjectGpuResources::ResidencyState::Resident:
          return "resident";
        }
        return "unknown";
      };

      const std::string strategyName = residencyStrategyName(_frameStrategy);
      const std::string sceneName =
          _sceneVariantName.empty() ? std::string("unknown_scene")
                                    : _sceneVariantName;
      const float blendWeight =
          useUnifiedNeural
              ? std::clamp(_residencyConfig.unifiedNeuralBlendWeight, 0.0f, 1.0f)
              : 0.0f;

      for (const UnifiedCandidate &candidate : candidates) {
        const size_t objectIndex = candidate.objectIndex;
        const bool currentlyActive =
            objectIndex < _objectActive.size() && _objectActive[objectIndex];
        const bool desiredActive =
            objectIndex < desiredObjectState.size() && desiredObjectState[objectIndex];
        bool visible =
            objectIndex < _objectVisible.size() && _objectVisible[objectIndex] != 0u;
        if (!visible && objectIndex < _objectBounds.size())
          visible = isInView(_objectBounds[objectIndex]);

        std::ostringstream row;
        appendCsvEscaped(row, sceneName);
        row << ',' << _renderedFrameCount << ',';
        appendCsvEscaped(row, strategyName);
        row << ',' << static_cast<int>(_frameStrategy) << ','
            << objectIndex << ',' << candidate.primitiveCount << ','
            << (currentlyActive ? 1 : 0) << ',' << (desiredActive ? 1 : 0) << ',';
        appendCsvEscaped(row, residencyStateName(objectIndex));
        row << ',' << (visible ? 1 : 0) << ','
            << formatFixed(candidate.adjustedCost, 6) << ','
            << formatFixed(candidate.heuristicUtility, 6) << ','
            << formatFixed(candidate.heuristicNorm, 6) << ','
            << formatFixed(candidate.neuralImpact, 6) << ','
            << formatFixed(candidate.neuralNorm, 6) << ','
            << formatFixed(blendWeight, 6) << ','
            << formatFixed(candidate.utility, 6) << ','
            << formatFixed(candidate.efficiency, 6);
        _unifiedNeuralScoreStream << row.str() << '\n';
      }
      _unifiedNeuralScoreStream.flush();
    }
  }

  bool changed = false;
  size_t toggledPrimitiveCount = 0;
  auto attemptToggleObject = [&](size_t objectIndex,
                                 bool shouldBeActive) -> size_t {
    if (objectIndex >= _allSceneObjects.size())
      return 0;

    bool currentlyActive =
        objectIndex < _objectActive.size() && _objectActive[objectIndex];
    if (currentlyActive == shouldBeActive)
      return 0;

    if (!forceAllToggles) {
      if (objectIndex < _objectCooldown.size() &&
          _objectCooldown[objectIndex] > 0)
        return 0;
      if (withinObjectStateCooldown(objectIndex))
        return 0;
      if (toggledPrimitiveCount >= adaptiveUnifiedToggleBudget)
        return 0;
    }

    const SceneObject &obj = _allSceneObjects[objectIndex];
    const size_t first = obj.firstPrimitive;
    const size_t last = first + obj.primitiveCount;
    size_t togglesNeeded = 0;
    for (size_t prim = first; prim < last && prim < _activePrimitive.size();
         ++prim) {
      if (_activePrimitive[prim] == shouldBeActive)
        continue;
      if (!forceAllToggles && prim < _primitiveCooldown.size() &&
          _primitiveCooldown[prim] > 0)
        return 0;
      ++togglesNeeded;
    }

    if (togglesNeeded == 0)
      return 0;

    size_t toggled = setObjectActive(objectIndex, shouldBeActive);
    if (toggled > 0 && !forceAllToggles) {
      toggledPrimitiveCount =
          std::min(toggledPrimitiveCount + toggled, adaptiveUnifiedToggleBudget);
    }
    return toggled;
  };

  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    if (primitiveCountForObject(objectIndex) == 0)
      continue;
    if (attemptToggleObject(objectIndex, desiredObjectState[objectIndex]) > 0)
      changed = true;
  }

  bool anyActiveObject = false;
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    if (primitiveCountForObject(objectIndex) == 0)
      continue;
    if (objectIndex < _objectActive.size() && _objectActive[objectIndex]) {
      anyActiveObject = true;
      break;
    }
  }

  if (!anyActiveObject) {
    for (const UnifiedCandidate &candidate : candidates) {
      if (attemptToggleObject(candidate.objectIndex, true) > 0) {
        changed = true;
        anyActiveObject = true;
        break;
      }
    }
  }

  if (!anyActiveObject && !_activePrimitive.empty()) {
    if (setPrimitiveActive(0, true))
      changed = true;
  }

  return changed;
}

bool Renderer::updateRayHitBudget(bool forceAllToggles) {
  if (_activePrimitive.empty())
    return false;

  const bool aggressiveEvict = _residencyConfig.rayHitAggressiveEvict;
  if (aggressiveEvict && !_rayHitAggressiveLogged) {
    std::printf(
        "[Renderer] Ray-hit aggressive eviction enabled; forcing residency toggles "
        "and waits may stall GPU work.\n");
    _rayHitAggressiveLogged = true;
  }
  const bool forceRayHitToggles = forceAllToggles || aggressiveEvict;

  const float cameraMotion =
      clampUnit(static_cast<float>(std::min(_frameCameraMotionMetric, 1.0)));
  const double totalCapMB = effectiveTotalGpuMemoryCapMB();
  const double gpuMB = currentGPUMemoryMB();
  const float memoryPressure =
      (totalCapMB > 0.0)
          ? clampUnit(static_cast<float>((gpuMB / totalCapMB - 0.80) / 0.20))
          : 0.0f;
  const float adaptiveRayHitTargetFraction =
      std::clamp(_residencyConfig.rayHitTargetFraction + 0.10f * cameraMotion -
                     0.24f * memoryPressure,
                 0.34f, 0.92f);
  const size_t baseRayHitToggleBudget =
      std::max<size_t>(_residencyConfig.rayHitMaxTogglesPerFrame, 1);
  const size_t adaptiveRayHitToggleBudget = std::max<size_t>(
      1, static_cast<size_t>(std::llround(
             static_cast<double>(baseRayHitToggleBudget) *
             (1.0 + 0.45 * static_cast<double>(cameraMotion) +
              0.60 * static_cast<double>(memoryPressure)))));

  if (_rayHitSortedIndices.size() != _activePrimitive.size()) {
    _rayHitSortedIndices.resize(_activePrimitive.size());
    std::iota(_rayHitSortedIndices.begin(), _rayHitSortedIndices.end(), size_t(0));
  }
  if (_primitiveHitScores.size() < _activePrimitive.size())
    _primitiveHitScores.resize(_activePrimitive.size(), 0.0f);

  if (_rayHitRebuildCooldown > 0) {
    if (forceRayHitToggles)
      _rayHitRebuildCooldown = 0;
    else {
      --_rayHitRebuildCooldown;
      return false;
    }
  }

  const size_t primCount = _activePrimitive.size();
  if (_primitiveHitScoresSnapshot.size() < primCount)
    _primitiveHitScoresSnapshot.resize(primCount, 0.0f);

  size_t copyCount = std::min(_primitiveHitScores.size(), primCount);
  std::copy_n(_primitiveHitScores.begin(), copyCount,
              _primitiveHitScoresSnapshot.begin());
  if (copyCount < primCount) {
    std::fill(_primitiveHitScoresSnapshot.begin() + copyCount,
              _primitiveHitScoresSnapshot.begin() + primCount, 0.0f);
  }

  if (_primitiveVisible.size() < primCount)
    _primitiveVisible.resize(primCount, 0);
  if (_primitiveHitLastFrame.size() < primCount)
    _primitiveHitLastFrame.resize(primCount, 0);

  auto &hitScores = _primitiveHitScoresSnapshot;
  parallelChunkedAsync(0, primCount, [&](size_t chunkBegin, size_t chunkEnd) {
    for (size_t i = chunkBegin; i < chunkEnd; ++i) {
      bool visible = false;
      if (i < _primitiveBounds.size())
        visible = isInView(_primitiveBounds[i]);
      _primitiveVisible[i] = visible ? 1 : 0;

      float score = (i < hitScores.size()) ? hitScores[i] : 0.0f;
      uint32_t hitsLast =
          (i < _primitiveHitLastFrame.size()) ? _primitiveHitLastFrame[i] : 0;

      if (!visible) {
        if (hitsLast == 0)
          score = 0.0f;
        else
          score *= lerpFloat(0.5f, 0.2f, memoryPressure);
      } else if (score <= 0.0f) {
        score = 1.0f;
      }

      if (_residencyConfig.enableRayHitPrior) {
        float probability = 0.5f;
        if (i < _primitiveToObject.size()) {
          size_t objectIndex = _primitiveToObject[i];
          if (objectIndex < _objectHitProbability.size())
            probability =
                std::clamp(_objectHitProbability[objectIndex], 0.0f, 1.0f);
        } else if (i < _primitiveHitProbability.size()) {
          probability =
              std::clamp(_primitiveHitProbability[i], 0.0f, 1.0f);
        }

        float priorScale = std::max(_residencyConfig.rayHitPriorScale, 0.0f);
        float bias = _residencyConfig.rayHitPriorFavorHighProbability ? -1.0f
                                                                       : 1.0f;
        float priorWeight = 1.0f + (probability - 0.5f) * priorScale * bias;
        float maxPrior =
            _residencyConfig.rayHitPriorFavorHighProbability ? 4.0f : 1.0f;
        priorWeight = std::clamp(priorWeight, 0.25f, maxPrior);
        score *= priorWeight;
      }

      if (i < hitScores.size())
        hitScores[i] = score;
    }
  });

  const auto &adjustedScores = hitScores;

  const size_t minActive =
      std::min(primCount, _residencyConfig.rayHitMinActivePrimitives);
  size_t targetActive =
      static_cast<size_t>(std::ceil(primCount * adaptiveRayHitTargetFraction));
  targetActive = std::max(targetActive, minActive);
  targetActive = std::min(targetActive, primCount);

  size_t sortCount = std::max(minActive, targetActive);
  sortCount = std::max<size_t>(sortCount, 1);
  sortCount = std::min(sortCount, primCount);

  if (sortCount > 0) {
    std::partial_sort(_rayHitSortedIndices.begin(),
                      _rayHitSortedIndices.begin() + sortCount,
                      _rayHitSortedIndices.end(),
                      [&adjustedScores](size_t a, size_t b) {
                        float rawA =
                            (a < adjustedScores.size()) ? adjustedScores[a] : 0.0f;
                        float rawB =
                            (b < adjustedScores.size()) ? adjustedScores[b] : 0.0f;
                        float scoreA = sanitizeSortValue(rawA);
                        float scoreB = sanitizeSortValue(rawB);
                        if (scoreA == scoreB)
                          return a < b;
                        return scoreA > scoreB;
                      });
  }

  std::vector<bool> desired(primCount, false);
  size_t enabled = 0;
  for (size_t idx : _rayHitSortedIndices) {
    if (enabled >= targetActive)
      break;
    desired[idx] = true;
    ++enabled;
  }

  const size_t visibilityReserve = std::min(
      primCount,
      static_cast<size_t>(std::ceil(primCount * (0.02f + 0.06f * cameraMotion))));
  size_t visiblePromotions = 0;
  for (size_t idx : _rayHitSortedIndices) {
    if (visiblePromotions >= visibilityReserve)
      break;
    bool visible = (idx < _primitiveVisible.size()) && (_primitiveVisible[idx] != 0);
    if (!visible)
      continue;
    if (!desired[idx]) {
      desired[idx] = true;
      ++enabled;
    }
    ++visiblePromotions;
  }

  for (size_t i = 0; i < minActive && i < _rayHitSortedIndices.size(); ++i)
    desired[_rayHitSortedIndices[i]] = true;

  // Enforce cooldowns and toggle budgets so prior-driven demotions cannot
  // churn primitives faster than configured.
  size_t toggles = 0;
  bool changed = false;
  for (size_t i = 0; i < primCount; ++i) {
    bool shouldBeActive = desired[i];
    if (shouldBeActive == _activePrimitive[i])
      continue;
    if (!forceRayHitToggles) {
      if (i < _primitiveCooldown.size() && _primitiveCooldown[i] > 0)
        continue;
      if (toggles >= adaptiveRayHitToggleBudget)
        break;
    }
    if (setPrimitiveActive(i, shouldBeActive)) {
      ++toggles;
      changed = true;
    }
  }

  size_t activeCount = 0;
  for (bool active : _activePrimitive)
    if (active)
      ++activeCount;

  if (activeCount == 0 && !_activePrimitive.empty()) {
    size_t fallback = !_rayHitSortedIndices.empty() ? _rayHitSortedIndices.front()
                                                    : size_t(0);
    if (setPrimitiveActive(fallback, true))
      changed = true;
  }

  if (changed)
    _rayHitRebuildCooldown = _residencyConfig.rayHitRebuildCooldownFrames;

  return changed;
}

bool Renderer::updateProbabilisticResidency(bool forceAllToggles) {
  if (_activePrimitive.empty())
    return false;

  const float cameraMotion =
      clampUnit(static_cast<float>(std::min(_frameCameraMotionMetric, 1.0)));
  const double totalCapMB = effectiveTotalGpuMemoryCapMB();
  const double gpuMB = currentGPUMemoryMB();
  const float memoryPressure =
      (totalCapMB > 0.0)
          ? clampUnit(static_cast<float>((gpuMB / totalCapMB - 0.80) / 0.20))
          : 0.0f;
  const float adaptiveProbabilityTargetFraction = std::clamp(
      _residencyConfig.probabilityTargetFraction + 0.08f * cameraMotion -
          0.12f * memoryPressure,
      0.45f, 0.93f);
  const size_t baseProbabilityToggleBudget =
      std::max<size_t>(_residencyConfig.probabilityMaxTogglesPerFrame, 1);
  const size_t adaptiveProbabilityToggleBudget = std::max<size_t>(
      1, static_cast<size_t>(std::llround(
             static_cast<double>(baseProbabilityToggleBudget) *
             (1.0 + 0.40 * static_cast<double>(cameraMotion) +
              0.50 * static_cast<double>(memoryPressure)))));

  _frameProbabilityTargetPrimitives = 0;
  _frameProbabilityInitialDesiredPrimitives = 0;
  _frameProbabilityFinalDesiredPrimitives = 0;
  _frameProbabilityTrimmedPrimitives = 0;
  _frameProbabilityBudgetHit = false;

  const size_t primCount = _activePrimitive.size();
  const size_t objectCount = _allSceneObjects.size();

  constexpr float kPosteriorFloor = 1.0e-3f;
  constexpr float kMinimalEvidenceThreshold = 1.0e-3f;
  constexpr float kVisibilityEvidenceDecay = 0.9f;
  const float configuredWindow = _residencyConfig.probabilityEvidenceWindow;
  const bool finiteEvidenceWindow =
      configuredWindow > 0.0f && std::isfinite(configuredWindow);
  float probabilityVisibleFloor =
      std::clamp(_residencyConfig.probabilityVisibleFloor, 0.0f, 1.0f);
  float scoringWindow = finiteEvidenceWindow
                            ? std::max(configuredWindow, kPosteriorFloor)
                            : std::max(64.0f, kPosteriorFloor);
  auto computeEvidenceFactor = [&](float mass, bool visible = true) {
    if (!(mass > 0.0f) || !std::isfinite(mass))
      return 0.0f;
    if (!visible)
      return 0.0f;
    if (finiteEvidenceWindow) {
      float normalized = mass / scoringWindow;
      return std::clamp(normalized, 0.0f, 1.0f);
    }
    float normalized = mass / (mass + scoringWindow);
    return std::clamp(normalized, 0.0f, 1.0f);
  };
  auto sanitizePosteriorProbability = [](float probability) {
    if (!std::isfinite(probability))
      return 0.5f;
    return std::clamp(probability, 0.0f, 1.0f);
  };
  auto sanitizePosteriorVariance = [](float variance) {
    if (!std::isfinite(variance) || variance < 0.0f)
      return 0.0f;
    return variance;
  };
  auto computeRegressedProbabilityFromEvidence =
      [&](float probability, float evidence) {
        float clampedEvidence = std::clamp(evidence, 0.0f, 1.0f);
        float sanitized = sanitizePosteriorProbability(probability);
        return sanitized * clampedEvidence + 0.5f * (1.0f - clampedEvidence);
      };
  auto computeRegressedProbability = [&](float probability, float mass,
                                         bool visible = true) {
    float evidence = computeEvidenceFactor(mass, visible);
    return computeRegressedProbabilityFromEvidence(probability, evidence);
  };
  auto computePosteriorScoreFromEvidence =
      [&](float probability, float variance, float evidence) {
        float regressed =
            computeRegressedProbabilityFromEvidence(probability, evidence);
        float sqrtVariance =
            std::sqrt(std::max(sanitizePosteriorVariance(variance), 0.0f));
        return regressed +
               _residencyConfig.probabilityUncertaintyBoost * sqrtVariance;
      };
  auto computePosteriorScore = [&](float probability, float variance, float mass,
                                   bool visible = true) {
    float evidence = computeEvidenceFactor(mass, visible);
    return computePosteriorScoreFromEvidence(probability, variance, evidence);
  };

  if (objectCount == 0) {
    _desiredObjectState.clear();
    _pendingDesiredObjects.clear();
    _desiredObjectPromotionFrame.clear();
    _desiredObjectDemotionFrame.clear();
    _objectVisibilityEvidence.clear();
    if (_primitiveHitProbability.size() < primCount)
      _primitiveHitProbability.resize(primCount, 0.5f);
    if (_probabilitySortedIndices.size() != primCount) {
      _probabilitySortedIndices.resize(primCount);
      std::iota(_probabilitySortedIndices.begin(), _probabilitySortedIndices.end(),
                size_t(0));
    }

    if (_primitiveVisible.size() < primCount)
      _primitiveVisible.resize(primCount, 0);
    if (_primitiveExplorationScore.size() < primCount)
      _primitiveExplorationScore.resize(primCount, 0.0f);

    std::vector<uint8_t> primitiveBecameVisible(primCount, 0);
    std::vector<bool> desired(primCount, false);
    float threshold = _residencyConfig.probabilityThreshold;
    for (size_t i = 0; i < primCount; ++i) {
      float probability = (i < _primitiveHitProbability.size())
                              ? _primitiveHitProbability[i]
                              : 0.5f;
      float mass = (i < _primitivePosteriorMass.size())
                       ? _primitivePosteriorMass[i]
                       : 0.0f;
      float effectiveProbability = computeRegressedProbability(probability, mass);
      if (effectiveProbability >= threshold)
        desired[i] = true;
    }

    size_t minActive =
        std::min(_residencyConfig.probabilityMinActivePrimitives, primCount);
    size_t partialCount = std::max<size_t>(
        1, std::min(primCount, std::max(minActive, size_t(1))));
    auto comparator = [this, &computePosteriorScore](size_t a, size_t b) {
      float probA = (a < _primitiveHitProbability.size())
                        ? _primitiveHitProbability[a]
                        : 0.5f;
      float varA = (a < _primitiveHitVariance.size())
                       ? _primitiveHitVariance[a]
                       : 0.0f;
      float massA = (a < _primitivePosteriorMass.size())
                        ? _primitivePosteriorMass[a]
                        : 0.0f;
      float scoreA = sanitizeSortValue(computePosteriorScore(probA, varA, massA));
      float probB = (b < _primitiveHitProbability.size())
                        ? _primitiveHitProbability[b]
                        : 0.5f;
      float varB = (b < _primitiveHitVariance.size())
                       ? _primitiveHitVariance[b]
                       : 0.0f;
      float massB = (b < _primitivePosteriorMass.size())
                        ? _primitivePosteriorMass[b]
                        : 0.0f;
      float scoreB = sanitizeSortValue(computePosteriorScore(probB, varB, massB));
      if (scoreA == scoreB)
        return a < b;
      return scoreA > scoreB;
    };
    std::partial_sort(_probabilitySortedIndices.begin(),
                      _probabilitySortedIndices.begin() + partialCount,
                      _probabilitySortedIndices.end(), comparator);

    auto computeVisibility = [this, &primitiveBecameVisible](size_t idx) {
      bool previousVisible =
          (idx < _primitiveVisible.size()) ? (_primitiveVisible[idx] != 0) : false;
      bool visible = false;
      if (idx < _primitiveBounds.size()) {
        visible = isInView(_primitiveBounds[idx]);
        if (idx < _primitiveVisible.size())
          _primitiveVisible[idx] = visible ? 1 : 0;
      } else if (idx < _primitiveVisible.size()) {
        visible = _primitiveVisible[idx] != 0;
      }
      if (visible && !previousVisible && idx < primitiveBecameVisible.size())
        primitiveBecameVisible[idx] = 1;
      return visible;
    };

    size_t desiredCount = 0;
    for (bool flag : desired)
      if (flag)
        ++desiredCount;

    std::vector<size_t> visibleExplore;
    std::vector<size_t> hiddenExplore;
    visibleExplore.reserve(primCount);
    hiddenExplore.reserve(primCount);

    for (size_t idx : _probabilitySortedIndices) {
      if (idx >= primCount)
        continue;
      if (idx < _activePrimitive.size() && _activePrimitive[idx])
        continue;
      if (desired[idx])
        continue;
      float probability = (idx < _primitiveHitProbability.size())
                              ? _primitiveHitProbability[idx]
                              : 0.5f;
      float mass = (idx < _primitivePosteriorMass.size())
                       ? _primitivePosteriorMass[idx]
                       : 0.0f;
      bool visible = computeVisibility(idx);
      float evidence = computeEvidenceFactor(mass, visible);
      bool lowEvidence = evidence <= kMinimalEvidenceThreshold;
      float effectiveProbability = computeRegressedProbabilityFromEvidence(
          probability, evidence);
      bool visibilityBootstrap = (idx < primitiveBecameVisible.size()) &&
                                 primitiveBecameVisible[idx] != 0 &&
                                 mass <= kMinimalEvidenceThreshold;
      float visibilityAdjustedProbability =
          (visible && lowEvidence)
              ? std::min(effectiveProbability, threshold)
              : effectiveProbability;
      float gatedProbability =
          visibilityBootstrap
              ? std::min(visibilityAdjustedProbability, threshold)
              : visibilityAdjustedProbability;
      if (gatedProbability > threshold)
        continue;
      float exploreScore =
          (idx < _primitiveExplorationScore.size())
              ? _primitiveExplorationScore[idx]
              : 0.0f;
      uint32_t raysTested =
          (idx < _primitiveRaysTestedLastFrame.size())
              ? _primitiveRaysTestedLastFrame[idx]
              : 0u;
      float effectiveExplore = exploreScore;
      if (visibilityBootstrap)
        effectiveExplore = std::max(effectiveExplore, kIdleVisibleExploreSeed);
      if (visible && (raysTested == 0 || lowEvidence))
        effectiveExplore = std::max(effectiveExplore, kIdleVisibleExploreSeed);
      if (visible && raysTested > 0 && !lowEvidence)
        effectiveExplore = std::max(effectiveExplore, kPosteriorFloor);
      if (idx < _primitiveExplorationScore.size())
        _primitiveExplorationScore[idx] = effectiveExplore;
      if (effectiveExplore <= 0.0f)
        continue;
      if (visible)
        visibleExplore.push_back(idx);
      else
        hiddenExplore.push_back(idx);
    }

    auto explorationComparator = [this](size_t a, size_t b) {
      float exploreA = (a < _primitiveExplorationScore.size())
                           ? sanitizeSortValue(_primitiveExplorationScore[a])
                           : -std::numeric_limits<float>::max();
      float exploreB = (b < _primitiveExplorationScore.size())
                           ? sanitizeSortValue(_primitiveExplorationScore[b])
                           : -std::numeric_limits<float>::max();
      if (exploreA == exploreB) {
        uint32_t raysA =
            (a < _primitiveRaysTestedLastFrame.size())
                ? _primitiveRaysTestedLastFrame[a]
                : 0u;
        uint32_t raysB =
            (b < _primitiveRaysTestedLastFrame.size())
                ? _primitiveRaysTestedLastFrame[b]
                : 0u;
        if (raysA == raysB)
          return a < b;
        return raysA > raysB;
      }
      return exploreA > exploreB;
    };

    std::sort(visibleExplore.begin(), visibleExplore.end(),
              explorationComparator);
    std::sort(hiddenExplore.begin(), hiddenExplore.end(),
              explorationComparator);

    auto promote = [&](const std::vector<size_t> &candidates, size_t &slots) {
      if (slots == 0)
        return;
      for (size_t idx : candidates) {
        if (slots == 0)
          break;
        if (idx >= primCount)
          continue;
        if (desired[idx])
          continue;
        desired[idx] = true;
        ++desiredCount;
        if (slots > 0)
          --slots;
        bool wasVisible =
            (idx < _primitiveVisible.size()) ? (_primitiveVisible[idx] != 0)
                                             : false;
        bool idle =
            (idx < _primitiveRaysTestedLastFrame.size())
                ? (_primitiveRaysTestedLastFrame[idx] == 0)
                : true;
        if (wasVisible && idle && idx < _primitiveExplorationScore.size() &&
            _primitiveExplorationScore[idx] < kIdleVisibleExploreSeed)
          _primitiveExplorationScore[idx] = kIdleVisibleExploreSeed;
      }
    };

    auto countRemaining = [&](const std::vector<size_t> &candidates) {
      size_t remaining = 0;
      for (size_t idx : candidates) {
        if (idx < primCount && !desired[idx])
          ++remaining;
      }
      return remaining;
    };

    if (desiredCount < minActive) {
      size_t slots =
          std::min(minActive - desiredCount, primCount - desiredCount);
      promote(visibleExplore, slots);
      promote(hiddenExplore, slots);
    }

    if (desiredCount < minActive) {
      for (size_t idx : _probabilitySortedIndices) {
        if (desiredCount >= minActive)
          break;
        if (idx >= primCount)
          continue;
        if (desired[idx])
          continue;
        desired[idx] = true;
        ++desiredCount;
      }
    }

    size_t remainingExplore = countRemaining(visibleExplore) +
                              countRemaining(hiddenExplore);
    if (remainingExplore > 0 && desiredCount < primCount) {
      size_t slots =
          std::min({std::max<size_t>(size_t(1), minActive / 2), remainingExplore,
                    primCount - desiredCount});
      promote(visibleExplore, slots);
      promote(hiddenExplore, slots);
    }

    size_t fallbackCandidate = primCount;
    if (!visibleExplore.empty())
      fallbackCandidate = visibleExplore.front();
    else if (!hiddenExplore.empty())
      fallbackCandidate = hiddenExplore.front();
    else if (!_probabilitySortedIndices.empty())
      fallbackCandidate = _probabilitySortedIndices.front();

    size_t toggles = 0;
    bool changed = false;
    size_t maxToggles = adaptiveProbabilityToggleBudget;
#ifndef NDEBUG
    assert(_probabilitySortedIndices.size() == primCount);
#endif
    auto tryToggle = [&](size_t idx) {
      if (idx >= primCount)
        return false;
      bool shouldBeActive = desired[idx];
      if (shouldBeActive == _activePrimitive[idx])
        return false;
      if (!forceAllToggles) {
        if (idx < _primitiveCooldown.size() && _primitiveCooldown[idx] > 0)
          return false;
        if (toggles >= maxToggles)
          return true;
      }
      if (setPrimitiveActive(idx, shouldBeActive)) {
        ++toggles;
        ++_frameProbabilisticToggles;
        changed = true;
        if (!shouldBeActive && idx < _primitiveExplorationScore.size())
          _primitiveExplorationScore[idx] = 0.0f;
      }
      return false;
    };

    auto walkList = [&](const std::vector<size_t> &candidates) {
      for (size_t idx : candidates) {
        if (!forceAllToggles && toggles >= maxToggles)
          return true;
        if (tryToggle(idx))
          return true;
      }
      return false;
    };

    // Walk probability-sorted primitives first so that we spend the toggle budget
    // on the most-likely contributors and then fall back to the exploration order
    // as needed.
    bool budgetReached = walkList(_probabilitySortedIndices);
    if (!budgetReached)
      budgetReached = walkList(visibleExplore);
    if (!budgetReached)
      budgetReached = walkList(hiddenExplore);

#ifndef NDEBUG
    if (!forceAllToggles)
      assert(toggles <= maxToggles);
#endif

    size_t activeCount = 0;
    for (bool active : _activePrimitive)
      if (active)
        ++activeCount;

    if (activeCount == 0 && primCount > 0) {
      size_t fallback = fallbackCandidate < primCount ? fallbackCandidate
                                                      : size_t(0);
      if (fallback >= primCount)
        fallback = primCount - 1;
      if (setPrimitiveActive(fallback, true)) {
        ++_frameProbabilisticToggles;
        changed = true;
      }
    }

    return changed;
  }

  if (_objectHitProbability.size() < objectCount)
    _objectHitProbability.resize(objectCount, 0.5f);
  if (_objectHitVariance.size() < objectCount)
    _objectHitVariance.resize(objectCount, 1.0f / 12.0f);
  if (_objectPosteriorMass.size() < objectCount)
    _objectPosteriorMass.resize(objectCount, 2.0f);
  if (_objectProbabilitySortedIndices.size() != objectCount) {
    _objectProbabilitySortedIndices.resize(objectCount);
    std::iota(_objectProbabilitySortedIndices.begin(),
              _objectProbabilitySortedIndices.end(), size_t(0));
  }
  if (_objectVisible.size() < objectCount)
    _objectVisible.resize(objectCount, 0);
  if (_objectVisibilityEvidence.size() < objectCount)
    _objectVisibilityEvidence.resize(objectCount, 0.0f);
  if (_objectExplorationScore.size() < objectCount)
    _objectExplorationScore.resize(objectCount, 0.0f);
  if (_objectRaysTestedLastFrame.size() < objectCount)
    _objectRaysTestedLastFrame.resize(objectCount, 0u);

  if (_desiredObjectState.size() < objectCount)
    _desiredObjectState.resize(objectCount, 0);
  else if (_desiredObjectState.size() > objectCount)
    _desiredObjectState.resize(objectCount);
  if (_pendingDesiredObjects.size() < objectCount)
    _pendingDesiredObjects.resize(objectCount, 0);
  else if (_pendingDesiredObjects.size() > objectCount)
    _pendingDesiredObjects.resize(objectCount);

  if (_desiredObjectPromotionFrame.size() < objectCount)
    _desiredObjectPromotionFrame.resize(objectCount, 0);
  else if (_desiredObjectPromotionFrame.size() > objectCount)
    _desiredObjectPromotionFrame.resize(objectCount);
  if (_desiredObjectDemotionFrame.size() < objectCount)
    _desiredObjectDemotionFrame.resize(objectCount, 0);
  else if (_desiredObjectDemotionFrame.size() > objectCount)
    _desiredObjectDemotionFrame.resize(objectCount);
  if (_objectDemotionDwell.size() < objectCount)
    _objectDemotionDwell.resize(objectCount, 0);
  else if (_objectDemotionDwell.size() > objectCount)
    _objectDemotionDwell.resize(objectCount);

  std::vector<uint8_t> desiredObjects = _desiredObjectState;
  auto &pendingDesiredObjects = _pendingDesiredObjects;
  std::vector<uint8_t> objectBecameVisible(objectCount, 0);
  auto computeObjectVisibility = [this, &objectBecameVisible](size_t idx) {
    bool previousVisible =
        (idx < _objectVisible.size()) ? (_objectVisible[idx] != 0) : false;
    bool visible = false;
    if (idx < _objectBounds.size()) {
      visible = isInView(_objectBounds[idx]);
      if (idx < _objectVisible.size())
        _objectVisible[idx] = visible ? 1 : 0;
    } else if (idx < _objectVisible.size()) {
      visible = _objectVisible[idx] != 0;
    }
    if (visible && !previousVisible && idx < objectBecameVisible.size())
      objectBecameVisible[idx] = 1;
    return visible;
  };

  std::vector<uint8_t> objectVisibility(objectCount, 0);
  for (size_t i = 0; i < objectCount; ++i) {
    bool visible = computeObjectVisibility(i);
    objectVisibility[i] = visible ? 1 : 0;
  }

  auto visibilityForIndex = [&](size_t idx) {
    if (idx < objectVisibility.size())
      return objectVisibility[idx] != 0;
    return computeObjectVisibility(idx);
  };
  auto bufferedEvidenceForIndex = [&](size_t idx, float rawEvidence,
                                      bool visible) {
    float previous =
        (idx < _objectVisibilityEvidence.size()) ? _objectVisibilityEvidence[idx]
                                                 : 0.0f;
    float decayed = previous * kVisibilityEvidenceDecay;
    float buffered = std::max(std::clamp(rawEvidence, 0.0f, 1.0f), decayed);
    if (idx < _objectVisibilityEvidence.size())
      _objectVisibilityEvidence[idx] = buffered;
    return buffered;
  };
  float threshold = _residencyConfig.probabilityThreshold;
  float hysteresis =
      std::clamp(_residencyConfig.probabilityDesiredHysteresis, 0.0f, 0.5f);
  float enterThreshold = std::clamp(threshold + hysteresis, 0.0f, 1.0f);
  float exitThreshold = std::clamp(threshold - hysteresis, 0.0f, 1.0f);
  uint32_t demotionDwellFrames =
      _residencyConfig.probabilityVisibleDemotionDwellFrames;

  size_t desiredPrimitiveCount = 0;
  for (size_t i = 0; i < objectCount; ++i) {
    float probability = (i < _objectHitProbability.size())
                            ? _objectHitProbability[i]
                            : 0.5f;
    float variance = (i < _objectHitVariance.size()) ? _objectHitVariance[i]
                                                     : 0.0f;
    float mass = (i < _objectPosteriorMass.size())
                     ? _objectPosteriorMass[i]
                     : 0.0f;
    bool visible = visibilityForIndex(i);
    float rawEvidence = computeEvidenceFactor(mass, visible);
    float bufferedEvidence = bufferedEvidenceForIndex(i, rawEvidence, visible);
    float sanitizedProbability = sanitizePosteriorProbability(probability);
    float effectiveProbability = computeRegressedProbabilityFromEvidence(
        sanitizedProbability, bufferedEvidence);
    float boostedProbability = computePosteriorScoreFromEvidence(
        probability, variance, bufferedEvidence);
    float enterScore = std::max(effectiveProbability, boostedProbability);
    float exitScore = effectiveProbability;
    bool previousDesired = desiredObjects[i] != 0;
    bool desired = previousDesired;
    bool cooldownExpired =
        (i >= _objectCooldown.size()) || _objectCooldown[i] == 0;
    float lowEvidenceMetric =
        std::min(bufferedEvidence, std::clamp(rawEvidence, 0.0f, 1.0f));
    bool lowEvidence = lowEvidenceMetric <= kMinimalEvidenceThreshold;
    float promotionProbability = enterScore;
    float demotionProbability = exitScore;
    float evaluationProbability = lowEvidence ? effectiveProbability
                                              : enterScore;

    bool dwellApplies = visible && demotionDwellFrames > 0;
    if (!dwellApplies || demotionProbability > exitThreshold)
      _objectDemotionDwell[i] = 0;

    if (visible) {
      demotionProbability =
          std::max(demotionProbability, probabilityVisibleFloor);
      evaluationProbability =
          std::max(evaluationProbability, probabilityVisibleFloor);
    }

    if (promotionProbability >= enterThreshold) {
      desired = true;
      _objectDemotionDwell[i] = 0;
    } else if (demotionProbability <= exitThreshold) {
      if (dwellApplies && previousDesired) {
        uint32_t &dwellCount = _objectDemotionDwell[i];
        if (dwellCount < demotionDwellFrames)
          ++dwellCount;
        if (dwellCount >= demotionDwellFrames)
          desired = false;
        else
          desired = true;
      } else {
        desired = false;
        _objectDemotionDwell[i] = 0;
      }
    } else if (cooldownExpired && !previousDesired) {
      desired = evaluationProbability >= threshold;
      if (desired)
        _objectDemotionDwell[i] = 0;
    }

    if (desired && !previousDesired && i < _desiredObjectPromotionFrame.size())
      _desiredObjectPromotionFrame[i] = _renderedFrameCount;
    else if (!desired && previousDesired &&
             i < _desiredObjectDemotionFrame.size())
      _desiredObjectDemotionFrame[i] = _renderedFrameCount;
    desiredObjects[i] = desired ? 1 : 0;
    if (!desired && i < pendingDesiredObjects.size())
      pendingDesiredObjects[i] = 0;
    if (desired) {
      bool pending =
          (i < pendingDesiredObjects.size()) ? pendingDesiredObjects[i] != 0
                                             : false;
      size_t contribution =
          (i < _objectPrimitiveCounts.size()) ? _objectPrimitiveCounts[i] : 0;
      if (contribution == 0)
        contribution = 1;
      if (!pending)
        desiredPrimitiveCount += contribution;
    }
  }

  size_t minActivePrimitives =
      std::min(_residencyConfig.probabilityMinActivePrimitives, primCount);

  auto primitiveContribution = [this](size_t idx) {
    size_t count =
        (idx < _objectPrimitiveCounts.size()) ? _objectPrimitiveCounts[idx] : 0;
    return std::max<size_t>(count, 1);
  };

  std::sort(_objectProbabilitySortedIndices.begin(),
            _objectProbabilitySortedIndices.end(),
            [this, &computePosteriorScore, &visibilityForIndex](size_t a,
                                                               size_t b) {
              float probA = (a < _objectHitProbability.size())
                                ? _objectHitProbability[a]
                                : 0.5f;
              float varA = (a < _objectHitVariance.size())
                               ? _objectHitVariance[a]
                               : 0.0f;
              float massA = (a < _objectPosteriorMass.size())
                                ? _objectPosteriorMass[a]
                                : 0.0f;
              float scoreA =
                  sanitizeSortValue(computePosteriorScore(
                      probA, varA, massA, visibilityForIndex(a)));
              float probB = (b < _objectHitProbability.size())
                                ? _objectHitProbability[b]
                                : 0.5f;
              float varB = (b < _objectHitVariance.size())
                               ? _objectHitVariance[b]
                               : 0.0f;
              float massB = (b < _objectPosteriorMass.size())
                                ? _objectPosteriorMass[b]
                                : 0.0f;
              float scoreB =
                  sanitizeSortValue(computePosteriorScore(
                      probB, varB, massB, visibilityForIndex(b)));
              if (scoreA == scoreB)
                return a < b;
              return scoreA > scoreB;
            });

  float targetFraction = adaptiveProbabilityTargetFraction;
  size_t targetPrimitiveBudget = static_cast<size_t>(std::ceil(
      static_cast<float>(primCount) * targetFraction));
  targetPrimitiveBudget = std::max(targetPrimitiveBudget, minActivePrimitives);
  if (targetPrimitiveBudget == 0 && primCount > 0)
    targetPrimitiveBudget = 1;
  targetPrimitiveBudget = std::min(targetPrimitiveBudget, primCount);

  _frameProbabilityTargetPrimitives = targetPrimitiveBudget;
  _frameProbabilityInitialDesiredPrimitives = desiredPrimitiveCount;

  size_t trimmedPrimitives = 0;
  if (targetPrimitiveBudget > 0 &&
      desiredPrimitiveCount > targetPrimitiveBudget) {
    for (size_t position = _objectProbabilitySortedIndices.size();
         position-- > 0;) {
      if (desiredPrimitiveCount <= targetPrimitiveBudget)
        break;
      size_t idx = _objectProbabilitySortedIndices[position];
      if (idx >= objectCount)
        continue;
      if (desiredObjects[idx] == 0)
        continue;
      bool pending =
          (idx < pendingDesiredObjects.size()) ? pendingDesiredObjects[idx] != 0
                                               : false;
      if (pending)
        continue;
      size_t contribution = primitiveContribution(idx);
      if (contribution == 0)
        continue;
      desiredObjects[idx] = 0;
      if (idx < pendingDesiredObjects.size())
        pendingDesiredObjects[idx] = 0;
      trimmedPrimitives += contribution;
      if (idx < _desiredObjectDemotionFrame.size())
        _desiredObjectDemotionFrame[idx] = _renderedFrameCount;
      desiredPrimitiveCount =
          (desiredPrimitiveCount > contribution)
              ? desiredPrimitiveCount - contribution
              : size_t(0);
    }
  }

  _frameProbabilityTrimmedPrimitives = trimmedPrimitives;
  _frameProbabilityBudgetHit = trimmedPrimitives > 0;

  std::vector<size_t> visibleExplore;
  std::vector<size_t> hiddenExplore;
  visibleExplore.reserve(objectCount);
  hiddenExplore.reserve(objectCount);

  for (size_t idx : _objectProbabilitySortedIndices) {
    if (idx >= objectCount)
      continue;
    bool currentlyActive =
        idx < _objectActive.size() ? _objectActive[idx] : false;
    if (currentlyActive)
      continue;
    if (desiredObjects[idx] != 0)
      continue;
    float probability = (idx < _objectHitProbability.size())
                            ? _objectHitProbability[idx]
                            : 0.5f;
    float mass = (idx < _objectPosteriorMass.size())
                     ? _objectPosteriorMass[idx]
                     : 0.0f;
    bool visible = visibilityForIndex(idx);
    float rawEvidence = computeEvidenceFactor(mass, visible);
    float bufferedEvidence = (idx < _objectVisibilityEvidence.size())
                                 ? _objectVisibilityEvidence[idx]
                                 : rawEvidence;
    float effectiveProbability = computeRegressedProbabilityFromEvidence(
        probability, bufferedEvidence);
    bool visibilityBootstrap = (idx < objectBecameVisible.size()) &&
                               objectBecameVisible[idx] != 0 &&
                               mass <= kMinimalEvidenceThreshold;
    float lowEvidenceMetric =
        std::min(bufferedEvidence, std::clamp(rawEvidence, 0.0f, 1.0f));
    bool lowEvidence = lowEvidenceMetric <= kMinimalEvidenceThreshold;
    float visibilityAdjustedProbability =
        (visible && lowEvidence) ? std::min(effectiveProbability, threshold)
                                 : effectiveProbability;
    float gatedProbability = visibilityBootstrap
                                 ? std::min(visibilityAdjustedProbability, threshold)
                                 : visibilityAdjustedProbability;
    if (gatedProbability > threshold)
      continue;
    float exploreScore = (idx < _objectExplorationScore.size())
                             ? _objectExplorationScore[idx]
                             : 0.0f;
    uint32_t raysTested = (idx < _objectRaysTestedLastFrame.size())
                              ? _objectRaysTestedLastFrame[idx]
                              : 0u;
    float effectiveExplore = exploreScore;
    if (visibilityBootstrap)
      effectiveExplore = std::max(effectiveExplore, kIdleVisibleExploreSeed);
    if (visible && (raysTested == 0 || lowEvidence))
      effectiveExplore = std::max(effectiveExplore, kIdleVisibleExploreSeed);
    if (visible && raysTested > 0 && !lowEvidence)
      effectiveExplore = std::max(effectiveExplore, kPosteriorFloor);
    if (idx < _objectExplorationScore.size())
      _objectExplorationScore[idx] = effectiveExplore;
    if (effectiveExplore <= 0.0f)
      continue;
    if (visible)
      visibleExplore.push_back(idx);
    else
      hiddenExplore.push_back(idx);
  }

  auto objectExploreComparator = [this](size_t a, size_t b) {
    float exploreA = (a < _objectExplorationScore.size())
                         ? sanitizeSortValue(_objectExplorationScore[a])
                         : -std::numeric_limits<float>::max();
    float exploreB = (b < _objectExplorationScore.size())
                         ? sanitizeSortValue(_objectExplorationScore[b])
                         : -std::numeric_limits<float>::max();
    if (exploreA == exploreB) {
      uint32_t raysA = (a < _objectRaysTestedLastFrame.size())
                           ? _objectRaysTestedLastFrame[a]
                           : 0u;
      uint32_t raysB = (b < _objectRaysTestedLastFrame.size())
                           ? _objectRaysTestedLastFrame[b]
                           : 0u;
      if (raysA == raysB)
        return a < b;
      return raysA > raysB;
    }
    return exploreA > exploreB;
  };

  std::sort(visibleExplore.begin(), visibleExplore.end(),
            objectExploreComparator);
  std::sort(hiddenExplore.begin(), hiddenExplore.end(),
            objectExploreComparator);

  auto isRecentlyPromoted = [&, window =
                                   _residencyConfig
                                       .probabilityRecentPromotionFrames](
                                  size_t idx) {
    if (window == 0)
      return false;
    uint64_t lastPromotion =
        idx < _desiredObjectPromotionFrame.size()
            ? _desiredObjectPromotionFrame[idx]
            : 0;
    uint64_t lastDemotion =
        idx < _desiredObjectDemotionFrame.size()
            ? _desiredObjectDemotionFrame[idx]
            : 0;
    uint64_t lastFrame = std::max(lastPromotion, lastDemotion);
    if (lastFrame == 0)
      return false;
    uint64_t currentFrame = _renderedFrameCount;
    if (currentFrame <= lastFrame)
      return true;
    return currentFrame - lastFrame < window;
  };

  auto markDesired = [&](size_t idx) {
    desiredObjects[idx] = 1;
    if (idx < _desiredObjectPromotionFrame.size())
      _desiredObjectPromotionFrame[idx] = _renderedFrameCount;
    size_t contribution = primitiveContribution(idx);
    bool pending =
        (idx < pendingDesiredObjects.size()) ? pendingDesiredObjects[idx] != 0
                                             : false;
    if (!pending)
      desiredPrimitiveCount += contribution;
    bool wasVisible =
        (idx < _objectVisible.size()) ? (_objectVisible[idx] != 0) : false;
    bool idle =
        (idx < _objectRaysTestedLastFrame.size())
            ? (_objectRaysTestedLastFrame[idx] == 0)
            : true;
    if (wasVisible && idle && idx < _objectExplorationScore.size() &&
        _objectExplorationScore[idx] < kIdleVisibleExploreSeed)
      _objectExplorationScore[idx] = kIdleVisibleExploreSeed;
    return contribution;
  };

  auto promoteObjects = [&](const std::vector<size_t> &candidates,
                            size_t &slots, bool allowSuppressed) {
    if (slots == 0)
      return;
    for (size_t idx : candidates) {
      if (slots == 0)
        break;
      if (idx >= objectCount)
        continue;
      if (desiredObjects[idx] != 0)
        continue;
      if (!allowSuppressed && isRecentlyPromoted(idx))
        continue;
      size_t contribution = markDesired(idx);
      if (slots <= contribution)
        slots = 0;
      else
        slots -= contribution;
    }
  };

  auto countRemainingPrimitives = [&](const std::vector<size_t> &candidates) {
    size_t remaining = 0;
    for (size_t idx : candidates) {
      if (idx >= objectCount)
        continue;
      if (desiredObjects[idx] != 0)
        continue;
      bool pending =
          (idx < pendingDesiredObjects.size()) ? pendingDesiredObjects[idx] != 0
                                               : false;
      if (pending)
        continue;
      remaining += primitiveContribution(idx);
    }
    return remaining;
  };

  if (desiredPrimitiveCount < minActivePrimitives) {
    size_t slots =
        std::min(minActivePrimitives - desiredPrimitiveCount,
                 primCount > desiredPrimitiveCount
                     ? primCount - desiredPrimitiveCount
                     : size_t(0));
    promoteObjects(visibleExplore, slots, false);
    promoteObjects(hiddenExplore, slots, false);
    if (slots > 0) {
      promoteObjects(visibleExplore, slots, true);
      promoteObjects(hiddenExplore, slots, true);
    }
  }

  if (desiredPrimitiveCount < minActivePrimitives) {
    auto promoteByProbability = [&](bool allowSuppressed) {
      for (size_t idx : _objectProbabilitySortedIndices) {
        if (desiredPrimitiveCount >= minActivePrimitives)
          break;
        if (idx >= objectCount)
          continue;
        if (desiredObjects[idx] != 0)
          continue;
        if (!allowSuppressed && isRecentlyPromoted(idx))
          continue;
        markDesired(idx);
      }
    };
    promoteByProbability(false);
    if (desiredPrimitiveCount < minActivePrimitives)
      promoteByProbability(true);
  }

  size_t remainingExplorePrimitives =
      countRemainingPrimitives(visibleExplore) +
      countRemainingPrimitives(hiddenExplore);
    if (remainingExplorePrimitives > 0 && desiredPrimitiveCount < primCount) {
      size_t slots = std::min({std::max<size_t>(size_t(1),
                                                minActivePrimitives / 2),
                               remainingExplorePrimitives,
                               primCount - desiredPrimitiveCount});
      promoteObjects(visibleExplore, slots, false);
      promoteObjects(hiddenExplore, slots, false);
      if (slots > 0) {
        promoteObjects(visibleExplore, slots, true);
        promoteObjects(hiddenExplore, slots, true);
      }
    }

  desiredPrimitiveCount = 0;
  for (size_t idx = 0; idx < objectCount; ++idx) {
    if (desiredObjects[idx] == 0)
      continue;
    bool pending =
        (idx < pendingDesiredObjects.size()) ? pendingDesiredObjects[idx] != 0
                                             : false;
    if (pending)
      continue;
    desiredPrimitiveCount += primitiveContribution(idx);
  }

  if (targetPrimitiveBudget > 0 && desiredPrimitiveCount > targetPrimitiveBudget) {
    for (size_t position = _objectProbabilitySortedIndices.size(); position-- > 0;) {
      if (desiredPrimitiveCount <= targetPrimitiveBudget)
        break;
      size_t idx = _objectProbabilitySortedIndices[position];
      if (idx >= objectCount)
        continue;
      if (desiredObjects[idx] == 0)
        continue;
      bool pending =
          (idx < pendingDesiredObjects.size()) ? pendingDesiredObjects[idx] != 0
                                               : false;
      if (pending)
        continue;
      size_t contribution = primitiveContribution(idx);
      if (contribution == 0)
        continue;
      desiredObjects[idx] = 0;
      if (idx < pendingDesiredObjects.size())
        pendingDesiredObjects[idx] = 0;
      trimmedPrimitives += contribution;
      if (idx < _desiredObjectDemotionFrame.size())
        _desiredObjectDemotionFrame[idx] = _renderedFrameCount;
      desiredPrimitiveCount =
          (desiredPrimitiveCount > contribution) ? desiredPrimitiveCount - contribution
                                                 : size_t(0);
    }
  }

  _frameProbabilityTrimmedPrimitives = trimmedPrimitives;
  _frameProbabilityBudgetHit = trimmedPrimitives > 0;

  size_t fallbackObject = objectCount;
  if (!visibleExplore.empty())
    fallbackObject = visibleExplore.front();
  else if (!hiddenExplore.empty())
    fallbackObject = hiddenExplore.front();
  else if (!_objectProbabilitySortedIndices.empty())
    fallbackObject = _objectProbabilitySortedIndices.front();

  bool changed = false;
  size_t toggledPrimitiveCount = 0;
  size_t maxPrimitiveToggles = adaptiveProbabilityToggleBudget;

  std::vector<size_t> toggleOrder;
  toggleOrder.reserve(objectCount);
  std::vector<uint8_t> added(objectCount, 0);
  std::vector<uint8_t> appliedDesired(objectCount, 0);
  size_t appliedDesiredPrimitiveCount = 0;

  auto appendList = [&](const std::vector<size_t> &candidates) {
    for (size_t idx : candidates) {
      if (idx >= objectCount)
        continue;
      if (added[idx])
        continue;
      toggleOrder.push_back(idx);
      added[idx] = 1;
    }
  };

  std::vector<size_t> desiredByPriority;
  desiredByPriority.reserve(objectCount);
  for (size_t idx : _objectProbabilitySortedIndices) {
    if (idx >= objectCount)
      continue;
    if (desiredObjects[idx] == 0)
      continue;
    desiredByPriority.push_back(idx);
  }

  std::vector<size_t> pendingCandidates;
  pendingCandidates.reserve(objectCount);
  for (size_t idx = 0; idx < objectCount; ++idx) {
    bool pending =
        (idx < pendingDesiredObjects.size()) ? pendingDesiredObjects[idx] != 0
                                             : false;
    if (pending)
      pendingCandidates.push_back(idx);
  }

  appendList(pendingCandidates);
  appendList(desiredByPriority);
  appendList(visibleExplore);
  appendList(hiddenExplore);
  appendList(_objectProbabilitySortedIndices);

  auto markPendingPromotion = [&](size_t idx) {
    if (idx < pendingDesiredObjects.size())
      pendingDesiredObjects[idx] = 1;
  };

  for (size_t objectIndex : toggleOrder) {
    if (!forceAllToggles) {
      if (maxPrimitiveToggles == 0 || toggledPrimitiveCount >= maxPrimitiveToggles)
        break;
    }

    bool shouldBeActive = desiredObjects[objectIndex] != 0;
    bool currentlyActive =
        objectIndex < _objectActive.size() && _objectActive[objectIndex];
    if (shouldBeActive == currentlyActive)
      continue;

    if (!forceAllToggles) {
      if (objectIndex < _objectCooldown.size() &&
          _objectCooldown[objectIndex] > 0) {
        if (shouldBeActive)
          markPendingPromotion(objectIndex);
        continue;
      }

      const SceneObject &obj = _allSceneObjects[objectIndex];
      size_t first = obj.firstPrimitive;
      size_t last = first + obj.primitiveCount;
      size_t pendingToggles = 0;
      bool canToggle = true;
      for (size_t prim = first; prim < last && prim < _activePrimitive.size();
           ++prim) {
        if (_activePrimitive[prim] == shouldBeActive)
          continue;
        ++pendingToggles;
        if (prim < _primitiveCooldown.size() && _primitiveCooldown[prim] > 0) {
          canToggle = false;
          break;
        }
      }

      if (!canToggle) {
        if (shouldBeActive)
          markPendingPromotion(objectIndex);
        continue;
      }
      if (pendingToggles == 0)
        continue;
      size_t remainingBudget =
          (maxPrimitiveToggles > toggledPrimitiveCount)
              ? (maxPrimitiveToggles - toggledPrimitiveCount)
              : size_t(0);
      if (pendingToggles > remainingBudget) {
        if (shouldBeActive)
          markPendingPromotion(objectIndex);
        continue;
      }
    }

    size_t toggled = setObjectActive(objectIndex, shouldBeActive);
    if (toggled > 0) {
      size_t applied = toggled;
      if (!forceAllToggles) {
        size_t remainingBudget =
            (maxPrimitiveToggles > toggledPrimitiveCount)
                ? (maxPrimitiveToggles - toggledPrimitiveCount)
                : size_t(0);
        applied = std::min(toggled, remainingBudget);
        toggledPrimitiveCount += applied;
      }
      if (shouldBeActive && !appliedDesired[objectIndex]) {
        appliedDesired[objectIndex] = 1;
        appliedDesiredPrimitiveCount += primitiveContribution(objectIndex);
        if (objectIndex < pendingDesiredObjects.size())
          pendingDesiredObjects[objectIndex] = 0;
        if (objectIndex < _desiredObjectState.size())
          _desiredObjectState[objectIndex] = 1;
      }
      _frameProbabilisticToggles += applied;
      changed = true;
      if (!shouldBeActive && objectIndex < _objectExplorationScore.size())
        _objectExplorationScore[objectIndex] = 0.0f;
      if (!shouldBeActive && objectIndex < pendingDesiredObjects.size())
        pendingDesiredObjects[objectIndex] = 0;
      if (!shouldBeActive && objectIndex < _desiredObjectState.size())
        _desiredObjectState[objectIndex] = 0;
    }
  }

  size_t activePrimitiveCount = 0;
  for (bool active : _activePrimitive)
    if (active)
      ++activePrimitiveCount;

  if (activePrimitiveCount == 0 && objectCount > 0) {
    size_t fallback = fallbackObject < objectCount ? fallbackObject : size_t(0);
    if (fallback >= objectCount)
      fallback = objectCount - 1;
    size_t remainingBudget =
        (maxPrimitiveToggles > toggledPrimitiveCount)
            ? (maxPrimitiveToggles - toggledPrimitiveCount)
            : size_t(0);
    bool canToggleFallback = forceAllToggles || remainingBudget > 0;
    size_t toggled = canToggleFallback ? setObjectActive(fallback, true) : 0;
    if (toggled > 0) {
      size_t applied = toggled;
      if (!forceAllToggles) {
        applied = std::min(toggled, remainingBudget);
        toggledPrimitiveCount += applied;
      }
      _frameProbabilisticToggles += applied;
      changed = true;
    }
    if (fallback < _desiredObjectState.size()) {
      _desiredObjectState[fallback] = 1;
      if (fallback < _desiredObjectPromotionFrame.size())
        _desiredObjectPromotionFrame[fallback] = _renderedFrameCount;
    }
  }

  _frameProbabilityFinalDesiredPrimitives = appliedDesiredPrimitiveCount;

  return changed;
}

bool Renderer::updateScreenSpaceFootprint(bool forceAllToggles) {
  if (_activePrimitive.empty())
    return false;

  const float cameraMotion =
      clampUnit(static_cast<float>(std::min(_frameCameraMotionMetric, 1.0)));
  const double totalCapMB = effectiveTotalGpuMemoryCapMB();
  const double gpuMB = currentGPUMemoryMB();
  const float memoryPressure =
      (totalCapMB > 0.0)
          ? clampUnit(static_cast<float>((gpuMB / totalCapMB - 0.80) / 0.20))
          : 0.0f;
  const float adaptiveScreenTargetFraction =
      std::clamp(_residencyConfig.screenFootprintTargetFraction +
                     0.08f * cameraMotion - 0.10f * memoryPressure,
                 0.50f, 0.94f);
  const size_t baseScreenToggleBudget =
      std::max<size_t>(_residencyConfig.screenFootprintMaxTogglesPerFrame, 1);
  const size_t adaptiveScreenToggleBudget = std::max<size_t>(
      1, static_cast<size_t>(std::llround(
             static_cast<double>(baseScreenToggleBudget) *
             (1.0 + 0.40 * static_cast<double>(cameraMotion) +
              0.50 * static_cast<double>(memoryPressure)))));

  const size_t primCount = _activePrimitive.size();
  if (_screenCoverageSortedIndices.size() != primCount) {
    _screenCoverageSortedIndices.resize(primCount);
    std::iota(_screenCoverageSortedIndices.begin(),
              _screenCoverageSortedIndices.end(), size_t(0));
  }

  updatePrimitiveScreenCoverageForFrame();

  float screenArea = Camera::screenSize.x * Camera::screenSize.y;
  if (screenArea <= 0.0f)
    screenArea = 1.0f;
  const size_t objectCount = _allSceneObjects.size();
  if (objectCount == 0)
    return false;

  std::vector<size_t> objectPrimitiveTotals(objectCount, 0);
  std::vector<float> objectCoverage(objectCount, 0.0f);
  auto objectCoverageForSphere =
      [this, screenArea](const BoundingSphere &b) -> float {
    if (!isInView(b))
      return 0.0f;

    float halfFov =
        Camera::verticalFov * static_cast<float>(M_PI) / 180.0f * 0.5f;
    float tanHalfFov = std::tan(halfFov);
    if (tanHalfFov <= 0.0f)
      tanHalfFov = 1e-3f;

    simd::float3 forward = simd::normalize(Camera::forward);
    simd::float3 up = simd::normalize(Camera::up);
    simd::float3 right = simd::cross(forward, up);
    float rightLenSq = simd::length_squared(right);
    if (rightLenSq < 1e-6f)
      right = {1.0f, 0.0f, 0.0f};
    else
      right /= std::sqrt(rightLenSq);

    float aspect = Camera::screenSize.y > 0.0f
                       ? Camera::screenSize.x / Camera::screenSize.y
                       : 1.0f;
    float horizontalHalfFov = std::atan(tanHalfFov * aspect);

    simd::float3 toCenter = b.center - Camera::position;
    float depth = simd::dot(toCenter, forward);
    if (depth <= 1e-3f)
      return 0.0f;

    float dist = simd::length(toCenter);
    float cosAngle = depth / std::max(dist, 1e-3f);
    float horiz = simd::dot(toCenter, right);
    float horizAngle = std::atan2(std::fabs(horiz), depth);
    if (horizAngle > horizontalHalfFov + 0.1f)
      return 0.0f;

    float radiusPixels =
        (b.radius / depth) / tanHalfFov * (Camera::screenSize.y * 0.5f);
    radiusPixels = std::max(radiusPixels, 0.0f);
    float area = static_cast<float>(M_PI) * radiusPixels * radiusPixels;
    float angleFactor = std::max(cosAngle, 0.0f);
    return std::min(area * angleFactor, screenArea);
  };

  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    size_t declaredPrimitiveCount =
        objectIndex < _allSceneObjects.size()
            ? _allSceneObjects[objectIndex].primitiveCount
            : size_t(0);
    objectPrimitiveTotals[objectIndex] = declaredPrimitiveCount;
    if (objectIndex < _objectBounds.size())
      objectCoverage[objectIndex] =
          objectCoverageForSphere(_objectBounds[objectIndex]);
  }

  const auto &meshGroups = _meshGroups;
  std::vector<float> meshGroupCoverage(meshGroups.size(), 0.0f);
  std::vector<size_t> meshGroupPrimitiveCount(meshGroups.size(), 0);
  for (size_t groupIndex = 0; groupIndex < meshGroups.size(); ++groupIndex) {
    const MeshGroupInfo &info = meshGroups[groupIndex];
    float coverageSum = 0.0f;
    size_t fallbackCount = 0;
    for (size_t objectIndex : info.objectIndices) {
      if (objectIndex < objectCoverage.size())
        coverageSum += objectCoverage[objectIndex];
      if (objectIndex < objectPrimitiveTotals.size())
        fallbackCount += objectPrimitiveTotals[objectIndex];
    }
    meshGroupCoverage[groupIndex] = coverageSum;
    size_t declaredCount = info.primitiveCount;
    if (declaredCount == 0)
      declaredCount = fallbackCount;
    meshGroupPrimitiveCount[groupIndex] = declaredCount;
  }

  std::vector<size_t> sortedGroups(meshGroups.size());
  std::iota(sortedGroups.begin(), sortedGroups.end(), size_t(0));
  std::sort(sortedGroups.begin(), sortedGroups.end(),
            [&meshGroupCoverage](size_t a, size_t b) {
              float ca = (a < meshGroupCoverage.size())
                             ? sanitizeSortValue(meshGroupCoverage[a])
                             : 0.0f;
              float cb = (b < meshGroupCoverage.size())
                             ? sanitizeSortValue(meshGroupCoverage[b])
                             : 0.0f;
              if (ca == cb)
                return a < b;
              return ca > cb;
            });

  std::vector<bool> desiredGroupState(meshGroups.size(), false);
  const size_t minActivePrimitives = std::min(
      primCount, _residencyConfig.screenFootprintMinActivePrimitives);
  const float targetCoverage = screenArea * adaptiveScreenTargetFraction;
  size_t primitivesEnabled = 0;
  float accumulatedCoverage = 0.0f;

  for (size_t groupIndex : sortedGroups) {
    if (groupIndex >= meshGroups.size())
      continue;
    size_t declaredPrimitiveCount =
        (groupIndex < meshGroupPrimitiveCount.size())
            ? meshGroupPrimitiveCount[groupIndex]
            : size_t(0);
    if (declaredPrimitiveCount == 0)
      continue;

    float coverage =
        (groupIndex < meshGroupCoverage.size()) ? meshGroupCoverage[groupIndex]
                                                : 0.0f;
    bool minPrimitivesSatisfied = primitivesEnabled >= minActivePrimitives;
    bool coverageSatisfied = accumulatedCoverage >= targetCoverage;
    if (minPrimitivesSatisfied && coverageSatisfied)
      break;
    if (minPrimitivesSatisfied &&
        coverage < _residencyConfig.screenFootprintMinPixelCoverage) {
      ++_frameScreenMinPixelCoverageSkips;
      continue;
    }
    if (minPrimitivesSatisfied && coverage <= 0.0f)
      continue;

    desiredGroupState[groupIndex] = true;
    primitivesEnabled += declaredPrimitiveCount;
    accumulatedCoverage += coverage;
  }

  for (size_t groupIndex : sortedGroups) {
    if (primitivesEnabled >= minActivePrimitives)
      break;
    if (groupIndex >= meshGroups.size())
      continue;
    if (desiredGroupState[groupIndex])
      continue;
    size_t declaredPrimitiveCount =
        (groupIndex < meshGroupPrimitiveCount.size())
            ? meshGroupPrimitiveCount[groupIndex]
            : size_t(0);
    if (declaredPrimitiveCount == 0)
      continue;
    desiredGroupState[groupIndex] = true;
    primitivesEnabled += declaredPrimitiveCount;
    accumulatedCoverage += (groupIndex < meshGroupCoverage.size())
                               ? meshGroupCoverage[groupIndex]
                               : 0.0f;
  }

  std::vector<bool> desiredObjectState(objectCount, false);
  for (size_t groupIndex = 0; groupIndex < meshGroups.size(); ++groupIndex) {
    bool groupDesired = desiredGroupState[groupIndex];
    for (size_t objectIndex : meshGroups[groupIndex].objectIndices) {
      if (objectIndex < desiredObjectState.size())
        desiredObjectState[objectIndex] = groupDesired;
    }
  }

  bool changed = false;
  size_t toggledPrimitiveCount = 0;
  for (size_t groupIndex : sortedGroups) {
    if (groupIndex >= meshGroups.size())
      continue;
    if (!forceAllToggles &&
        toggledPrimitiveCount >= adaptiveScreenToggleBudget)
      break;

    bool groupDesired = desiredGroupState[groupIndex];
    const auto &objectIndices = meshGroups[groupIndex].objectIndices;
    std::vector<size_t> objectsToToggle;
    objectsToToggle.reserve(objectIndices.size());
    bool canToggleGroup = true;

    for (size_t objectIndex : objectIndices) {
      if (objectIndex >= desiredObjectState.size())
        continue;
      bool shouldBeActive = desiredObjectState[objectIndex];
      bool currentlyActive =
          objectIndex < _objectActive.size() && _objectActive[objectIndex];
      if (shouldBeActive == currentlyActive)
        continue;

      if (!forceAllToggles && objectIndex < _objectCooldown.size() &&
          _objectCooldown[objectIndex] > 0) {
        canToggleGroup = false;
        break;
      }

      const SceneObject &obj = _allSceneObjects[objectIndex];
      size_t first = obj.firstPrimitive;
      size_t last = first + obj.primitiveCount;
      for (size_t prim = first; prim < last && prim < _activePrimitive.size();
           ++prim) {
        if (_activePrimitive[prim] == shouldBeActive)
          continue;
        if (!forceAllToggles && prim < _primitiveCooldown.size() &&
            _primitiveCooldown[prim] > 0) {
          canToggleGroup = false;
          break;
        }
      }

      if (!canToggleGroup)
        break;

      objectsToToggle.push_back(objectIndex);
    }

    if (!canToggleGroup || objectsToToggle.empty())
      continue;

    for (size_t objectIndex : objectsToToggle) {
      size_t toggled = setObjectActive(objectIndex, groupDesired);
      if (toggled > 0) {
        toggledPrimitiveCount =
            std::min(toggledPrimitiveCount + toggled, adaptiveScreenToggleBudget);
        changed = true;
      }
    }
  }

  bool anyActivePrimitive = false;
  for (bool active : _activePrimitive) {
    if (active) {
      anyActivePrimitive = true;
      break;
    }
  }

  if (!anyActivePrimitive) {
    size_t fallbackPrimitives = 0;
    for (size_t groupIndex : sortedGroups) {
      if (groupIndex >= meshGroups.size())
        continue;
      size_t declaredPrimitiveCount =
          (groupIndex < meshGroupPrimitiveCount.size())
              ? meshGroupPrimitiveCount[groupIndex]
              : size_t(0);
      if (declaredPrimitiveCount == 0)
        continue;

      bool groupActivated = false;
      for (size_t objectIndex : meshGroups[groupIndex].objectIndices) {
        if (objectIndex >= _allSceneObjects.size())
          continue;
        if (setObjectActive(objectIndex, true) > 0) {
          changed = true;
          groupActivated = true;
        }
      }

      if (groupActivated) {
        fallbackPrimitives += declaredPrimitiveCount;
        anyActivePrimitive = true;
      }

      if (anyActivePrimitive && fallbackPrimitives >= minActivePrimitives)
        break;
    }
  }

  return changed;
}

bool Renderer::updatePredictiveResidency(bool forceAllToggles) {
  size_t objectCount = _allSceneObjects.size();
  size_t primitiveCount = _activePrimitive.size();
  if (objectCount == 0 || primitiveCount == 0)
    return false;

  const float cameraMotion =
      clampUnit(static_cast<float>(std::min(_frameCameraMotionMetric, 1.0)));
  const double totalCapMB = effectiveTotalGpuMemoryCapMB();
  const double gpuMB = currentGPUMemoryMB();
  const float memoryPressure =
      (totalCapMB > 0.0)
          ? clampUnit(static_cast<float>((gpuMB / totalCapMB - 0.80) / 0.20))
          : 0.0f;
  const float adaptiveEnvironmentEscapeThreshold =
      std::clamp(_residencyConfig.environmentEscapeThreshold +
                     0.03f * cameraMotion + 0.08f * memoryPressure,
                 0.12f, 0.95f);
  const float adaptiveEnvironmentTargetFraction =
      std::clamp(_residencyConfig.environmentTargetActiveFraction +
                     0.08f * cameraMotion - 0.10f * memoryPressure,
                 0.12f, 0.92f);
  const size_t baseEnvironmentToggleBudget =
      std::max<size_t>(_residencyConfig.environmentMaxTogglesPerFrame, 1);
  const size_t adaptiveEnvironmentToggleBudget = std::max<size_t>(
      1, static_cast<size_t>(std::llround(
             static_cast<double>(baseEnvironmentToggleBudget) *
             (1.0 + 0.40 * static_cast<double>(cameraMotion) +
              0.70 * static_cast<double>(memoryPressure)))));

  if (_objectProbabilitySortedIndices.size() != objectCount) {
    _objectProbabilitySortedIndices.resize(objectCount);
    std::iota(_objectProbabilitySortedIndices.begin(),
              _objectProbabilitySortedIndices.end(), size_t(0));
  }

  if (_objectImportance.size() != objectCount)
    _objectImportance.assign(objectCount, 0.0f);
  else
    std::fill(_objectImportance.begin(), _objectImportance.end(), 0.0f);

  if (_objectVisible.size() < objectCount)
    _objectVisible.resize(objectCount, 0u);

  std::vector<uint8_t> desiredObjectState(objectCount, 0);
  std::vector<float> weightedPredictive(objectCount, 0.0f);

  auto computeDepthWeight = [&](size_t objectIndex) -> float {
    const auto &weights = _residencyConfig.environmentDepthWeights;
    if (weights.empty())
      return 1.0f;
    const auto &radii = _residencyConfig.environmentDepthRadii;
    size_t pairCount = std::min(weights.size(), radii.size());
    if (pairCount == 0)
      return weights.back();
    float fallbackWeight = weights[pairCount - 1];
    if (objectIndex >= _objectBounds.size())
      return fallbackWeight;

    simd::float3 toCenter = _objectBounds[objectIndex].center - Camera::position;
    float distance = simd::length(toCenter);
    if (!(distance > 0.0f))
      return weights.front();

    auto end = radii.begin() + pairCount;
    auto lowerBound = std::lower_bound(radii.begin(), end, distance);
    if (lowerBound != end)
      return weights[std::distance(radii.begin(), lowerBound)];
    return fallbackWeight;
  };

  constexpr float kExplorationPriorFloor = 1.0e-4f;
  constexpr float kExplorationPriorScale = 0.5f;

  constexpr float kVisibleWeightBoost = 2.0f;
  constexpr float kHiddenWeightPenalty = 0.05f;

  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    float hitProbability =
        (objectIndex < _objectHitProbability.size())
            ? std::clamp(_objectHitProbability[objectIndex], 0.0f, 1.0f)
            : 0.0f;
    bool hasEvidence =
        (objectIndex < _objectRaysTestedLastFrame.size())
            ? (_objectRaysTestedLastFrame[objectIndex] > 0)
            : false;
    bool currentlyActive =
        (objectIndex < _objectActive.size()) ? _objectActive[objectIndex] : false;

    bool visible =
        (objectIndex < _objectBounds.size()) ? isInView(_objectBounds[objectIndex])
                                             : false;
    if (objectIndex < _objectVisible.size())
      _objectVisible[objectIndex] = visible ? 1u : 0u;

    if (!hasEvidence && !currentlyActive) {
      float exploration =
          (objectIndex < _objectExplorationScore.size())
              ? std::max(_objectExplorationScore[objectIndex], 0.0f)
              : 0.0f;
      float visibilityHint =
          (objectIndex < _objectVisible.size()) ? _objectVisible[objectIndex] : 0u;
      float explorationPrior = 1.0f -
                               std::exp(-exploration * kExplorationPriorScale);
      if (visibilityHint) {
        float hintedPrior = 1.0f -
                            std::exp(-kIdleVisibleExploreSeed *
                                     kExplorationPriorScale);
        explorationPrior = std::max(explorationPrior, hintedPrior);
      }
      explorationPrior =
          std::clamp(explorationPrior, kExplorationPriorFloor, 1.0f);
      hitProbability = std::max(hitProbability, explorationPrior);
    }

    float rayHitScore =
        (objectIndex < _objectRayHitScore.size()) ? _objectRayHitScore[objectIndex]
                                                  : 0.0f;
    size_t primitiveBudget =
        (objectIndex < _allSceneObjects.size())
            ? _allSceneObjects[objectIndex].primitiveCount
            : size_t(0);
    float normalizedHitScore = rayHitScore;
    if (primitiveBudget > 0)
      normalizedHitScore /= static_cast<float>(primitiveBudget);
    if (normalizedHitScore <= 0.0f && (currentlyActive || visible))
      normalizedHitScore = 1.0f;

    _objectImportance[objectIndex] = hitProbability;

    float predictiveScore = normalizedHitScore * hitProbability;
    float depthWeight = computeDepthWeight(objectIndex);
    float visibilityWeight = visible ? kVisibleWeightBoost : kHiddenWeightPenalty;
    float weighted = predictiveScore * std::max(depthWeight, 0.0f) *
                     std::max(visibilityWeight, 0.0f);
    weightedPredictive[objectIndex] = std::isfinite(weighted) ? weighted : 0.0f;
  }

  auto &sortedIndices = _objectProbabilitySortedIndices;
  std::sort(sortedIndices.begin(), sortedIndices.end(),
            [&](size_t a, size_t b) {
              float scoreA =
                  (a < weightedPredictive.size()) ? weightedPredictive[a] : 0.0f;
              float scoreB =
                  (b < weightedPredictive.size()) ? weightedPredictive[b] : 0.0f;
              if (scoreA == scoreB) {
                float importanceA =
                    (a < _objectImportance.size()) ? _objectImportance[a] : 0.0f;
                float importanceB =
                    (b < _objectImportance.size()) ? _objectImportance[b] : 0.0f;
                if (importanceA == importanceB)
                  return a < b;
                return importanceA > importanceB;
              }
              return scoreA > scoreB;
            });

  float escapeThreshold = adaptiveEnvironmentEscapeThreshold;
  size_t minActivePrimitives =
      std::min(primitiveCount, _residencyConfig.environmentMinActivePrimitives);
  if (minActivePrimitives == 0 && primitiveCount > 0)
    minActivePrimitives = 1;

  float targetFraction = adaptiveEnvironmentTargetFraction;
  size_t targetPrimitiveCount = 0;
  if (targetFraction > 0.0f && primitiveCount > 0) {
    targetPrimitiveCount = static_cast<size_t>(std::ceil(
        targetFraction * static_cast<float>(primitiveCount)));
    targetPrimitiveCount = std::min(targetPrimitiveCount, primitiveCount);
  }
  size_t activationFloor = std::max(minActivePrimitives, targetPrimitiveCount);
  if (activationFloor == 0 && primitiveCount > 0)
    activationFloor = 1;

  size_t desiredPrimitiveCount = 0;
  float weightedEscape = 0.0f;
  auto averageEscape = [](size_t primCount, float weighted) {
    if (primCount == 0)
      return 1.0f;
    return weighted / static_cast<float>(primCount);
  };

  for (size_t objectIndex : sortedIndices) {
    if (objectIndex >= _allSceneObjects.size())
      continue;
    const SceneObject &object = _allSceneObjects[objectIndex];
    if (object.primitiveCount == 0)
      continue;

    bool needMorePrimitives = desiredPrimitiveCount < activationFloor;
    bool needsEscapeReduction =
        averageEscape(desiredPrimitiveCount, weightedEscape) > escapeThreshold;
    if (!needMorePrimitives && !needsEscapeReduction)
      break;

    desiredObjectState[objectIndex] = 1;
    desiredPrimitiveCount += object.primitiveCount;
    float escapeProbability =
        1.0f - ((objectIndex < _objectImportance.size())
                    ? _objectImportance[objectIndex]
                    : 0.0f);
    weightedEscape += escapeProbability * static_cast<float>(object.primitiveCount);
  }

  if (desiredPrimitiveCount == 0 && primitiveCount > 0) {
    for (size_t objectIndex : sortedIndices) {
      if (objectIndex >= _allSceneObjects.size())
        continue;
      const SceneObject &object = _allSceneObjects[objectIndex];
      if (object.primitiveCount == 0)
        continue;
      desiredObjectState[objectIndex] = 1;
      desiredPrimitiveCount = object.primitiveCount;
      float escapeProbability =
          1.0f - ((objectIndex < _objectImportance.size())
                      ? _objectImportance[objectIndex]
                      : 0.0f);
      weightedEscape =
          escapeProbability * static_cast<float>(object.primitiveCount);
      break;
    }
  }

  bool changed = false;
  size_t offscreenActivePrimitiveCount = 0;
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    bool active =
        (objectIndex < _objectActive.size()) ? _objectActive[objectIndex] : false;
    bool visible = (objectIndex < _objectVisible.size())
                       ? (_objectVisible[objectIndex] != 0)
                       : false;
    if (!active || visible)
      continue;

    size_t primitives = 0;
    if (objectIndex < _objectPrimitiveCounts.size())
      primitives = _objectPrimitiveCounts[objectIndex];
    else if (objectIndex < _allSceneObjects.size())
      primitives = _allSceneObjects[objectIndex].primitiveCount;

    offscreenActivePrimitiveCount += primitives;
  }

  size_t baseToggleBudget = adaptiveEnvironmentToggleBudget;
  size_t maxToggleBudget = baseToggleBudget;
  if (offscreenActivePrimitiveCount > baseToggleBudget) {
    size_t catchUpBudget = (offscreenActivePrimitiveCount + 1) / 2;
    maxToggleBudget = std::max(baseToggleBudget, catchUpBudget);
  }

  size_t reservedOffscreenBudget =
      (maxToggleBudget == 0) ? 0 : std::max<size_t>(1, maxToggleBudget / 4);
  size_t generalBudget =
      (maxToggleBudget > reservedOffscreenBudget)
          ? (maxToggleBudget - reservedOffscreenBudget)
          : 0;
  size_t totalBudget = maxToggleBudget + reservedOffscreenBudget;

  size_t toggledPrimitiveCount = 0;

  for (size_t objectIndex : sortedIndices) {
    bool shouldBeActive =
        (objectIndex < desiredObjectState.size()) ? desiredObjectState[objectIndex]
                                                  : 0;
    bool currentlyActive =
        (objectIndex < _objectActive.size()) ? _objectActive[objectIndex] : false;
    if (shouldBeActive == currentlyActive)
      continue;

    bool visible = (objectIndex < _objectVisible.size())
                       ? (_objectVisible[objectIndex] != 0)
                       : false;
    float importance =
        (objectIndex < _objectImportance.size()) ? _objectImportance[objectIndex]
                                                 : 0.0f;
    bool lowImportanceOffscreen = !visible && (importance < escapeThreshold);
    bool bypassCooldowns = currentlyActive && !shouldBeActive &&
                           lowImportanceOffscreen;

    if (!forceAllToggles && !bypassCooldowns && objectIndex < _objectCooldown.size() &&
        _objectCooldown[objectIndex] > 0)
      continue;

    const SceneObject &object = _allSceneObjects[objectIndex];
    size_t first = object.firstPrimitive;
    size_t last = std::min(first + object.primitiveCount, _activePrimitive.size());
    if (last <= first)
      continue;

    bool canToggle = true;
    size_t pendingToggles = 0;
    for (size_t prim = first; prim < last; ++prim) {
      if (_activePrimitive[prim] == shouldBeActive)
        continue;
      if (!forceAllToggles && !bypassCooldowns && prim < _primitiveCooldown.size() &&
          _primitiveCooldown[prim] > 0) {
        canToggle = false;
        break;
      }
      ++pendingToggles;
    }

    if (!canToggle || pendingToggles == 0)
      continue;

    if (!forceAllToggles) {
      if (bypassCooldowns) {
        if (toggledPrimitiveCount >= totalBudget)
          continue;
      } else {
        if (toggledPrimitiveCount >= generalBudget)
          continue;
      }
    }

    size_t toggled = setObjectActive(objectIndex, shouldBeActive);
    if (toggled > 0) {
      changed = true;
      if (!forceAllToggles) {
        toggledPrimitiveCount =
            std::min(toggledPrimitiveCount + toggled, totalBudget);
      }
    }
  }

  bool anyActiveObject = false;
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    if (objectIndex >= _objectActive.size())
      continue;
    if (objectIndex >= _objectPrimitiveCounts.size())
      continue;
    if (_objectPrimitiveCounts[objectIndex] == 0)
      continue;
    if (_objectActive[objectIndex]) {
      anyActiveObject = true;
      break;
    }
  }

  if (!anyActiveObject) {
    for (size_t objectIndex : sortedIndices) {
      if (objectIndex >= _objectPrimitiveCounts.size())
        continue;
      if (_objectPrimitiveCounts[objectIndex] == 0)
        continue;
      if (setObjectActive(objectIndex, true) > 0) {
        changed = true;
        anyActiveObject = true;
      }
      if (anyActiveObject)
        break;
    }
  }

  if (!anyActiveObject && !_activePrimitive.empty()) {
    if (setPrimitiveActive(0, true))
      changed = true;
  }

  return changed;
}

bool Renderer::updateEnvironmentHitResidency(bool forceAllToggles) {
  size_t objectCount = _allSceneObjects.size();
  size_t primitiveCount = _activePrimitive.size();
  if (objectCount == 0 || primitiveCount == 0)
    return false;

  const float cameraMotion =
      clampUnit(static_cast<float>(std::min(_frameCameraMotionMetric, 1.0)));
  const double totalCapMB = effectiveTotalGpuMemoryCapMB();
  const double gpuMB = currentGPUMemoryMB();
  const float memoryPressure =
      (totalCapMB > 0.0)
          ? clampUnit(static_cast<float>((gpuMB / totalCapMB - 0.80) / 0.20))
          : 0.0f;
  const float adaptiveEnvironmentEscapeThreshold =
      std::clamp(_residencyConfig.environmentEscapeThreshold +
                     0.03f * cameraMotion + 0.08f * memoryPressure,
                 0.12f, 0.95f);
  const float adaptiveEnvironmentTargetFraction =
      std::clamp(_residencyConfig.environmentTargetActiveFraction +
                     0.08f * cameraMotion - 0.10f * memoryPressure,
                 0.12f, 0.94f);
  const size_t baseEnvironmentToggleBudget =
      std::max<size_t>(_residencyConfig.environmentMaxTogglesPerFrame, 1);
  const size_t adaptiveEnvironmentToggleBudget = std::max<size_t>(
      1, static_cast<size_t>(std::llround(
             static_cast<double>(baseEnvironmentToggleBudget) *
             (1.0 + 0.40 * static_cast<double>(cameraMotion) +
              0.70 * static_cast<double>(memoryPressure)))));

  _frameEnvironmentActivationFloor = 0;

  if (_objectProbabilitySortedIndices.size() != objectCount) {
    _objectProbabilitySortedIndices.resize(objectCount);
    std::iota(_objectProbabilitySortedIndices.begin(),
              _objectProbabilitySortedIndices.end(), size_t(0));
  }

  if (_objectImportance.size() != objectCount)
    _objectImportance.assign(objectCount, 0.0f);
  else
    std::fill(_objectImportance.begin(), _objectImportance.end(), 0.0f);

  if (_objectVisible.size() < objectCount)
    _objectVisible.resize(objectCount, 0u);

  std::vector<uint8_t> desiredObjectState(objectCount, 0);
  std::vector<float> weightedImportance(objectCount, 0.0f);

  if (_objectDepthWeight.size() < objectCount)
    _objectDepthWeight.resize(objectCount, 1.0f);
  if (_objectDepthWeightVersion.size() < objectCount)
    _objectDepthWeightVersion.resize(objectCount, 0);

  bool cameraChanged = _depthWeightCameraVersion != _cameraVersion;

  auto computeDepthWeight = [&](size_t objectIndex) -> float {
    const auto &weights = _residencyConfig.environmentDepthWeights;
    if (weights.empty())
      return 1.0f;
    const auto &radii = _residencyConfig.environmentDepthRadii;
    size_t pairCount = std::min(weights.size(), radii.size());
    if (pairCount == 0)
      return weights.back();
    float fallbackWeight = weights[pairCount - 1];
    if (objectIndex >= _objectBounds.size())
      return fallbackWeight;

    simd::float3 toCenter = _objectBounds[objectIndex].center - Camera::position;
    float distance = simd::length(toCenter);
    if (!(distance > 0.0f))
      return weights.front();

    auto end = radii.begin() + pairCount;
    auto lowerBound = std::lower_bound(radii.begin(), end, distance);
    if (lowerBound != end)
      return weights[std::distance(radii.begin(), lowerBound)];
    return fallbackWeight;
  };

  auto depthWeightForObject = [&](size_t objectIndex) -> float {
    uint64_t objectVersion =
        (objectIndex < _objectBoundsVersion.size()) ? _objectBoundsVersion[objectIndex] : 0;
    uint64_t cachedVersion =
        (objectIndex < _objectDepthWeightVersion.size()) ? _objectDepthWeightVersion[objectIndex]
                                                         : 0;

    if (cameraChanged || cachedVersion != objectVersion) {
      float depthWeight = computeDepthWeight(objectIndex);
      if (objectIndex < _objectDepthWeight.size())
        _objectDepthWeight[objectIndex] = depthWeight;
      if (objectIndex < _objectDepthWeightVersion.size())
        _objectDepthWeightVersion[objectIndex] = objectVersion;
      return depthWeight;
    }

    if (objectIndex < _objectDepthWeight.size())
      return _objectDepthWeight[objectIndex];
    return 1.0f;
  };

  constexpr float kExplorationPriorFloor = 1.0e-4f;
  constexpr float kExplorationPriorScale = 0.5f;

  constexpr float kVisibleWeightBoost = 2.0f;
  constexpr float kHiddenWeightPenalty = 0.05f;

  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    float hitProbability =
        (objectIndex < _objectHitProbability.size())
            ? std::clamp(_objectHitProbability[objectIndex], 0.0f, 1.0f)
            : 0.0f;
    bool hasEvidence =
        (objectIndex < _objectRaysTestedLastFrame.size())
            ? (_objectRaysTestedLastFrame[objectIndex] > 0)
            : false;
    bool currentlyActive =
        (objectIndex < _objectActive.size()) ? _objectActive[objectIndex] : false;

    bool visible =
        (objectIndex < _objectBounds.size()) ? isInView(_objectBounds[objectIndex])
                                             : false;
    if (objectIndex < _objectVisible.size())
      _objectVisible[objectIndex] = visible ? 1u : 0u;

    if (!hasEvidence && !currentlyActive) {
      float exploration =
          (objectIndex < _objectExplorationScore.size())
              ? std::max(_objectExplorationScore[objectIndex], 0.0f)
              : 0.0f;
      float visibilityHint =
          (objectIndex < _objectVisible.size()) ? _objectVisible[objectIndex] : 0u;
      float explorationPrior = 1.0f -
                               std::exp(-exploration * kExplorationPriorScale);
      if (visibilityHint) {
        float hintedPrior = 1.0f -
                            std::exp(-kIdleVisibleExploreSeed *
                                     kExplorationPriorScale);
        explorationPrior = std::max(explorationPrior, hintedPrior);
      }
      explorationPrior =
          std::clamp(explorationPrior, kExplorationPriorFloor, 1.0f);
      hitProbability = std::max(hitProbability, explorationPrior);
    }

    _objectImportance[objectIndex] = hitProbability;

    float depthWeight = depthWeightForObject(objectIndex);
    float visibilityWeight = visible ? kVisibleWeightBoost : kHiddenWeightPenalty;
    float weighted = hitProbability * std::max(depthWeight, 0.0f) *
                     std::max(visibilityWeight, 0.0f);
    weightedImportance[objectIndex] = std::isfinite(weighted) ? weighted : 0.0f;
  }

  _depthWeightCameraVersion = _cameraVersion;

  auto &sortedIndices = _objectProbabilitySortedIndices;
  std::sort(sortedIndices.begin(), sortedIndices.end(),
            [&](size_t a, size_t b) {
              float scoreA =
                  (a < weightedImportance.size()) ? weightedImportance[a] : 0.0f;
              float scoreB =
                  (b < weightedImportance.size()) ? weightedImportance[b] : 0.0f;
              if (scoreA == scoreB) {
                float importanceA =
                    (a < _objectImportance.size()) ? _objectImportance[a] : 0.0f;
                float importanceB =
                    (b < _objectImportance.size()) ? _objectImportance[b] : 0.0f;
                if (importanceA == importanceB)
                  return a < b;
                return importanceA > importanceB;
              }
              return scoreA > scoreB;
            });

  float escapeThreshold = adaptiveEnvironmentEscapeThreshold;
  size_t minActivePrimitives =
      std::min(primitiveCount, _residencyConfig.environmentMinActivePrimitives);
  if (minActivePrimitives == 0 && primitiveCount > 0)
    minActivePrimitives = 1;

  float targetFraction = adaptiveEnvironmentTargetFraction;
  size_t targetPrimitiveCount = 0;
  if (targetFraction > 0.0f && primitiveCount > 0) {
    targetPrimitiveCount = static_cast<size_t>(std::ceil(
        targetFraction * static_cast<float>(primitiveCount)));
    targetPrimitiveCount = std::min(targetPrimitiveCount, primitiveCount);
  }
  size_t activationFloor = std::max(minActivePrimitives, targetPrimitiveCount);

  float globalEscape = _lastFrameGlobalEnvEscape;
  float high = std::clamp(_residencyConfig.envHighEscapeThreshold, 0.0f, 1.0f);
  float low = std::clamp(_residencyConfig.envLowEscapeThreshold, 0.0f, high);

  if (globalEscape > high) {
    activationFloor = std::min(
        primitiveCount, static_cast<size_t>(activationFloor * 1.5f));
  } else if (globalEscape < low) {
    activationFloor = std::max(
        minActivePrimitives, static_cast<size_t>(activationFloor * 0.8f));
  }

  if (activationFloor == 0 && primitiveCount > 0)
    activationFloor = 1;
  _frameEnvironmentActivationFloor = activationFloor;

  size_t desiredPrimitiveCount = 0;
  float weightedEscape = 0.0f;
  auto averageEscape = [](size_t primCount, float weighted) {
    if (primCount == 0)
      return 1.0f;
    return weighted / static_cast<float>(primCount);
  };

  for (size_t objectIndex : sortedIndices) {
    if (objectIndex >= _allSceneObjects.size())
      continue;
    const SceneObject &object = _allSceneObjects[objectIndex];
    if (object.primitiveCount == 0)
      continue;

    bool needMorePrimitives = desiredPrimitiveCount < activationFloor;
    bool needsEscapeReduction =
        averageEscape(desiredPrimitiveCount, weightedEscape) > escapeThreshold;
    if (!needMorePrimitives && !needsEscapeReduction)
      break;

    desiredObjectState[objectIndex] = 1;
    desiredPrimitiveCount += object.primitiveCount;
    float escapeProbability =
        1.0f - ((objectIndex < _objectImportance.size())
                    ? _objectImportance[objectIndex]
                    : 0.0f);
    weightedEscape += escapeProbability * static_cast<float>(object.primitiveCount);
  }

  if (desiredPrimitiveCount == 0 && primitiveCount > 0) {
    for (size_t objectIndex : sortedIndices) {
      if (objectIndex >= _allSceneObjects.size())
        continue;
      const SceneObject &object = _allSceneObjects[objectIndex];
      if (object.primitiveCount == 0)
        continue;
      desiredObjectState[objectIndex] = 1;
      desiredPrimitiveCount = object.primitiveCount;
      float escapeProbability =
          1.0f - ((objectIndex < _objectImportance.size())
                      ? _objectImportance[objectIndex]
                      : 0.0f);
      weightedEscape =
          escapeProbability * static_cast<float>(object.primitiveCount);
      break;
    }
  }

  bool changed = false;
  size_t offscreenActivePrimitiveCount = 0;
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    bool active =
        (objectIndex < _objectActive.size()) ? _objectActive[objectIndex] : false;
    bool visible = (objectIndex < _objectVisible.size())
                       ? (_objectVisible[objectIndex] != 0)
                       : false;
    if (!active || visible)
      continue;

    size_t primitives = 0;
    if (objectIndex < _objectPrimitiveCounts.size())
      primitives = _objectPrimitiveCounts[objectIndex];
    else if (objectIndex < _allSceneObjects.size())
      primitives = _allSceneObjects[objectIndex].primitiveCount;

    offscreenActivePrimitiveCount += primitives;
  }

  size_t baseToggleBudget = adaptiveEnvironmentToggleBudget;
  size_t maxToggleBudget = baseToggleBudget;
  if (offscreenActivePrimitiveCount > baseToggleBudget) {
    size_t catchUpBudget = (offscreenActivePrimitiveCount + 1) / 2;
    maxToggleBudget = std::max(baseToggleBudget, catchUpBudget);
  }

  size_t reservedOffscreenBudget =
      (maxToggleBudget == 0) ? 0 : std::max<size_t>(1, maxToggleBudget / 4);
  size_t generalBudget =
      (maxToggleBudget > reservedOffscreenBudget)
          ? (maxToggleBudget - reservedOffscreenBudget)
          : 0;
  size_t totalBudget = maxToggleBudget + reservedOffscreenBudget;

  size_t toggledPrimitiveCount = 0;

  for (size_t objectIndex : sortedIndices) {
    bool shouldBeActive =
        (objectIndex < desiredObjectState.size()) ? desiredObjectState[objectIndex]
                                                  : 0;
    bool currentlyActive =
        (objectIndex < _objectActive.size()) ? _objectActive[objectIndex] : false;
    if (shouldBeActive == currentlyActive)
      continue;

    bool visible = (objectIndex < _objectVisible.size())
                       ? (_objectVisible[objectIndex] != 0)
                       : false;
    float importance =
        (objectIndex < _objectImportance.size()) ? _objectImportance[objectIndex]
                                                 : 0.0f;
    bool lowImportanceOffscreen = !visible && (importance < escapeThreshold);
    bool bypassCooldowns = currentlyActive && !shouldBeActive &&
                           lowImportanceOffscreen;

    if (!forceAllToggles && !bypassCooldowns && objectIndex < _objectCooldown.size() &&
        _objectCooldown[objectIndex] > 0)
      continue;

    const SceneObject &object = _allSceneObjects[objectIndex];
    size_t first = object.firstPrimitive;
    size_t last = std::min(first + object.primitiveCount, _activePrimitive.size());
    if (last <= first)
      continue;

    bool canToggle = true;
    size_t pendingToggles = 0;
    for (size_t prim = first; prim < last; ++prim) {
      if (_activePrimitive[prim] == shouldBeActive)
        continue;
      if (!forceAllToggles && !bypassCooldowns && prim < _primitiveCooldown.size() &&
          _primitiveCooldown[prim] > 0) {
        canToggle = false;
        break;
      }
      ++pendingToggles;
    }

    if (!canToggle || pendingToggles == 0)
      continue;

    if (!forceAllToggles) {
      if (bypassCooldowns) {
        if (toggledPrimitiveCount >= totalBudget)
          continue;
      } else {
        if (toggledPrimitiveCount >= generalBudget)
          continue;
      }
    }

      size_t toggled = setObjectActive(objectIndex, shouldBeActive);
      if (toggled > 0) {
        changed = true;
        if (!forceAllToggles) {
          toggledPrimitiveCount =
              std::min(toggledPrimitiveCount + toggled, totalBudget);
        }
      }
    }

  bool anyActiveObject = false;
  for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
    if (objectIndex >= _objectActive.size())
      continue;
    if (objectIndex >= _objectPrimitiveCounts.size())
      continue;
    if (_objectPrimitiveCounts[objectIndex] == 0)
      continue;
    if (_objectActive[objectIndex]) {
      anyActiveObject = true;
      break;
    }
  }

  if (!anyActiveObject) {
    for (size_t objectIndex : sortedIndices) {
      if (objectIndex >= _objectPrimitiveCounts.size())
        continue;
      if (_objectPrimitiveCounts[objectIndex] == 0)
        continue;
      if (setObjectActive(objectIndex, true) > 0) {
        changed = true;
        anyActiveObject = true;
      }
      if (anyActiveObject)
        break;
    }
  }

  if (!anyActiveObject && !_activePrimitive.empty()) {
    if (setPrimitiveActive(0, true))
      changed = true;
  }

  return changed;
}

// Propagate any pending primitive/object toggles to GPU memory.  The method
// skips expensive repacks if nothing changed, otherwise it forwards to
// rebuildResidentResources to patch only the dirty ranges (or rebuild
// everything when forced).
void Renderer::flushResidencyChanges(bool forceFullRebuild) {
  bool hasRecentChanges = !_recentlyActivated.empty() ||
                          !_recentlyDeactivated.empty() ||
                          !_dirtyResidentObjects.empty();
  const bool aggressiveEvict = _residencyConfig.rayHitAggressiveEvict;

  if (!forceFullRebuild && !hasRecentChanges) {
    for (size_t objectIndex = 0;
         objectIndex < _residentObjectGpuResources.size(); ++objectIndex) {
      auto &resident = _residentObjectGpuResources[objectIndex];
      auto &record = _instanceRecords[objectIndex];
      resident.clearPendingCommand();
      bool pending = resident.hasPendingCommands();

      if (!_residencyPreviewOnly &&
          _frameStrategy != ResidencyStrategy::AlwaysResident &&
          !resident.isResident() && !pending) {
        transitionResidentToCold(objectIndex, resident, record);
      }
    }
    return;
  }

  std::chrono::steady_clock::time_point frameWaitSnapshot;
  if (!waitForPendingFrameCommands(kFrameCommandBufferWaitTimeout,
                                   &frameWaitSnapshot)) {
    if (aggressiveEvict) {
      std::printf(
          "[Renderer] Ray-hit aggressive eviction: pending frame commands did not "
          "finish in time; skipping residency rebuild to avoid GPU hazards.\n");
    }
    return;
  }

  rebuildResidentResources(forceFullRebuild);
  _lastResidentFlushCameraVersion = _cameraVersion;
}

} // namespace NomadPathTracer
