#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace {

struct ProbabilityResidencyConfig {
  float decay = 0.9f;
  float threshold = 0.5f;
  std::size_t minActive = 16;
  std::size_t maxToggles = 16;
  float uncertaintyBoost = 0.25f;
  float evidenceWindow = 64.0f;
};

class ProbabilityResidencyHarness {
public:
  explicit ProbabilityResidencyHarness(std::size_t primitiveCount)
      : _alpha(primitiveCount, 1.0f),
        _beta(primitiveCount, 1.0f),
        _probability(primitiveCount, 0.5f),
        _variance(primitiveCount, 1.0f / 12.0f),
        _mass(primitiveCount, 2.0f),
        _active(primitiveCount, false),
        _sorted(primitiveCount) {
    std::iota(_sorted.begin(), _sorted.end(), std::size_t(0));
  }

  ProbabilityResidencyConfig config;

  void advanceFrame(const std::vector<std::uint32_t> &hits) {
    updateProbabilities(hits);
    applyDecisions();
  }

  float probabilityAt(std::size_t index) const { return _probability[index]; }

  float regressedProbabilityAt(std::size_t index) const {
    return computeRegressedProbability(index);
  }

  bool isActive(std::size_t index) const { return _active[index]; }

  std::size_t activeCount() const {
    return static_cast<std::size_t>(
        std::count(_active.begin(), _active.end(), true));
  }

private:
  static float sanitizeSortValue(float value) {
    if (!std::isfinite(value))
      return -std::numeric_limits<float>::max();
    return value;
  }

  static float sanitizeProbability(float value) {
    if (!std::isfinite(value))
      return 0.5f;
    return std::clamp(value, 0.0f, 1.0f);
  }

  float computeEvidence(float mass) const {
    if (!(mass > 0.0f) || !std::isfinite(mass))
      return 0.0f;
    float window = config.evidenceWindow > 0.0f
                       ? std::max(config.evidenceWindow, 1.0e-3f)
                       : 64.0f;
    if (config.evidenceWindow > 0.0f)
      return std::clamp(mass / window, 0.0f, 1.0f);
    return std::clamp(mass / (mass + window), 0.0f, 1.0f);
  }

  float computeRegressedProbability(std::size_t index) const {
    float probability = sanitizeProbability(_probability[index]);
    float evidence = computeEvidence(_mass[index]);
    return probability * evidence + 0.5f * (1.0f - evidence);
  }

  float computePosteriorScore(std::size_t index) const {
    float regressed = computeRegressedProbability(index);
    float variance = _variance[index];
    if (!std::isfinite(variance) || variance < 0.0f)
      variance = 0.0f;
    float sqrtVar = std::sqrt(variance);
    return regressed + config.uncertaintyBoost * sqrtVar;
  }

  void updateProbabilities(const std::vector<std::uint32_t> &hits) {
    const std::size_t primCount = _probability.size();
    assert(hits.size() == primCount);

    for (std::size_t i = 0; i < primCount; ++i) {
      float success = static_cast<float>(hits[i]);
      float failure = 1.0f;
      float alpha = _alpha[i] * config.decay + success;
      float beta = _beta[i] * config.decay + failure;
      float sum = alpha + beta;
      constexpr float kPosteriorFloor = 1.0e-3f;
      if (!(sum > 0.0f)) {
        alpha = beta = kPosteriorFloor * 0.5f;
        sum = kPosteriorFloor;
      }
      if (sum < kPosteriorFloor) {
        float scale = kPosteriorFloor / sum;
        alpha *= scale;
        beta *= scale;
        sum = kPosteriorFloor;
      }
      if (config.evidenceWindow > 0.0f) {
        float maxMass = std::max(config.evidenceWindow, kPosteriorFloor);
        if (sum > maxMass) {
          float scale = maxMass / sum;
          alpha *= scale;
          beta *= scale;
          sum = maxMass;
        }
      }
      _alpha[i] = alpha;
      _beta[i] = beta;
      _mass[i] = sum;
      _probability[i] = (sum > 0.0f) ? (alpha / sum) : 0.5f;
      if (sum > 1.0f)
        _variance[i] = (alpha * beta) / ((sum * sum) * (sum + 1.0f));
      else
        _variance[i] = 0.0f;
    }
  }

  void applyDecisions() {
    const std::size_t primCount = _probability.size();
    std::vector<bool> desired(primCount, false);
    for (std::size_t i = 0; i < primCount; ++i)
      if (computeRegressedProbability(i) >= config.threshold)
        desired[i] = true;

    std::size_t minActive = std::min(config.minActive, primCount);
    std::iota(_sorted.begin(), _sorted.end(), std::size_t(0));
    std::sort(_sorted.begin(), _sorted.end(), [this](std::size_t a, std::size_t b) {
      float scoreA = sanitizeSortValue(computePosteriorScore(a));
      float scoreB = sanitizeSortValue(computePosteriorScore(b));
      if (scoreA == scoreB)
        return a < b;
      return scoreA > scoreB;
    });

    for (std::size_t i = 0; i < minActive && i < _sorted.size(); ++i)
      desired[_sorted[i]] = true;

    std::size_t toggles = 0;
    for (std::size_t i = 0; i < primCount; ++i) {
      bool shouldBeActive = desired[i];
      if (shouldBeActive == _active[i])
        continue;
      if (toggles >= config.maxToggles)
        break;
      _active[i] = shouldBeActive;
      ++toggles;
    }

    if (activeCount() == 0 && primCount > 0)
      _active[_sorted.front()] = true;
  }

  std::vector<float> _alpha;
  std::vector<float> _beta;
  std::vector<float> _probability;
  std::vector<float> _variance;
  std::vector<float> _mass;
  std::vector<bool> _active;
  std::vector<std::size_t> _sorted;
};

class ObjectFallbackHarness {
public:
  explicit ObjectFallbackHarness(const std::vector<std::size_t> &objectPrimitiveCounts)
      : _objectPrimitiveCounts(objectPrimitiveCounts),
        _objectOffsets(objectPrimitiveCounts.size(), 0) {
    std::size_t totalPrimitives = 0;
    for (std::size_t i = 0; i < objectPrimitiveCounts.size(); ++i) {
      _objectOffsets[i] = totalPrimitives;
      totalPrimitives += objectPrimitiveCounts[i];
    }
    _activePrimitive.assign(totalPrimitives, false);
    _objectActive.assign(objectPrimitiveCounts.size(), false);
  }

  void setMaxPrimitiveToggles(std::size_t maxToggles) {
    _maxPrimitiveToggles = maxToggles;
  }

  void resetFrameCounters() {
    _toggledPrimitiveCount = 0;
    _frameProbabilisticToggles = 0;
  }

  std::size_t setObjectActive(std::size_t objectIndex, bool shouldBeActive) {
    if (objectIndex >= _objectActive.size())
      return 0;
    bool currentlyActive = _objectActive[objectIndex];
    if (currentlyActive == shouldBeActive)
      return 0;
    std::size_t start = _objectOffsets[objectIndex];
    std::size_t count = _objectPrimitiveCounts[objectIndex];
    std::size_t toggled = 0;
    for (std::size_t i = 0; i < count; ++i) {
      std::size_t primIndex = start + i;
      if (primIndex >= _activePrimitive.size())
        break;
      if (_activePrimitive[primIndex] != shouldBeActive) {
        _activePrimitive[primIndex] = shouldBeActive;
        ++toggled;
      }
    }
    _objectActive[objectIndex] = shouldBeActive;
    return toggled;
  }

  std::size_t activePrimitiveCount() const {
    return static_cast<std::size_t>(
        std::count(_activePrimitive.begin(), _activePrimitive.end(), true));
  }

  bool applyFallback(bool forceAllToggles, std::size_t fallbackCandidate) {
    std::size_t objectCount = _objectActive.size();
    if (activePrimitiveCount() != 0 || objectCount == 0)
      return false;

    std::size_t fallback =
        fallbackCandidate < objectCount ? fallbackCandidate : std::size_t(0);
    if (fallback >= objectCount)
      fallback = objectCount - 1;
    std::size_t remainingBudget =
        (_maxPrimitiveToggles > _toggledPrimitiveCount)
            ? (_maxPrimitiveToggles - _toggledPrimitiveCount)
            : std::size_t(0);
    std::size_t toggled = setObjectActive(fallback, true);
    if (toggled == 0)
      return false;

    if (!forceAllToggles) {
      std::size_t applied = std::min(toggled, remainingBudget);
      _toggledPrimitiveCount += applied;
    }
    _frameProbabilisticToggles += toggled;
    return true;
  }

  std::size_t frameProbabilisticToggles() const {
    return _frameProbabilisticToggles;
  }

  std::size_t toggledPrimitiveCount() const { return _toggledPrimitiveCount; }

private:
  std::vector<std::size_t> _objectPrimitiveCounts;
  std::vector<std::size_t> _objectOffsets;
  std::vector<bool> _activePrimitive;
  std::vector<bool> _objectActive;
  std::size_t _maxPrimitiveToggles = 0;
  std::size_t _toggledPrimitiveCount = 0;
  std::size_t _frameProbabilisticToggles = 0;
};

void testPosteriorProbabilityTrend() {
  ProbabilityResidencyHarness harness(1);
  harness.config.decay = 0.9f;
  harness.config.threshold = 0.6f;
  harness.config.minActive = 1;
  harness.config.maxToggles = 4;

  std::vector<float> history;
  float alpha = 1.0f;
  float beta = 1.0f;
  const std::vector<std::uint32_t> frames = {4, 4, 4};

  for (std::uint32_t hits : frames) {
    harness.advanceFrame({hits});
    alpha = alpha * harness.config.decay + static_cast<float>(hits);
    beta = beta * harness.config.decay + 1.0f;
    float expected = alpha / (alpha + beta);
    float actual = harness.probabilityAt(0);
    history.push_back(actual);
    assert(std::fabs(actual - expected) < 1e-5f);
    assert(harness.isActive(0));
  }

  assert(history[1] > history[0]);
  assert(history[2] > history[1]);
}

void testAlternatingHitsRespectDecayAndFallback() {
  ProbabilityResidencyHarness harness(3);
  harness.config.decay = 0.8f;
  harness.config.threshold = 0.65f;
  harness.config.minActive = 2;
  harness.config.maxToggles = 3;

  std::vector<float> alpha(3, 1.0f);
  std::vector<float> beta(3, 1.0f);

  auto step = [&](const std::vector<std::uint32_t> &hits) {
    harness.advanceFrame(hits);
    for (std::size_t i = 0; i < hits.size(); ++i) {
      alpha[i] = alpha[i] * harness.config.decay + static_cast<float>(hits[i]);
      beta[i] = beta[i] * harness.config.decay + 1.0f;
      float expected = alpha[i] / (alpha[i] + beta[i]);
      float actual = harness.probabilityAt(i);
      assert(std::fabs(actual - expected) < 1e-5f);
    }
  };

  step({5, 0, 0});
  assert(harness.isActive(0));
  assert(harness.isActive(1));
  assert(!harness.isActive(2));
  float firstP0 = harness.probabilityAt(0);

  step({0, 5, 0});
  assert(harness.isActive(0));
  assert(harness.isActive(1));
  assert(!harness.isActive(2));
  float secondP0 = harness.probabilityAt(0);
  assert(secondP0 < firstP0);

  step({0, 0, 0});
  assert(harness.activeCount() == 2);
  assert(harness.isActive(0));
  assert(harness.isActive(1));
  assert(!harness.isActive(2));
  float thirdP0 = harness.probabilityAt(0);

  step({5, 0, 0});
  assert(harness.probabilityAt(0) > thirdP0);
  assert(harness.activeCount() == 2);
  assert(harness.isActive(0));
  assert(harness.isActive(1));
  assert(!harness.isActive(2));
}

void testZeroActiveFallbackEnsuresOnePrimitive() {
  ProbabilityResidencyHarness harness(2);
  harness.config.decay = 0.75f;
  harness.config.threshold = 0.95f;
  harness.config.minActive = 0;
  harness.config.maxToggles = 4;

  harness.advanceFrame({0, 0});
  assert(harness.activeCount() == 1);
  assert(harness.isActive(0));
  assert(!harness.isActive(1));
}

void testObjectFallbackOverridesToggleBudget() {
  ObjectFallbackHarness harness({1, 1});
  harness.setMaxPrimitiveToggles(0);
  harness.resetFrameCounters();

  harness.setObjectActive(0, true);
  harness.setObjectActive(0, false);
  harness.setObjectActive(1, false);

  bool changed = harness.applyFallback(false, 0);
  assert(changed);
  assert(harness.activePrimitiveCount() >= 1);
  assert(harness.frameProbabilisticToggles() == 1);
  assert(harness.toggledPrimitiveCount() == 0);
}

void testLowEvidenceHysteresisDemotion() {
  constexpr float kThreshold = 0.6f;
  constexpr float kHysteresis = 0.1f;
  const float enterThreshold = std::clamp(kThreshold + kHysteresis, 0.0f, 1.0f);
  const float exitThreshold = std::clamp(kThreshold - kHysteresis, 0.0f, 1.0f);
  constexpr float kEvidenceWindow = 64.0f;
  constexpr float kMinimalEvidenceThreshold = 1.0e-3f;

  auto sanitizeProbability = [](float probability) {
    if (!std::isfinite(probability))
      return 0.5f;
    return std::clamp(probability, 0.0f, 1.0f);
  };

  auto computeEvidenceFactor = [=](float mass) {
    if (!(mass > 0.0f) || !std::isfinite(mass))
      return 0.0f;
    float window = std::max(kEvidenceWindow, 1.0e-3f);
    float normalized = mass / window;
    return std::clamp(normalized, 0.0f, 1.0f);
  };

  auto evaluateDesiredState = [&](float probability, float mass,
                                  bool previousDesired,
                                  uint32_t cooldown) {
    float sanitizedProbability = sanitizeProbability(probability);
    float evidence = computeEvidenceFactor(mass);
    float regressedProbability = sanitizedProbability * evidence +
                                 0.5f * (1.0f - evidence);
    bool desired = previousDesired;
    bool cooldownExpired = cooldown == 0;
    bool lowEvidence = evidence <= kMinimalEvidenceThreshold;
    float evaluationProbability =
        lowEvidence ? regressedProbability : sanitizedProbability;
    if (sanitizedProbability >= enterThreshold)
      desired = true;
    else if (regressedProbability <= exitThreshold)
      desired = false;
    else if (cooldownExpired && !previousDesired)
      desired = evaluationProbability >= kThreshold;
    return desired;
  };

  bool desired = true;
  uint32_t cooldown = 0;
  constexpr float kLowEvidenceMass = kMinimalEvidenceThreshold * 0.5f;

  desired = evaluateDesiredState(0.65f, kLowEvidenceMass, desired, cooldown);
  assert(desired);

  desired = evaluateDesiredState(0.4f, kLowEvidenceMass, desired, cooldown);
  assert(!desired);

  desired = evaluateDesiredState(0.75f, kLowEvidenceMass, desired, cooldown);
  assert(desired);

  desired = evaluateDesiredState(0.62f, kLowEvidenceMass, desired, cooldown);
  assert(desired);
}

void testIdleResidencyCooldownClearsDesiredState() {
  constexpr uint32_t kStateCooldownFrames = 3;
  constexpr uint32_t kIdleCooldownFrames = 2;
  constexpr float kThreshold = 0.55f;
  constexpr float kHysteresis = 0.05f;
  const float enterThreshold = std::clamp(kThreshold + kHysteresis, 0.0f, 1.0f);
  const float exitThreshold = std::clamp(kThreshold - kHysteresis, 0.0f, 1.0f);
  constexpr float kEvidenceWindow = 64.0f;
  constexpr float kMinimalEvidenceThreshold = 1.0e-3f;

  auto sanitizeProbability = [](float probability) {
    if (!std::isfinite(probability))
      return 0.5f;
    return std::clamp(probability, 0.0f, 1.0f);
  };

  auto computeEvidenceFactor = [=](float mass) {
    if (!(mass > 0.0f) || !std::isfinite(mass))
      return 0.0f;
    float window = std::max(kEvidenceWindow, 1.0e-3f);
    float normalized = mass / window;
    return std::clamp(normalized, 0.0f, 1.0f);
  };

  auto evaluateDesiredState = [&](float probability, float mass,
                                  bool previousDesired,
                                  uint32_t cooldown) {
    float sanitizedProbability = sanitizeProbability(probability);
    float evidence = computeEvidenceFactor(mass);
    float regressedProbability = sanitizedProbability * evidence +
                                 0.5f * (1.0f - evidence);
    bool desired = previousDesired;
    bool cooldownExpired = cooldown == 0;
    bool lowEvidence = evidence <= kMinimalEvidenceThreshold;
    float evaluationProbability =
        lowEvidence ? regressedProbability : sanitizedProbability;
    if (sanitizedProbability >= enterThreshold)
      desired = true;
    else if (regressedProbability <= exitThreshold)
      desired = false;
    else if (cooldownExpired && !previousDesired)
      desired = evaluationProbability >= kThreshold;
    return desired;
  };

  bool desired = true;
  uint32_t cooldown = kStateCooldownFrames;
  const uint32_t totalIdleFrames = kStateCooldownFrames + kIdleCooldownFrames;
  float probability = kThreshold;
  float mass = kEvidenceWindow;

  for (uint32_t frame = 0; frame < totalIdleFrames; ++frame) {
    float frameMass = mass;
    if (frame + 1 == totalIdleFrames)
      frameMass = 1.0e-6f;
    bool next = evaluateDesiredState(probability, frameMass, desired, cooldown);
    if (frame + 1 < totalIdleFrames)
      assert(next);
    desired = next;
    if (cooldown > 0)
      --cooldown;
  }

  assert(!desired);
}

} // namespace

void RunProbabilisticResidencyTests() {
  testPosteriorProbabilityTrend();
  testAlternatingHitsRespectDecayAndFallback();
  testZeroActiveFallbackEnsuresOnePrimitive();
  testObjectFallbackOverridesToggleBudget();
  testLowEvidenceHysteresisDemotion();
  testIdleResidencyCooldownClearsDesiredState();
}
