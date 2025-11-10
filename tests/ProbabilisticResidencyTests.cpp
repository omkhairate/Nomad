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
};

class ProbabilityResidencyHarness {
public:
  explicit ProbabilityResidencyHarness(std::size_t primitiveCount)
      : _alpha(primitiveCount, 1.0f),
        _beta(primitiveCount, 1.0f),
        _probability(primitiveCount, 0.5f),
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

  void updateProbabilities(const std::vector<std::uint32_t> &hits) {
    const std::size_t primCount = _probability.size();
    assert(hits.size() == primCount);

    for (std::size_t i = 0; i < primCount; ++i) {
      float success = static_cast<float>(hits[i]);
      float failure = 1.0f;
      float alpha = _alpha[i] * config.decay + success;
      float beta = _beta[i] * config.decay + failure;
      _alpha[i] = alpha;
      _beta[i] = beta;
      float sum = alpha + beta;
      _probability[i] = (sum > 0.0f) ? (alpha / sum) : 0.5f;
    }
  }

  void applyDecisions() {
    const std::size_t primCount = _probability.size();
    std::vector<bool> desired(primCount, false);
    for (std::size_t i = 0; i < primCount; ++i)
      if (_probability[i] >= config.threshold)
        desired[i] = true;

    std::size_t minActive = std::min(config.minActive, primCount);
    std::iota(_sorted.begin(), _sorted.end(), std::size_t(0));
    std::sort(_sorted.begin(), _sorted.end(), [this](std::size_t a, std::size_t b) {
      float probA = sanitizeSortValue(_probability[a]);
      float probB = sanitizeSortValue(_probability[b]);
      if (probA == probB)
        return a < b;
      return probA > probB;
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
  std::vector<bool> _active;
  std::vector<std::size_t> _sorted;
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

} // namespace

void RunProbabilisticResidencyTests() {
  testPosteriorProbabilityTrend();
  testAlternatingHitsRespectDecayAndFallback();
  testZeroActiveFallbackEnsuresOnePrimitive();
}
