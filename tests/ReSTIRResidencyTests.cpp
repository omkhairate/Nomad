#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <random>
#include <vector>

namespace {

class RestirReservoirSimulator {
public:
  RestirReservoirSimulator(std::size_t primitiveCount, std::size_t reservoirSize,
                           std::size_t candidateCount, float reuseWeight,
                           std::uint32_t seed = 1337u)
      : _primitiveCount(primitiveCount),
        _reservoirSize(std::max<std::size_t>(1, reservoirSize)),
        _candidateCount(std::max<std::size_t>(1, candidateCount)),
        _reuseWeight(std::clamp(reuseWeight, 0.0f, 1.0f)), _rng(seed),
        _dist(0.0f, 1.0f) {}

  void step(const std::vector<float> &weights) {
    assert(weights.size() == _primitiveCount);

    std::vector<float> adjusted(weights.size(), 0.0f);
    float totalWeight = 0.0f;
    for (std::size_t i = 0; i < weights.size(); ++i) {
      float w = std::max(weights[i], 0.0f) + 1.0e-3f;
      w += 1.0e-3f * 0.25f * _dist(_rng);
      adjusted[i] = w;
      totalWeight += w;
    }

    if (!(totalWeight > 0.0f))
      return;

    auto drawKey = [&](float weight) {
      float u = std::max(_dist(_rng), 1.0e-6f);
      float key = std::pow(u, 1.0f / std::max(weight, 1.0e-6f));
      key += (_dist(_rng) - 0.5f) * 1.0e-4f;
      return key;
    };

    auto sampleIndex = [&](float r) {
      float accum = 0.0f;
      for (std::size_t i = 0; i < adjusted.size(); ++i) {
        accum += adjusted[i];
        if (r <= accum)
          return i;
      }
      return adjusted.size() - 1;
    };

    using HeapEntry = std::pair<float, std::pair<std::size_t, float>>;
    auto cmp = [](const HeapEntry &a, const HeapEntry &b) {
      return a.first > b.first;
    };
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, decltype(cmp)> heap(cmp);

    for (std::size_t i = 0; i < _reservoirIndices.size(); ++i) {
      std::size_t idx = _reservoirIndices[i];
      float weight = _reservoirWeights[i] * _reuseWeight;
      if (!(weight > 0.0f))
        continue;
      float key = drawKey(weight);
      if (heap.size() < _reservoirSize)
        heap.push({key, {idx, weight}});
      else if (key > heap.top().first) {
        heap.pop();
        heap.push({key, {idx, weight}});
      }
    }

    for (std::size_t c = 0; c < _candidateCount; ++c) {
      float r = _dist(_rng) * totalWeight;
      std::size_t idx = sampleIndex(r);
      float weight = adjusted[idx];
      float key = drawKey(weight);
      if (heap.size() < _reservoirSize)
        heap.push({key, {idx, weight}});
      else if (key > heap.top().first) {
        heap.pop();
        heap.push({key, {idx, weight}});
      }
    }

    _reservoirIndices.clear();
    _reservoirWeights.clear();
    while (!heap.empty()) {
      auto entry = heap.top();
      heap.pop();
      _reservoirIndices.push_back(entry.second.first);
      _reservoirWeights.push_back(entry.second.second);
    }

    ++_frameCount;
  }

  const std::vector<std::size_t> &reservoir() const { return _reservoirIndices; }
  std::size_t selectionsPerFrame() const { return _reservoirSize; }
  std::size_t framesSimulated() const { return _frameCount; }

private:
  std::size_t _primitiveCount = 0;
  std::size_t _reservoirSize = 1;
  std::size_t _candidateCount = 1;
  float _reuseWeight = 0.0f;
  std::mt19937 _rng;
  std::uniform_real_distribution<float> _dist;
  std::vector<std::size_t> _reservoirIndices;
  std::vector<float> _reservoirWeights;
  std::size_t _frameCount = 0;
};

} // namespace

void RunReSTIRResidencyTests() {
  constexpr std::size_t kPrimitiveCount = 32;
  RestirReservoirSimulator simulator(kPrimitiveCount, 6, 4, 0.6f);
  std::vector<float> uniformWeights(kPrimitiveCount, 1.0f);
  std::vector<std::size_t> hitCount(kPrimitiveCount, 0);

  constexpr std::size_t kFrames = 200;
  for (std::size_t frame = 0; frame < kFrames; ++frame) {
    simulator.step(uniformWeights);
    for (std::size_t idx : simulator.reservoir()) {
      assert(idx < hitCount.size());
      ++hitCount[idx];
    }
  }

  std::size_t uniqueHits = 0;
  for (std::size_t count : hitCount)
    if (count > 0)
      ++uniqueHits;
  assert(uniqueHits > kPrimitiveCount / 2);

  std::size_t totalSelections = simulator.selectionsPerFrame() * kFrames;
  std::size_t maxCount =
      *std::max_element(hitCount.begin(), hitCount.end());
  float maxShare = (totalSelections > 0)
                       ? static_cast<float>(maxCount) /
                             static_cast<float>(totalSelections)
                       : 0.0f;
  assert(maxShare < 0.35f);
}

