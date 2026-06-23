# UnifiedNeural Runtime

This note describes the renderer-side inference path for using the exported tiny MLP as the scoring backend for `ResidencyStrategy::UnifiedNeural`.

## Goal

At runtime, compute a learned per-object score:

```text
features(object, current clip/window state) -> predicted delta_cgvqm
```

Interpretation:

- high prediction: the object is costly to remove, so bias toward keeping it resident
- low prediction: the object is safer to demote or offload

## Exported Model Format

The exporter script:

- `Benchmarks/export_unified_neural_model.py`

writes JSON in a compact renderer-friendly format:

```json
{
  "format": "unified_neural_mlp_v1",
  "target": "delta_cgvqm",
  "target_transform": "exp_m1_of_linear_output",
  "input_normalization": {
    "features": ["total_object_hits", "..."],
    "mean": [0.0, "..."],
    "std": [1.0, "..."]
  },
  "network": {
    "hidden_dims": [16, 8],
    "layers": [
      {
        "type": "dense",
        "activation": "relu",
        "weights": [[...]],
        "bias": [...]
      },
      {
        "type": "dense",
        "activation": "relu",
        "weights": [[...]],
        "bias": [...]
      },
      {
        "type": "dense",
        "activation": "linear",
        "weights": [[...]],
        "bias": [...]
      }
    ]
  }
}
```

## Runtime Integration Path

### 1. Scene / config input

The renderer now accepts scene attributes such as:

```xml
<Scene residencyStrategy="unifiedNeural"
       unifiedNeuralModel="../Benchmarks/.../unified_neural_mlp_16x8.json"
       unifiedNeuralBlendWeight="0.5" />
```

Implemented attributes:

- `unifiedNeuralModel`
- `unifiedNeuralMode`
- `unifiedNeuralBlendWeight`

Example scene:

- `Nomad Path Tracer/scene_bistro_test_v2_unified_neural.xml`

### 2. Renderer-side model container

The renderer keeps a tiny CPU-side model container:

```cpp
struct NeuralDenseLayer {
  std::vector<float> weights; // row-major: out_dim * in_dim
  std::vector<float> bias;    // out_dim
  size_t inputDim = 0;
  size_t outputDim = 0;
  bool relu = false;
};

struct UnifiedNeuralModel {
  std::vector<std::string> featureNames;
  std::vector<float> mean;
  std::vector<float> stddev;
  std::vector<NeuralDenseLayer> layers;
  bool useExpM1Output = true;
  bool loaded = false;
};
```

### 3. Load once at scene startup

When the scene is loaded:

- parse the JSON file
- validate dimensions
- populate `_unifiedNeuralModel`
- if loading fails, fall back to heuristic unified scoring

### 4. Build the feature vector per object

Use the same feature ordering as the exported model.

For the current `16,8` model, the vector is:

1. `total_object_hits`
2. `mean_object_rayhit_score`
3. `mean_object_importance`
4. `mean_hit_probability`
5. `total_object_rays_tested`
6. `mean_distance`
7. `min_distance`
8. `mean_visible_coverage`
9. `max_visible_coverage`
10. `primitive_count`
11. `mean_estimated_object_bytes`

At runtime, most of these already exist or can be derived from the same state used by the feature logger.

Implemented helper:

```cpp
bool Renderer::buildUnifiedNeuralFeatureVector(
    size_t objectIndex,
    size_t primitiveCount,
    std::vector<float>& outFeatures);
```

### 5. Normalize

Apply:

```cpp
normalized[i] = (feature[i] - mean[i]) / max(stddev[i], 1e-6f);
```

### 6. Forward pass

Run a tiny CPU-side dense network:

```cpp
std::vector<float> activations(input.begin(), input.end());
for (const NeuralDenseLayer& layer : _unifiedNeuralModel.layers) {
  std::vector<float> out(layer.outputDim, 0.0f);
  for (size_t o = 0; o < layer.outputDim; ++o) {
    float sum = layer.bias[o];
    for (size_t i = 0; i < layer.inputDim; ++i)
      sum += layer.weights[o * layer.inputDim + i] * activations[i];
    if (layer.relu)
      sum = std::max(0.0f, sum);
    out[o] = sum;
  }
  activations.swap(out);
}
float y = activations.empty() ? 0.0f : activations[0];
if (_unifiedNeuralModel.useExpM1Output)
  y = std::max(0.0f, std::exp(y) - 1.0f);
```

That final `y` is the predicted `delta_cgvqm`.

### 7. Use prediction as the learned score

The current integration supports two modes:

- `unifiedNeuralMode="blend"`: normalized heuristic/neural blend
- `unifiedNeuralMode="neural_only"`: use the neural normalized score directly

Blend mode uses:

```cpp
heuristicNorm = normalize(heuristicUtility);
neuralNorm = normalize(predictedDeltaCgvqm);
finalScore = lerp(heuristicNorm, neuralNorm, blendWeight);
```

Neural-only mode uses:

```cpp
heuristicNorm = normalize(heuristicUtility);
neuralNorm = normalize(predictedDeltaCgvqm);
finalScore = neuralNorm;
```

That means:

- the old unified heuristic still matters
- the neural model nudges ranking instead of fully replacing it
- different score scales do not need to match exactly

## Recommended First Integration

For the first renderer-side experiment we now do:

1. load the exported `16,8` model
2. compute predicted `delta_cgvqm` per object
3. normalize heuristic and neural scores across candidates
4. blend them with `unifiedNeuralBlendWeight`
4. keep all current visibility / transport-critical guardrails intact

That gives a low-risk path:

- the MLP influences ranking
- existing hard safety rules still prevent catastrophic demotion

## Suggested Fallback Behavior

If the model file is missing, invalid, or expects unsupported features:

- log a warning once
- fall back to heuristic unified scoring inside `updateUnifiedResidency()`

That keeps scene files stable even when the model asset is absent.

## Current Best Tiny Model

From the architecture comparison on the 50-row dataset:

- best current small MLP: `16,8`

Use that exported JSON as the first runtime candidate.
