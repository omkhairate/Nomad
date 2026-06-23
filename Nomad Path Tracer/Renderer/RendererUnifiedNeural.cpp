#include "Renderer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

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

    outError = "Unexpected token in JSON input.";
    return false;
  }

  bool parseObject(TinyJsonValue &outValue, std::string &outError) {
    if (!consume('{')) {
      outError = "Expected '{'.";
      return false;
    }
    outValue = TinyJsonValue{};
    outValue.type = TinyJsonValue::Type::Object;

    skipWhitespace();
    if (consume('}'))
      return true;

    while (true) {
      TinyJsonValue keyValue;
      if (!parseStringValue(keyValue, outError))
        return false;
      skipWhitespace();
      if (!consume(':')) {
        outError = "Expected ':' after object key.";
        return false;
      }
      TinyJsonValue value;
      if (!parseValue(value, outError))
        return false;
      outValue.objectValue[keyValue.stringValue] = std::move(value);
      skipWhitespace();
      if (consume('}'))
        return true;
      if (!consume(',')) {
        outError = "Expected ',' or '}' in object.";
        return false;
      }
    }
  }

  bool parseArray(TinyJsonValue &outValue, std::string &outError) {
    if (!consume('[')) {
      outError = "Expected '['.";
      return false;
    }
    outValue = TinyJsonValue{};
    outValue.type = TinyJsonValue::Type::Array;

    skipWhitespace();
    if (consume(']'))
      return true;

    while (true) {
      TinyJsonValue value;
      if (!parseValue(value, outError))
        return false;
      outValue.arrayValue.push_back(std::move(value));
      skipWhitespace();
      if (consume(']'))
        return true;
      if (!consume(',')) {
        outError = "Expected ',' or ']' in array.";
        return false;
      }
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
    if (!consume('"')) {
      outError = "Expected string literal.";
      return false;
    }

    outString.clear();
    while (_pos < _text.size()) {
      const char c = _text[_pos++];
      if (c == '"')
        return true;
      if (c == '\\') {
        if (_pos >= _text.size()) {
          outError = "Unterminated escape sequence in string.";
          return false;
        }
        const char esc = _text[_pos++];
        switch (esc) {
        case '"':
        case '\\':
        case '/':
          outString.push_back(esc);
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
        case 'u': {
          if (_pos + 4 > _text.size()) {
            outError = "Incomplete unicode escape in string.";
            return false;
          }
          unsigned int codePoint = 0;
          for (int i = 0; i < 4; ++i) {
            const char hex = _text[_pos++];
            codePoint <<= 4;
            if (hex >= '0' && hex <= '9')
              codePoint |= static_cast<unsigned int>(hex - '0');
            else if (hex >= 'a' && hex <= 'f')
              codePoint |= static_cast<unsigned int>(hex - 'a' + 10);
            else if (hex >= 'A' && hex <= 'F')
              codePoint |= static_cast<unsigned int>(hex - 'A' + 10);
            else {
              outError = "Invalid unicode escape in string.";
              return false;
            }
          }
          if (codePoint <= 0x7F) {
            outString.push_back(static_cast<char>(codePoint));
          } else {
            outError = "Only ASCII unicode escapes are supported.";
            return false;
          }
          break;
        }
        default:
          outError = "Unsupported escape sequence in string.";
          return false;
        }
      } else {
        outString.push_back(c);
      }
    }

    outError = "Unterminated string literal.";
    return false;
  }

  bool parseNumber(TinyJsonValue &outValue, std::string &outError) {
    const size_t start = _pos;
    if (peek('-'))
      ++_pos;
    if (_pos >= _text.size()) {
      outError = "Unexpected end of number.";
      return false;
    }
    if (_text[_pos] == '0') {
      ++_pos;
    } else {
      if (!std::isdigit(static_cast<unsigned char>(_text[_pos]))) {
        outError = "Invalid number literal.";
        return false;
      }
      while (_pos < _text.size() &&
             std::isdigit(static_cast<unsigned char>(_text[_pos]))) {
        ++_pos;
      }
    }
    if (peek('.')) {
      ++_pos;
      if (_pos >= _text.size() ||
          !std::isdigit(static_cast<unsigned char>(_text[_pos]))) {
        outError = "Invalid fractional part in number.";
        return false;
      }
      while (_pos < _text.size() &&
             std::isdigit(static_cast<unsigned char>(_text[_pos]))) {
        ++_pos;
      }
    }
    if (peek('e') || peek('E')) {
      ++_pos;
      if (peek('+') || peek('-'))
        ++_pos;
      if (_pos >= _text.size() ||
          !std::isdigit(static_cast<unsigned char>(_text[_pos]))) {
        outError = "Invalid exponent in number.";
        return false;
      }
      while (_pos < _text.size() &&
             std::isdigit(static_cast<unsigned char>(_text[_pos]))) {
        ++_pos;
      }
    }

    char *endPtr = nullptr;
    const std::string token = _text.substr(start, _pos - start);
    const double value = std::strtod(token.c_str(), &endPtr);
    if (!endPtr || *endPtr != '\0' || !std::isfinite(value)) {
      outError = "Invalid numeric value.";
      return false;
    }

    outValue = TinyJsonValue{};
    outValue.type = TinyJsonValue::Type::Number;
    outValue.numberValue = value;
    return true;
  }

  bool matchLiteral(const char *literal) {
    size_t offset = 0;
    while (literal[offset]) {
      if (_pos + offset >= _text.size() || _text[_pos + offset] != literal[offset])
        return false;
      ++offset;
    }
    _pos += offset;
    return true;
  }

  bool consume(char expected) {
    skipWhitespace();
    if (_pos < _text.size() && _text[_pos] == expected) {
      ++_pos;
      return true;
    }
    return false;
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

const TinyJsonValue *findJsonMember(const TinyJsonValue &value, const char *name) {
  if (value.type != TinyJsonValue::Type::Object)
    return nullptr;
  auto it = value.objectValue.find(name);
  if (it == value.objectValue.end())
    return nullptr;
  return &it->second;
}

bool jsonStringValue(const TinyJsonValue &value, std::string &outString) {
  if (value.type != TinyJsonValue::Type::String)
    return false;
  outString = value.stringValue;
  return true;
}

bool jsonNumberValue(const TinyJsonValue &value, float &outNumber) {
  if (value.type != TinyJsonValue::Type::Number)
    return false;
  outNumber = static_cast<float>(value.numberValue);
  return std::isfinite(outNumber);
}

bool jsonBoolValue(const TinyJsonValue &value, bool &outBool) {
  if (value.type != TinyJsonValue::Type::Bool)
    return false;
  outBool = value.boolValue;
  return true;
}

bool jsonStringArray(const TinyJsonValue &value, std::vector<std::string> &out) {
  if (value.type != TinyJsonValue::Type::Array)
    return false;
  out.clear();
  out.reserve(value.arrayValue.size());
  for (const TinyJsonValue &entry : value.arrayValue) {
    if (entry.type != TinyJsonValue::Type::String)
      return false;
    out.push_back(entry.stringValue);
  }
  return true;
}

bool jsonNumberArray(const TinyJsonValue &value, std::vector<float> &out) {
  if (value.type != TinyJsonValue::Type::Array)
    return false;
  out.clear();
  out.reserve(value.arrayValue.size());
  for (const TinyJsonValue &entry : value.arrayValue) {
    float number = 0.0f;
    if (!jsonNumberValue(entry, number))
      return false;
    out.push_back(number);
  }
  return true;
}

} // namespace

namespace NomadPathTracer {

void Renderer::clearUnifiedNeuralModel() {
  _unifiedNeuralModel = UnifiedNeuralModel{};
  _unifiedNeuralModelLoadAttempted = false;
}

bool Renderer::loadUnifiedNeuralModel(const std::string &path) {
  _unifiedNeuralModel = UnifiedNeuralModel{};
  _unifiedNeuralModelLoadAttempted = true;

  if (path.empty()) {
    std::printf(
        "[Renderer] UnifiedNeural selected but no unifiedNeuralModel path was "
        "provided; falling back to heuristic unified scoring.\n");
    return false;
  }

  std::ifstream input(path);
  if (!input.is_open()) {
    std::printf(
        "[Renderer] Failed to open UnifiedNeural model JSON at %s; falling "
        "back to heuristic unified scoring.\n",
        path.c_str());
    return false;
  }

  std::ostringstream buffer;
  buffer << input.rdbuf();
  const std::string text = buffer.str();

  TinyJsonParser parser(text);
  TinyJsonValue root;
  std::string error;
  if (!parser.parse(root, error)) {
    std::printf(
        "[Renderer] Failed to parse UnifiedNeural model JSON at %s: %s\n",
        path.c_str(), error.c_str());
    return false;
  }

  const TinyJsonValue *formatValue = findJsonMember(root, "format");
  const TinyJsonValue *targetValue = findJsonMember(root, "target");
  const TinyJsonValue *targetTransformValue =
      findJsonMember(root, "target_transform");
  const TinyJsonValue *inputNormalizationValue =
      findJsonMember(root, "input_normalization");
  const TinyJsonValue *networkValue = findJsonMember(root, "network");
  if (!formatValue || !targetValue || !targetTransformValue ||
      !inputNormalizationValue || !networkValue) {
    std::printf(
        "[Renderer] UnifiedNeural model JSON at %s is missing required top-level "
        "fields.\n",
        path.c_str());
    return false;
  }

  UnifiedNeuralModel loadedModel;
  loadedModel.sourcePath = path;
  if (!jsonStringValue(*formatValue, loadedModel.format) ||
      !jsonStringValue(*targetValue, loadedModel.target) ||
      !jsonStringValue(*targetTransformValue, loadedModel.targetTransform)) {
    std::printf(
        "[Renderer] UnifiedNeural model JSON at %s has invalid metadata types.\n",
        path.c_str());
    return false;
  }

  const TinyJsonValue *trainingValue = findJsonMember(root, "training");
  if (trainingValue) {
    const TinyJsonValue *allowNegativeValue =
        findJsonMember(*trainingValue, "allow_negative_target");
    if (allowNegativeValue &&
        !jsonBoolValue(*allowNegativeValue, loadedModel.allowNegativeTarget)) {
      std::printf(
          "[Renderer] UnifiedNeural training metadata at %s has invalid "
          "allow_negative_target flag.\n",
          path.c_str());
      return false;
    }
  }

  const TinyJsonValue *residualTeacherValue =
      findJsonMember(root, "residual_teacher");
  if (residualTeacherValue) {
    const TinyJsonValue *kindValue = findJsonMember(*residualTeacherValue, "kind");
    const TinyJsonValue *scaleValue = findJsonMember(*residualTeacherValue, "scale");
    const TinyJsonValue *biasValue = findJsonMember(*residualTeacherValue, "bias");
    if (!kindValue || !scaleValue || !biasValue ||
        !jsonStringValue(*kindValue, loadedModel.residualTeacher.kind) ||
        !jsonNumberValue(*scaleValue, loadedModel.residualTeacher.scale) ||
        !jsonNumberValue(*biasValue, loadedModel.residualTeacher.bias)) {
      std::printf(
          "[Renderer] UnifiedNeural residual_teacher block at %s is invalid.\n",
          path.c_str());
      return false;
    }

    const TinyJsonValue *costExponentValue =
        findJsonMember(*residualTeacherValue, "cost_exponent");
    const TinyJsonValue *visibilityWeightValue =
        findJsonMember(*residualTeacherValue, "visibility_weight");
    const TinyJsonValue *hitProbabilityWeightValue =
        findJsonMember(*residualTeacherValue, "hit_probability_weight");
    if (costExponentValue &&
        !jsonNumberValue(*costExponentValue,
                         loadedModel.residualTeacher.costExponent)) {
      std::printf(
          "[Renderer] UnifiedNeural residual teacher cost exponent at %s is "
          "invalid.\n",
          path.c_str());
      return false;
    }
    if (visibilityWeightValue &&
        !jsonNumberValue(*visibilityWeightValue,
                         loadedModel.residualTeacher.visibilityWeight)) {
      std::printf(
          "[Renderer] UnifiedNeural residual teacher visibility weight at %s "
          "is invalid.\n",
          path.c_str());
      return false;
    }
    if (hitProbabilityWeightValue &&
        !jsonNumberValue(*hitProbabilityWeightValue,
                         loadedModel.residualTeacher.hitProbabilityWeight)) {
      std::printf(
          "[Renderer] UnifiedNeural residual teacher hit-probability weight at "
          "%s is invalid.\n",
          path.c_str());
      return false;
    }
    loadedModel.residualTeacher.enabled = true;
  }

  const TinyJsonValue *featuresValue =
      findJsonMember(*inputNormalizationValue, "features");
  const TinyJsonValue *meanValue = findJsonMember(*inputNormalizationValue, "mean");
  const TinyJsonValue *stdValue = findJsonMember(*inputNormalizationValue, "std");
  if (!featuresValue || !meanValue || !stdValue ||
      !jsonStringArray(*featuresValue, loadedModel.featureNames) ||
      !jsonNumberArray(*meanValue, loadedModel.mean) ||
      !jsonNumberArray(*stdValue, loadedModel.stddev)) {
    std::printf(
        "[Renderer] UnifiedNeural input normalization block at %s is invalid.\n",
        path.c_str());
    return false;
  }

  if (loadedModel.featureNames.empty() ||
      loadedModel.featureNames.size() != loadedModel.mean.size() ||
      loadedModel.featureNames.size() != loadedModel.stddev.size()) {
    std::printf(
        "[Renderer] UnifiedNeural feature metadata at %s has inconsistent "
        "dimensions.\n",
        path.c_str());
    return false;
  }

  const TinyJsonValue *layersValue = findJsonMember(*networkValue, "layers");
  if (!layersValue || layersValue->type != TinyJsonValue::Type::Array ||
      layersValue->arrayValue.empty()) {
    std::printf("[Renderer] UnifiedNeural network at %s has no layers.\n",
                path.c_str());
    return false;
  }

  size_t expectedCols = loadedModel.featureNames.size();
  for (const TinyJsonValue &layerValue : layersValue->arrayValue) {
    const TinyJsonValue *typeValue = findJsonMember(layerValue, "type");
    const TinyJsonValue *activationValue =
        findJsonMember(layerValue, "activation");
    const TinyJsonValue *rowsValue = findJsonMember(layerValue, "weight_rows");
    const TinyJsonValue *colsValue = findJsonMember(layerValue, "weight_cols");
    const TinyJsonValue *weightsValue = findJsonMember(layerValue, "weights");
    const TinyJsonValue *biasValue = findJsonMember(layerValue, "bias");
    if (!typeValue || !activationValue || !rowsValue || !colsValue ||
        !weightsValue || !biasValue) {
      std::printf(
          "[Renderer] UnifiedNeural layer description at %s is missing required "
          "fields.\n",
          path.c_str());
      return false;
    }

    std::string layerType;
    std::string activation;
    float rowsNumber = 0.0f;
    float colsNumber = 0.0f;
    if (!jsonStringValue(*typeValue, layerType) ||
        !jsonStringValue(*activationValue, activation) ||
        !jsonNumberValue(*rowsValue, rowsNumber) ||
        !jsonNumberValue(*colsValue, colsNumber) || layerType != "dense") {
      std::printf("[Renderer] UnifiedNeural layer metadata at %s is invalid.\n",
                  path.c_str());
      return false;
    }

    const size_t rows = std::max<size_t>(0, static_cast<size_t>(rowsNumber));
    const size_t cols = std::max<size_t>(0, static_cast<size_t>(colsNumber));
    if (rows == 0 || cols == 0 || cols != expectedCols ||
        weightsValue->type != TinyJsonValue::Type::Array) {
      std::printf(
          "[Renderer] UnifiedNeural layer dimensions at %s do not match the "
          "expected network shape.\n",
          path.c_str());
      return false;
    }

    UnifiedNeuralDenseLayer layer;
    layer.rows = rows;
    layer.cols = cols;
    layer.relu = (activation == "relu");
    layer.weights.reserve(rows * cols);

    if (weightsValue->arrayValue.size() != rows) {
      std::printf("[Renderer] UnifiedNeural weight matrix at %s has %zu rows, "
                  "expected %zu.\n",
                  path.c_str(), weightsValue->arrayValue.size(), rows);
      return false;
    }
    for (const TinyJsonValue &weightRow : weightsValue->arrayValue) {
      std::vector<float> parsedRow;
      if (!jsonNumberArray(weightRow, parsedRow) || parsedRow.size() != cols) {
        std::printf("[Renderer] UnifiedNeural weight matrix row at %s is invalid.\n",
                    path.c_str());
        return false;
      }
      layer.weights.insert(layer.weights.end(), parsedRow.begin(), parsedRow.end());
    }

    if (!jsonNumberArray(*biasValue, layer.bias) || layer.bias.size() != rows) {
      std::printf("[Renderer] UnifiedNeural bias vector at %s is invalid.\n",
                  path.c_str());
      return false;
    }

    loadedModel.layers.push_back(std::move(layer));
    expectedCols = rows;
  }

  loadedModel.valid = true;
  _unifiedNeuralModel = std::move(loadedModel);
  std::printf(
      "[Renderer] Loaded UnifiedNeural model from %s (%zu features, %zu layers, "
      "target=%s, transform=%s).\n",
      _unifiedNeuralModel.sourcePath.c_str(),
      _unifiedNeuralModel.featureNames.size(), _unifiedNeuralModel.layers.size(),
      _unifiedNeuralModel.target.c_str(),
      _unifiedNeuralModel.targetTransform.c_str());
  if (_unifiedNeuralModel.residualTeacher.enabled) {
    std::printf(
        "[Renderer] UnifiedNeural residual teacher enabled (%s, scale=%f, "
        "bias=%f).\n",
        _unifiedNeuralModel.residualTeacher.kind.c_str(),
        static_cast<double>(_unifiedNeuralModel.residualTeacher.scale),
        static_cast<double>(_unifiedNeuralModel.residualTeacher.bias));
  }
  return true;
}

bool Renderer::computeUnifiedNeuralRuntimeSignals(
    size_t objectIndex, size_t primitiveCount,
    UnifiedNeuralRuntimeSignals &outSignals) {
  if (!_unifiedNeuralModel.valid || objectIndex >= _allSceneObjects.size())
    return false;

  const SceneObject &obj = _allSceneObjects[objectIndex];
  const size_t first = obj.firstPrimitive;
  const size_t last =
      std::min(first + obj.primitiveCount, _primitiveScreenCoverage.size());

  bool visible =
      objectIndex < _objectVisible.size() && _objectVisible[objectIndex] != 0u;
  if (!visible && objectIndex < _objectBounds.size())
    visible = isInView(_objectBounds[objectIndex]);

  double visibleCoverage = 0.0;
  for (size_t primIndex = first; primIndex < last; ++primIndex)
    visibleCoverage +=
        std::max(static_cast<double>(_primitiveScreenCoverage[primIndex]), 0.0);
  if (!visible)
    visibleCoverage = 0.0;

  double distance = 0.0;
  if (objectIndex < _objectBounds.size()) {
    simd::float3 delta = _objectBounds[objectIndex].center - Camera::position;
    distance = static_cast<double>(simd::length(delta));
  }

  const double hitProbability =
      objectIndex < _objectHitProbability.size()
          ? std::clamp(static_cast<double>(_objectHitProbability[objectIndex]), 0.0,
                       1.0)
          : 0.5;
  const double rayHitScore =
      objectIndex < _objectRayHitScore.size()
          ? std::max(static_cast<double>(_objectRayHitScore[objectIndex]), 0.0)
          : 0.0;
  const double totalHits =
      objectIndex < _objectHitLastFrame.size() ? _objectHitLastFrame[objectIndex]
                                               : 0.0;
  const double totalRaysTested =
      objectIndex < _objectRaysTestedLastFrame.size()
          ? _objectRaysTestedLastFrame[objectIndex]
          : 0.0;

  double estimatedBytes = 0.0;
  if (objectIndex < _residentObjectGpuResources.size()) {
    const auto &resident = _residentObjectGpuResources[objectIndex];
    estimatedBytes = static_cast<double>(resident.byteSize > 0
                                             ? resident.byteSize
                                             : resident.resources.currentRequiredBytes());
  }

  double objectImportance = 0.0;
  if (objectIndex < _objectImportance.size())
    objectImportance =
        std::max(static_cast<double>(_objectImportance[objectIndex]), 0.0);
  if (!(objectImportance > 0.0)) {
    const size_t importanceLast =
        std::min(first + obj.primitiveCount, _primitiveImportance.size());
    for (size_t primIndex = first; primIndex < importanceLast; ++primIndex)
      objectImportance +=
          std::max(static_cast<double>(_primitiveImportance[primIndex]), 0.0);
  }

  outSignals.visible = visible;
  outSignals.visibleCoverage = visibleCoverage;
  outSignals.distance = distance;
  outSignals.hitProbability = hitProbability;
  outSignals.rayHitScore = rayHitScore;
  outSignals.totalHits = totalHits;
  outSignals.totalRaysTested = totalRaysTested;
  outSignals.estimatedBytes = estimatedBytes;
  outSignals.objectImportance = objectImportance;
  (void)primitiveCount;
  return true;
}

bool Renderer::buildUnifiedNeuralFeatureVector(
    size_t objectIndex, size_t primitiveCount, std::vector<float> &outFeatures) {
  UnifiedNeuralRuntimeSignals signals;
  if (!computeUnifiedNeuralRuntimeSignals(objectIndex, primitiveCount, signals))
    return false;

  outFeatures.clear();
  outFeatures.reserve(_unifiedNeuralModel.featureNames.size());
  for (const std::string &featureName : _unifiedNeuralModel.featureNames) {
    double value = 0.0;
    if (featureName == "total_object_hits") {
      value = signals.totalHits;
    } else if (featureName == "mean_object_rayhit_score") {
      value = signals.rayHitScore;
    } else if (featureName == "mean_object_importance") {
      value = signals.objectImportance;
    } else if (featureName == "mean_hit_probability") {
      value = signals.hitProbability;
    } else if (featureName == "total_object_rays_tested") {
      value = signals.totalRaysTested;
    } else if (featureName == "mean_distance" ||
               featureName == "min_distance") {
      value = signals.distance;
    } else if (featureName == "mean_visible_coverage" ||
               featureName == "max_visible_coverage") {
      value = signals.visibleCoverage;
    } else if (featureName == "primitive_count") {
      value = static_cast<double>(primitiveCount);
    } else if (featureName == "mean_estimated_object_bytes") {
      value = signals.estimatedBytes;
    } else {
      std::printf(
          "[Renderer] UnifiedNeural model expects unsupported feature '%s'; "
          "falling back to heuristic unified scoring.\n",
          featureName.c_str());
      return false;
    }

    if (!std::isfinite(value))
      value = 0.0;
    outFeatures.push_back(static_cast<float>(value));
  }

  return true;
}

float Renderer::computeUnifiedNeuralTeacherRawScore(
    const UnifiedNeuralRuntimeSignals &signals) const {
  if (!_unifiedNeuralModel.residualTeacher.enabled)
    return 0.0f;

  const std::string &kind = _unifiedNeuralModel.residualTeacher.kind;
  if (kind != "rayhit_efficiency_percentile_v1")
    return static_cast<float>(std::max(signals.rayHitScore, 0.0));

  const float visibilitySignal =
      signals.visible
          ? std::clamp(static_cast<float>(signals.visibleCoverage), 0.0f, 1.0f)
          : 0.0f;
  const float hitProbability =
      std::clamp(static_cast<float>(signals.hitProbability), 0.0f, 1.0f);
  const float rayHitScore = std::max(static_cast<float>(signals.rayHitScore), 0.0f);
  const float estimatedBytes =
      std::max(static_cast<float>(signals.estimatedBytes), 1.0e-6f);
  const float costExponent =
      std::max(_unifiedNeuralModel.residualTeacher.costExponent, 0.0f);
  const float visibilityBonus =
      1.0f +
      visibilitySignal * _unifiedNeuralModel.residualTeacher.visibilityWeight;
  const float probabilityBonus =
      1.0f + hitProbability *
                  _unifiedNeuralModel.residualTeacher.hitProbabilityWeight;
  const float costTerm = std::pow(estimatedBytes, costExponent);
  return rayHitScore * visibilityBonus * probabilityBonus /
         std::max(costTerm, 1.0e-6f);
}

float Renderer::predictUnifiedNeuralImpact(const std::vector<float> &features) const {
  if (!_unifiedNeuralModel.valid ||
      features.size() != _unifiedNeuralModel.featureNames.size()) {
    return 0.0f;
  }

  std::vector<float> activations(features.size(), 0.0f);
  for (size_t i = 0; i < features.size(); ++i) {
    const float mean = _unifiedNeuralModel.mean[i];
    const float stddev = _unifiedNeuralModel.stddev[i];
    if (std::fabs(stddev) <= 1.0e-8f) {
      activations[i] = 0.0f;
      continue;
    }
    activations[i] = (features[i] - mean) / stddev;
    if (!std::isfinite(activations[i]))
      activations[i] = 0.0f;
  }

  for (const UnifiedNeuralDenseLayer &layer : _unifiedNeuralModel.layers) {
    if (activations.size() != layer.cols)
      return 0.0f;

    std::vector<float> next(layer.rows, 0.0f);
    for (size_t row = 0; row < layer.rows; ++row) {
      float sum = row < layer.bias.size() ? layer.bias[row] : 0.0f;
      const size_t rowOffset = row * layer.cols;
      for (size_t col = 0; col < layer.cols; ++col)
        sum += layer.weights[rowOffset + col] * activations[col];
      if (layer.relu)
        sum = std::max(sum, 0.0f);
      next[row] = std::isfinite(sum) ? sum : 0.0f;
    }
    activations = std::move(next);
  }

  if (activations.empty())
    return 0.0f;

  float prediction = activations[0];
  if (_unifiedNeuralModel.targetTransform == "exp_m1_of_linear_output")
    prediction = std::exp(prediction) - 1.0f;
  if (!std::isfinite(prediction))
    prediction = 0.0f;
  return prediction;
}

} // namespace NomadPathTracer
