# UnifiedNeural Active Learning

This is the practical training loop for UnifiedNeural with a cheap `rayhit` teacher.

## Core idea

We split training into two stages:

1. `Teacher pretrain`
   Build a cheap pseudo-label target from the rayhit signal already present in `neural_object_features.csv`.
   This gives the MLP a strong starting policy without running any ablations.

2. `Perceptual fine-tuning`
   Use CGVQM-labeled ablations only for the difficult correction cases.
   Active learning selects the next objects to label based on uncertainty, disagreement, predicted impact, and visibility.

So the expensive perceptual labels become a correction signal rather than the only source of supervision.

The orchestrator is:

`/Users/apollo/Downloads/MetalPathtracer-dev_alt_4/Benchmarks/run_unified_neural_active_learning.py`

## Cheap teacher target

The rayhit teacher builder writes three related columns:

- `teacher_rayhit_raw`
- `teacher_rayhit_efficiency`
- `teacher_rayhit_percentile`

Recommended default:

- `teacher_rayhit_percentile`

Why this one:

- it is clip-local
- it is stable across clips
- it roughly captures which objects rayhit would prioritize
- it includes a mild memory-cost penalty through the efficiency score

The teacher dataset is created by:

`/Users/apollo/Downloads/MetalPathtracer-dev_alt_4/Benchmarks/build_rayhit_teacher_labels.py`

## Commands

### 1. Teacher pretrain

```bash
python3 "/Users/apollo/Downloads/MetalPathtracer-dev_alt_4/Benchmarks/run_unified_neural_active_learning.py" \
  teacher-pretrain \
  --session-root "/absolute/session/root"
```

This will:

- build `active_learning/teacher/rayhit_teacher_labels.csv`
- train the tiny MLP on the teacher target
- export renderer-ready JSON weights

### 2. Active-learning round with teacher initialization

```bash
python3 "/Users/apollo/Downloads/MetalPathtracer-dev_alt_4/Benchmarks/run_unified_neural_active_learning.py" \
  active-round \
  --session-root "/absolute/session/root" \
  --seed-labels-csv "/absolute/path/to/neural_training_labels_50.csv" \
  --baseline-run-dir "/absolute/path/to/baseline_run" \
  --app "/absolute/path/to/Nomad Path Tracer" \
  --train-init-model "/absolute/session/root/active_learning/teacher/model/full_model.pt" \
  --batch-size 20 \
  --per-clip-limit 5
```

This will:

- select informative unlabeled candidates
- run only those ablations
- score only those with CGVQM
- merge the new labels into the master CSV
- fine-tune the MLP from the teacher checkpoint
- export updated renderer-ready weights

### 3. Retrain and export from merged labels

```bash
python3 "/Users/apollo/Downloads/MetalPathtracer-dev_alt_4/Benchmarks/run_unified_neural_active_learning.py" \
  retrain-export \
  --session-root "/absolute/session/root" \
  --init-model "/absolute/session/root/active_learning/teacher/model/full_model.pt"
```

## Outputs

### Teacher stage

- `active_learning/teacher/rayhit_teacher_labels.csv`
- `active_learning/teacher/rayhit_teacher_labels.summary.json`
- `active_learning/teacher/model/full_model.pt`
- `active_learning/teacher/model/unified_neural_mlp.json`

### Active-learning stage

Each round writes to:

`<session-root>/active_learning/round_XXX`

Important artifacts:

- `selector/active_manifest.csv`: selected candidates for this round
- `selector/active_candidates.csv`: full scored pool
- `cgvqm/neural_training_labels_round.csv`: new labels created this round
- `model/full_model.pt`: fine-tuned checkpoint
- `model/unified_neural_mlp.json`: renderer-friendly export
- `round_summary.json`: round bookkeeping

The canonical merged label table lives at:

`<session-root>/active_learning/labels_master.csv`

The latest promoted artifacts are copied to:

`<session-root>/active_learning/latest`

## Why this is more practical

The expensive step is ablation plus CGVQM labeling.

With the rayhit-teacher setup:

- the first model is cheap to obtain
- CGVQM labels are only used where they teach the model something new
- the model starts from a strong heuristic instead of learning from scratch
- the overall pipeline is much more scalable than full brute-force ablation

## Residual Learning

The next refinement after teacher pretraining is residual learning.

Instead of asking the neural model to predict the full perceptual target directly, we:

1. Fit a simple affine mapping from the teacher target to the CGVQM labels.
2. Treat that mapped teacher prediction as the baseline policy.
3. Train the MLP only on the signed residual error that remains.

So the learning problem becomes:

`final_prediction = teacher_baseline + neural_residual`

This is usually more data-efficient than direct prediction, especially with a small label set.

### Residual command

```bash
python3 "/Users/apollo/Downloads/MetalPathtracer-dev_alt_4/Benchmarks/run_unified_neural_active_learning.py" \
  residual-train \
  --session-root "/absolute/session/root" \
  --labels-csv "/absolute/path/to/neural_training_labels_50.csv" \
  --teacher-csv "/absolute/session/root/active_learning/teacher/rayhit_teacher_labels.csv" \
  --init-model "/absolute/session/root/active_learning/teacher/model/full_model.pt"
```

This writes:

- `active_learning/residual_model/residual_training_labels.csv`
- `active_learning/residual_model/residual_eval_predictions.csv`
- `active_learning/residual_model/residual_summary.json`
- `active_learning/residual_model/full_model.pt`

### What to inspect

The key numbers are in:

`active_learning/residual_model/residual_summary.json`

Compare:

- `teacher_baseline_metrics`
- `teacher_plus_residual_metrics`

If residual learning is helping, the teacher-plus-residual metrics should improve over the teacher baseline, and ideally also over the direct fine-tuned model.
