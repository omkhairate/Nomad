# Benchmarks and Frame Capture Controls

The path tracer can emit additional metrics and frame captures while running benchmarks. Configure the behavior with the following environment variables before launching the app:

- `MPT_RUNS_PATH` (default: unset) – directory where run artifacts such as acceleration structure dumps and CSV metrics are written.
- `MPT_MAX_FRAMES` (default: unset) – stop rendering after the specified number of frames when keyframes are present.
- `MPT_CAPTURE_EXR` (default: `0`) – set to `1`, `true`, or `yes` to enable EXR frame capture from the renderer.
- `MPT_CAPTURE_INTERVAL` (default: `1`) – capture every _n_ frames when EXR capture is enabled. Values less than `1` are ignored and treated as `1`.

These variables can be exported in your shell or added to any run scripts you maintain for benchmarking sessions.
