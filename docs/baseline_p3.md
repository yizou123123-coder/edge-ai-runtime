# P3-07 Release Single-frame Baseline

## Environment

- Platform: WSL2 Ubuntu 24.04 x86_64
- CPU: AMD Ryzen 5 7500F (6C/12T)
- Compiler: GCC 13.3.0
- Build: Release
- ONNX Runtime C++: 1.30.0
- ORT intra-op threads: default
- Model: YOLOv8n ONNX, FP32, 640x640
- Input: test.mp4 first frame, 1920x1080

## Method

- Warmup: 10 iterations
- Measurement: 100 iterations
- Session initialized only once
- Video decoding excluded from timing
- Same frame used for every iteration

## Results

| Stage | Mean (ms) | P95 (ms) |
|---|---:|---:|
| Preprocess | 2.304 | 3.404 |
| Inference | 33.528 | 49.716 |
| Postprocess | 0.636 | 0.766 |

## Findings

- Inference accounts for approximately 91.9% of the
  summed average stage latency.
- Inference P95 is noticeably higher than its mean.
- Root cause of the latency variation is not yet known.
- These results are a single-frame microbenchmark,
  not end-to-end camera latency.

## Next Steps

- Phase 4: Real-time input, frame age, drop policies.
- Phase 5: ORT thread sweep and performance profiling.
- Phase 6: ARM64 performance comparison.