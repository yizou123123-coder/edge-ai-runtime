# Changelog

## Unreleased

### Phase 6 - Engineering Infrastructure

- Added x86 Debug, Release, ASan and TSan CMake Presets.
- Added reusable scripts/build.sh.
- Added GitHub Actions CI for build and selected tests.
- Added Pull Request template.
- Removed local video dependency from preprocessing unit test.
- Standardized C++ formatting with clang-format.

## Phase 5 - Debug, Profiling and Optimization

Tag: `phase5-pass`

- Completed ASan / TSan investigations.
- Profiled CPU hotspots using perf.
- Added configurable ONNX Runtime intra-op threads.
- Added preprocessing, inference and postprocessing timing statistics.
- Optimized YOLO preprocessing channel conversion.
- Passed 21 Release CTest tests.

See docs/phase5_report.md for measurements and limitations.
