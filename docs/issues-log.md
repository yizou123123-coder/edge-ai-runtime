# Engineering Issues Log

## Issue 001 - CI preprocessing test failed

### Symptom
GitHub Actions build succeeded, but preprocess_test failed.

### Root Cause
The test depended on a local absolute video path that
did not exist on the GitHub runner.

### Fix
Replaced the external video input with a synthetic cv::Mat.

### Prevention
Keep unit tests independent of local datasets.
Use explicit test fixtures for integration tests.
