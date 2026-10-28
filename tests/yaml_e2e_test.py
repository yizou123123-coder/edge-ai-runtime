
#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/debug/edge_runtime"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def write_config(
    path,
    video_path,
    deadline=100.0,
    capacity=4,
    policy="latest_frame",
    display=False,
):
    """生成独立 YAML，不修改项目正式配置。"""

    content = f"""
source:
  type: video
  video_path: {json.dumps(str(video_path))}
  camera_index: 0
  target_fps: 30.0

runtime:
  queue_capacity: {capacity}
  queue_policy: {policy}
  deadline_ms: {deadline}

model:
  path: models/yolov8n.onnx
  confidence_threshold: 0.25
  nms_iou_threshold: 0.45

display:
  enabled: {str(display).lower()}
"""

    path.write_text(content, encoding="utf-8")


def run_command(arguments):
    """启动真实 edge_runtime，并捕获输出。"""

    result = subprocess.run(
        [str(BINARY), *map(str, arguments)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
    )

    return result.returncode, result.stdout + result.stderr


def read_metric(output, name):
    """从 Runtime 日志中读取整数指标。"""

    pattern = rf"\[Processing\] {re.escape(name)}: (\d+)"
    match = re.search(pattern, output)

    require(match is not None, f"Missing metric: {name}")

    return int(match.group(1))


def test_deadline(config_path, expected_budget, all_missed):
    code, output = run_command(
        ["--config", config_path]
    )

    require(code == 0, f"Runtime failed:\n{output}")

    expected_log = (
        f"Deadline Budget: {expected_budget:.6f} ms"
    )

    require(
        expected_log in output,
        f"Deadline not applied:\n{output}",
    )

    processed = read_metric(output, "Frames processed")
    misses = read_metric(output, "Deadline Misses")
    on_time = read_metric(output, "On-time Frames")

    require(processed > 0, "No frames processed")

    if all_missed:
        require(
            misses == processed and on_time == 0,
            "1 ms deadline statistics mismatch",
        )
    else:
        require(
            misses == 0 and on_time == processed,
            "Large deadline statistics mismatch",
        )

    print(
        f"[PASS] Deadline {expected_budget:g} ms: "
        f"processed={processed}, misses={misses}"
    )


def test_invalid_config(path, expected_error, name):
    code, output = run_command(
        ["--config", path]
    )

    require(code != 0, f"{name}: unexpectedly succeeded")

    require(
        expected_error in output,
        f"{name}: unexpected error:\n{output}",
    )

    require(
        "[Runtime] capture thread started" not in output,
        f"{name}: capture thread started unexpectedly",
    )

    print(f"[PASS] {name}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
        help="Path to test.mp4",
    )

    args = parser.parse_args()
    video = Path(args.video).expanduser().resolve()

    require(BINARY.is_file(), f"Binary missing: {BINARY}")
    require(video.is_file(), f"Video missing: {video}")
    require(
        (ROOT / "models/yolov8n.onnx").is_file(),
        "ONNX model missing",
    )

    with tempfile.TemporaryDirectory(
        prefix="edge_ai_yaml_e2e_"
    ) as directory:
        temp = Path(directory)

        # Test 1: 极小 Deadline
        small = temp / "small_deadline.yaml"
        write_config(
            small, video, deadline=1.0
        )

        test_deadline(
            small, expected_budget=1.0, all_missed=True
        )

        # Test 2: 极大 Deadline
        large = temp / "large_deadline.yaml"
        write_config(
            large, video, deadline=1000000.0
        )

        test_deadline(
            large, expected_budget=1000000.0,
            all_missed=False
        )

        # Test 3: 非法队列容量
        invalid_capacity = temp / "invalid_capacity.yaml"
        write_config(
            invalid_capacity, video, capacity=0
        )

        test_invalid_config(
            invalid_capacity,
            "queue_capacity must be greater than 0",
            "Invalid queue capacity rejected",
        )

        # Test 4: 非法队列策略
        invalid_policy = temp / "invalid_policy.yaml"
        write_config(
            invalid_policy, video, policy="unknown_policy"
        )

        test_invalid_config(
            invalid_policy,
            "Unknown queue policy",
            "Invalid queue policy rejected",
        )

        
        # Test 5: 非法显示布尔值
        # 不启动 GUI，兼容无图形界面的 CI 环境
        invalid_display = temp / "invalid_display.yaml"

        write_config(invalid_display, video)

        content = invalid_display.read_text(encoding="utf-8")

        require(
            "enabled: false" in content,
            "Display field missing",
        )

        invalid_display.write_text(
            content.replace(
                "enabled: false",
                "enabled: invalid_bool",
                1
            ),
            encoding="utf-8",
        )

        test_invalid_config(
            invalid_display,
            "bad conversion",
            "Invalid display boolean rejected",
        )

        # Test 6: 配置文件不存在
        missing = temp / "not_exist.yaml"
        code, output = run_command(
            ["--config", missing]
        )

        require(
            code != 0 and "Configuration startup failed" in output,
            "Missing config was not rejected",
        )

        print("[PASS] Missing YAML rejected")

        # Test 7: 旧 CLI 已删除，不应返回成功
        code, output = run_command(
            ["--video", video]
        )

        require(
            code != 0,
            "Removed --video option unexpectedly succeeded",
        )

        print("[PASS] Removed --video option rejected")

    print("\nYAML End-to-End: ALL PASS")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, subprocess.TimeoutExpired) as error:
        print(f"\n[FAIL] {error}")
        raise SystemExit(1)
