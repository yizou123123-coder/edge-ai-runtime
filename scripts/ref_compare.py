import numpy as np
import onnxruntime as ort
import cv2
from pathlib import Path
# =========================
# 1. 固定模型与数据路径
# =========================

MODEL_PATH = "models/yolov8n.onnx"
INPUT_PATH = "build/reference/input_f32.bin"
CPP_OUTPUT_PATH = "build/reference/cpp_output_f32.bin"

INPUT_SHAPE = (1, 3, 640, 640)
OUTPUT_SHAPE = (1, 84, 8400)


# =========================
# 2. 读取 C++ 导出的数据
# =========================

input_data = np.fromfile(INPUT_PATH, dtype=np.float32)
cpp_output_data = np.fromfile(CPP_OUTPUT_PATH, dtype=np.float32)

# 检查元素数量
assert input_data.size == np.prod(INPUT_SHAPE), (
    f"Input size mismatch: {input_data.size}"
)

assert cpp_output_data.size == np.prod(OUTPUT_SHAPE), (
    f"Output size mismatch: {cpp_output_data.size}"
)

# 恢复 Tensor 形状
input_tensor = input_data.reshape(INPUT_SHAPE)
cpp_output = cpp_output_data.reshape(OUTPUT_SHAPE)


# =========================
# 3. Python ONNX Runtime 推理
# =========================

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

python_outputs = session.run(
    None,
    {input_name: input_tensor}
)

python_output = python_outputs[0]

# 检查形状和数据类型
assert python_output.shape == OUTPUT_SHAPE, (
    f"Unexpected Python output shape: {python_output.shape}"
)

assert python_output.dtype == np.float32, (
    f"Unexpected Python output dtype: {python_output.dtype}"
)


# =========================
# 4. 数值误差比较
# =========================

diff = np.abs(cpp_output - python_output)

max_error = np.max(diff)
mean_error = np.mean(diff)

# 额外检查是否存在 NaN / Inf
assert np.isfinite(cpp_output).all(), "C++ output contains NaN or Inf"
assert np.isfinite(python_output).all(), "Python output contains NaN or Inf"


# =========================
# 5. 打印验证结果
# =========================

print("========== Reference Alignment ==========")

print(f"Model: {MODEL_PATH}")
print(f"Provider: {session.get_providers()}")

print(f"Input shape: {input_tensor.shape}")
print(f"Input dtype: {input_tensor.dtype}")

print(f"C++ output shape: {cpp_output.shape}")
print(f"Python output shape: {python_output.shape}")

print("\nC++ output first 10:")
print(cpp_output.flatten()[:10])

print("\nPython output first 10:")
print(python_output.flatten()[:10])

print("\n========== Numerical Error ==========")

print(f"Max absolute error : {max_error:.10f}")
print(f"Mean absolute error: {mean_error:.10f}")

print("=========================================")

print("Exactly equal:", np.array_equal(cpp_output, python_output))
print("Different elements:", np.count_nonzero(cpp_output != python_output))
print(f"Max error (scientific): {max_error:.12e}")

# ==================================================
# P3-05-03: C++ / Python Preprocess Alignment
# ==================================================

VIDEO_PATH = Path.home() / "datasets/edge-ai-runtime/test.mp4"


def preprocess_yolo_python(image, target_width=640, target_height=640):
    if image is None or image.size == 0:
        raise ValueError("Empty input image")

    original_height, original_width = image.shape[:2]

    # 1. 缩放比例：模拟 C++ float 运算
    scale_w = np.float32(target_width) / np.float32(original_width)
    scale_h = np.float32(target_height) / np.float32(original_height)
    scale = np.minimum(scale_w, scale_h)

    # 2. 缩放尺寸：与 C++ static_cast<int> 一致
    resized_width = int(scale * np.float32(original_width))
    resized_height = int(scale * np.float32(original_height))

    resized_image = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR
    )

    # 3. Letterbox Padding
    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2

    top = pad_y
    bottom = target_height - resized_height - top
    left = pad_x
    right = target_width - resized_width - left

    padded_image = cv2.copyMakeBorder(
        resized_image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    # 4. BGR -> RGB
    rgb_image = cv2.cvtColor(
        padded_image,
        cv2.COLOR_BGR2RGB
    )

    # 5. uint8 -> float32，归一化
    float_image = rgb_image.astype(np.float32)
    float_image *= np.float32(1.0 / 255.0)

    # 6. HWC -> CHW
    chw_image = np.transpose(float_image, (2, 0, 1))

    # 7. 增加 Batch 维度
    input_tensor_python = np.ascontiguousarray(
        chw_image[np.newaxis, :, :, :]
    )

    print("\n========== Python Preprocess Info ==========")
    print(f"Original size: {original_width} x {original_height}")
    print(f"Resized size : {resized_width} x {resized_height}")
    print(f"Scale        : {scale}")
    print(f"Padding      : left={left}, right={right}, "
          f"top={top}, bottom={bottom}")

    return input_tensor_python


# 读取与 C++ 相同视频的第一帧
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

ret = cap.read()
frame = cv2.imread(
    "build/reference/frame_cpp.png",
    cv2.IMREAD_COLOR
)

if frame is None:
    raise RuntimeError("Failed to read frame_cpp.png")
cap.release()

if not ret or frame is None:
    raise RuntimeError("Failed to read first video frame")


# 独立执行 Python 预处理
python_input = preprocess_yolo_python(frame, 640, 640)

# input_tensor 是前半部分已读取的 C++ 输入 Tensor
assert python_input.shape == input_tensor.shape
assert python_input.dtype == np.float32
assert np.isfinite(python_input).all()


# 逐元素比较
preprocess_diff = np.abs(input_tensor - python_input)

preprocess_max_error = np.max(preprocess_diff)
preprocess_mean_error = np.mean(preprocess_diff)

different_elements = np.count_nonzero(
    input_tensor != python_input
)

exactly_equal = np.array_equal(
    input_tensor,
    python_input
)

max_index = np.unravel_index(
    np.argmax(preprocess_diff),
    preprocess_diff.shape
)

print("\n========== Preprocess Alignment ==========")

print(f"C++ input shape    : {input_tensor.shape}")
print(f"Python input shape : {python_input.shape}")

print(f"Exactly equal      : {exactly_equal}")
print(f"Different elements : {different_elements}")

print(f"Max absolute error : {preprocess_max_error:.12e}")
print(f"Mean absolute error: {preprocess_mean_error:.12e}")

print(f"Max error index    : {max_index}")
print(f"C++ value          : {input_tensor[max_index]}")
print(f"Python value       : {python_input[max_index]}")

print("==========================================")