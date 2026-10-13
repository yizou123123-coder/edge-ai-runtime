from ultralytics import YOLO

MODEL_PATH = "models/yolov8n.onnx"
IMAGE_PATH = "build/reference/frame_cpp.png"

# 使用相同 ONNX 模型
model = YOLO(MODEL_PATH, task="detect")

results = model.predict(
    source=IMAGE_PATH,
    imgsz=640,
    conf=0.25,
    iou=0.45,
    rect=False,
    agnostic_nms=False,
    device="cpu",
    verbose=False,
)

result = results[0]

print("========== Ultralytics Reference ==========")
print(f"Detections: {len(result.boxes)}")

for box in result.boxes:
    class_id = int(box.cls.item())
    confidence = float(box.conf.item())

    # Ultralytics 使用 xyxy 格式
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    # 转换为与 C++ cv::Rect2f 相同的 xywh 格式
    x = x1
    y = y1
    width = x2 - x1
    height = y2 - y1

    print(f"Class ID: {class_id}")
    print(f"Confidence: {confidence:.6f}")
    print(
        f"Box (xywh): "
        f"({x:.3f}, {y:.3f}, "
        f"{width:.3f}, {height:.3f})"
    )