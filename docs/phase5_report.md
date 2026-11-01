cat > docs/phase5_report.md <<'EOF'
# Phase 5 性能分析与优化报告

## 1. 测试环境
- 平台：WSL2 Ubuntu 24.04
- CPU：AMD Ryzen 5 7500F（6C/12T）
- 模型：YOLOv8n ONNX，640x640
- 推理后端：ONNX Runtime CPU
- 编译：Release

## 2. Sanitizer 检查
- ASan：未在已执行的核心测试中发现自身代码的内存访问错误。
- LSan：存在 ONNX Runtime 等第三方组件泄漏报告，尚未完全排除。
- TSan：队列并发测试通过；ORT、FFmpeg 相关报告已通过独立程序复现，未认定为自身代码竞争错误，也未证明第三方实现无问题。

## 3. CPU 热点
perf 采样显示，ONNX Runtime 约占用户态 CPU 采样的 94%。
推理计算是当前主要 CPU 热点。

## 4. 优化一：ORT intra-op 线程数
测试线程配置：0、1、2、3、4、5、6、7、8、9、10、12。
每种配置重复三次。

默认配置：
- Processing FPS：27.09
- Frame Age P99：87.68 ms
- Queue Drop Rate：8.65%

9 线程配置：
- Processing FPS：29.67
- Frame Age P99：54.61 ms
- Queue Drop Rate：0.28%

以上为三次运行对应指标的平均值。
9 线程是当前测试环境下的推荐配置。

完整实验数据：docs/p5_05_ort_threads.csv

## 5. 优化二：YOLO 图像预处理
移除单独的 BGR→RGB 转换，
通过调整 cv::split 的目标通道顺序保持 RGB CHW 排列。

三次运行平均结果：
- Preprocess Mean：2.110 → 1.977 ms
- Preprocess P99：5.787 → 4.675 ms
- Frame Age P99：66.90 → 61.52 ms

预处理 Mean 降低约 6.3%。
端到端指标存在运行波动，不能完全归因于本次修改。

## 6. 回归测试
Release 构建成功。
CTest：21/21 PASS。

## 7. 结论与限制
当前 WSL2 测试条件下，经过优化的配置已在多次测试中达到
Frame Age P99 <= 100 ms 的目标。

未完成的验证：
- 第三方库 Sanitizer 报告进一步归因
- ARM64 真实设备性能复测
- 长时间负载下的性能稳定性验证
EOF