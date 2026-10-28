# Phase 4 — Real-Time Edge AI Runtime 验收与性能基线

**项目：** Edge AI Runtime  
**阶段：** Phase 4 — 实时调度、输入源抽象、配置化与可视化  
**日期：** 2026-10-28  
**验收结论：** 功能 PASS；100 ms P99 性能目标未达成

## 1. 阶段目标

在 Phase 3 YOLOv8n CPU 推理能力的基础上，实现具备实时调度能力的 Linux C++17 Edge AI Runtime。

主要目标包括：

- 多线程采集与推理流水线。
- 有界队列及 Blocking、DropOldest、LatestFrame 三种策略。
- 30 FPS 视频回放节拍和源端过期帧跳过。
- 视频文件与摄像头的统一 FrameSource 接口。
- Frame Age、吞吐率、队列丢帧等性能统计。
- YAML 配置解析、校验和动态创建输入源。
- YOLO 模型路径、检测阈值和 Deadline 配置化。
- 可选 OpenCV GUI 检测结果显示。
- 完整单元测试、异常测试和端到端回归。

## 2. 实验环境

| 项目 | 配置 |
|---|---|
| 开发环境 | WSL2 Ubuntu 24.04 / x86_64 |
| CPU | AMD Ryzen 5 7500F |
| 编程语言 | C++17 |
| 构建系统 | CMake |
| 推理框架 | ONNX Runtime C++ |
| 模型 | YOLOv8n FP32 ONNX |
| 模型输入 | `[1, 3, 640, 640]` |
| 模型输出 | `[1, 84, 8400]` |
| 输入视频 | 1920×1080、478 帧、60 FPS 元数据 |
| 目标输入节拍 | 30 FPS |
| 实验 Deadline | 100 ms |

性能结果来自当前 WSL2 x86_64 环境，不代表后续 Raspberry Pi 5 的实际性能。

## 3. 系统架构

```text
VideoSource / CameraSource
             |
             v
       Capture Thread
             |
             v
    BlockingQueue<Frame>
             |
             v
      Processing Thread
             |
             v
       YOLOv8n / ORT
             |
       +-----+------+
       |            |
       v            v
 Performance     DisplayQueue
 Statistics          |
                     v
                 Main Thread
                 OpenCV GUI
```

FrameSource 抽象接口隔离视频文件与摄像头输入差异。

推理队列实现三种策略：

- **Blocking：** 队列满时阻塞生产者，不主动丢弃队列中的帧。
- **DropOldest：** 队列满时丢弃最旧的待处理帧。
- **LatestFrame：** 新帧进入时清除队列中的旧待处理帧，优先保证结果新鲜度。

显示队列独立于推理队列，容量为 1，采用 LatestFrame 策略，避免 GUI 速度反向阻塞推理线程。

## 4. 实时指标定义

**Frame Age**

从 `source.read()` 成功返回后记录的采集时间戳，到该帧推理和后处理完成时的时间差。

它代表应用层帧年龄，不包含摄像头曝光、设备驱动内部缓冲等不可见延迟。

**Deadline Miss Rate**

超过配置 Deadline 的输出帧数除以成功处理的帧数。

**On-time Rate**

Deadline 内完成的输出帧数除以成功提交到推理队列的帧数。

**Queue Drop Rate**

推理队列主动丢弃的帧数除以成功提交到推理队列的帧数。

**Source Skipped Slots**

视频源因为回放时间槽过期而跳过的目标时间槽数量，与队列丢帧分开统计；不包含 60 FPS 视频转换为 30 FPS 时常规跳过的交替源帧。

正常 EOF 且队列排空时验证：

`Captured = Processed + QueueDropped`

## 5. 队列策略实验结论

此前 P4-03 策略对比实验显示：

- Blocking 保留输入数据，但容易形成排队延迟和源端反压。
- DropOldest 通过丢弃较旧的等待帧降低积压。
- LatestFrame 更适合优先关注当前画面、允许跳帧的实时目标检测场景。

历史策略实验中，LatestFrame 的 Frame Age 明显低于 Blocking。

不同批次实验的运行环境和 CPU 负载可能存在差异，因此不能直接将历史策略实验数据与本次最终 Release 结果视为完全受控的性能对比。

## 6. Phase 4 最终 Release 基线

测试条件：

- Release 构建。
- YOLOv8n CPU 推理。
- LatestFrame 队列策略。
- 30 FPS 视频回放。
- GUI 关闭（Headless）。
- Deadline = 100 ms。
- 同一视频连续运行三次。

| 指标 | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Frames Captured | 239 | 238 | 239 |
| Frames Processed | 89 | 84 | 127 |
| Queue Dropped | 150 | 154 | 112 |
| Source Skipped Slots | 0 | 1 | 0 |
| Processing FPS | 10.90 | 10.30 | 15.74 |
| Frame Age P50 (ms) | 86.53 | 97.91 | 78.22 |
| Frame Age P95 (ms) | 202.46 | 187.87 | 113.57 |
| Frame Age P99 (ms) | 367.20 | 413.43 | 133.48 |
| Deadline Miss Rate | 34.83% | 50.00% | 17.32% |
| On-time Rate | 24.27% | 17.65% | 43.93% |
| Queue Drop Rate | 62.76% | 64.71% | 46.86% |

三次运行中位数：

- Processing FPS：**10.90**
- Frame Age P50：**86.53 ms**
- Frame Age P95：**187.87 ms**
- Frame Age P99：**367.20 ms**

**性能验收：0/3 次达到 P99 ≤ 100 ms。**

三次运行的吞吐率和长尾延迟波动明显，说明当前环境的运行稳定性和延迟预算仍需要优化。

本阶段不将该性能目标标记为 PASS。

## 7. 功能与测试验收

完成并验证：

- BlockingQueue 和三种队列策略。
- 多生产者与消费者并发测试。
- VideoSource 节拍、过期时间槽跳过。
- CameraSource 打开失败处理。
- FrameSource Factory 和多态 Runtime。
- Runtime 一次启动限制、停止、线程回收。
- Live Source 读取故障处理。
- Runtime 可配置队列容量。
- 模型路径和 Deadline 参数校验。
- Confidence Filtering 与 NMS IoU 专项测试。
- YAML ConfigLoader、数值校验与启动参数校验。
- YAML 端到端自动化测试。
- DisplayQueue 最新结果、关闭与排空测试。
- OpenCV GUI 检测框、类别 ID 和置信度显示。
- GUI 正常播放、提前退出以及 Headless 模式。

最终 Debug 回归输出：

`PHASE 4 DEBUG REGRESSION: ALL PASS`

YAML 端到端回归输出：

`YAML End-to-End: ALL PASS`

## 8. 当前限制与后续工作

1. Release P99 未达到 100 ms 实验目标。
2. WSL2 环境存在明显性能波动，需要进一步分析。
3. 真实 USB UVC 摄像头尚未在 Raspberry Pi 5 上完成最终验证。
4. 当前 Frame Age 不包含驱动及物理采集侧的全部延迟。
5. 模型与视频处理仍依赖当前实现所支持的输入格式、模型形状和节拍条件。
6. 后续需要进行 AddressSanitizer、ThreadSanitizer 和异常生命周期压力测试。

## 9. 阶段结论

Phase 4 已完成实时推理 Runtime 的核心工程功能，包括有界队列调度、帧丢弃策略、多态输入源、YAML 配置化、ONNX Runtime CPU 推理、延迟指标统计及可选 GUI 显示。

功能测试及 Debug 全量回归通过。

在最终 Release 实验中，Frame Age P99 的 100 ms 目标尚未达到，保留为后续性能优化任务。

**Phase 4 状态：Functional PASS / Performance Target NOT MET。**
