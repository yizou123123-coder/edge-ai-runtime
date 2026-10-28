# P4-03 Real-time Queue Policy Comparison

**项目：** ARM Linux Edge AI Real-time Inference Runtime  
**阶段：** Phase 4 / P4-03  
**测试平台：** WSL2 Ubuntu 24.04 x86_64  
**构建模式：** Release  
**实验状态：** Completed — 3 Policies × 3 Runs

## 1. 实验目标

评估不同有界队列策略对端侧目标检测系统实时性的影响。

在视频输入速率高于系统持续推理能力时，对比三种策略：

- **Blocking**：队列满时阻塞生产者。
- **DropOldest**：队列满时删除最旧待处理帧。
- **LatestFrame**：新帧到来时替换全部待处理旧帧。

重点研究队列策略对 Frame Age、输出吞吐量、源端跳帧和队列丢帧的影响。

本实验不以提高模型计算速度为目标，而是评价不同数据调度策略对结果新鲜度的影响。

## 2. 实验环境与条件

| 项目 | 配置 |
|---|---|
| CPU | AMD Ryzen 5 7500F，6 核 12 线程 |
| 平台 | WSL2 / Ubuntu 24.04 / x86_64 |
| 编译器 | GCC 13.3.0 |
| 构建 | CMake Release |
| ONNX Runtime C++ | 1.30.0 |
| Execution Provider | CPU |
| ORT 线程配置 | 默认，未显式设置 |
| 模型 | YOLOv8n ONNX FP32 |
| 网络输入 | 1×3×640×640 |
| 测试视频 | test.mp4，1920×1080，60 FPS |
| 目标回放速率 | 30 FPS |
| 目标时间槽 | 239 / run |
| Queue Capacity | 4 |
| Confidence Threshold | 0.25 |
| NMS IoU Threshold | 0.45 |
| Frame Age 预算 | P99 ≤ 100 ms |
| 重复次数 | 每种策略 3 次 |

每轮测试使用同一视频、模型和目标输入速率，运行至正常 EOF，并完成队列排空。

## 3. 指标定义

**Frame Age**

检测结果产生时刻减去对应图像的应用层采集时间戳：

`Frame Age = output_ts - capture_ts`

其中 `capture_ts` 在视频 `read()` 成功返回后记录。因此该指标不包含摄像头曝光、硬件传输和驱动内部缓存时间。

**Source Skipped Slots**

VideoSource 为保持实时视频时间轴，额外跳过的目标帧时间槽。不包含 60 FPS 转换到 30 FPS 时的正常抽帧。

**Queue Dropped**

已成功提交到队列、后来被队列策略主动淘汰的帧数。

**Deadline Miss Rate**

`超过 100 ms 的已输出帧数 / 全部已输出帧数 × 100%`

**Target-slot On-time Rate**

`100 ms 内完成输出的帧数 / 目标输入时间槽总数 × 100%`

该指标同时考虑 Source Skip 和 Queue Drop 的损失，适合不同策略之间的场景级比较。

## 4. Release 原始实验结果

### 4.1 Blocking

| 指标 | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Frames Captured | 195 | 185 | 192 |
| Frames Processed | 195 | 185 | 192 |
| Source Skip | 44 | 54 | 47 |
| Queue Drop | 0 | 0 | 0 |
| Processing FPS | 23.96 | 22.31 | 23.64 |
| Frame Age P50 (ms) | 200.76 | 248.44 | 201.34 |
| Frame Age P95 (ms) | 423.56 | 353.05 | 435.00 |
| Frame Age P99 (ms) | 480.99 | 422.43 | 509.80 |
| On-time Frames | 0 | 4 | 0 |

三轮均正常退出。

Blocking 保留所有已经成功入队的帧，因此 Queue Drop 为 0。但生产者受到反压，三轮分别额外跳过 44、54、47 个视频时间槽。

队列峰值均达到容量 4。

### 4.2 DropOldest

| 指标 | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Frames Captured | 239 | 239 | 239 |
| Frames Processed | 200 | 183 | 126 |
| Source Skip | 0 | 0 | 0 |
| Queue Drop | 39 | 56 | 113 |
| Processing FPS | 24.67 | 22.41 | 15.42 |
| Frame Age P50 (ms) | 146.60 | 157.65 | 173.92 |
| Frame Age P95 (ms) | 181.16 | 190.48 | 237.81 |
| Frame Age P99 (ms) | 195.49 | 210.59 | 276.34 |
| On-time Frames | 21 | 8 | 0 |

三轮均正常退出。

DropOldest 允许 VideoSource 按预定节拍运行，Source Skip 均为 0。

但由于只有队列满时才删除最旧元素，系统仍然可能保留多帧待处理的历史图像。

第三轮 Processing FPS 明显下降，说明测试期间存在较大的运行性能波动，其具体原因尚未确认。

### 4.3 LatestFrame

| 指标 | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Frames Captured | 239 | 239 | 239 |
| Frames Processed | 209 | 189 | 223 |
| Source Skip | 0 | 0 | 0 |
| Queue Drop | 30 | 50 | 16 |
| Processing FPS | 25.94 | 23.51 | 27.76 |
| Frame Age P50 (ms) | 48.11 | 59.19 | 42.33 |
| Frame Age P95 (ms) | 78.22 | 85.35 | 71.62 |
| Frame Age P99 (ms) | 93.50 | 100.59 | 80.25 |
| On-time Frames | 208 | 187 | 223 |

三轮均正常退出，队列峰值始终为 1。

LatestFrame 避免积压多个待处理旧帧，使消费者完成当前推理后，可以获取更接近当前时刻的图像。

三轮中有两轮满足 100 ms P99 预算，第二轮 P99 为 100.59 ms，轻微超出目标。

## 5. 综合对比

以下 FPS、Frame Age 和帧数量为三轮算术平均。三轮平均 P99 不等于合并所有原始帧龄样本后的 P99。

| 指标 | Blocking | DropOldest | LatestFrame |
|---|---:|---:|---:|
| Processing FPS | 23.30 | 20.83 | **25.74** |
| Frames Processed | 190.67 | 169.67 | 207.00 |
| Source Skip | 48.33 | 0 | 0 |
| Queue Drop | 0 | 69.33 | 32.00 |
| Frame Age P50 (ms) | 216.85 | 159.39 | **49.88** |
| Frame Age P95 (ms) | 403.87 | 203.15 | **78.40** |
| Frame Age P99 (ms) | 471.07 | 227.47 | **91.45** |
| Target-slot On-time Rate | 0.56% | 4.04% | **86.19%** |
| 满足 P99 ≤ 100 ms 的轮次 | 0/3 | 0/3 | **2/3** |

场景级按时输出率采用三轮合计统计，每种策略的分母均为：

`239 × 3 = 717 个目标时间槽`

这避免了 Blocking 因 Source Skip 导致分母变小，从而影响策略之间的公平比较。

## 6. 结果分析

### 6.1 Blocking：数据完整性与实时性之间的矛盾

Blocking 不主动删除已经入队的帧，但不能保证所有目标时间槽的数据都进入系统。

生产者阻塞时，VideoSource 需要跳过已经过期的时间槽。三轮合计产生 145 次 Source Skip。

与此同时，队列中旧数据可能长时间等待，造成较大的 Frame Age。

因此，Queue Drop 为 0 并不意味着系统没有损失输入数据，也不意味着结果具有良好的实时性。

### 6.2 DropOldest：减少反压，但无法彻底消除排队延迟

DropOldest 在三轮中均实现了完整的 239 个目标时间槽输入，没有 Source Skip。

与 Blocking 相比，其三轮平均 P99 从 471.07 ms 下降至 227.47 ms，降幅约为 51.7%。

不过 DropOldest 允许队列保留最多 4 个待处理元素。即使队列未满，消费者仍然可能从较旧的帧开始处理。

因此，它改善了积压，但仍未满足 100 ms P99 预算。

### 6.3 LatestFrame：优先保证结果新鲜度

LatestFrame 的三轮平均 Frame Age P99 为 91.45 ms。

相比 Blocking，平均 P99 降低约 **80.6%**；相比 DropOldest，降低约 **59.8%**。

这一改善主要来自减少待处理帧的排队等待，而不是对 ONNX 推理算子本身进行优化。

LatestFrame 的三轮目标时间槽按时输出率为 86.19%，明显高于另外两种策略。

在本次实验中，LatestFrame 也获得了较高的平均 Processing FPS，但不能仅据此认定该策略提高了模型本身的计算速度。不同轮次的 CPU 负载及调度波动可能影响这一指标。

## 7. 实时性预算验收

初始实验预算：

**Frame Age P99 ≤ 100 ms**

LatestFrame 的三轮结果分别为：

- Run 1：93.50 ms，达标。
- Run 2：100.59 ms，未达标。
- Run 3：80.25 ms，达标。

因此：

**LatestFrame 在本次实验中表现最好，但尚未稳定满足 100 ms P99 目标。**

不应仅根据平均 P99 低于 100 ms，就将系统认定为稳定达标。

后续需要通过性能分析、线程配置实验和更多次独立测试，进一步确认尾部延迟的来源。

## 8. 实验限制

1. 测试平台为 WSL2 x86_64，不代表 Raspberry Pi 5 ARM64 的实际性能。
2. 每种策略仅重复 3 次，尚不足以得出严格的统计显著性结论。
3. ORT 使用默认线程配置，尚未进行线程数、CPU 亲和性或调度策略优化。
4. 本实验使用固定视频节拍回放，并非真实 USB 摄像头的数据采集链路。
5. Frame Age 为应用层测量，不是曝光到感知输出的完整物理延迟。
6. Blocking、DropOldest 与 LatestFrame 的 Processing FPS 在不同轮次存在波动，不能将吞吐差异全部归因于队列策略。
7. 本实验保留所有原始运行结果，不剔除表现较差的轮次。

**原始记录注意事项：** 最后三轮实际运行策略为 LatestFrame，但执行脚本仍使用了 `drop_oldest_*.log` 文件名。因此该轮数据以程序打印的 `[Experiment] Policy: LatestFrame` 为准。日志归档时应按实际策略重新命名或重新保存，避免覆盖混淆。

## 9. 工程结论与策略选择

对于当前前视视觉感知场景，系统更关注检测结果的新鲜度，而不是保留每一帧的完整历史记录。

基于三轮 Release 实验：

**选择 LatestFrame 作为当前场景的优先候选队列策略。**

选择依据：

- Frame Age P50/P95/P99 均明显低于其他策略。
- 队列峰值为 1，避免积压多个待处理旧帧。
- 场景级按时输出率显著提高。
- 能在有限推理算力下维持接近目标输入时间轴的数据获取行为。

但 LatestFrame 会主动丢弃尚未开始处理的帧，因此不适合要求完整保留全部事件或视频帧的业务。

当前策略选择属于**针对前视感知新鲜度需求的工程权衡**，而不是对所有使用场景的通用结论。

## 10. 后续工作

**Phase 4：**
- 完成 FrameSource / CameraSource 抽象。
- 通过 YAML 配置选择 Queue Policy、输入源及队列容量。
- 完成摄像头与完整 Runtime 的集成验证。

**Phase 5：**
- 使用 perf、Sanitizer 和 ORT 线程配置实验分析性能。
- 调查推理耗时波动及 Frame Age 尾部延迟。
- 在可复现条件下进行优化前后对比。

**Phase 6：**
- 部署到 Raspberry Pi 5 ARM64。
- 使用 USB UVC 摄像头进行实际运行测试。
- 对比 x86_64 与 ARM64 的性能，并重新评估实时性预算。

---

**P4-03 结论：** 三种队列策略已实现并通过功能和并发测试；Release 重复实验表明，LatestFrame 在本次 CPU 推理场景中显著改善应用层 Frame Age，但 100 ms P99 预算尚未稳定达标。