# Edge AI Runtime — Scenario Requirements

**版本：** Draft 0.1  
**阶段：** P4-00  
**场景：** 移动机器人 / 车载前视摄像头感知节点

## 1. 应用场景

使用 USB UVC 摄像头或按实时节拍回放的视频作为输入，通过 YOLOv8n ONNX 模型输出目标检测结果。

本项目定位为 Linux 用户态软实时感知 Runtime，不承担安全关键控制任务，也不保证硬实时截止时间。

## 2. 输入与输出

**输入：**
- 目标输入速率：30 FPS
- 视频测试源：`test.mp4`，1920×1080，元数据帧率 60 FPS
- 回放目标帧率：30 FPS
- 后续 ARM 目标：Raspberry Pi 5 + USB UVC Camera

**输出：**
- `frame_id`
- `capture_ts`
- 检测类别、置信度与边界框
- Runtime 性能统计

## 3. 实时性指标

| 指标 | x86_64 初始目标 | Raspberry Pi 5 |
|---|---|---|
| 目标输入速率 | 30 FPS | 30 FPS 场景 |
| Frame Age P99 | 候选 100 ms，待确认 | 实测后修订 |
| Output Hz | P4 实验确定 | P6 实测后确定 |
| Drop Rate | P4 实验确定 | P6 实测后确定 |
| Deadline Miss Rate | P4 实验确定 | P6 实测后确定 |
| On-time Rate | P4 实验确定 | P6 实测后确定 |

100 ms 是当前拟定的实验预算，相当于约三个 30 FPS 帧周期；它不是来自真实机器人安全需求，也尚未证明可以达到。

P3-07 Release 基线中三阶段平均耗时合计约 36.47 ms，用作可行性分析依据，不直接当作连续输入下的 Frame Age。

## 4. 指标定义

- `capture_ts`：`read()` 成功返回后，使用 `steady_clock` 记录的应用层时间戳。
- `Frame Age`：检测结果输出时刻减去 `capture_ts`。
- `Drop Rate`：队列策略主动丢弃的帧数除以提交给队列的帧数。
- `Deadline Miss Rate`：已输出但 Frame Age 超过预算的帧数除以已输出帧数。
- `On-time Rate`：在预算内完成输出的帧数除以提交给队列的帧数。
- `Tail Latency Spread`：Frame Age P99 − P50。

所有比率需要使用明确且一致的统计窗口。

## 5. 测试规则

- 固定视频、模型、输入节拍与配置，对比 Blocking、Drop Oldest、Latest Frame。
- 同时记录实际输入速率、输出频率、队列丢帧、Frame Age P50/P95/P99、Deadline Miss Rate、On-time Rate。
- 视频源按 30 FPS 节拍输入；60 FPS 视频如何抽帧在 P4-03 固定规则。
- 视频抽帧与队列主动丢帧分开统计，不能混为一谈。
- Blocking 导致采集线程无法达到目标输入速率时，必须如实记录，不能只用零丢帧率宣称效果更好。
- x86_64 与 ARM64 分别测试，不直接套用同一硬件性能预算。

## 6. 资源与可靠性

- 队列必须有界，运行中内存不应持续无限增长。
- 正常 EOF、主动 Stop、推理错误都必须能够协调退出并 Join。
- CPU、RSS 和长时间稳定性将在 Phase 5/6 建立实际测试基线。
- 摄像头断线恢复、SIGTERM 与 systemd 故障注入属于可选工程亮点，按实际完成情况验收。

## 7. 验收方式

Phase 4 通过实时节拍输入和三种队列策略对比，验证实际输入速率、帧龄、丢帧行为与系统退出可靠性，并根据实测结果形成明确的策略选择结论。

未确定的量化目标须在实验前给出依据，不得事后为迎合结果随意修改。