---
name: npu-performance-analysis
description: 分析 NPU 推理 profiling 和执行图，定位搬运、计算、Host 调度及 GE 融合瓶颈，并提出可验证的优化实验；适用于 GE Session 或 ATC→OM→ACL 的选定路线。
---

# NPU 推理性能分析

面向 SE 的分析能力：从本次执行证据形成瓶颈判断和实验建议。它不接管工作流调度，不代替独立测试，不要求安装来源仓或其他外部 skill。

## 先固定分析对象

读取任务设计和基线，确认模型/代码身份、唯一 source→export→runtime 路线、dtype、shape/长度分布、设备与软件版本、核/内存预算、实例数/并发/batch，以及 p99 上限与计时边界。继承已确认契约，不因某种工具方便而切换路线。

完成实测瓶颈分析需要本次路线的 profiling。缺数据时按[采集与瓶颈分解](references/profiling.md)提出或执行已授权的有限采集；只有模型文件时可先做结构假设，但明确尚未完成实测归因。缺请求级记录时不能从算子均值生成 p99。

## 按问题加载方法

- 有 profiling，需判断时间消耗与优化优先级：读[采集与瓶颈分解](references/profiling.md)。包含推荐/打分模型的输入搬运、Embedding/Gather、小算子、Host 调度和多实例分析。
- 有源图或 GE Build 图，需关联 kernel、查找重复模式：读[图与 kernel 归因](references/graph-attribution.md)。图为可选输入，纯 ONNX 不要求 PyTorch modeling 脚本。
- 需提出源图、GE pass、解析插件或运行时优化：读[GE 优化方法](references/ge-optimization.md)。接口与参数须核实目标安装版本。
- 正式测量遵循[统一测量约定](../npu-model-workflow/references/measurement.md)，精度与并发复验遵循[正确性约定](../npu-model-workflow/references/correctness.md)。

## 输出与交接

向掌控者返回报告路径和简短结论；报告包含：

1. 对象与数据有效性：版本、路线、采集范围、请求/迭代对应关系、选择/排除依据、未覆盖项。
2. 瓶颈与证据：墙钟范围、热点、关键依赖、搬运/Host/设备时间线，区分事实、近似归因和假设。
3. 优先候选：每项一句话内容、触发证据、作用层、适用条件、具体修改与回退方式、预计作用机制、精度/p99/资源风险、验证方法和实验预算。
4. 待核实事项：参数/API/版本支持、可选图或缺失工具；证据不足时提出最小补采集，不填猜测数字。

已有 OPT 编号时更新其指定记录；尚未分配时返回候选，由掌控者统一编号。每次实施都保留成功、退化、无收益、失败和未完成记录，不能只汇报收益项。调优记录和交付汇总沿用 workflow 模板，不另造格式。

最终标准是在固定资源、负载分布和精度约束下提高 p99 上限内的可持续吞吐。候选需先通过对应正确性与最终配置压力检查；搜索后冻结配置，由测试用独立序列/轮次验收。profiling 解释原因，无采集开销的正式 benchmark 才提供交付性能值。

方法的内化来源与适配记录见 [sources](../npu-model-workflow/references/sources.md)，随包保留适用许可；不以来源仓或外部同名技能为运行前提。
