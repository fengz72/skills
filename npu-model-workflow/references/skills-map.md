# 本包技能路由

| 当前需要 | 读取 |
|---|---|
| 端到端推进、需求、角色调度、验收与交付 | [npu-model-workflow](../SKILL.md) |
| 已有基线/profiling，分析搬运、计算、Host、重复结构并提出优化 | [npu-performance-analysis](../../npu-performance-analysis/SKILL.md) |
| 在精度门槛内评估 FP16、BF16、局部降精度及 FP32 回退 | [低精度与混合精度](../../npu-performance-analysis/references/precision-optimization.md) |
| 用户给延迟上限但不知道合理吞吐，估计资源容量并匹配历史经验 | [npu-performance-target](../../npu-performance-target/SKILL.md) |
| GE 动态 shape、内存/地址、流或编译/执行崩溃 | [npu-ge-diagnose](../../npu-ge-diagnose/SKILL.md) |

只加载当前问题需要的正文和引用。候选应说明触发证据、所选路线/作用层、适用 shape/dtype/版本、风险、修改范围、验证、回退与预算。缺可选来源继续其他候选；缺必需参考或执行工具就明确未验收，不能假装已有能力。

外部 skills 可增强具体算子、pass、kernel 或版本查询；它们不是基本流程前提。PyTorch eager/TorchAir 优化只有真正进入选定 AIR/ONNX/GE/OM 后才可能算收益。ONNX 插件映射不等于实现 kernel；Session 与 ATC 的 pass/plugin 接入分别核实。
