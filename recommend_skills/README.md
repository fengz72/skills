# recommend_skills

- 说明
用于存放推荐领域相关的 skills，主要用于推荐模型的功能适配、性能分析优化、自定义交付件开发等场景。

- 目录结构

```
recommend_skills/
│── cannbot-infra.md   # 推荐工作流主文档（推荐模型 · GE 图模式，14 个条目 + 持续优化循环）
│── skills/            # 工作流配套全部 18 个 skills
│── README.md          # 本说明
```

- 推荐工作流

`cannbot-infra.md`：昇腾 NPU 推荐模型（搜索/推荐/排序/视觉编码类网络）推理自动分析与改进工作流。
执行顺序：环境自检 → 模型分析 → 单卡基线 → 并发限核分析 → 精度门禁 → 持续优化循环（性能分析 → GE 图模式优化 → 算子与图适配 → 多线程优化 → 访存优化 → 融合规划 → Kernel 生成 → 自定义算子开发，运行时诊断贯穿，回归验收收口）。
所有改动须过「精度 + 性能」双门禁，产出 perf_ledger 与最终优化报告。

- 当前已归档的 skills（18 个，与工作流条目对应）

| 工作流条目 | Skill | 用途 |
|---|---|---|
| 0 环境基线 + 路由 | ascend-env-baseline | 环境自检、产物目录约定 |
| 1 模型分析 | model-analysis | onnx 结构档案、算子构成、显存估算、可执行判定 |
| 2 单卡基线运行 | baseline-run | 首基线采集（时延/QPS/HBM/精度/profiling） |
| 3 并发与限核策略 | concurrency-strategy-analysis | THREAD_NUM/CORE_NUM/并行度推荐 |
| 4 精度对比 | precision-comparison | golden 标杆、精度门禁、分节点定位 |
| 5 性能分析 | performance-analysis | 三表分析、bound 判定 |
| 5 性能深度分析 | profiling-deep-analysis | 四项专项（算子执行/多流/H2D-D2H/host 瓶颈） |
| 循环编排 | optimization-loop | 持续优化闭环迭代与终止判定 |
| 6 GE 图模式与 shape 策略 | ge-graph-optimization | shape 策略、GE 编译选项、torchair/torch.compile |
| 7 算子与图优化适配 | fusion-operator-adaptation | 既有算子替换、图融合 pass、等价改写 |
| 7 FA 族融合 | flash-attention-fusion | FlashAttention 族融合确认与接入 |
| 8 多线程与图并行 | multi-thread-optimization | 并发/限核实施与 sweep 验收 |
| 9 访存优化 | weight-prefetch | memory-bound 热点搬运与计算重叠 |
| 10 融合规划与理论估计 | fusion-planning-estimation | 融合候选清单、Roofline bound 判定、收益排序 |
| 11 Kernel 生成 | kernel-generation | AscendC/cannbot-dsl 融合 kernel 开发 |
| 12 自定义算子开发 | custom-op-development | 功能补齐向算子开发与 GE 接入 |
| 13 运行时诊断 | runtime-debug | 报错/卡死定界修复路径 |
| 14 回归验收与基线管理 | regression-baseline-mgmt | 优化项台账、双门禁验收、收口报告 |

- 来源

归档自 cannbot-infra 工作流仓（`.agents/skills/` + 工作流主文档），skills 间相对引用已保持有效。
