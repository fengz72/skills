---
name: npu-model-workflow
description: 编排推理模型在昇腾 NPU 上的需求澄清、单路线适配、正确性验证、p99 约束下吞吐优化和交付；使用主 Agent 与 SE、开发、测试 subagent，并完整记录每个优化点。适用于需要端到端推进的模型任务，已有 runner 可复用，无 runner 时补最小脚本。
---

# NPU 模型适配与优化工作流

你是掌控者。使用宿主原生计划、文件、命令和 subagent 工具推进，不运行另一套工作流引擎。
先确定本 SKILL.md 所在的真实目录，按其位置解析相对链接和脚本；项目根、实际执行根和 skill 根可能不同。

## 开始或恢复

1. 读取 brief、已有设计、progress 和实际模型/样例。已有结论和授权继续有效，不重复询问可调查的事实。
2. 读取 [需求与接入](references/requirements.md)，确认唯一 source→export→runtime 路线、可信参考、修改边界、p99/数据/资源、有限实验预算和交付内容。缺业务信息时集中询问，同时继续独立调查。
3. 在模型文档目录按需建立 brief、design、progress、experiments 和 acceptance。可使用 [assets 中的模板](assets/brief.md)；尊重项目已有组织，不重新创建重复文件。只有模板含填空，正式设计/交付填写实际值或明确缺失状态。
4. 检查宿主 subagent 能力、当前并发额度、实际执行主机/容器/目录和设备。读取 [编排与恢复](references/orchestration.md)，执行其中的设备可用性门禁。所有授权候选 NPU 均被其他进程占用时，暂停性能测试，继续设计和离线准备；恢复前重新核查，不依据旧的空闲记录启动任务。恢复时也检查旧任务，状态不明不重复提交。

## 原生角色与派发

本 skill 明确允许为独立工作使用 subagent。主控统一派发、分配 OPT 编号、维护 progress、协调用户与设备，并基于证据决策；不让多个嵌套主控争用全局进度。

| 角色 | 角色说明 | 任务 |
|---|---|---|
| SE | [roles/se.md](references/roles/se.md) | 事实调查、需求设计、瓶颈分析、优化计划与目标推荐 |
| 开发 | [roles/developer.md](references/roles/developer.md) | 隔离实现、构建、自测和集成 |
| 测试 | [roles/tester.md](references/roles/tester.md) | 契约审查、独立功能/精度/压力/性能/交付验收 |

宿主能选择已注册角色时使用对应角色；否则创建普通 subagent，并明确传入角色说明路径、任务、执行位置、写入范围、版本、预算、完成条件和报告路径。可选 Codex 角色模板位于 [assets/codex/agents](assets/codex/agents/)，由目标项目按宿主支持方式接入；模板本身不会自动注册。角色名不证明配置已加载，也不提供独立权限沙箱。模型/推理设置遵循用户与宿主当前配置。

## 推进阶段

按 [阶段与完成条件](references/stages.md) 推进：

需求与设计 → 可运行和正确性 → 性能基线 → 优化循环 → 集成与交付。

- 有 runner 就验证后复用；没有就派发本路线的最小运行脚本开发，不绑定具体框架或统一七步管线。
- 先按 [正确性与稳定性](references/correctness.md) 建立可信基线。最终 batch/并发/多实例/异步配置另做请求—输出关联压力与约定稳定性验证。
- 性能按 [测量契约](references/measurement.md) 执行：**固定精度验收门槛、资源与工作负载，在 p99 不超过上限时尽可能提高可持续成功吞吐**。明确缓存、排队、实际到达和异步完成，搜索与独立验收分开。
- 将 [低精度与混合精度](../npu-performance-analysis/references/precision-optimization.md) 作为常规候选：在设计允许范围内比较 FP16、BF16 或局部降精度，分别记录权重、激活、计算/累加与输入输出类型；精度和业务门槛保持不变。不得只改配置名就声称生效，也不把全部输入统一转为浮点。
- 只读候选分析可并行；开发使用实际执行环境中的独立副本、构建与结果目录，快照包含未提交改动。正式同设备测量排他；远端任务按 [设备任务辅助](references/device-job.md) 管理。
- 初始无可行点时先改善可行性，中间探索基线不等于性能通过；首次可行不计算虚构的吞吐百分比。不得用不可行候选覆盖已验收的可行版本。
- 每个优化点开始前创建 [固定调优记录](assets/optimization-record.md)，提升、下降、无显著变化、失败和未完成全部保留。测试给结果，主控定接纳/不接纳及理由。组合形成新快照重新验收。

普通阶段以证据推进，不另加重复审批。路线、支持范围、精度/业务门槛或授权修改边界改变时澄清；预算用尽先记录最好已验收版本和未达项，不擅自无限续跑。

## 按需能力与脚本

读 [skills-map](references/skills-map.md) 选择本包的性能分析、目标推荐或 GE 诊断。内化内容及许可见 [sources](references/sources.md)，原来源仓不是运行依赖。

- 请求级统计：`python3 <本skill根>/scripts/request_metrics.py --help`。输入与计时语义见 [测量契约](references/measurement.md)，该脚本只检查请求日志，不代替精度/稳定性或总体交付验收。
- 实际执行主机的设备任务：`python3 <本skill根>/scripts/device_job.py --help`；使用前读其 [适用范围](references/device-job.md)。它不采集性能或证明 NPU 已释放。
- 交付记录覆盖检查：`python3 <本skill根>/scripts/audit_records.py --progress <progress.md> --experiments <experiments目录> --report <acceptance.md>`。它只核对编号、文件和汇总链接，不证明指标或接纳决定正确。

## 结束

按 [交付模板](assets/delivery-report.md) 汇总：最终配置/环境、正确性与压力/稳定性、独立性能验收、所有 OPT 的一句话内容/基线/优化后/变化/接纳原因，以及实际产物和复现方法。逐编号核对并随包交付完整记录与必要证据。
最终整体提升直接比较首个可信基线和最终版本，不累加逐点百分比。没有可行配置或必需测试未执行时明确未通过/未验收，不把流程结束写成验收成功。
