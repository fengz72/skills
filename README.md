# Skills

可复用的 Agent 技能。每个一级目录是一份 skill，入口为各自的 `SKILL.md`。

| Skill | 用途 |
|---|---|
| [karpathy-guidelines](karpathy-guidelines/SKILL.md) | 编码、审查与重构的行为准则 |
| [npu-model-workflow](npu-model-workflow/SKILL.md) | 编排 NPU 推理需求、设计、正确性、吞吐优化和交付 |
| [npu-performance-analysis](npu-performance-analysis/SKILL.md) | 从 profiling、执行图和运行证据提出优化实验 |
| [npu-performance-target](npu-performance-target/SKILL.md) | 结合 Roofline、历史证据和基线推荐吞吐目标 |
| [npu-ge-diagnose](npu-ge-diagnose/SKILL.md) | 诊断 GE 编译、动态 shape、内存、流与执行问题 |

## NPU 工作流的组织与使用

分发 NPU 工作流时保留上述四个 `npu-*` 目录的同级关系，技能之间通过相对路径引用。模板、SE/开发/测试角色正文、设备任务辅助、请求统计、记录覆盖检查和 CPU 回归均随包提供，不依赖 cannbot-skills 或具体模型适配框架。

向 Agent 提供 `npu-model-workflow/SKILL.md` 的实际位置和模型需求文档即可开始。项目根、skill 根和远端执行根分别解析，不假定开发机目录或某种安装位置。示例请求：

```text
使用 /absolute/skills/npu-model-workflow/SKILL.md，读取 /model-project/docs/brief.md。
先完善设计并验证正确性，再在有限预算内优化吞吐。
计时包含输入输出搬运；每个优化点保留固定记录，最终汇总全部结果。
```

主 Agent 派发普通 subagent 时，明确传入 skill 中的角色正文和任务边界。可选的 Codex 角色模板位于 [assets/codex/agents](npu-model-workflow/assets/codex/agents/)；按目标宿主的角色配置方式接入，模板不会因存放在 assets 中自动注册。

工作流会先检查设备占用。所有授权候选 NPU 被其他进程占用时，暂停性能测试，继续设计和离线准备；恢复前重新核查。用户明确不设置 p99 上限时，报告观察到的吞吐和 p99，不伪造延迟 SLA 通过。

## CPU 验证

在本仓库根目录执行：

```sh
python3 -m unittest discover -s npu-model-workflow/tests -p 'test_*.py' -v
```

辅助脚本使用 Python 标准库；设备任务辅助要求 POSIX 进程组和 flock。这些测试不占用 NPU，也不能代替真实设备正确性、性能或释放核查。

NPU 方法的来源、适配和许可见 [sources.md](npu-model-workflow/references/sources.md)，适用许可全文随包保留。模型、业务样例和本地运行日志不包含在本仓库中。
