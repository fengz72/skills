# 来源、适配与许可

这些是内化方法的来源记录，不是运行依赖。正式技能只读取本包内容，不要求来源仓、外部同名 skill 或开发机目录存在。当前脚本为本工作流实现，没有迁入来源脚本、hooks、模板或状态服务。

主要来源是 cannbot-skills，2026-10-24 查阅记录中的远端 HEAD 为 `ec8a7b0462e3acdd0d1d2f857f1dae652e19f4f2`。下表 SHA256 标识实际所读文件，不能据此假定来源工作区没有未提交修改：

| 来源文件 | SHA256 | 内化位置 |
|---|---|---|
| plugins-official/model-infer-optimize/AGENTS.md | 941e7521e0eda4f7f43cda4168c1597bafd568825fd24f1e82a107a80c34af99 | 主控唯一编排、角色分工、实施与独立验收 |
| docs/agent/sota-approach-skill-design.md | a0ebc90eb992b6e6b6821e70572d22912c60fd3f9982a781e0bc77b82eb156ab | 多来源候选、主控归并、组合复验与进度单写 |
| model/model-recommend-analysis/SKILL.md | 794e46b84fa3597335913947dbc315fba6526239d47cfd21d19fdf409615695d | performance-analysis 中推理搬运/计算/Host/重复结构方法 |
| graph/ge-dynamic-shape-diagnose/SKILL.md | 90cb3afb76533f92de4828f38dfd45550703a86ff17139611826cb6d48bbae5a | ge-diagnose 动态 shape |
| graph/ge-memory-analysis/SKILL.md | 46102b3b47b7491fed74fd306f3161c1f7bb84076aac2f2f6b218d3378136005 | ge-diagnose 内存与缓冲 |
| graph/ge-stream-log-analysis/SKILL.md | e19808417cc17781db3f09ed346f880f730762e3652f60e9a645485383589158 | ge-diagnose 流身份与证据 |
| graph/ge-coredump-diagnose/SKILL.md | c6d784aec02b0aed040a0eb99d86169567622f9cbcfb78278c1966792eb885e6 | ge-diagnose 编译/执行崩溃 |
| graph/ge-fusion-pass-skill/SKILL.md | eed1a22b656cc313fe1cfcc5f8d391e15f56e3eaf866bfc2657873804690823a | performance-analysis GE 优化的模式/API/生效验证 |
| graph/ge-onnx-plugin-development/SKILL.md | f8889bda436c0f3a49c857b516f1442b322eb1ea68c87477c789893c48242876 | performance-analysis 解析映射与 kernel 边界 |

适配：只保留推理；每任务一条 GE 路线；不嵌套来源编排或强制特定框架；源图、最终 GE 图和运行时分层取证；参数查目标版本，不照搬旧阈值/开关；不使用平均时间公式或自动删慢迭代证明 p99。新增实际执行身份/取消、压力正确性、稳定性、负载有效性、缓存契约、搜索/独立验收，以及完整固定 OPT 记录。

来源文档按 CANN Open Software License Agreement Version 2.0 提供，随包保留 [许可全文](../licenses/CANN-OSL-2.0.txt)。来源中的适用通知继续有效，本记录不将其重新许可。分发新增的外部材料时分别登记身份、改动和适用许可，不仅复制来源名称。

Codex 的目录与 TOML 角色适配依据 [Subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents)，skills 组织依据 [Build skills](https://learn.chatgpt.com/docs/build-skills)。目标客户端仍需核实发现/加载能力，普通 subagent 显式读取角色说明是回退路径；这些网页不是运行步骤的必需依赖。

不预置虚构历史性能案例。当前模型经验先保存在本项目，确认可共享范围后，将摘要、比较条件和必要证据按相对路径放进案例包。来源溯源不构成访问原仓、修改共享 skill 或发布数据的授权。
