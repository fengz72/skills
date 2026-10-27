# CANNBot-Infra（推荐模型 · GE 图模式）推理网络自动分析与改进工作流

推荐模型（搜索/推荐/排序/视觉编码类网络）在昇腾 NPU 上的推理自动分析与优化工作流。**GE 图模式为最终执行基座**，按模型输入格式分三条接入路径（§0.1）。

> **文档说明**：
> - 每个条目按「目标 / 输入 / 输出 / 执行步骤」组织；判定标准、昇腾专属要点、参考实现、依据等详细内容见附录一对应 skill，不在本文档展开
> - onnx 输入经 GE_SESSION_DEMO 的 `prepare.sh` / `compile.sh` / `run.sh` / `test.sh` 入口运行，禁止绕过脚本直接执行二进制（会破坏工作目录与缓存治理）
> - **依据标注约定**：【实测】= 本机实际执行命令获取；【框架】= GE_SESSION_DEMO 已验证能力；【官网】= www.hiascend.com（CANN 文档）；【GitCode】= gitcode.com/Ascend（算子参考仓）

---

## 0. 目标环境基线（所有条目的前置条件）

任一工作流执行前先完成环境自检（入口 skill：`skills/ascend-env-baseline/`），异常时停止下游条目并报告。

| 项目 | 获取方式 | 样例输出 |
|------|---------|---------|
| NPU | `npu-smi info` | Ascend910B 系列（Name 显示 Ascend910，PCI 0xD803），容器内单卡可见 |
| HBM | `npu-smi info` | 每芯片 65536 MB |
| 核心形态 | `npu-smi info`；实际核心数运行时经 `PlatformAscendC` 获取 | AIC/AIV 分离（限核 `CORE_NUM="AIC\|AIV"`，默认 `24\|48`），禁止硬编码核心数 |
| CANN | `echo $ASCEND_HOME_PATH` | 9.1.0（aarch64-linux，`/usr/local/Ascend/ascend-toolkit/latest`） |
| GE | `ls $ASCEND_HOME_PATH/$(uname -m)-linux/lib64/libge_runner_v2.so` | GE Session v2（`GeSession` + `RunGraphAsync` + `aclgrphParseONNX`） |
| 算子工具链 | `which atc msopgen msprof`；A5 平台加查 cannbot-dsl | atc / msopgen / msprof 可用 |
| Python 侧 | `python3 -c "import onnx"` 等 | onnx 1.22.0；onnxruntime 可用性；torch / torch_npu / torchair 版本（pytorch/hf 输入必查，torchair 缺失先补环境） |
| 执行框架 | `cd GE_SESSION_DEMO && bash compile.sh` | 编译通过，产出 `./GEsession_test_acl`（onnx 输入路径） |

### 0.1 模型输入格式与优化约束（三种，条目 1 首先识别）

| 输入格式 | 执行路径 | 优化约束 |
|---|---|---|
| **onnx 图** | GE_SESSION_DEMO：`aclgrphParseONNX` 解析，配置驱动，C++ 无模型分支 | **任何优化不得修改原始 onnx 图**。合法手段：GE 内存图 pass、GE 编译选项、并发/限核参数、精度模式开关；onnx 落盘改写/算子替换重写文件禁止 |
| **pytorch 脚本** | torch_npu 执行；**torchair 转 GE 引擎为必选优化项** | 可修改脚本（git 可回退 + 开关保护）；可使能 torch.compile；torchair 动转静接入 |
| **HuggingFace 模型** | 同 pytorch 脚本（权重与实现脚本均在仓内） | 同 pytorch 脚本 |

### 0.2 关键约束（各条目共用）

- **单卡执行**：`DEVICE_NUM` 恒为 1；卡选择经 `GE_DEVICE_ID` / `START_DEVICE`
- **精度验证仅串行单并发**（`THREAD_NUM=1`）；并发（`THREAD_NUM>1`）仅做性能压测，不做 dump 与精度验证
- **不做量化**：性能手段为图优化、算子融合、并发/限核、访存优化、自定义算子、torchair/torch.compile；精度手段仅 fp16/hf32 精度模式开关，启用必须过精度门禁
- **默认保持原精度**：一切优化默认在原图/原始脚本精度下进行；低精度选项（FORCE_16 / ENABLE_HF32 / precision_mode / op_precision_mode / 算子级低精度路由等）默认关闭，启用须「**用户人工确认** + 精度门禁（记录误差指标）」双重前置，缺一不可；未经确认的低精度收益不得纳入已验收配置
- **自定义算子语言按平台路由**：A2/A3 平台用 **AscendC**（msopgen 工程）；A5 平台推荐 **cannbot-dsl**
- **硬件基准实测锚定**：算力（TFLOPS）与带宽（TB/s）以本机小规模 matmul + HBM copy 实测为准，供 bound 判定与 Roofline 使用
- 所有优化必须**可开关、可回退**；不可回退的改动禁止实施（框架侧需求记录移交，不擅自改）

### 0.3 信息渠道

| 渠道 | 入口 | 用途 |
|------|------|------|
| 昇腾社区官网 | www.hiascend.com（CANN 文档） | GE/ACL/AscendC/cannbot-dsl 文档、算子清单、错误码 |
| GitCode CANN 社区 | gitcode.com/Ascend（ops-* / op-plugin） | 算子参考实现查证 |
| GE_SESSION_DEMO | `doc/README.md`、`doc/DESIGN.md` | onnx 输入执行框架唯一真源 |
| 环境实测 | npu-smi / atc / msprof / 试运行 | 最终仲裁依据 |

### 0.4 产物目录约定（全局强制）

| 目录 | 用途 |
|------|------|
| `docs/{model_name}/{skill-name}/` | 文档类正式交付件：结构档案、方案、分析/验收报告、台账 |
| `data/{model_name}/...` | 数据类交付件：基线、性能/精度对比数据（供后续环节程序化引用） |
| `prof/{model_name}/...` | profiling 归档（完整 PROF_* 目录 + kernel 汇总） |
| `ops/{op_name}/` | 生成的算子工程（源码 / golden / 测试 / benchmark / 集成产物） |
| `tmp/{model_name}/{skill-name}/` | 全部中间产物（可随时清理，不作为交付依据） |

`{model_name}` 取工作区标识（onnx 输入为 GE_SESSION_DEMO 编号 `M<id>`；pytorch/hf 输入为工作区模型名）。判定标准：人读文档 → docs/；机器可读 → data/；算子工程 → ops/（按算子名组织，与模型无关）；可再生 → tmp/。禁止中间产物写入 docs/data/ops，禁止交付件散落 tmp/ 或工作目录根。GE_SESSION_DEMO 框架自管理目录（`models/M<id>/{output,report,.cache,sweep}`）不挪动。全量定义见 ascend-env-baseline skill。

---

## 总体流水（编号 = 执行顺序）

```
(0)环境自检
  → (1)模型分析（输入格式识别 + 结构档案 + 算子构成 + 显存估算 + 可执行判定）
  → (2)单卡基线运行
  → (3)并发与限核策略分析（只分析不改代码）
  → (4)精度对比基线门禁（此后每个优化项均须过此门禁）
  ╔══ 持续优化循环（optimization-loop，直至终止条件）══╗
  ║ (5)性能分析（三表 + bound + 四项专项深度分析）          ║
  ║ (6)GE 图模式与 shape 策略优化（含 torchair/torch.compile）║
  ║ (7)既有算子与图优化适配（含 FA 族融合）── 未适配项 → (10)/(12) ║
  ║ (8)多线程与图并行优化 ←── (5) overlap 机会               ║
  ║ (9)访存优化 ←── (5) memory-bound 热点                    ║
  ║ (10)融合规划与理论估计（双轨分流：回流 (7) / 进 (11)）    ║
  ║ (11)Kernel 生成（A2/A3: AscendC；A5: cannbot-dsl）       ║
  ║ (12)自定义算子开发与 GE 接入（功能补齐向）                ║
  ║ (13)运行时诊断（贯穿，任一环节报错即进入）                ║
  ║ (14)回归验收与基线管理 → 最终报告 + 下一步动作表 → 下一轮  ║
  ╚═════════════════════════════════════════════════════════╝
```

### 持续优化循环

首轮线性执行 0→14 后，按「选动作 → 执行优化（双门禁）→ profiling 采集 → 深度分析（四项专项）→ 收益判定 → 终止检查 → 动作表更新」迭代。**终止条件（满足其一）**：

1. 达到指定性能目标：目标指标（用户指定 avg/TP99/QPS 阈值）连续 2 次同口径复测达标
2. 连续 3 轮无收益（主指标改善 ≤ 噪声带；回退轮计入，纯分析轮不计但连续 2 轮无新候选触发安全阀）
3. 安全阀：无可执行动作 / 轮次预算上限（默认 10 轮）

循环状态：`data/{model_name}/optimization_log/loop_state.json`（支持中断后幂等恢复）。每轮纪律：**一轮一动作**（归因唯一）、双门禁全程有效、同口径复测、四项专项必做且优化点并入动作表、交付件当轮同步更新（台账实时写、每轮性能报告当轮落盘、workflow_status.md 当轮更新、禁止事后批量补写）。完整编排见 optimization-loop skill。

---

## 全局强制门禁：改动校验双门禁

**任何对模型执行的改动，落地前必须同时通过「精度门禁」与「性能门禁」，任一不通过即回退。** 适用范围：图/模型改动、配置改动、编译选项、算子实现变更、运行时修复中涉及上述任何一项的改动。分析类只读动作不触发；结论转化为改动时即触发。

1. **精度门禁**：串行单并发（`THREAD_NUM=1`）verify 全部输出 pass（整型/bool 精确相等；浮点 `allclose`，阈值按模型精度要求裁定并记录依据）；精度相关选项（fp16/hf32/precision_mode）须额外记录误差指标，且**低精度选项启用前须用户人工确认**（默认关闭，见 §0.2）；并发/限核参数仍须做一次串行 verify。精度不过 → **直接回退，不进入性能测量**。
2. **性能门禁**：同口径对比（同卡/同 batch/同 THREAD_NUM/CORE_NUM/同 CircleCount 与统计策略，对照基线或上一已验收配置）；目标场景主指标（单并发看 avg/TP99；吞吐场景看 QPS/TP99）**不得劣化超过噪声带**（以重复测量方差确定，典型 ±2%～±5%）；改善小于噪声带不得宣称提升；允许局部指标劣化换主指标显著改善，但必须显式声明、量化、经用户确认；算子级改动除算子级加速比外**必须补整图端到端对比**。**性能结论双口径强制**：时延（单并发 avg/TP99/TP999）与吞吐（目标并发/批量配置的 QPS 与 TP99）两组数据均须产出（吞吐口径配置取当前已验收配置或用户指定场景），只测其一视为测量未完成。

**落地流程**：改动前确认同口径对照基准（无则先采）→ 实施（可开关/可回退）→ 精度门禁 →（过）→ 性能门禁 →（过）→ 落地并写 perf_ledger（`reverted=false`）；任一不过 → 回退并记录原因。判定细则见 precision-comparison / performance-analysis / regression-baseline-mgmt skill。

---

## 全局强制输出要求

1. **perf_ledger.json（逐条目强制）**：每条目执行完毕在 `data/{model_name}/optimization_log/` 追加记录：`step / timestamp / action / perf_before / perf_after / delta_pct / precision / profiling / reverted / note`。before/after 必须同口径；分析类条目各字段填 `null` 并在 note 注明；回退项 `reverted: true` 并记原因。完整 schema 见 regression-baseline-mgmt skill。
2. **最终优化报告（条目 14 收口强制）**：`docs/{model_name}/final_optimization_report.md`，必含：优化手段总表（含回退项，标注条目来源/类型/启用状态/精度判定）、逐步性能记录表（按时间线 before/after/增量）、累计收益表（基线 = 条目 2 首基线）、各项贡献分解（单独开启增益 + 叠加交叉影响）、精度总结（逐输出结论/阈值/max_rel_err）、未实施项与遗留风险、下一步动作表、环境信息（可复现）。
3. **下一步优化动作表（强制）**：随报告同章产出，机器可读版同步 `data/{model_name}/regression-baseline-mgmt/next_actions.json`。每行字段：优先级（P0/P1/P2）/ 优化动作（具体可执行，禁止空话）/ 手段类型 / 依据（可追溯数据源）/ 预期收益（量化区间 + 推导方法，不与已实施项重复计算）/ 置信度 / 对应工作流（条目 + skill）/ 前置依赖。动作来源六类逐类核对（缺失须说明）：① 条目 10 未实施候选 ② 台账回退项（含重试条件）③ 未实施机会 ④ 条目 5 四项专项 findings ⑤ 条目 7 未适配项 ⑥ 框架需求（标注移交，不入执行队列）。空表必须给出理由。
4. **更新时机**：条目 2 写入首基线；条目 6-12 每实施一项追加一条（循环轮次标注轮次号）；条目 14 汇总收口。
5. **每轮性能报告（强制）**：每轮优化动作完成后产出 `docs/{model_name}/performance-reports/round{k}.md`，按**独立模板文件** `performance_report_template.md`（与本工作流文档同目录）书写，八章齐全（模型介绍与测试环境 / 性能汇总 / 优化手段总表 / 每轮优化动作汇总表 / 精度总结 / 下一步动作 / 修复问题汇总 / 优化改动交付件及复现命令），性能汇总须含时延与吞吐双口径。
6. **工作流执行状态检查表（强制，Markdown 表格，全矩阵）**：`data/{model_name}/optimization_log/workflow_status.md` 记录每一轮×全部条目（0-14）的执行情况——**每轮必须覆盖全部 15 个条目，未触达的条目也必须记录判断**（not_applicable=无触发 / skipped=有判断未做+理由 / blocked=外部受阻），**表中无记录即视为未执行，违背工作流指示**；即使本轮无优化收益，判断本身也要落表。四张表：①逐轮×全条目执行矩阵（轮次/条目/状态/判断与产物）②轮次总览③遗留待办（收口对照）④收口核验清单；每轮收口（L7）时更新，条目 14 收口核验时对照检查（空白单元格=流程未完成）。

---

## （1）模型分析

**目标**：产出「结构档案 + 算子构成 + 模块链路 + 显存估算 + 单卡可执行判定」，作为条目 3/6/7/10 的公共输入。

**输入**：模型工作区（onnx 文件 / pytorch 脚本 / hf 仓；onnx 外部权重须与模型文件同目录）。

**输出**：结构档案（输入输出清单 name/dtype/shape/动态维、opset/框架版本、图规模）；算子构成（AIC/AIV 计数与占比、频次 Top-N、超大 initializer、dtype/domain/bool 预警）；模块链路清单（按 AIC/AIV/访存/控制归属）；显存估算（权重 + 激活 + workspace + 并发叠加）；单卡可执行判定。

**执行步骤**：
1. 识别输入格式（onnx / pytorch / hf），确定执行路径与优化约束（§0.1）
2. 结构参数提取：onnx 经 `prepare.sh <id> --config-only` 生成配置；pytorch/hf 解析脚本模块组成，产出等价结构档案
3. 算子构成统计与兼容性预警（dtype 支持集合、非标 domain、bool 输入算子、动态输出）
4. 模块链路拆解并标注归属（直接支撑条目 3 限核配比与条目 10 融合规划）
5. 显存静态估算（数值经脚本核验，脚本入 tmp/，结果入 data/）
6. 单卡可执行判定：可执行 → 流转条目 2；放不下 → 如实输出问题清单与结论（分片/稀疏化超出范围，需用户决策）

## （2）单卡基线运行

**目标**：模型单卡跑通并采集首基线（时延 avg/TP99/TP999、QPS、HBM 峰值、profiling、精度结论），作为一切优化的同口径对照。

**输入**：条目 1 判定可执行的模型工作区 + 与用户逐项确认的决策清单。

**输出**：`data/{model_name}/baseline/baseline_metadata.json`（配置 + 口径 + 计时 + HBM + 精度 + prof 归档指针）、基线报告、`prof/{model_name}/baseline/` 归档。

**执行步骤**：
1. 关键决策与用户逐项确认（禁止默认拍板）：输入来源、shape 策略、batch、标杆来源、采集内容、目标卡
2. 设备空闲确认；首次执行时算力/带宽实测锚定（记入 `data/env-baseline/`）
3. 模型准备与配置校验（onnx：prepare + `inspect_config.py` 全绿）
4. 冒烟测试（小 circle 跑通建图/编译/执行/落盘全链路；失败转条目 13）
5. 正式基线：串行单并发，CircleCount 足够大（≥1000），HBM 后台采样，记录编译产物状态
6. 精度验证（仅串行；标杆优先级：用户 `reference/` → onnxruntime 生成 → 报错；bf16 与自定义算子模型必须用户提供标杆）
7. profiling 采集与归档（`--prof ge|msprof`，含 mindstudio_profiler_output 汇总与 kernel 汇总）
8. 报告落盘（含口径记录与流转结论 → 条目 3/4/5）

## （3）并发与限核策略分析

**目标**：输出 `THREAD_NUM` / `CORE_NUM`（AIC|AIV）/ `GRAPH_MAX_PARALLEL` / `batch_size` 推荐组合及定量依据。**只分析不改代码**（实施由条目 8 承接）。

**输入**：条目 1 算子构成；目标场景（高吞吐 QPS / 低时延 TP99 / 均衡）；基线 HBM 占用。

**输出**：结构化并发配置推荐 + 方案文档（限核配比理由、图实例显存估算、线程收益上限、TP99 影响分析）。

**执行步骤**：
1. 复用结构档案：AIC 主导 vs AIV 主导判定
2. 定性分类：图粒度（单请求计算量大小 → 多线程收益方向）、限核配比方向、batch 与并发权衡
3. 定量估算：图实例显存（workspace 随 `GRAPH_MAX_PARALLEL` 增长，对照 HBM 余量）、线程数上限、限核对时延与 QPS 的影响
4. 方案审查 → 输出单套推荐配置（含 TP99 上限约束下的最优 QPS 组合）
5. sweep 校准（必做）：`tools/sweep.py` 三阶段（限核寻优 → 并发 1..8 → 可选 batch），回修估算并说明误差
6. 变更触发重分析：场景 / batch / TP99 上限变更，或优化项改变核资源画像（如 FA 融合后）

## （4）模型精度对比

**目标**：建立 golden 标杆与判定标准（**先于一切优化**），并作为此后每个优化项的验证门禁。所有精度判定基于与标杆的数值对比，不可凭直觉；**每次只改一个变量**。

**输入**：基线输出（`output_i.bin` + `output_shapes.json`）、优化后输出、优化阶段记录。

**输出**：精度验证结论（通过/不通过 + 指标）；不通过时输出定位报告（首个偏差节点 → 参数）与修复验证。

**执行步骤**：
1. 收集信息：症状、优化阶段、精度模式（fp16/hf32/precision_mode）、标杆来源
2. 快速诊断：症状分类表匹配 → 快速验证清单 → 发现问题直接修复（症状表见 precision-comparison skill）
3. 分节点定位：`--dump_data` 开启 GE 数据 dump → 与 CPU 参考逐节点对比 → 定位首个偏差节点
4. 精细定位（按需）：暴露中间输出逐步缩小范围（onnx 输入时仅限内存图/临时副本，不改原图文件）
5. 修复与全量精度重验
6. 输出调试报告（问题概要/根因/修复/验证/遗留风险）

## （5）性能分析（含四项专项深度分析）

**目标**：端到端与算子级性能采集、瓶颈定位（bound 分类）、四项专项深度分析，产出结构化优化点并入动作表，驱动条目 6-12 迭代。

**输入**：可运行基线 + 采集场景定义（batch / THREAD_NUM / CORE_NUM / CircleCount）。

**输出**：性能报告（时延/QPS 分场景数据、三表、bound 结论）；`data/{model_name}/profiling-deep-analysis/iter{k}/findings.json`（结构化优化点：id/category/finding/evidence/suggestion/expected_gain/confidence/workflow_entry），**必须并入 next_actions**（去重标注）。

**执行步骤**：
1. 定义采集场景：单并发时延基线与目标并发吞吐基线**分开采集且均须产出**（双口径强制）；CircleCount 足够大保证打点稳定
2. 模型级采集：`run.sh <id> --prof ge|msprof`，归档 `prof/{model_name}/{case}/`（完整 PROF_* + 提取 kernel 汇总）；采集前确认卡空闲
3. 算子级深度采集：msprof 对热点 kernel 深度采集（aic-metrics 覆盖 Cube/Vector/MTE 流水）
4. 三表分析：kernel 表（耗时 Top-N，累计 ≥1%）、overlap 机会表（多线程下利用率/间隙/拷贝窗口）、融合模式表（source-backed 对照 GE 融合结果判定"应融合未融合"）
5. bound 判定与归因：compute-bound（利用率 vs 实测锚定峰值）/ memory-bound（带宽利用率 vs 实测 copy 带宽）/ host-launch-bound（kernel 总时长 ÷ 墙钟）/ 排队争抢（并发 TP99 劣化但 avg 改善）
6. 四项专项（全部必做，数据源 op_summary / api_statistic / fusion_op / step_trace）：**A 算子执行时间**（类型聚合、AIC/AIV 分列、碎片化分布、单 kernel Roofline 效率）；**B 多流掩盖**（流分布、时间线重叠度、串行化链、并发流争抢）；**C H2D/D2H**（拷贝分解、固定开销判定、关键路径占比）；**D host 瓶颈**（kernel/墙钟 < 80% 时触发：墙钟缺口量化 + host API 耗时分解 + 同步等待分解；未触发时给判定依据）
7. 对比与回修：优化前后同口径对比；结论回修条目 3 并发分析与条目 10 融合规划估算模型；数据对账（kernels + H2D/D2H + host 缺口 ≈ 墙钟）

## （6）GE 图模式与 shape 策略优化

**目标**：为模型选定最优 shape 策略与图/编译选项组合，最大化执行性能并控制编译开销；pytorch/hf 输入完成 torchair 转 GE 引擎（**必选**）与 torch.compile 评估。

**输入**：基线跑通的模型；条目 1 动态维分布；条目 5 性能数据。

**输出**：部署配置 + 前后性能对比 + 图编译质量验证记录（dump_graph / 融合统计）。

**执行步骤**：
1. shape 策略决策：onnx 三模式（`all_static` / `all_dynamic` / `custom`，决策表见 skill）；pytorch/hf 输入走 torchair 动转静接入（必选优化项，过双门禁）+ torch.compile 使能评估
2. GE 编译选项调优（逐项单变量）：`CORE_NUM` / `FORCE_16` / `ENABLE_HF32` / `ge.exec.precision_mode` / `op_precision_mode` / 融合开关文件 / `GRAPH_MAX_PARALLEL`；精度相关选项须过精度门禁并记录误差
3. **自动融合增强评估（必做）**：GE 场景尝试开启 autofuse 自动融合（torchair 场景 `experimental_config.auto_fusion_pass`；onnx/GE 场景经融合开关文件与相关编译选项验证自动融合增开空间）；torch.compile 场景尝试 **inductor + autofuse** 组合（`inductor_config.enable_inductor` + auto_fusion_pass）；逐项与默认配置对比，过双门禁后记录收益（环境/版本不支持时如实记录原因并归档结论，不阻塞流水）
4. 图编译质量验证：dump_graph 查看编译各阶段图，确认预期 pass 生效；`fusion_result.json` 量化融合命中
5. 重编译控制评估：形状集合有限时 kernel_meta 同配置复用；形状多变场景评估编译耗时占比，必要时转 custom/all_dynamic
6. 双门禁验收落地；切换策略后 `--show` 与 io_config 一致性确认

## （7）既有算子与图优化适配（含 FA 族融合）

**目标**：用**零开发成本**手段落地可替换/可改善的计算模式：GE 图编译内置优化、图融合 pass（含 FA 族）、等价改写、精度模式路由。

**输入**：条目 1 算子构成与链路清单；条目 5 热点数据。

**输出**：逐模块适配方案（模块 → 手段 + 约束核对）；分开关归因数据；替换前后 profiling 对比 + op_improvement 归档；**未适配项清单**（模块/链路 → 失败原因分类【无算子/硬约束/语义不匹配/精度未达标】→ 所需算子能力 → 已排查候选），性能替换型并入条目 10，功能补齐型转条目 12。**本条目禁止自行启动新算子开发**。

**执行步骤**：
1. 模块拆解（复用条目 1）到可替换链路级
2. 逐链路匹配：GE/ACL 支持算子清单 + 算子参考仓（ops-transformer / ops-nn / ops-math）查证 + 官方文档校验（记录文档路径与参数约束）
3. FA 族融合确认（attention 链为热点时必做）：功能确认用最小验证模型实证参数矩阵（序列长/头数/头维/dtype/布局/mask/scale）
4. 图融合 pass 接入：在 `aclgrphParseONNX` 之后、`AddGraph` 之前的 GE 内存图上做模式匹配与算子替换（不依赖 onnx 融合算子表达）；**onnx 输入禁止改原图**，一切改写仅发生在内存图；开关独立、默认关闭
5. 等价改写：onnx 输入仅限 GE 内存图 pass 或运行时副本；pytorch/hf 输入可直接修改脚本（git 可回退 + 逐项 enable 开关）
6. 精度模式路由：`FORCE_16` / `ENABLE_HF32` / `op_precision_mode` 逐项评估
7. 分开关隔离归因（必做）：逐项单独启用测端到端 + kernel 时间再组合；融合收益是 shape/并发相关的，单并发与目标并发分别测
8. 双门禁验收（逐项 verify 全 pass + 端到端主指标不劣化，回退项如实记录）+ 未适配项移交条目 10/12

## （8）多线程与图并行优化

**目标**：按条目 3 推荐配置实施并发/限核组合，用 sweep 与 profiling 验收 QPS/TP99 收益（推荐模型单请求计算量通常不大，QPS 主靠并发变现）。

**输入**：条目 3 推荐配置；条目 5 overlap 机会表。

**输出**：`multi-thread-analysis/<model>.md` 分析报告 + 验收数据（QPS/TP99/TP999 前后对比）。

**执行步骤**：
1. 先分析后实施：整网 DAG 确认请求间独立；图内 AIC/AIV 资源画像决定限核配比
2. sweep 三阶段实施（`tools/sweep.py`），或按条目 3 推荐直接验证
3. 参数天然可回退；先证明正确（单并发精度 + 并发结果稳定性，无异常尖刺）再追性能；overlap ≠ 收益（资源争抢/拖尾/TP99 劣化需继续调限核与线程数）
4. 调试期 profiling 确认多线程下 AIC/AIV 利用率变化与 kernel 排队
5. 双门禁验收：配置组合串行 verify 全 pass；目标场景 QPS/TP99 不劣化超过噪声带

## （9）访存优化

**目标**：对条目 5 确认的 memory-bound 热点（查表/大 GEMM 权重读取/密集 elementwise 访存链）实施访存优化：搬运与计算重叠、减少 GM 往返。

**输入**：memory-bound 热点清单 + 依赖窗口分析。

**输出**：带开关保护的优化实现 + 性能报告（目标算子耗时、关键路径、带宽利用率前后对比）。

**执行步骤**：
1. 方案设计：热点 + 依赖窗口分析（前序算子不能同为 memory-bound）→ 用户确认
2. 手段分层：图结构侧（GE 内存图 pass / 脚本改写减少 GM 往返，零开发，首选）→ kernel 侧（自定义算子内双缓冲/预取，`DataCopy` 与计算重叠）→ 精度模式联动（fp16 减少搬运字节，须过精度门禁）→ 框架侧需求（GE 无图级显式 stream prefetch，记录移交）
3. 实施（enable 开关保护，默认关闭）
4. 验证：**单点先行**（先一个热点，成功再扩展）；目标算子耗时 + 端到端 + 精度
5. 双门禁验收

## （10）算子融合规划与理论性能估计

**目标**：对无现成解法的热点链路**和算子适配失败项**规划融合/自定义算子方案，并在动手开发前给出理论收益排序（决定条目 11 开发优先级）。**本条目是条目 11 新算子开发的唯一入口。**

**输入**：条目 5 算子级耗时分布（kernel 表 + 融合模式表）+ 条目 7 未适配项清单 + 条目 1 AIC/AIV 归属。

**输出**：融合候选清单（链路 → 方案 → 理论收益区间 → bound 判定 → 开发路径 → 来源标记），按收益/成本比排序，**双轨分流**：既有算子可解（改写/调用修正/参数组合）回流条目 7；需新算子形成条目 11 开发清单；功能补齐型（图跑不起来）转条目 12。

**执行步骤**：
1. 热点链路识别与未适配项并入：从 profiling 提取相邻 kernel 序列，识别「小算子串」（launch 开销主导）与「GM 往返链路」（中间张量落回 HBM）；条目 7 未适配项逐项核对（有 profiling 证据用实测锚定，无证据用结构推导并标注来源）
2. bound 判定（Roofline）：对每个候选链路计算算术强度（FLOPs/Bytes），对照本机实测锚定判定 compute/memory-bound
3. 理论收益估计：融合收益 = 消除的 launch 开销 + 减少的 GM 读写 + 减少的 dtype 转换，给出保守/乐观区间
4. 融合机制选型：GE 图内融合 pass / 等价改写 / AscendC·cannbot-dsl 纯 Vector 融合算子 / Matmul 高阶 API Cube 融合 / AIC-AIV 协同 kernel（按 §0.2 平台路由）
5. 排序输出 Top-N 开发清单（结合模板复用度等开发成本）

## （11）Kernel 生成

**目标**：为条目 10 选定的融合候选生成可运行的融合 kernel，并完成精度/性能验证。**唯一合法入口 = 条目 10 开发清单**（例外：纯 bug 修复、既有算子调用修正、精度修复，属条目 7/4 范畴）。

**输入**：开发项（算法规格、bound 判定、理论收益区间、来源标记）+ 目标交付形态。

**输出**：算子工程 + 精度测试 + benchmark（vs 原算子串加速比，对照理论收益区间复核）+ GE 集成产物。

**执行步骤**：
1. 需求理解 → 设计文档（API 映射、精度路由、Tiling 推导、循环结构、数据流）
2. 路径决策（按 §0.2 平台路由）：A2/A3 → 标准 AscendC（纯 Vector 逐元素/归约/查表走常规 kernel；Matmul 族走 Matmul 高阶 API；Cube+Vector 融合按标准编程模型，从已验证模板改造不裸写）；A5 → cannbot-dsl
3. 工程实现（A2/A3：msopgen 生成工程，`--soc-version` 以本机为准；硬件参数运行时经 `PlatformAscendC` 获取，禁止硬编码）
4. 精度验证：对照 golden（numpy / CPU onnxruntime 参考），按 dtype 的 atol/rtol 标准判定；尺寸覆盖 batch 1/2/N 与动态维边界
5. 性能测试：msprof 采集 vs 原算子串标杆，输出加速比报告（geomean/mean/min/max）
6. 交付集成（算子包部署 + 图接入，见条目 12）+ 双门禁（整图 verify 全 pass + 端到端主指标不劣化，profiling 确认融合 kernel 真实替换原算子串）

## （12）自定义算子开发与 GE 接入

**目标**：补齐 GE parser 无法解析或无原生实现的算子（**功能补齐向**：图跑不起来直接进入本条目），完成「实现 → 算子包部署 → 图接入」全链路。性能替换型必须走条目 10 → 11。

**输入**：含无法解析/无原生实现算子的模型。

**输出**：算子工程源码 + DESIGN.md（强制）+ test_*.py + benchmark + GE 接入说明（部署与环境注入配置）。

**执行步骤**：
1. 方案设计：算法规格（shape/dtype/公式/FLOPs）、分解策略（Cube 段 / Vector 段），输出 DESIGN.md
2. 实现路径（先组合后下沉）：v1 优先现有算子组合 / 脚本改写（pytorch/hf 输入）验证语义；不满足再下沉自定义算子（语言按 §0.2 平台路由）
3. GE 接入：算子包编译部署（经 `ASCEND_CUSTOM_OPP_PATH` 或模型环境 `M<id>_env.sh` 注入）；onnx 输入经 GE 内存图 pass 插入对应节点（**不改原图**，命名与注册信息严格一致）；pytorch/hf 输入经脚本注册接入
4. 精度门禁（强制）：golden 真源为 CPU 参考实现；用例覆盖 batch/动态维/边界；按 dtype atol/rtol 判定，全量通过后才进性能阶段
5. 性能门禁（强制）：底线不低于原等价算子串；达标按 Roofline（有效 TFLOPS 或带宽利用率 ≥ 0.8，峰值取实测锚定）；未达标输出瓶颈分析
6. 交付件清单核验 + 端到端双门禁（vs v1 组合与改动前整图分别对比）

## （13）运行时问题诊断（贯穿全流程）

**目标**：任一环节报错/卡死时按标准路径快速定界修复，不阻塞主流程。

**输入**：故障现象（有明确报错 / 无输出卡死）。

**输出**：修复方案；案例沉淀为可检索的排查记录。

**执行步骤**：
1. 有明确错误：error.log（GE_ERROR_LOG）+ plog（`--info` 等级 2 / `--debug` 等级 0，自动落 report/）→ 按类别归因：初始化失败（PYTHONPATH）/ onnx 解析失败（自定义算子、非标 domain → 条目 12）/ dtype 越界 / aicore timeout（二分法定位）/ OOM（降 GRAPH_MAX_PARALLEL/THREAD_NUM/batch）/ shape 不匹配 / aclnn 错误码（环境/Tiling/Kernel 三类）
2. 无明确错误：npu-smi 状态判定（有进程 + HBM 不变 + CPU 高 = 建图编译中；HBM 不变 + CPU 0 = hang → 查残留 → plog；0 进程 → error.log）
3. 修复实施；若改动模型执行（图/配置/选项/算子），同样过双门禁后才关闭案例
4. 同类错误二次出现按路径直达根因；排查记录回写 runtime-debug skill

## （14）优化回归验收与基线管理（收口环节）

**目标**：建立统一基线与逐项验收机制，管理所有优化开关；收口产出最终优化报告与下一步动作表（下一轮执行入口）。

**输入**：全部已实施优化项与台账。

**输出**：最终配置（= 已验收项组合）、`final_optimization_report.md`、`next_actions.json`、全量回归结论。

**执行步骤**：
1. 基线标准化：每阶段 `baseline_metadata.json`（配置 + 输出摘要 + 性能数据 + 采集命令与场景记录，保证可复现）
2. 优化项台账：每个优化项登记（开关名/参数、默认状态、依赖前置项、验收数据、回退路径）
3. 逐项验收门禁：精度（条目 4 verify 全 pass）+ 性能（条目 5 同口径对比）+ 功能一致（输出 shape/数值分布合理）三关全过才可合入；失败项回退并记录
4. 变更触发重分析：场景 / batch / TP99 上限变更 → 重做条目 3；CANN / 框架版本变更 → 重采全量基线；图 / 配置变更 → 重跑受影响项验收
5. 经验沉淀：新算子约束、等价改写陷阱、排错案例按模板回写 skill
6. 下一步动作表产出（六类来源逐类核对，见全局强制输出要求；机器可读版落 next_actions.json）
7. 收口核验：任意优化项可单独开关并复现收益；每笔改动有双门禁记录；perf_ledger 完整覆盖所有已执行条目；最终报告与动作表已产出

---

## 附录一：Skill 目录映射

| 条目 | Skill 路径 |
|------|-----------|
| 0 环境基线 + 路由 | `skills/ascend-env-baseline/` |
| 1 模型分析 | `skills/model-analysis/` |
| 2 单卡基线运行 | `skills/baseline-run/` |
| 3 并发与限核策略分析 | `skills/concurrency-strategy-analysis/` |
| 4 精度对比 | `skills/precision-comparison/` |
| 5 性能分析（含四项专项深度分析） | `skills/performance-analysis/`、`skills/profiling-deep-analysis/` |
| 循环编排（持续优化闭环） | `skills/optimization-loop/` |
| 6 GE 图模式与 shape 策略优化（含 torchair / torch.compile） | `skills/ge-graph-optimization/` |
| 7 算子与图优化适配（含 FA 族融合） | `skills/fusion-operator-adaptation/`、`skills/flash-attention-fusion/` |
| 8 多线程与图并行优化 | `skills/multi-thread-optimization/` |
| 9 访存优化 | `skills/weight-prefetch/` |
| 10 融合规划与理论估计 | `skills/fusion-planning-estimation/` |
| 11 Kernel 生成（A2/A3: AscendC；A5: cannbot-dsl） | `skills/kernel-generation/` |
| 12 自定义算子开发与 GE 接入 | `skills/custom-op-development/` |
| 13 运行时诊断 | `skills/runtime-debug/` |
| 14 回归验收与基线管理 | `skills/regression-baseline-mgmt/` |

## 附录二：证据清单

| 编号 | 证据 | 类型 | 位置 |
|------|------|------|------|
| E1 | NPU 设备/HBM/健康 | 环境实测 | `npu-smi info` |
| E2 | CANN 版本 | 环境实测 | `$ASCEND_HOME_PATH` |
| E3 | GE v2 组件（libge_runner_v2 / GeSession / RunGraphAsync） | 环境实测 | `$ASCEND_HOME_PATH/${ARCH}-linux/lib64/` |
| E4 | 工具链 atc / msopgen / msprof（及 A5 平台 cannbot-dsl 可用性） | 环境实测 | `which` / `pip list` |
| E5 | GE_SESSION_DEMO 框架能力（配置驱动、dtype 无损、verify、dump、profiling、并发约束） | 框架验证 | `GE_SESSION_DEMO/doc/README.md` §11 |
| E6 | onnx / onnxruntime / torch / torch_npu / torchair 栈 | 环境实测 | `python3 -c "import ..."` |
| E7 | AscendC / cannbot-dsl 自定义算子开发路径 | 官网文档 | hiascend.com CANN 文档 |
| E8 | 算子参考实现仓 | GitCode | gitcode.com/Ascend（ops-* / op-plugin） |
