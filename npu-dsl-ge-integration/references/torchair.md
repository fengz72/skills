# TorchAir（PyTorch）前端接入与实测

## 已验证范围

`assets/ge-torchair/` 已对 cannbot-dsl 与 TileLang 的连续一维 FP32 Add 完成 TorchAir 图编译、优化后 GE 节点断言及 NPU 精度验证。固定 shape 分别为 `[256]` 与 `[524288]`；动态示例支持 `1 <= N <= 4096`，本轮执行重复和缩小序列。上述尺寸属于示例特化，不是硬件参数；不将结论推广到其他 dtype、layout、算子或增长路径。

完整实测包含固定 shape 4 个场景、动态 shape 8 个场景，共 120 次调用、6,435,944 个元素，最大绝对误差与 RMSE 均为 0。使用此前已验证的 DSL 导出件，本轮重新构建 GE 实现与 OPP，再编译前端图；未重新导出 DSL kernel。具体环境、命令、源文件和 OPP 哈希保存在任务报告，不固定在 skill 中。

## 前置条件与运行

- 初始化目标 CANN 环境，确认配套的 torch、torch_npu 与 TorchAir 能实际导入。有些安装由 torch_npu 提供 bundled TorchAir，应先导入 torch/torch_npu 再判断；单凭导入前的 `find_spec("torchair")` 为空不能认定未安装。
- 按 [ge-smoke.md](ge-smoke.md) 和 [dynamic-shrink.md](dynamic-shrink.md) 构建独立的固定/动态 OPP，并通过 `scripts/check_opp.py`。环境与参数来源见 [environment.md](environment.md)。
- GE 库匹配 GE 的 C++ ABI；若另写 torch C++ 扩展，其 ABI 匹配 torch。两者可不同，本示例仅经 ctypes/C 接口读取计数，不构建 torch C++ 扩展。
- `python assets/ge-torchair/torch_add.py` 的 `probe()` 记录实际导入路径及公开 API 签名，不注册 converter。确认 `register_fx_node_ge_converter`、`ge.custom_op`、`ge.TensorMove`、`CompilerConfig` 与 backend 工厂存在后适配。

```bash
source "${CANN_ENV_SCRIPT:?set CANN_ENV_SCRIPT}"
python "${SKILL_DIR:?set SKILL_DIR}/scripts/run_torchair.py" \
  --fixed-opp-root "${FIXED_OPP_ROOT:?set FIXED_OPP_ROOT}" \
  --dynamic-opp-root "${DYNAMIC_OPP_ROOT:?set DYNAMIC_OPP_ROOT}" \
  --device "${DEVICE_ID:?set DEVICE_ID}" \
  --work-dir "${WORK_DIR:?set a new WORK_DIR}" \
  --stage-timeout "${STAGE_TIMEOUT:-900}"
```

运行器在两个独立子进程中分别测试固定与动态 OPP；仅在这些验证进程内设置 `ASCEND_CUSTOM_OPP_PATH`，不改父进程/全局 OPP。两个 fixture 库使用同名 C 导出且 `SmokeLaunch` 签名不同，不能在同一进程以全局符号混载。生产工程可使用独立符号与库命名；这个隔离要求来自示例的符号设计。

## 前端实现与一致性

1. **schema/Meta**：`torch_add.py` 提供两个 DSL 的固定 Add、动态 Add 和动态 AddOut 共六个 op。普通 schema 返回 Tensor；可变输出使用 `(Tensor x, Tensor y, Tensor(a!) out_buffer) -> ()`，模型调用后返回 `out_buffer`。Meta 校验连续一维 FP32、输入尺寸和示例范围；动态维用 `torch._check`，不转换为 Python int。可变 op 的 Meta 与 converter 都返回 None，匹配无返回值 schema，由 functionalization 传递修改。
2. **加载和注册**：在 TorchAir/GE 初始化前设置目标 OPP，调用 `load_opp()` 保持注册库句柄，再显式调用 `register_converter()`。模板不提供 eager 计算，PrivateUse1 eager 路径直接报错；验证使用 `fullgraph=True`。确需 eager bridge 时单独实现并验证。
3. **GE 映射**：固定节点是 `CannbotAddSmoke`/`TilelangAddSmoke`；动态节点为 `CannbotDynamicAdd`/`TilelangDynamicAdd`，可变输出追加 `Out`。converter 的 inputs/outputs/attrs 对齐 GE 原型，AddOut 的引用输入与输出均名为 `out_buffer`。torch schema 的参数名可不同，但顺序、类型与默认值三侧一致；export 可省略默认实参，converter 应补齐原型默认值。
4. **输出所有权**：本例动态普通 Add 的 converter 在自定义节点后显式接 `ge.TensorMove`，使返回结果拥有可保留的存储。未加时当前输出精度虽正确，持有的旧输出会在后续执行被覆盖。不能以编译图内的 `aten.clone` 替代：本次安装中其 converter 返回 `ge.Identity`。TensorMove 已在默认优化配置下验证，无需关闭通用优化。
5. **编译与证据**：用计数 backend 包装 `torchair.get_npu_backend(compiler_config=config)`，保存 FX 与优化后 GE pbtxt，并对每张优化图断言唯一目标 GE type；动态普通 Add 还断言 TensorMove。数值通过、编译次数、GE 节点与 C++ 回调共同证明入图执行，不仅看输出。

## 验证矩阵与模型边界

| 场景 | 每个 DSL 的调用 | 检查 |
|---|---:|---|
| 固定 shape，输入地址复用/轮换 | 各 6 次 | 每场景编译一次；输出持有、精度、输入及旧分配不变 |
| 动态普通 Add，输入地址复用/轮换 | 各 12 次 | shape、模型复用、实际 kernel 输入地址、输出所有权 |
| 动态 AddOut，I/O 地址复用/轮换 | 各 12 次 | 可变输出别名、kernel I/O 地址、外部输出前后及逻辑尾部哨兵 |

动态序列为 `4096,4096,4096,2049,2049,1025,257,3,2,2,1,1`。`N >= 2` 通过 `mark_dynamic(..., min=2, max=4096)` 在同一编译图中执行前 10 步：编译一次、每步一次动态 Execute、node context 相同。长度 1 被本次 TorchDynamo 特化为第二张图，单独记录两次调用；不能把整段序列声称为一张 GE 模型。总计 20 次前端图编译。换版本时若特化行为变化，应检查 FX/guards/GE 图，再调整测试期望，不能仅删除断言。

固定图在加载时录制一次 host Execute，后续设备任务重放，因此 6 次模型调用不要求 6 次 host 回调。动态非单位维路径每次读取当前 context；多个节点/模型可能共享 C++ 实现对象，不能用实现对象地址代替模型身份，也不能在该共享对象保存跨模型的 last-shape。缩小序列由测试运行器控制。

地址检查分两层：调用方保留旧 Tensor，确保轮换确实得到不同地址；C++ 日志记录 kernel 的 x/y/z。动态 AddOut 轮换场景的前 10 步，两个 DSL 均实际观察到 kernel x/y/z 各 10 个不同地址。TorchAir functionalization 可能先写临时输出再回写用户目标，本次 kernel z 与用户目标不同；验证了前端可变语义，不宣称零拷贝。普通动态 Add 的 kernel z 复用内部存储，TensorMove 后的用户输出各自持有。

精度使用独立 CPU FP64 加法再转 FP32 的参考，容差 `1e-6 + 1e-6 * abs(reference)`；逐步检查输入、旧分配、持有的普通输出。可变输出检查前缀、整个逻辑尾部及后部哨兵；复用可变目标时允许其旧值按语义更新。轮换输入使用精确逻辑尺寸分配，没有额外外部 guard；GE 内部输出也没有外部哨兵。顺序执行多个模型的通过结果不证明并发支持。

`results.json` 保存编译次数、每步 shape/精度、调用方及 kernel 地址、图文件哈希、实际 API 与 OPP 哈希；日志和 FX/GE 图保留在工作目录。常见失败与修正见 [troubleshooting.md](troubleshooting.md)。

## 参考

公开接口定位参考 [TorchAir API 列表](https://github.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/api/api_list.md)与[官方 converter 示例](https://www.hiascend.com/document/detail/zh/Pytorch/600/modthirdparty/torchairuseguide/torchair_0044.html)，具体签名以实际安装及 probe 为准。更复杂的 schema/eager/converter 示例见 [prefix-attention-example.md](prefix-attention-example.md)。
