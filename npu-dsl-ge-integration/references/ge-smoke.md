# 两个 DSL 的真实 NPU / GE 验证工程

仅在用户请求实际 NPU/GE 验证时使用。此工程是固定规格的接入参考，不是任意算子的自动包装器。

## 运行

先确认设备可用和逻辑 device ID，进入已配置的目标容器，source 当前 CANN 环境。`--work-dir` 必须是新目录；脚本不改写全局 OPP 或用户的 DSL/GE 仓库。

变量设置见 [environment.md](environment.md)。

```bash
source "${CANN_ENV_SCRIPT:?set CANN_ENV_SCRIPT}"
python "${SKILL_DIR:?set SKILL_DIR}/scripts/run_ge_smoke.py" \
  --work-dir "${WORK_DIR:?set WORK_DIR}" \
  --device "${DEVICE_ID:?set DEVICE_ID}" --arch "${NPU_ARCH:?set NPU_ARCH}" \
  --cxx11-abi "${CXX11_ABI:?set CXX11_ABI}" \
  --tilelang-root "${TILELANG_ROOT:?set TILELANG_ROOT}"
```

新运行换一个工作目录名。脚本从当前 skill 的 `assets/ge-smoke/` 复制工程，同时复制 `assets/ge-common/`，使用两个导出脚本重新生成源码/binary，构建独立 OPP 和验证程序，再启动六个独立测试进程：

| DSL | 算子和固定输入 | NPU direct | GE 外部输出 | GE 自动分配地址 |
|---|---|---|---|---|
| cannbot-dsl | SIMT FP32 Add，`[256]`，ND | 六组输入 | 同一模型六次执行 | 同一模型六次执行 |
| TileLang Ascend 后端 | SIMT FP32 Add，`[524288]`，ND | 六组输入 | 同一模型六次执行 | 同一模型六次执行 |

每条路径使用独立 CPU 参考（double 加法后舍入到 float32），逐元素比较、检查非有限值和输入未被改写。调用方提供输出时前后各 256 bytes 哨兵检查越界。GE 自动分配的输出不声称有哨兵覆盖。

## 工程阅读顺序

- `cannbot_add.py`：AOT driver 与最小真实计算，使用安装版本的 `vf(mode='simt')`。
- `prepare.py`：从 manifest 提取 ABI 符号，将 TileLang executable ELF 嵌入编译单元；从匹配的 host IR 提取 grid/dynamic UBUF。它仅接受这一个已知 vecadd IR 形式，其他算子需重新分析，不能放宽正则来猜测参数。
- `launcher.cc`：cannbot schema/frame 调用；TileLang binary-load + function lookup + 带动态 UBUF 配置的 launch；按 context 隔离句柄并显式卸载。
- `custom_op.cc` / `ops.h`：两个独立 GE type 的注册、shape/dtype/format 校验、`MallocOutputTensor` 和 GE stream 调用。
- `main.cc`：直接 NPU 与 GE Session 两条入口，构图为两个 Data 节点加一个自定义节点；记录保存的 AIR 和 GE 回调次数。
- `CMakeLists.txt`：使用传入的 C++ ABI，构建独立 OPP 并配置 kernel 依赖库相对路径。
- `assets/ge-common/`：从产物识别 cannbot ABI、选择已实现的 frame 解码器；frame 容量和对齐由 runtime schema 提供。

## 实测修正的接入细节

### TileLang dynamic UBUF 不等于 workspace

本特化 host IR 的末尾是 `grid=64, dyn_ubuf=196608`。grid 作为 launch 的 block 数；UBUF 通过 `ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE` 和 `attr.value.dynUBufSize` 设置，配置传给 `aclrtLaunchKernelWithHostArgs`。设备参数只有三个指针。不要为 UBUF 调用 GE `MallocWorkSpace` 并增加一个指针参数。

### 静态图的默认输出描述符不会自动变成预期 shape

本环境静态图 `ExecuteGraphWithStreamAsync` 返回了 GE 分配的地址，但默认构造的输出 Tensor 仍保留默认标量 shape，导致 `GetShapeSize()==1`。正确做法是在调用前填好输出的 origin/storage shape、format、placement、dtype；需要 GE 分配地址时地址设为 `nullptr`，需要外部输出时填入该缓冲区地址。输出有效性在 stream 同步后检查。

这与动态执行器的输出元信息更新行为不能混为一谈；也不能从算子内部调用 `MallocOutputTensor` 推断外层 Session API 一定会更新所有输出描述字段。

### Execute 可能是加载期录制回调

本静态 shape 模型中 `Execute` 在 `LoadGraph` 期间调用一次，随后六次模型执行均使用新输入内容，GE 回调计数仍为 1。不要把回调日志出现六次当作 GE 已执行六次的必要条件。区分“host 回调录制任务”和“设备模型重放”。

### 构图、C++ 与链接

- 显式 `graph.AddOp` 加入节点，再设置输入/输出，使图枚举与保存 AIR 都有可核验的三个节点。构图时只有 SetInputs/SetOutputs 的惰性图，AddGraph 前枚举可返回 0，不能据此证明没有算子。
- 本版本 `gert::Tensor` 不可复制。`vector<Tensor>{tensor1,tensor2}` 的 initializer_list 要求复制而编译失败；先创建定长 vector，再 move 赋值。
- cannbot 导出的 `add_host.so` 没有 `lib` 前缀，直接交给当前 CMake 会被转成 `-ladd_host` 并找不到库。工程复制为 `libcannbot_add.so`，让 DT_NEEDED 使用可解析的库名，并配置 `$ORIGIN/artifacts` 的 RPATH。核对 `readelf -d` 的 NEEDED/RPATH/RUNPATH 和 `ldd`。

## 验收与限制

成功报告 `results.json` 应有六条通过路径，共 36 次调用、9,441,792 个元素比较。每条 GE 路径必须有 AddGraph/CompileGraph/LoadGraph 成功、实际 custom Execute 回调、六组数值结果与清理成功；不能只看进程退出码或构图成功。

`environment.log`、`device.log`、`ldd-runner.log`、产物 manifest、库哈希、保存的 AIR 和各路径日志都在工作目录。精度门限为 `1e-6 + 1e-6 * abs(reference)`；本次实测所有路径最大绝对误差为 0。

范围限于固定 FP32 ND、单 device/context、串行执行的原生 GE 静态模型。没有验证任意动态 shape、零拷贝地址刷新、多模型并发、TorchAir/ONNX 前端或离线 OM。不能把本结果直接扩大到这些场景。

其他工程的独立验证范围见 [torchair.md](torchair.md) 与 [onnx-om.md](onnx-om.md)；上述限制描述本原生 GE smoke 工程，不表示这些前端没有执行证据。
