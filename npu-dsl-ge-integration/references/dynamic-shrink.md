# 动态 shape 缩小与地址变化验证

## 示例范围

本工程验证同一动态 GE 模型在 shape 重复、缩小及输入输出地址变化时的执行结果。脚本与 C++ 运行器使用单调不增的 shape 序列，外部缓冲区固定保留首轮最大容量。增长后的输出越界问题另见[错误排查案例](troubleshooting.md)。

## 运行方式

使用当前 skill 的 `scripts/run_dynamic_shrink.py`。脚本复制 `assets/ge-dynamic-shrink/` 与 `assets/ge-common/` 到新目录，从两个 DSL 源码重新生成各一份动态 artifact，构建独立 OPP 并验证 NPU 和 GE。模板依赖相应的 cannbot AOT、TileLang Ascend 动态维与 GE custom_op_v2 接口；运行前按实际安装核对 API 和 ABI。

变量设置见 [environment.md](environment.md)。

```bash
source "${CANN_ENV_SCRIPT:?set CANN_ENV_SCRIPT}"
python "${SKILL_DIR:?set SKILL_DIR}/scripts/run_dynamic_shrink.py" \
  --work-dir "${WORK_DIR:?set WORK_DIR}" \
  --device "${DEVICE_ID:?set DEVICE_ID}" --arch "${NPU_ARCH:?set NPU_ARCH}" \
  --cxx11-abi "${CXX11_ABI:?set CXX11_ABI}" \
  --tilelang-root "${TILELANG_ROOT:?set TILELANG_ROOT}"
```

工作目录必须不存在；再次运行需另选目录。先 source 目标 CANN 环境并确认设备，脚本不修改全局 OPP、工具链、GE 或 DSL 仓库。

## 用例与证据

FP32 ND 向量 Add 的运行时维度序列为：

```text
4096, 4096, 4096, 2049, 2049, 1025, 257, 1, 1
```

重复 4096/2049/1 用于隔离“相同 shape 的多次执行/地址变化”与真正的 shape 缩小。2049/1025/257/1 覆盖非 tile 对齐和短尾。每个 DSL 只生成一份支持 runtime N 的 kernel，GE 输入描述为 `[-1]`；每条 GE 路径只 AddGraph/CompileGraph/LoadGraph 一次，连续执行九步，不按 shape 重编译或重建模型。

| 模式 | 地址与内存策略 | 要验证的内容 |
|---|---|---|
| direct | 每轮换一套输入/输出 | 动态 kernel 本身的数值、尾块和实际参数更新 |
| reuse | 普通 Add，同一套调用方地址 | 同一动态 GE 模型缩小时，地址稳定、逻辑大小改变 |
| rotate | 普通 Add，每轮换调用方输入/输出地址 | GE 公共 I/O 地址变化；同时记录真正的 kernel 地址 |
| auto | 每轮换输入，GE 分配输出 | GE 内部输出从首轮最大容量开始复用，返回 shape/容量正确 |
| alias | AddOut 引用每轮更换的目标输入缓冲区 | kernel 的实际 x/y/z 都改变，验证真实目标地址传递 |

外部物理缓冲区始终保留首轮最大 4096 元素容量，前后各加 4096 bytes 哨兵；有效输出预填 NaN，当前逻辑输出末尾直到物理容量末尾均填哨兵。既比较有效元素，也检查缩小后没有继续写旧尺寸的尾部。旧缓冲区保留到模型卸载，逐次比对快照，避免 allocator 重用同一地址而形成“地址变了”的假阳性，并检查旧地址未被写入。

GE 自动分配的内部输出没有外部哨兵覆盖，不宣称该内存有边界检查；其数值、描述符、容量和地址被记录。输入始终检查有效内容、padding 和前后边界未被改写。

每步使用独立 CPU double 加法后舍入 float32 的参考结果，门限 `1e-6 + 1e-6 * abs(reference)`，拒绝非有限输出。日志分别记录：

- `[STEP]`：调用方实际地址、shape、逻辑输出和检查结果。
- `[GE_DYNAMIC_EXECUTE]`：kernel 真正接收的地址、GE node context/实现对象、ACL context、内部容量。
- `[RESULT]`：调用次数、唯一地址数、误差、只编译一次模型及 `growth_executed=0`。

不要只数 malloc/free 次数来证明地址变化；要核对实际指针。普通动态 Add 的 kernel 输出在当前环境复用内部 buffer，外部输出地址轮换由 GE 拷贝处理，因此该路径不证明 kernel 输出指针变化。alias 路径必须确认 kernel x/y/z 均有九个不同值，并逐步与调用方地址相同。

## 实际发现并修正的问题

### cannbot-dsl：动态 shape 的读取位置

在 `vf(mode='simt')` 内直接读取 `x.shape[0]` 会在当前版本生成 VF 内不允许的 `cannir.extract_layout`，codegen 失败。先在 VF 外读取 `n=x.shape[0]`，再在 SIMT VF 内使用 n。AOT 使用同一个 `Dim('N', min=1, max=4096)` 描述三个张量，frame 每次填写实际 shape 和 stride，不能继续使用固定示例的 256。

### TileLang：动态维成为设备参数

动态 factory 使用 `T.dynamic('N', dtype='int64')`。生成设备入口为三个指针后接 `int64_t N`；host IR 为 `main_kernel(A,B,C,N,grid,dynamic_ubuf)`。该示例曾生成 grid=1、dynamic UBUF=3072；准备脚本从本次 host IR 读取实际值，后二者仍是启动元数据。launcher 必须将本轮 N 按 int64 写入参数结构，不能沿用固定 shape 时的三指针参数包。

`T.copy` 的尾块在当前后端生成按剩余逻辑长度截断的 DMA；此结论由本轮 2049/1025/257/1 的数值和逻辑尾部哨兵验证，不能推断其他算子的尾块实现也正确。

### MakeOutputRefInput：原型输入输出必须同名

当前实现会校验 `op_desc` 中 input_index 对应的名称等于 output_index 的名称。将输入命名为 `out_buffer`、输出仍命名为 `z` 会报 `[MakeOutputRefInput] output name does not exist in input`。

AddOut 原型使用输入 `out_buffer` 与同名输出 `out_buffer`，再调用 `MakeOutputRefInput(0,2)`。这是一条显式的目标缓冲区引用契约，允许写入第三个输入；普通 Add 的两个数值输入仍不可改写。不要仅为测试而偷偷修改普通算子的别名语义。

### 动态与静态执行回调不同

前一轮固定 shape 模型在加载时录制一次 Execute；本轮动态模型每次执行均进入 Execute。记录同一场景中的 node context、编译/加载次数与实现对象。TorchAir 可让多个模型节点共享同一个实现对象，因此实现对象地址相同本身不证明复用了同一个模型。算子 Execute 读取当前 context，不在共享实现对象上保存跨模型的 last-shape；单调缩小由测试运行器控制。

## 结果解释

每个 DSL 五条路径、每条九步，共 90 次调用、176,700 个元素比较。该数量含两条 direct 路径；八条 GE 路径各为一个模型连续执行九次。实际结果以工作目录 `results.json` 为准，数值与地址证据不以 skill 文字替代。

历史验证从新目录完整复现通过：最大绝对误差/RMSE 均为 0，所有外部哨兵、逻辑尾部及旧缓冲区检查通过。两条 alias 路径的 kernel x/y/z 均有九个唯一地址并与调用方逐步一致；GE node context/实现对象在各场景九步内各保持一个。

`[STEP] capacity` 是调用方返回 Tensor 的 GetSize，`[GE_DYNAMIC_EXECUTE] capacity` 是节点内部 Tensor 的 GetSize；二者可能不同。不能把返回视图的逻辑尺寸变小解释成底层物理分配真的缩小。

本次覆盖上述形状集合内的动态缩小、公共 I/O 地址变化，以及显式 AddOut 路径的实际 kernel 地址变化。增长、底层物理分配缩小、静态下沉模型的 ArgsUpdater/AnnotatedArgs、并发及跨 context 不在本示例的验证范围内。

TorchAir 前端上的动态 shape、长度 1 特化、可变输出与地址变化另见 [torchair.md](torchair.md)。前端 functionalization 可能引入临时输出，不能直接沿用原生 Session 的零拷贝结论。
