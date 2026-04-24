# 多进程数据并行

## 概述

EasyStacker 的多进程执行采用**数据并行**（Data Parallelism）架构：将 DAG 中可并行的完整管段（I/O + Map 链 + 多终端 Reduce / DiskBuffer + 迭代式 Reduce）整体复制到 N 个 worker 进程，每个 worker 处理 1/N 帧并独立完成解码 → 处理 → 局部归约 → 迭代 Reduce 的全流程。主进程仅做轻量的路径分发、partial merge 和收敛判断。

与传统的 Op 粒度多进程方案（将各 Op 分配到不同进程，跨进程边界传输图像数据）不同，数据并行方案的 IPC 开销极低：

- **入口**：仅传输文件路径字符串（~100 bytes/帧），不传图像数据
- **出口**：仅 N 个 partial result（N = worker 数），不是每帧一个
- **Phase 2 迭代**：broadcast 单块 SharedMemory，所有 worker 只读 attach（物理页面仅 1 份）

典型场景下 IPC 总量对比（100 帧 6000x4000 RGB float32 图像，每帧 ~288MB）：

| 场景 | Op 粒度多进程 | 数据并行 |
|------|-------------|---------|
| 简单叠加（3 个跨进程边界） | 100 x 288MB x 3 = **~86 GB** | 100 x 100B + 3 x 288MB = **~864 MB** |

YAML 配置层完全不感知多进程，用户无需修改任何 DAG 定义。

## 架构总览

```
                        Main Process (asyncio event loop)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                                                                          │
 │   Feeders ──► SegmentAdapter ──► 非段化 Ops ──► 结果收集                   │
 │                  │        ↑                                              │
 │      dispatch()  │        │  collect()                                   │
 │                  │        │    Phase 1: merge partials                   │
 │    IPC 入口:     │        │    Phase 2: broadcast ref → collect → merge  │
 │    文件路径      │        │    Phase 结束: finish → cleanup               │
 │    (~100B/帧)    │        │                                              │
 └──────────────────│────────│──────────────────────────────────────────────┘
                    │        │
    ┌───────────────▼────────┴─────────────────────────────────────────────┐
    │  Worker 0:                                                            │
    │    Phase 1:  I/O → Map → Reduce + DiskBuffer → partial_0             │
    │    Phase 2:  iterate(ref) → local_buffer → clip_partial_0            │
    ├──────────────────────────────────────────────────────────────────────┤
    │  Worker 1:  (同上)                                                    │
    ├──────────────────────────────────────────────────────────────────────┤
    │  Worker 2:  (同上)                                                    │
    └──────────────────────────────────────────────────────────────────────┘
```

## 执行流程

入口函数 `run_from_yaml()` 的执行流程：

```
run_from_yaml(yaml_path, inputs, configs, num_workers=N)
     │
     ├─ 1. 加载 YAML spec
     │
     ├─ 2. meta_resolve()  [仅 .meta.yaml: 编译路由选择]
     │
     ├─ 3. flatten_sub_dags()  [展平 .yaml SubDAG 引用为扁平拓扑]
     │
     ├─ 4. validate_and_build_order() → ValidatedDag
     │
     ├─ 5. run_dag_multiprocess()
     │      │
     │      ├─ 5a. instantiate_and_wire()
     │      │       标准布线：实例化 Op、连接队列、创建 feeder 协程
     │      │
     │      ├─ 5b. apply_data_parallelism()
     │      │       段检测 → 段替换 → 返回新的 ops 列表
     │      │       (如果无可并行段，回退到单进程 run_dag)
     │      │
     │      ├─ 5c. DAGExecutor(new_ops).execute()
     │      │       并发执行所有 ops（SegmentAdapter + 非段化 ops）
     │      │
     │      └─ 5d. 收集全局输出
     │
     └─ 返回 results
```

**关键：`flatten_sub_dags()` 在 `validate_and_build_order()` 之前执行**，将所有 `.yaml` 子图引用展平为带命名空间前缀的顶层节点。这使得段检测可以穿透子图边界，对 `sigma_clip.yaml`、`huber_mean.yaml` 等子图内部的节点也能进行数据并行化。

当 `num_workers <= 1` 或未检测到可并行段时，自动回退到单进程模式。

## SubDAG 预展开

`flatten_sub_dags()` 在编译期（spec 级别）递归展开所有 `.yaml` 引用的子图：

### 展开规则

1. 子图内部节点添加 `"{parent_name}."` 前缀
2. 子图内部引用（`node.output`）添加前缀
3. 子图 `inputs.xxx` 替换为父图实际 src（未布线的可选输入标记为 `__inactive__`）
4. 子图 `configs.xxx` 替换为父图实际 src（未覆盖且有默认值的省略，由 Op 自身 CONFIGS default 补齐）
5. 父图中引用此 SubDAG 输出的消费者重定向到展平后的实际链接
6. 支持嵌套：展开后若仍有 `.yaml` 引用则继续迭代（最多 10 层深度）

### 展开示例

`sigma_clip.yaml` 作为 `simgaclipstacker` 展开前后：

```yaml
# 展开前（父图）:
simgaclipstacker:
  op: sigma_clip.yaml
  inputs: { data: data_loader.result }
  configs: { int_weight: configs.int_weight, rej_high: configs.rej_high, ... }

# 展开后（3 个独立节点）:
simgaclipstacker.mean_stacker:
  op: MeanStackerOp
  inputs: { data: data_loader.result }
  configs: { int_weight: configs.int_weight }

simgaclipstacker.disk_buffer:
  op: DiskBufferWriterOp
  inputs: { data: data_loader.result, fnames: __inactive__ }

simgaclipstacker.sigma_clip_iter:
  op: SigmaClipIteratorOp
  configs:
    fgp_total: simgaclipstacker.mean_stacker.statistics
    buffer_handle: simgaclipstacker.disk_buffer.buffer_handle
    rej_high: configs.rej_high
    ...
```

### Meta SubDAG 处理

如果子图包含 `routes` / `route_key`（Meta YAML），展开前自动调用 `meta_resolve()` 解析路由选择。

### `_parse_link` 兼容命名空间

`build.py` 的 `_parse_link()` 使用 `rsplit(".", 1)` 解析 link，因此嵌套命名空间如 `simgaclipstacker.mean_stacker.statistics` 会被正确解析为 `node="simgaclipstacker.mean_stacker"`, `output="statistics"`。

## Op 分类与并行策略

引擎通过 Op 的类属性决定其在数据并行中的角色：

### 类属性

| 属性 | 类型 | 含义 |
|------|------|------|
| `DATA_PARALLEL` | `bool` | 是否允许被纳入数据并行段 |
| `DECOMPOSABLE` | `bool` | Reduce op 是否支持分布式归约 |
| `IS_DISK_BUFFER` | `bool` | 是否为磁盘帧缓冲 Op（段终端类型之一） |
| `BUFFER_ITERATOR` | `bool` | 是否为消费 buffer 的迭代式 Reduce |
| `ITERATOR_TYPE` | `str` | 迭代类型标识：`"sigma_clip"` / `"huber_mean"` / `"median"` |
| `EXECUTOR` | `str\|None` | 执行模式：`"cpu"` = CPU 密集，`None` = 主进程 |

### 分类规则

| 类别 | 判定条件 | 段内角色 | 示例 |
|------|----------|----------|------|
| **帧级 I/O** | `DATA_PARALLEL=True` 且非 `ParallelBaseOp`/`FilterBaseOp` | 段头 | `ImgDataLoaderOp` |
| **Map (N→N)** | `ParallelBaseOp` 且 `DATA_PARALLEL=True` | 段中 | `CalibrationSubtractOp`, `CalibrationDivideOp` |
| **可分解 Reduce** | `DECOMPOSABLE=True` 且 `EXECUTOR="cpu"` | 段尾终端 | `TrailStackerOp`, `MinStackerOp`, `MeanStackerOp` |
| **磁盘帧缓冲** | `IS_DISK_BUFFER=True` | 段尾终端 | `DiskBufferWriterOp` |
| **迭代式 Reduce** | `BUFFER_ITERATOR=True` | 段关联 iterator_op | `SigmaClipIteratorOp`, `HuberMeanIteratorOp` |
| **不可分布式迭代** | `BUFFER_ITERATOR=True`, `ITERATOR_TYPE="median"` | 主进程回退 | `MedianReduceOp` |
| **Transform/Generate** | 无序列输入或非 CPU | 主进程 | `WeightGeneratorOp`, `LoadMaskImageOp` |
| **Filter (N→M)** | `FilterBaseOp` | 未来扩展 | `StarAlignmentOp` |

### 终端类型

段尾支持两种终端类型，可在分支点处共存形成**多终端段**：

```python
class TerminalType(Enum):
    DECOMPOSABLE_REDUCE = "decomposable_reduce"  # MeanStacker, TrailStacker
    DISK_BUFFER = "disk_buffer"                  # DiskBufferWriterOp
```

### merge_partial 协议

可分解 Reduce op 必须实现 `merge_partial()` 类方法，用于合并 N 个 worker 的局部结果：

```python
class TrailStackerOp(BaseOp):
    DECOMPOSABLE = True

    @classmethod
    def merge_partial(cls, partial_results: list[dict[str, Any]]) -> dict[str, Any]:
        # 复用 Merger._merge 做逐对归约
        merger = cls.MERGER(int_weight=False)
        merger.result = partial_results[0][merge_key]
        for partial in partial_results[1:]:
            merger.result = merger._merge(merger.result, partial[merge_key])
        return {"result": merger.merged_image}
```

各 Reduce op 的 merge 语义：

| Op | merge 操作 |
|----|-----------|
| `TrailStackerOp` (Max) | `np.maximum` 逐像素取最大值 |
| `MinStackerOp` | `np.minimum` 逐像素取最小值 |
| `MeanStackerOp` | `FastGaussianParam.__add__` 合并统计量，再提取均值 |

## 段检测算法

`detect_parallel_segments()` 从 DAG 拓扑序中检测可并行段。

### 段结构

```
段: io_ops → map_ops → terminal(s)
                           ↓
                     iterator_ops (消费 DiskBuffer 终端的迭代式 Reduce)
```

### 算法步骤

1. 遍历拓扑序，找到所有 `DATA_PARALLEL=True` 的帧级 I/O op 作为段头起点
2. 从段头沿正向邻接表（provider → consumer）向下延伸
3. 连续收集满足条件的 Map ops
4. 检查末端：
   - **单消费者**：
     - 可分解 Reduce → 纳入段尾（单终端段）
     - DiskBuffer → 纳入段尾 + 扫描下游 iterator ops
     - Map/I/O eligible → 继续延伸
   - **多消费者（分支点）**：
     - **所有**分支都是可分解 Reduce 或 DiskBuffer → 构建多终端段
     - 任一分支不可段化 → 段终止
5. DiskBuffer 终端的下游 `BUFFER_ITERATOR` ops 纳入 `iterator_ops`

### 段边界切断条件

- 下一个 Op 不满足 `DATA_PARALLEL` 或 `ParallelBaseOp` 条件
- 遇到 `FilterBaseOp`
- 遇到不可分解的非 DiskBuffer/非 BUFFER_ITERATOR op
- 分支点处有任一不可段化的消费者
- 下一个 Op 有来自段外的不可处理的序列输入（Reduce 的额外输入除外，如 `weight`）

### Reduce 额外输入处理

部分 Reduce op 除主序列输入外，还有段外的额外序列输入。例如 `TrailStackerOp` 有：
- `data`：主序列输入（来自 Map 链输出 / I/O）
- `weight`：额外序列输入（来自段外的 `WeightGeneratorOp`）

段检测时将额外输入记录在 `SegmentTerminal.extra_inputs` 中。Dispatcher 在分发每帧时，同步读取额外输入并与主输入打包一起发给 worker。

## SegmentAdapter

`SegmentAdapter` 是 `BaseOp` 的子类，在 DAG 中替代整个并行段（包括所有终端和 iterator ops）。对 `DAGExecutor` 而言，它表现为一个普通的 Op。

### 内部组件

```
SegmentAdapter._async_execute()
    │
    ├─ 创建 per-worker IPCQueue（input + output）
    ├─ 构建 segment_info（可 pickle 的段描述 dict）
    ├─ 启动 N 个 mp.Process(target=_segment_worker_main)
    ├─ 等待所有 worker 就绪（ready_event）
    │
    ├─ asyncio.gather(
    │       _dispatch(input_ipcs),     # 分发帧到 workers
    │       _collect(output_ipcs,      # Phase 1: 收集 + merge partials
    │                input_ipcs),      # Phase 2: broadcast → collect → merge
    │   )                              # Phase 结束: finish → cleanup
    │
    └─ finally:
            done_events.set()  # 通知 workers 可以退出
            join + cleanup
```

### Dispatch（分发）

Dispatcher 在主进程运行，负责从上游队列读取输入并 round-robin 分发到各 worker：

```
Frame 0  → Worker 0
Frame 1  → Worker 1
Frame 2  → Worker 2
Frame 3  → Worker 0
Frame 4  → Worker 1
...
```

分发的数据是文件路径（~100 bytes），不是解码后的图像。每个 worker 独立解码自己分到的帧。

### Collect（收集）—— 多阶段协议

**Phase 1: 收集 N 个 worker 的流式处理 partial**

每个 worker 完成 Phase 1 后发送 partial result dict，包含：
- 每个 DECOMPOSABLE_REDUCE 终端的归约结果
- DiskBuffer 终端的描述符（`__disk_buffer`）

Collector 收集 N 个 partials 后：
- 对每个 DECOMPOSABLE_REDUCE 终端调用 `reduce_cls.merge_partial()` 合并为最终结果，推送到下游
- 解析段内 config 连接（如 `sigma_clip_iter.fgp_total ← mean_stacker.statistics`）

**Phase 2: 分布式迭代 Reduce**

对段内的每个 iterator op，根据 `ITERATOR_TYPE` 执行分布式迭代：

- **sigma_clip**：多轮迭代
  1. `broadcast_pack(ref_fgp)` → 单块 SharedMemory，所有 worker 只读 attach
  2. 每个 worker 遍历 local buffer → SigmaClippingMerger → partial rejected FGP
  3. 主进程 merge rejected → `accepted = fgp_total - total_rejected`
  4. 收敛检查 → 未收敛则 `ref_fgp = accepted`，重复
- **huber_mean**：单轮迭代，流程同上但无收敛循环
- **median**：不可分布式，从 worker 描述符重建 buffer 在主进程串行计算

**Phase 结束: 通知 worker 退出**

```python
for ipc in input_ipcs:
    await ipc.put({"action": "finish"})
```

Worker 收到 finish 后清理 local buffer 并退出。

## Worker 进程

### 入口

```python
_segment_worker_main(segment_info, all_configs, input_ipc, output_ipc,
                     cancel_event, tracker_queue, worker_id,
                     dag_search_paths_str, ready_event, done_event)
```

### 内部执行流程

```
1. import 注册表，实例化段内所有 ops
2. ready_event.set()  # 通知主进程就绪
3. 运行 asyncio event loop:

   ── Phase 1: 流式处理 ──
     a. 从 input_ipc 获取帧数
     b. 为每个 Reduce 终端构建本地队列 + 启动 reduce op 协程
     c. 若有 DiskBuffer 终端：创建本地 DiskFrameBuffer / SourceReplayBuffer
     d. 循环处理帧：
          input_ipc.get() → I/O ops → Map ops
          → 对每个 Reduce 终端: 喂入 local_reduce_queue（含 extra_inputs）
          → 对 DiskBuffer 终端: local_buffer.append(img, weight)
     e. 收集所有 Reduce 终端的 partial result
     f. 打包发送：大对象走 ShmTransportable，轻量 manifest 走 pickle
     g. 释放 Phase 1 大对象（仅保留 local_buffer）

   ── Phase 2+: 命令驱动循环 ──
     while True:
       cmd = await input_ipc.get()

       if cmd.action == "finish":
           break

       if cmd.action == "iterate":
           ref_data = broadcast_unpack(cmd.broadcast_shm, ...)  # 只读 attach
           if iter_type == "sigma_clip":
               clip_merger = SigmaClippingMerger(ref_data, ...)
               for idx in range(len(local_buffer)):
                   raw, weight = local_buffer[idx]
                   clip_merger.merge(raw, weight)
               await output_ipc.put(clip_merger.result)
           elif iter_type == "huber_mean":
               huber_merger = HuberWeightedMerger(ref_data, ...)
               for idx in range(len(local_buffer)):
                   raw, weight = local_buffer[idx]
                   huber_merger.merge(raw, weight)
               await output_ipc.put(huber_merger.result)

     local_buffer.cleanup()

4. done_event.wait()  # 等待主进程读完 SharedMemory 再退出
```

### Worker 内 Reduce 布线

worker 内的 reduce op 通过本地 `RichContextQueue` 接收 Map 链的输出。这是流式归约——每收到一帧就合并到累积结果中，内存开销恒定（1 帧 + 1 个累积结果），不随帧数增长。

```
Map chain output                        Reduce op (per terminal)
     │                                     │
     ▼                                     ▼
 local_reduce_input (RichContextQueue) ──► reduce_op.execute()
                                              │
                                              ▼
                                    local_reduce_outputs ──► partial result
```

在多终端段中，每帧同时喂给所有终端（Reduce + DiskBuffer）。

额外序列输入（如 `weight`）使用独立的 `RichContextQueue`，从 `frame_input` 中提取后逐帧喂入。

### Phase 1 Partial 发送协议

为避免将整个 partial dict pickle 一次性序列化（内存放大），Worker 采用分离发送：

1. 遍历 partial dict，ShmTransportable 对象（如 FastGaussianParam）逐个通过 `output_ipc.put()` 走 SharedMemory 高效路径
2. 最后发送轻量 manifest dict（仅含描述符和 key 顺序），由主进程重组

## IPCQueue：跨进程通信

`IPCQueue` 继承 `BaseQueue`，提供与 `RichContextQueue` 完全一致的接口。Op 代码无需感知是否跨进程。

### 传输策略

| 数据类型 | 传输方式 | 开销 |
|----------|----------|------|
| `np.ndarray`（大于阈值） | SharedMemory 零拷贝 | 1x memcpy |
| `ShmTransportable` 对象（如 `FloatImage`, `FastGaussianParam`, `HuberMeanParam`） | 数组部分走 SharedMemory，元数据走 pickle | 1x memcpy + 少量 pickle |
| 小对象（< 32KB） | pickle via Pipe | 极低 |
| 大 pickle 对象（> 32KB） | pickle 字节写入 SharedMemory，Pipe 仅传引用 | 1x memcpy |
| Sentinel / CancellationToken | Pipe 控制帧 | 极低 |

### ShmTransportable 协议

`ShmTransportable` 是抽象基类，声明对象支持 SharedMemory 高效传输：

```python
class ShmTransportable(ABC):
    def shm_nbytes(self) -> int: ...       # 预计算所需 shm 字节数
    def shm_pack_into(self, buf) -> bytes: ... # 直写 shm buffer，返回元数据
    @classmethod
    def shm_unpack_from(cls, buf, meta): ...   # 从 shm buffer + 元数据重建
```

子类通过 `__init_subclass__` 自动注册到 `_SHM_REGISTRY`。当前注册的类型：
- `FloatImage` — 语义级图像（data + dtype）
- `FastGaussianParam` — 流均值/方差统计量（sum_mu + square_sum + n）
- `HuberMeanParam` — Huber 加权和（weighted_sum + weight_total）

### Broadcast Helpers

Phase 2 迭代 Reduce 需要将同一个 ref_fgp 发送给 N 个 worker。使用独立于 IPCQueue 的 broadcast 辅助函数：

```python
# 生产者（主进程）
bc_shm, bc_cls, bc_meta = broadcast_pack(ref_fgp)  # 单次 pack → 单块 shm
try:
    for ipc in input_ipcs:
        await ipc.put({
            "action": "iterate",
            "broadcast_shm": bc_shm.name,   # shm 名称（字符串）
            "broadcast_cls": bc_cls,
            "broadcast_meta": bc_meta,
            ...
        })
    # ... 收集 partials ...
finally:
    _safe_close_shm(bc_shm, unlink=True)    # 所有 worker 读完后释放

# 消费者（worker 进程）
ref_data = broadcast_unpack(
    cmd["broadcast_shm"], cmd["broadcast_cls"], cmd["broadcast_meta"])
# shm.close() 但不 unlink —— 由生产者统一 unlink
```

N 个 worker attach 同一块 SharedMemory，操作系统通过虚拟内存映射共享物理页面，无 N 倍内存放大。

### 背压控制

通过双信号量实现，语义与 `asyncio.Queue(maxsize=N)` 一致：

- `_empty_sem`：可用 slot 数（生产者 `put` 前 acquire）
- `_filled_sem`：就绪 item 数（消费者 `get` 前 acquire）

### SharedMemory 生命周期管理（Windows 兼容）

Windows 上 Named File Mapping 是引用计数的：最后一个 handle 关闭时立即销毁。这与 POSIX shared memory（`shm_unlink` 后仍可通过已有 fd 访问）不同。因此需要多层保护：

**1. 生产者持有 handle**：`put()` 创建 SharedMemory 后，handle 保存在 `_slot_shm` 队列中，直到确认消费者已读取后才关闭。

**2. `_put_count` 守卫**：前 `maxsize` 次 `put()` 的信号量 acquire 来自初始计数（不是消费者释放），此时消费者尚未读取任何数据。只有第 `maxsize + 1` 次 acquire 才对应消费者的第 1 次释放。因此清理阈值为 `_put_count > maxsize`（严格大于）。

**3. 消费者先读后释放**：`get()` 中 `_empty_sem.release()` 放在 `finally` 块中，确保 SharedMemory 内容已完整读取后才通知生产者可以关闭 handle。

**4. `done_event` 同步**：worker 进程在发送完 partial result 后，等待主进程设置 `done_event` 后才退出。防止进程退出导致所有 SharedMemory handle 被 OS 回收。

```
时序图（正常流程，含 Phase 2）：

Worker                          Main Process
  │                                 │
  ├── Phase 1 partial ─shm──►      │
  │                                 ├── collect Phase 1 partials
  │                                 ├── merge Reduce partials → push downstream
  │                                 │
  │   ◄── {action:iterate, bc_shm} ┤  broadcast ref_fgp
  ├── attach bc_shm (只读)          │
  ├── iterate local_buffer          │
  ├── clip_partial ───shm───►      │
  │                                 ├── collect clip partials → merge → converge?
  │                                 │   (重复 iterate 直到收敛)
  │                                 │
  │   ◄── {action:finish} ─────────┤
  ├── cleanup local_buffer          │
  │                                 │
  │   ◄── done_event.set() ────────┤
  ├── 释放 output_ipc shm handles  │
  ├── 退出                          ├── join() + cleanup()
```

## 完整示例：`mix_startrail.yaml`

以 `mix_startrail.yaml` 为例，这是项目中最复杂的 DAG，展示 SubDAG 展开 + 多终端段 + 迭代式 Reduce 的完整数据并行流程。

### 原始 DAG 拓扑

```
                  ┬──► data_loader ────────┬──► simgaclipstacker ──┐
                  │                        │      (SubDag)          │
 inputs.fnames ──┤                        ├──► trailstacker(max) ─┼──► mne ──► image_saver
                  │                        │                       │
                  ├──► weight_generator ───┘                       │
                  │                                                │
 configs.mask ───│─────► load_mask ────────────────────────────────┘
                  │
                  └──► exif_loader ──► exif_reducer ──────────────────► image_saver
```

### flatten 后的拓扑

`simgaclipstacker`（`sigma_clip.yaml`）被展开为 3 个独立节点：

```
                            ┌─► trailstacker (MaxMerger, DECOMPOSABLE)
                            │
data_loader ───────────────┼─► simgaclipstacker.mean_stacker (MeanMerger, DECOMPOSABLE)
                            │
                            └─► simgaclipstacker.disk_buffer (IS_DISK_BUFFER)
                                         ↓
                            simgaclipstacker.sigma_clip_iter (BUFFER_ITERATOR)

段外: weight_generator, exif_loader→exif_reducer, load_mask, mne, image_saver
```

### 节点分类（展开后）

| 节点 | Op 类 | 关键属性 | 段内角色 |
|------|--------|----------|----------|
| `data_loader` | `ImgDataLoaderOp` | `DATA_PARALLEL=True` | 段头 I/O |
| `trailstacker` | `TrailStackerOp` | `DECOMPOSABLE=True` | 终端 (DECOMPOSABLE_REDUCE) |
| `simgaclipstacker.mean_stacker` | `MeanStackerOp` | `DECOMPOSABLE=True` | 终端 (DECOMPOSABLE_REDUCE) |
| `simgaclipstacker.disk_buffer` | `DiskBufferWriterOp` | `IS_DISK_BUFFER=True` | 终端 (DISK_BUFFER) |
| `simgaclipstacker.sigma_clip_iter` | `SigmaClipIteratorOp` | `BUFFER_ITERATOR=True` | iterator_op |
| `weight_generator` | `WeightGeneratorOp` | `DATA_PARALLEL=False` | 段外（主进程） |
| `exif_loader` / `exif_reducer` | - | - | 段外（主进程） |
| `load_mask` / `mne` / `image_saver` | - | - | 段外（主进程） |

### 段检测结果

```
段: io=[data_loader], map=[],
    terminals=[
        (trailstacker, DECOMPOSABLE_REDUCE, extra={weight: weight_generator.result}),
        (simgaclipstacker.mean_stacker, DECOMPOSABLE_REDUCE),
        (simgaclipstacker.disk_buffer, DISK_BUFFER),
    ],
    iterator_ops=[simgaclipstacker.sigma_clip_iter]
```

这是一个**三终端段 + 一个 iterator op** — data_loader 的输出分支到 3 个可段化消费者，且 DiskBuffer 终端的下游 sigma_clip_iter 是 BUFFER_ITERATOR，纳入多阶段协议。

### 段化后的执行（3 workers, 200 帧）

```
                      主进程                                  Worker 0..2
                      ──────                                  ──────────
段外节点正常执行:
  WeightGenerator → weight 序列
  ExifLoader → ExifReducer
  LoadMask
                 │
                 ▼
SegmentAdapter:

Phase 1 (流式处理):
  Dispatcher: 帧路径 + weight ──────────────► Worker:
                                                decode(img)
                                                trail_merger.merge(img, weight)
                                                mean_merger.merge(img)
                                                disk_buffer.append(img)

  ◄── partial {trail, mean, buffer_desc} ─────

  merge trail partials → np.maximum → final trail → 推送 trailstacker.result
  merge mean  partials → FGP.__add__ → global_fgp → 推送 mean_stacker.result/statistics
  解析内部 config: sigma_clip_iter.fgp_total ← mean_stacker.statistics

Phase 2 (sigma clip 迭代, 最多 max_iter 轮):
  broadcast_pack(ref_fgp) → 1 块 shm ─────► Worker:
                                                broadcast_unpack(shm) → ref_fgp
                                                SigmaClippingMerger(ref_fgp)
                                                for frame in local_buffer:
                                                    clip_merger.merge(frame)
  ◄── clip_partial (rejected FGP) ─────────

  merge: total_rejected = sum(clip_partials)
  accepted = fgp_total - total_rejected
  accepted.apply_zero_var(fgp_total)
  收敛检查 → if not converged: ref_fgp = accepted, 重复

Phase 结束:
  {action: "finish"} ────────────────────► Worker:
                                                cleanup local_buffer + 退出

  推送 sigma_clip_iter.result/statistics → 下游

段外继续:
  MNE(trail_result, sigma_statistics, mask) → ImageSaver
```

### IPC 数据量估算

以 200 帧 6000x4000 uint16 RGB、3 workers、5 轮迭代为例：

| 阶段 | 数据量 |
|------|--------|
| Phase 1 入口 | 200 × ~100B (路径 + weight) ≈ 20 KB |
| Phase 1 出口 | 3 × (trail partial + FGP partial) ≈ 3 × (274 + 960) MB ≈ 3.7 GB |
| Phase 2 每轮 broadcast | 1 × ~960 MB shm（物理仅 1 份） |
| Phase 2 每轮出口 | 3 × ~960 MB (clip partial FGP) |
| Phase 2 合计 (5 轮) | 5 × (960 + 3×960) ≈ 19.2 GB 累计 shm 映射 |

## 复杂示例：`calibration_stack.meta.yaml` (route: mean)

这是一个更能体现数据并行优势的 DAG。Meta YAML 的路由机制在构建时展开为确定的拓扑。

### 展开后拓扑（calibration route=mean，所有校准帧可用）

```
bias_fnames ──► bias_stacker(MeanStackerOp) ──────────┬──► bias_subtract ──┐
dark_fnames ──► dark_stacker(MeanStackerOp) ──────────┼──► dark_subtract ──┤
flat_fnames ──► flat_stacker(MeanStackerOp) ► flat_bias_sub ┼─► flat_divide ──┤
                                                      │                    │
light_fnames ──► light_loader(ImgDataLoaderOp) ───────┘                    │
                                                                           ▼
                                                   main_stacker(MeanStackerOp)
                                                           │
                                                       image_saver
```

### 段检测

```
段: io=[light_loader], map=[bias_subtract, dark_subtract, flat_divide],
    terminals=[(main_stacker, DECOMPOSABLE_REDUCE)]
```

这条从 `light_loader` 到 `main_stacker` 的完整路径被检测为一个可并行段：
- **段头 I/O**: `light_loader` — 帧级解码
- **段中 Map**: `bias_subtract` → `dark_subtract` → `flat_divide` — 逐帧校准
- **段尾终端**: `main_stacker` (MeanStackerOp) — 可分解均值堆叠（单终端段）

`bias_stacker`、`dark_stacker`、`flat_stacker`、`flat_bias_sub` 不在段内——它们的输出是单个图像（不是序列），作为 configs 传入段内 Map ops，不影响段检测。

### 段化后的执行

```
                      主进程
                    ┌─────────────────────────────────┐
bias_fnames ──► bias_stacker ──►┐                     │
dark_fnames ──► dark_stacker ──►├─ (单图结果作为 config)│
flat_fnames ──► flat_stacker ──►│                     │
                flat_bias_sub ──┘                     │
                                                      │
                      SegmentAdapter                   │
                    ┌──────────────────────────────┐  │
  light_fnames ──►  │  W0: decode → sub_b → sub_d  │  │
     (路径分发)      │      → div_f → mean_reduce  │  │
                    │  W1: decode → sub_b → sub_d  │  │
                    │      → div_f → mean_reduce  │  │
                    │  W2: decode → sub_b → sub_d  │  │
                    │      → div_f → mean_reduce  │  │
                    └──────────────────────────────┘  │
                         │  3 个 FastGaussianParam     │
                         ▼                             │
                    merge(FGP_0 + FGP_1 + FGP_2)      │
                         │                             │
                         ▼                             │
                    final mean image ──► image_saver   │
                    └─────────────────────────────────┘
```

每个 worker 独立完成：

```
for frame_path in my_frames:
    img = decode(frame_path)              # I/O: 磁盘读取 + 解码
    img = img - bias_master               # Map: bias 减法
    img = img - dark_master               # Map: dark 减法
    img = img / flat_master               # Map: flat 除法
    mean_merger.merge(img)                # Reduce: 流式均值累积

output: FastGaussianParam (sum_mu, square_sum, n, source_dtype)
```

主进程合并：`FGP_0 + FGP_1 + FGP_2 → final FGP → mean image`。

这是一个纯 Phase 1 段——没有 DiskBuffer 和 iterator_ops，worker 完成流式处理后即退出。

## 内存分析

以 200 帧 6000x4000 uint16 RGB 图像、`mix_startrail.yaml` 流程为例。

### 基础常量

- 单帧原始大小: `6000 × 4000 × 3 × 2B (uint16)` ≈ **137 MB**（记为 F）
- FastGaussianParam (uint16 源): sum_mu(uint32) + square_sum(uint64) + n(uint16) ≈ **960 MB**（记为 G）
- MaxMerger result (int_weight, uint32): ≈ **274 MB**

### 单进程峰值

| 阶段 | 主要活跃数据 | 峰值 |
|------|-------------|------|
| Phase 1 (流式 200 帧) | 当前帧 + MaxMerger + MeanMerger(FGP) + DiskBuffer(磁盘) | ~1.4 GB |
| Phase 2 (sigma clip 迭代) | fgp_total + ref_fgp + clip_merger(rej 阈值 + result) + accepted + trail | ~3.5-4.5 GB |
| Phase 3 (MNE) | trail + accepted + float64 临时 | ~2-3 GB |

**单进程总峰值 ≈ 3.5-4.5 GB**

### N 进程峰值

| 组件 | 内存占用 |
|------|---------|
| **主进程常驻** (fgp_total + broadcast shm + merge 缓冲) | ~3 GB |
| **每个 Worker Phase 2 峰值** (rej_high/low + clip_merger.result + 当前帧) | ~1.4 GB |
| **broadcast shm** (物理仅 1 份，N worker 只读映射) | ~1 GB (已计入主进程) |

近似关系:

```
M(N) ≈ M_main + N × M_worker ≈ 3 GB + 1.4N GB
```

| N (worker 数) | 估算系统峰值 | 相对单进程倍率 |
|:---:|---:|:---:|
| 1 | ~3.5-4.5 GB | 1× |
| 2 | ~5.8 GB | ~1.4× |
| 3 | ~7.2 GB | ~1.8× |
| 4 | ~8.6 GB | ~2.1× |

**峰值瓶颈在 Phase 2 sigma clip 迭代**：每个 worker 独立持有 SigmaClippingMerger 的完整 FGP result (~960 MB)，这是线性增长的主因。

## 不参与数据并行的场景

以下场景在当前版本中不支持数据并行：

| 场景 | 原因 | 示例 |
|------|------|------|
| **FilterBaseOp** | 帧间依赖性使数据并行更复杂 | `StarAlignmentOp` |
| **MedianReduceOp** | 不可分布式归约，需要回退主进程串行 | `median_stack` |
| **仅 I/O 无 Map/Reduce** | 不值得段化的开销 | 孤立的 `data_loader` |
| **分支点有不可段化的消费者** | 段内数据由 worker 处理，无法同时复制给主进程 | （当前实际 DAG 中无此情况） |

**注意**：以下场景在此前版本中不支持，但现在已支持：

| 已支持场景 | 实现方式 |
|-----------|---------|
| SubDag 内部节点 | `flatten_sub_dags()` 编译期展开 |
| 拓扑分支点 | 多终端段（所有分支都可段化时） |
| DiskBufferWriterOp | 作为 DISK_BUFFER 终端，worker 本地创建 buffer |
| 迭代式 Reduce (SigmaClip/Huber) | 多阶段 Worker 协议，分布式迭代 |

## Worker 数量

自动检测：`max(2, os.cpu_count() - 2)`，保留 1 核给主进程 + 1 核给系统。

用户可通过 `run_from_yaml(..., num_workers=N)` 手动指定。`num_workers=0` 或 `num_workers=1` 禁用数据并行。

## 错误处理与取消

- **Worker 异常**：Worker 设置 `cancel_event`（`mp.Event`），主进程检测后传播 `CancellationToken` 到下游
- **上游取消**：Dispatcher 向所有 worker 的 `input_ipc` 发送 `CancellationToken`，worker 读到后退出
- **外部取消**：共享 `mp.Event` 通知所有 worker；多阶段 worker 在命令等待中也检查 cancel
- **Worker 超时**：主进程 `join(timeout=10)` 后 `terminate()` 兜底

## 进度追踪

Worker 内的 ops 通过 `ProxyTracker` + `mp.Queue` 向主进程汇报进度事件。主进程的 `TrackerEventConsumer` 异步消费事件队列并更新 UI 进度条。

## 文件结构

| 文件 | 职责 |
|------|------|
| `engine/segment_detect.py` | 段检测算法、终端类型枚举、ParallelSegment 数据结构 |
| `engine/segment_adapter.py` | SegmentAdapter（多阶段 Collector）、apply_data_parallelism() |
| `engine/segment_worker.py` | worker 入口 `_segment_worker_main`（多阶段协议） |
| `engine/flatten.py` | `flatten_sub_dags()` SubDAG 预展开 |
| `engine/meta.py` | `meta_resolve()` Meta YAML 路由解析 |
| `engine/multiprocess.py` | `run_dag_multiprocess()` 入口 |
| `engine/wiring.py` | 标准布线 + `run_from_yaml()` 路由 |
| `engine/build.py` | `validate_and_build_order()`、`_parse_link()` (rsplit) |
| `engine/executor.py` | DAGExecutor，并发执行所有 ops |
| `component/ipc_queue.py` | IPCQueue + ShmTransportable ABC + broadcast helpers |
| `component/queue.py` | BaseQueue / RichContextQueue 基类 |
| `component/merger.py` | Merger 基类 + SigmaClippingMerger / HuberWeightedMerger |
| `component/frame_buffer.py` | DiskFrameBuffer / SourceReplayBuffer + 描述符 |
| `component/data_container.py` | FloatImage、FastGaussianParam、HuberMeanParam (ShmTransportable) |
| `ops/base.py` | BaseOp 类属性 (DECOMPOSABLE / DATA_PARALLEL / IS_DISK_BUFFER / BUFFER_ITERATOR) |
| `ops/trailstacker.py` | TrailStackerOp / MeanStackerOp / MNE 的 merge_partial 实现 |
| `ops/sigma_clip_ops.py` | DiskBufferWriterOp / SigmaClipIteratorOp / HuberMeanIteratorOp / MedianReduceOp |
| `ops/dataloader.py` | ImgDataLoaderOp (DATA_PARALLEL=True) |
