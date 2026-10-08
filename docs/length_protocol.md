# 长度广播协议：规范、缺陷清单与维护指导

> 覆盖代码：`hoshicore/component/queue.py`、`hoshicore/ops/base.py`、`hoshicore/engine/wiring.py`、`hoshicore/engine/executor.py`
> 相关文档：[DAG Engine 架构概览](./dag_engine.md)、[Op 算子设计](./op.md)、[DAG 节点定义规范](./dag_node_definition.md)、[框架层改造建议](./framework_refactor_notes.md)
>
> 本文档的定位：**协议规范 + 缺陷登记 + 维护清单**。协议行为已逐条对照源码核实；缺陷 D1 已在本地用最小脚本复现（复现脚本见附录 A，未入库）。修改本协议相关代码前请先读第 5 节的「新增 Op 契约」与第 7 节的「改动影响面」。

---

## 1. 协议定义

长度（`length`）不是数据，而是**挂在队列对象上的一条独立控制通道**，与数据共用同一个 `BaseQueue` 实例。

### 1.1 队列侧字段

| 成员 | 位置 | 语义 |
|------|------|------|
| `length: Optional[int]` | `queue.py:68` | `int N` = 已知长度；`None` = 未知（sentinel 驱动） |
| `_length_event: asyncio.Event` | `queue.py:69` | 长度就绪门闩，只被 `set_length()` / `force_cancel()` 置位 |
| `active: bool` | `queue.py:70` | 是否有生产者连接；`False` 时 `pre_execute` 跳过该输入（不参与长度协商） |
| `set_length(length)` | `queue.py:134-141` | 生产者调用；持 `_put_lock`；冲突即抛；置位 `_length_event` |
| `get_length()` | `queue.py:143-147` | 消费者 `await _length_event.wait()` 后返回 `self.length`，再 `_check_cancelled()` |

### 1.2 三条关键性质

1. **不占队列槽位**。背压为 `maxsize=1` 时长度仍能穿透整条链——它是纯内存事件，不走 `asyncio.Queue`。
2. **首次广播即定值**。`_length_event` 一旦置位便永久置位；`get_length()` 返回的是**首次广播值**，后续 `set_length` 不会改写已返回的消费者视角（见 D2）。
3. **值域只有三种**：
   - `int N`：序列已知长度（= 生产者自身输入序列长度，见 `base.py:151-154`）；
   - `None`：变长 / sentinel 驱动（Filter 类）；
   - `1`：非 `sequence` 端口（`base.py:134-135` 强制）。

### 1.3 负载不变式：每队列单生产者

协议的正确性依赖一条**未被断言**的前提：**每个下游输入队列有且只有一个生产者**。

- `build.py:104-115` 的 `_iter_node_src_links` 对每个 `(section, arg_name)` 只产出一个 `src`，因此一个输入端口只有一个来源；
- `wiring.py:274` 的 `instances[provider].outputs[port].append(target_queue)` 对同一 `target_queue` 至多执行一次；
- feeder 为每个目标队列创建独立协程（`wiring.py:161-166`），也只写自己的队列。

因此 `set_length` 对每个队列实际只被调用一次（外加可选的 `force_cancel` 唤醒）。**这是 L1 冲突检查能够成立的前提，请勿在无检查的情况下引入扇入（fan-in）。**

---

## 2. 传播路径

### 2.1 唯一的传播节点：`BaseOp.pre_execute()`

`base.py:84-140` 的步骤顺序是有意设计的：

```
1. 对每个 active 输入队列  await queue.get_length()   ← 阻塞点：等上游广播
2. 按 INPUTS[name].type == "sequence" 分桶 known_seq / none_seq
3. 混用检查（known + none） → ValueError                 ← L3 检查①
4. 已知长度集合大小 > 1 → ValueError                     ← L3 检查②
5. self.length = 唯一已知长度，或 None
6. 对每个 OUTPUTS 端口 set_length(output_length 或 1)     ← 向下游广播
7. 最后才 await config 队列                              ← 长度传播不受 config 背压阻塞
```

第 6 步排在第 7 步之前是刻意的：**长度广播不被 config feeder 的背压阻塞**，否则一条慢配置链会拖住整图的长度协商。

### 2.2 拓扑序上的「先收后发」链

```
feeder(wiring.py:161-166)
  set_length(len(data)) + put(item)*          ← 全局输入的授权长度来源
        │
        ▼
  Op A.pre_execute(): get_length() → 广播 A 的每个输出端口
        │
        ▼
  Op B.pre_execute(): get_length() → 广播 B 的每个输出端口
        │
        ▼
  ... 直到汇点；全局输出收集队列由产出节点的 pre_execute 赋予长度
```

每个节点都是「先等齐所有上游长度 → 再发布自己的输出长度」，所以长度沿拓扑序一次性扩散完成，**先于任何数据帧**。实测（3 节点链，喂入前不放数据）：

```
last_node_length_before_data = [4]      # 末节点在首帧到达前已拿到 4
lengths = [4, 4, 4]
```

### 2.3 变长（sentinel 驱动）分支

`FilterBaseOp._infer_output_length()` 恒返回 `None`（`base.py:519`），于是：

- 该类 Op 的**所有** `sequence` 端口都广播 `None`（含 `aligned_exifs`、`center_indices` 这类副端口）；
- 下游 `self.length is None` → `_input_range()` 返回 `itertools.count()`（`base.py:156-165`），必须配合 `except StreamExhausted: break`；
- 进度条被 `wiring.py:680-681` 主动剔除（`variable_input_nodes` 集合）；
- `_select_reporting_nodes` 的剔除对象是**变长输入的消费方**，不是变长源本身。

### 2.4 长度的实际用途

| 用途 | 位置 |
|------|------|
| 有界迭代（`range(N)` 而非 sentinel） | `base.py:163-164` |
| 长度控制通道（只取长度、丢弃数据） | `weight_generator.py:96-119`（权重斜坡按 N 预生成）；`dag/startrail.meta.yaml:109-117` 把 `inputs.fnames` 直连 `WeightGeneratorOp.sequence` |
| 进度条总数 | `base.py:272-273`、`base.py:300-301`、`image_saver.py:211`、各 `tot_num = self.length` |
| 资源规划（**独立通道**，见 D8） | `preflight.py:290`、`runtime_plan.py:169-179` |

---

## 3. 一致检查：四层

| 层 | 位置 | 检查内容 | 触发时机 | 有无测试 |
|----|------|----------|----------|----------|
| **L1 队列** | `queue.py:138` | 同一队列二次 `set_length` 且两个 int 不等 → `ValueError: Length mismatch: A vs B` | 广播时 | ❌ 无直接测试 |
| **L2 编译期** | `wiring.py:745-830` `_check_variable_source_conflicts` | ①多个不同变长源汇入同一节点 → 抛；②固定 + 变长混入 → 抛；沿拓扑序传播 `(provider, port) → 变长源` 标记 | 布线期，跑数据前 | ❌ 无测试（`tests/engine/` 无 wiring 测试文件） |
| **L3 运行期** | `base.py:109-121` | ①`known_seq` 与 `none_seq` 混用 → 抛（消息直接点名修法 `Use FilterGate pattern`）；②已知长度集合大小 > 1 → 抛 | `pre_execute` | ⚠️ 仅被间接覆盖 |
| **L4 Op 自检** | `bundle_ops.py:770-773`、`bundle_ops.py:842-847` | `seen_count` / `frame_count` / `next_center` 三重对账；多一帧即 `ValueError("second image load is longer than AlignmentPlan")` | 算法内部 | ⚠️ 部分覆盖 |

实测触发结果（全部符合预期）：

```
L3  mix     → "mixer: cannot mix known-length and sentinel-driven sequence inputs"
    unequal → "Input sequence length mismatch: {'src': 3, 'aux': 5}"
L2  two_var_sources → "receives sequence inputs from multiple variable-length sources: ['va','vb']"
    fixed_plus_var  → "mixes fixed-length and variable-length sequence inputs (variable source: va)"
    transitive      → ['shr']     # 变长性沿拓扑传递到下游节点
L1  set_length(5) → set_length(7) → "Length mismatch: 5 vs 7"
```

### 3.1 L2 与 L3 的关系

二者是**同一约束的两个视角**，不是冗余：

- **L2** 用类属性 `VARIABLE_OUTPUT` 做静态拓扑推理 → 能在跑数据前报错、能给出可执行的修复建议（FilterGate）；
- **L3** 用运行时 `get_length()` 的真实结果兜底 → 覆盖手工构造的图、单元测试图、以及 L2 因信息不足而漏判的场景。

⚠️ 两点必须记住：

1. 两者的判据来自**不同来源**（类属性 vs 方法返回值），见 **D3**；
2. **L2 的「多个不同变长源」检查没有运行时兜底**：当两条变长流同时汇入一个节点时，L3 看到的 `known_seq` 为空集，混用检查（`base.py:110`）不会触发。该约束目前**只**由 `VARIABLE_OUTPUT` 保证。

### 3.2 多序列端口的锁步约定

多输出端口的 Filter Op 必须保证各端口**同增同减**，否则下游 zip 错位：

- `alignment_ops.py:260-266`：`result` 与 `aligned_exifs` 在同一个 `try` 内一次性广播，失败帧两个端口一起丢；
- `bundle_ops.py:883-887`：`result` / `aligned_exifs` / `center_indices` 同时广播。

跨流对账（非变长场景）也有显式防御：`bundle_ops.py:403-404`、`bundle_ops.py:765-767`、`bundle_ops.py:917-919` 把「另一条流提前结束」翻译成 `ValueError("data/exifs sequences have different lengths")`，而不是让它退化成静默截断。

---

## 4. 终止与长度协议的交互

| 场景 | 机制 | 位置 |
|------|------|------|
| 生产者未广播长度就失败 | `executor.cancel_all()` → `queue.force_cancel(token)` → `_length_event.set()` 唤醒卡在 `get_length()` 的消费者，唤醒后 `_check_cancelled()` 抛 `CancellationError` | `executor.py:96-113`、`queue.py:93`、`queue.py:146` |
| 队列已取消后再广播 | `set_length()` 入口 `_check_cancelled()` 直接抛 | `queue.py:136` |
| 取消 vs 已就绪长度 | `force_cancel` 后即使 `length` 已有值，`get_length()` 仍抛 `CancellationError`（取消优先） | `queue.py:143-147` |
| 长度协商卡死 | 已由 `tests/engine/test_executor.py:339-357` 覆盖 | — |

**已知限制**：挂死（hang）与取消不同——见 D1，框架层无超时/看门狗，挂死不会进入 `cancelled_nodes`，也不会抛 `DAGExecutionError`。

---

## 5. 新增 / 修改 Op 的契约（Checklist）

发布新 Op 或改动已有 Op 的发射逻辑时，逐条自检：

1. **声明长度必须等于实际发射数量。**
   - 输出数量恒等于输入序列长度 → 用默认 `_infer_output_length`（不要覆盖）；
   - 输出数量在运行前未知（过滤、去重、条件丢弃） → 继承 `FilterBaseOp`（自动 `VARIABLE_OUTPUT = True` + 广播 `None`）；
   - 输出数量可从某个特定输入推导 → 覆盖 `_infer_output_length` 并**显式按键取值**（如 `satellite_clean_op.py:70-71`、`simple_ops.py:404-405` 的 `input_lengths.get('data')`），**不要**依赖默认实现的「首个非 None」。
2. **发射循环用 `self._input_range()`**，并把 `StreamExhausted` 当作「上游提前结束」处理（打 warning 或翻译成有意义的错误），不要让它裸抛成节点失败。
3. **声明了固定长度时，加一个「多发即抛」的护栏。** 参考 `_BundleWindowStackOp` 的写法：

   ```python
   if frame_count is not None and index >= frame_count:
       raise ValueError("... longer than AlignmentPlan: "
                        f"got at least {seen_count}, expected {frame_count}")
   ```

   并在循环结束后断言计数自洽（`if next_center != seen_count: raise RuntimeError(...)`）。**这是目前唯一能防止 D1 挂死的办法。**
4. **不要在 `_async_execute` 中循环外额外补发数据**（flush/补齐/预热帧）。若确有必要，必须相应覆盖 `_infer_output_length` 或改用变长（`None`）语义。
5. **多序列端口必须锁步广播**（见 §3.2）。
6. **不要覆盖 `VARIABLE_OUTPUT` 与 `_infer_output_length` 使其语义矛盾**（见 D3）。
7. **非 `sequence` 端口不需要考虑长度**（统一为 `1`），也不要尝试 `get_length()` 一个 config 队列——config 队列**不参与长度协议**，`get_length()` 会永久阻塞。
8. 若新增 Op 会让**两条不同来源的序列汇入同一节点**，先确认 L2 会正确拒绝它；需要真正合流时按 FilterGate 模式先对齐。

---

## 6. 缺陷清单

共登记 9 项（D1–D9）。严重度：**P0** = 会造成静默挂死/结果错误；**P1** = 协议语义漏洞，当前未触发但极易踩中；**P2** = 健壮性/可诊断性；**P3** = 一致性/可维护性。

> 每项的「现状」段落区分了两类结论：**已复现**（有最小脚本或实测输出）与**已核对但当前不可达**（潜伏）。请勿把后者当作现存 bug 处理，但也请勿在改动相关代码时忽略它们。

### D1（P0）超发帧令生产者永久阻塞，整图静默挂死

**现象**：Op 声明长度为 N，却实际发射了 N+1 帧时，消费者按 `range(N)` 收满即正常退出；第 N+1 帧留在 `maxsize=1` 的队列里，生产者随后 `_send_sentinel()` 的 `await queue.put(SENTINEL)`（`base.py:202`）因队列已满**永久阻塞**。没有超时、没有告警、不进入 `cancelled_nodes`，`run_dag` 的 `asyncio.gather` 永不返回。

**复现**（最小脚本见附录 A）：

```
done: ['consumer', 'sink_drain'] | stuck: ['producer'] | consumer consumed: 2
```

**影响面**：当前 46 个注册 Op 中，有 20 个声明了固定长度的 `sequence` 输出（另 3 个为变长：`StarAlignmentOp`、`BundleReferenceRemapOp`、`WindowFrameFilterGateOp`）。这 20 个的发射路径已逐个核对，**尚未发现触发点**：它们要么以 `_input_range()` 为界天然 1:1 / ≤1:1 发射，要么对 `frame_count` 有显式护栏。因此 D1 属于**框架级防御缺失**而非现存生产 bug；但它一旦被触发表现为「无日志挂死」，排障成本极高，且新增 flush 型 / 补齐型 Op 或修改现有 Op 的发射循环都极易踩中。

**建议修法**（任选，建议 ①+②）：

① `BaseOp` 增加发射计数护栏：

```python
# __init__: self._emitted = 0
async def _broadcast_outputs(self, results):
    ...
    if self.length is not None and "sequence-port-in-results":
        self._emitted += 1
        if self._emitted > self.length:
            raise ValueError(
                f"{self.name}: emitted more than the declared length "
                f"({self.length}); downstream consumers already stopped")
```

② `_send_sentinel()` 改为非阻塞：`put_nowait`，`QueueFull` 时记录 warning 并放弃（sentinel 只对「仍在等待的消费者」有意义）。

③ 给 `run_dag` 增加可选的整体看门狗（超时/无进展检测），把挂死转换为可诊断错误。

**回归测试**：`tests/ops/test_base_length_protocol.py`，用一个故意超发的 Op 断言 `ValueError` 而不是挂死。

---

### D2（P1）`set_length(None)` 静默覆盖已知 int，且无「封版」语义

**现象**：

```python
set_length(5); set_length(None)   # 不报错，length 变为 None      ← 不对称守卫
set_length(5); set_length(7)      # ValueError: Length mismatch   ← 只有 int/int 才检查
```

`queue.py:138` 的守卫是 `if self.length is not None and length is not None and self.length != length`，只覆盖 int↔int 冲突。另外 `_length_event` 首次置位即「定值」：若先到 `None`，已经等在 `get_length()` 里的消费者会立刻拿到 `None` 并转入 sentinel 驱动，之后到达的真长度不会纠正它。

**为何今天不出问题**：§1.3 的单生产者不变式保证每个队列只被广播一次。

**建议修法**：把守卫改为「一旦写过就不允许改写」——用独立哨兵区分「未广播」与「已广播为 None」：

```python
if self._length_broadcast and self.length != length:
    raise ValueError(f"Length mismatch: {self.length} vs {length}")
self.length = length
self._length_broadcast = True
```

并在 `instantiate_and_wire()` 里对「同一队列被两个来源写入」显式报错，把 §1.3 的隐式前提变成显式约束。

**回归测试**：`tests/component/test_queue.py`（该文件目前完全没有长度协议测试）。

---

### D3（P1）`VARIABLE_OUTPUT` 与 `_infer_output_length` 双真相源

**现象**：L2 静态检查依据类属性 `VARIABLE_OUTPUT`（`wiring.py:819`），L3 运行时行为依据 `_infer_output_length()` 的返回值（`base.py:132`）。二者无任何一致性校验。

危险方向（`VARIABLE_OUTPUT = False` 但 `_infer_output_length` 返回 `None`）：

- L2 会把它当成**固定长度源**，于是 L2 的两条检查**都不会触发**；
- L3 也补不上：`base.py:110` 的混用检查条件是 `if known_seq and none_seq`，当某节点**只有一个**变长输入时两个 L2 检查本就该放行（语义正确），但当它接入**两个来自不同来源的变长输入**时，`known_seq` 为空、`none_seq` 有两个 → **L3 同样不报错**，节点会静默地 zip 两条长度不同的流。

也就是说：**「多个不同变长源汇入同一节点」这一条约束，目前完全依赖 `VARIABLE_OUTPUT` 类属性的正确性，L3 在结构上无法兜底。** 反向矛盾（`True` + 返回 int）则会让 L2 过度拒绝合法图，属可容忍的失败方向。

**现状**：46 个注册 Op 逐个核对后自洽（`FilterBaseOp` 子类全部为 `True`/`None`；覆盖了 `_infer_output_length` 的 `SatelliteCleanOp`、`SequenceSortOp` 均为固定长度且 `VARIABLE_OUTPUT = False`）。已把下面的断言脚本对全部注册 Op 实跑一遍：**29 个含序列输入的 Op 全部通过，无假阳性**，可直接落地为 CI 契约测试。

**建议修法**：加一条廉价断言，把「类属性」与「方法返回值」绑死。放在 `instantiate_and_wire()` 的实例化循环里（运行期保护），或做成遍历 `REGISTERED_OP` 的契约测试（CI 保护，推荐后者）：

```python
# 用一个虚拟的已知长度探测该类在「输入长度已知」时的输出声明
probe = {name: 7 for name, spec in op_cls.INPUTS.items()
         if spec.get("type") == "sequence"}
inst = op_cls(name="__contract_probe__")      # 仅构造，不执行
if probe and (inst._infer_output_length(probe) is None) != op_cls.VARIABLE_OUTPUT:
    raise ValueError(
        f"{op_cls.__name__}: VARIABLE_OUTPUT={op_cls.VARIABLE_OUTPUT} "
        f"contradicts _infer_output_length(...)=None "
        f"— the static variable-source check cannot be trusted for this Op")
```

**回归测试**：`tests/ops/test_base_length_protocol.py` 中做一个遍历 `REGISTERED_OP` 的契约测试。

---

### D4（P1）`_infer_output_length` 默认实现依赖 INPUTS 声明顺序

**现象**（`base.py:151-154`）：

```python
for name, length in input_lengths.items():
    if length is not None:
        return length          # 首个非 None，不区分端口类型
```

`input_lengths` 含**所有 active 输入**，包括非 `sequence` 端口（其长度被上游广播为 `1`）。若某 Op 声明 `{"buffer": {"type": "image"}, "data": {"type": "sequence"}}`，输出序列长度会静默变成 `1`。

**现状**：已扫描全部 46 个注册 Op，**没有任何 Op 同时声明 `sequence` 与非 `sequence` 输入**，所以当前不可达，属潜伏问题。

**建议修法**：默认实现只考虑 `sequence` 端口，其余不参与：

```python
for name, length in input_lengths.items():
    if self.INPUTS[name].get("type") != "sequence":
        continue
    if length is not None:
        return length
return None if any(self.INPUTS[n].get("type") == "sequence"
                   for n in input_lengths) else 1
```

（注意保持「有序列输入但全部为 `None` → 返回 `None`；完全无输入 → 返回 `1`」的既有语义。）

---

### D5（P2）`get_length()` 无超时，`active` 漏设即挂死

`get_length()` 只有在 `set_length()` 或 `force_cancel()` 时才可能返回。兜底完全依赖：

- `wiring.py:245-253`：`__inactive__` 标记 → `queue.active = False`；
- `wiring.py:328-337`：未布线且 `required: False` 的输入 → `active = False`。

任一环节漏设（例如新引入一种「可选输入」语法而忘记同步 wiring），节点会在 `pre_execute` 永久阻塞，且同上不会产生任何诊断。

**建议修法**：给 `pre_execute` 的 `get_length()` 加可配置超时（默认关闭或较长），超时后抛出带 **节点名 + 输入端口名 + 上游节点名** 的错误；或在 `run_dag` 启动后 N 秒打印「仍在等待长度协商的节点」诊断快照。

---

### D6（P2）少发帧的处理不对称

| Op | 行为 | 位置 |
|----|------|------|
| `TrailStackerOp` | `logger.warning("upstream ended at i/N")` + `break`，降级继续 | `trailstacker.py:91-95` |
| `ApplyMaskOp` | 同上 | `simple_ops.py:560-564` |
| `SatelliteCleanOp`、`SlidingWindowMaxOp`、`EmaDecayMaxOp`、`WeightedSlidingWindowMaxOp`、`ExifReadOp`、各 Bundle 窗口 Op | 多数显式处理 | 见 `grep -n "except StreamExhausted" hoshicore/ops` |
| `SequenceSortOp` | **不捕获** → `StreamExhausted` 裸抛 → 节点判为**失败** | `simple_ops.py:416-418` |
| `WeightGeneratorOp` | **不捕获**（`drain_input`），另有 `assert length is not None`（`-O` 下失效） | `weight_generator.py:98-112` |

`SequenceSortOp` 需要完整序列才能排序，失败尚可辩护，但错误信息是 `"Stream ended normally"`，对用户无意义。

**建议修法**：统一为「翻译成带业务语义的错误」或「warning + 降级」，二选一并保持全库一致；`WeightGeneratorOp` 的 `assert` 换成显式 `raise ValueError`。

---

### D7（P2）`_collect_outputs` 对 `length == 0` 静默丢 key

`wiring.py:492` 的 `if items:` 使长度为 0 的全局输出在 `results` 中**完全不存在**，调用方无法区分「输出为空」与「输出未声明」。

**建议修法**：`results[name] = items[0] if len(items) == 1 else items`（去掉 `if items` 守卫），让空序列显式返回 `[]`。注意这会改变 `run_dag` 的对外契约，需要同步 CLI/GUI 的取值处。

---

### D8（P3）第三套长度来源：`len(global_inputs["fnames"])`

`preflight.py:290` 与 `runtime_plan.py:169-179` 用编译期的 `len(fnames)` 作为 `n_frames` 参与资源估算与 `chunk_rows` 规划，与运行时长度协议是**两条独立通道**：

- 硬编码键名 `"fnames"`（其他命名的全局输入拿不到帧数，退化为「无估算」）；
- 要求 `isinstance(fnames, (list, tuple))`（`@<list file>`、JSON 数组等其它输入形态在 CLI 侧已被规整，但库层调用方可绕过）。

当前二者同源（feeder 用的就是同一个对象），因此一致。**维护要求**：新增全局输入命名、或改变 `global_inputs` 的装载方式时，必须同时检查这两处；否则会出现「预检按 A 个帧估算内存、实际跑 B 个帧」的偏差。

---

### D9（P3）文档错误：`None` 不是 sentinel，且 `None` 是合法负载

`docs/dag_engine.md:73` 写「**Sentinel**：`None` 标记序列结束，下游据此退出循环」——**与实现不符，且照此实现会造成数据丢失**。

实际实现是**独立哨兵对象 + 身份比较**：

```python
# queue.py:44
_SENTINEL = object()          # 所有子类共享
# queue.py:128
if item is BaseQueue._SENTINEL:
    ... raise StreamExhausted
```

`None` 是**合法数据负载**，必须原样透传给下游，例如：

- `NoneOutputOp`（`simple_ops.py:451-458`）广播 `{"result": None}`，用于 Meta YAML `route="none"` 触发下游 passthrough；
- `WindowFrameFilterGateOp`（`bundle_ops.py:885`）广播 `frame.exif`，可能是 `None`；
- `StarAlignmentOp`（`alignment_ops.py:261`）在 `exifs` 输入未激活时广播 `exif_obj = None`。

**维护要求**：任何「用 `None` 表示结束」的改动都会破坏上述路径。若确需简化协议，须先全库审计 `_broadcast_outputs({...: None})` 与 `put(None)`，并同步 `docs/op.md` §「信号传播机制」、`docs/dag_engine.md` §「队列通信协议」。`docs/README.md` §2.3 的「信号传播」表述正确，可作参照。

---

## 7. 改动影响面

修改以下任一处时，需要同步检查的对象：

| 改动对象 | 必须同步检查 |
|----------|--------------|
| `queue.py:set_length/get_length` | `ForceCancel` 唤醒路径（`queue.py:81-99`）、`executor.cancel_all()`、`tests/engine/test_executor.py:107-120/339-357`、`docs/dag_engine.md` §「取消语义」表格 |
| `BaseOp.pre_execute` / `_infer_output_length` | L3 两条检查的语义、`ParallelBaseOp._execute_serial/_execute_concurrent` 的 `self.length is None` 分支、`FilterBaseOp`、`wiring._select_reporting_nodes` 的进度条剔除逻辑 |
| `_check_variable_source_conflicts` | `docs/op.md` §「静态冲突检测」、`docs/dag_node_definition.md` §6.3、`docs/README.md` §3.5、`CLAUDE.md`「Length before data」不变量描述 |
| `_send_sentinel` / `_broadcast_outputs` | `docs/op.md` §「信号传播机制」（含 SENTINEL 回填语义）、`docs/dag_engine.md` §「队列通信协议」（**含 D9 的错误描述，改协议时一并修正**）、`tests/engine/test_executor.py:287-299`（非 sequence 端口不发 sentinel） |
| 新增 Filter 类 Op | `VARIABLE_OUTPUT`、`_infer_output_length`、多端口锁步、下游进度条剔除 |
| 新增全局输入命名 | D8 的两处 `"fnames"` 硬编码 |

---

## 8. 建议的加固顺序

| 阶段 | 内容 | 对应缺陷 |
|------|------|----------|
| 1 | `_send_sentinel` 非阻塞 + `_broadcast_outputs` 超发护栏；补 P0 回归测试 | D1 |
| 2 | 长度「一次写入」语义 + 单生产者显式断言；补队列层测试；顺手修正 sentinel 文档错误 | D2、D9 |
| 3 | `VARIABLE_OUTPUT` / `_infer_output_length` 契约测试；默认实现只认 sequence 端口 | D3、D4 |
| 4 | `get_length()` 超时或「等待长度协商」诊断快照 | D5 |
| 5 | 少发帧处理统一化；`results` 空输出显式化 | D6、D7 |
| 6 | 长度来源收敛为单一入口（`n_frames` 从 `ValidatedDag.global_inputs` 推导） | D8 |

阶段 1–3 是纯粹的防御性加固，不改对外契约，建议优先；阶段 5 会改变 `run_dag` 返回值语义，需与 CLI/GUI 同步。

---

## 9. 现有测试覆盖与缺口

| 能力 | 现有覆盖 |
|------|----------|
| `force_cancel` 唤醒 `get_length` | ✅ `tests/engine/test_executor.py:107-120` |
| `pre_execute` 卡在 `get_length` 时被取消唤醒 | ✅ `tests/engine/test_executor.py:339-357` |
| 长度驱动串行/并发执行 | ✅ `tests/ops/test_parallel_base_op.py:42` |
| 非 sequence 端口不发 sentinel | ✅ `tests/engine/test_executor.py:287-299` |
| 进度条按长度注册 | ✅ `tests/ops/test_progress_reporting.py:103` |
| 输出长度传递 | ⚠️ `tests/ops/test_timelapse_ops.py:53-62` 等零散断言 |
| `set_length` 冲突 / `None` 覆盖 | ❌ **无** |
| L2 静态变长源冲突（`tests/engine/` 无 wiring 测试文件） | ❌ **无** |
| L3 混用 / 不等长 | ❌ **无**（仅在集成路径间接触发） |
| 超发帧 / 少发帧行为 | ❌ **无** |

建议新增：

- `tests/component/test_queue.py`：扩展 `set_length`/`get_length` 用例（int↔int 冲突、`None` 覆盖、首次广播定值、`force_cancel` 抢占）；
- `tests/engine/test_wiring_length.py`（新文件）：`_check_variable_source_conflicts` 的四类拓扑（单变长源、多变长源、固定+变长、变长传递）与 `_collect_outputs` 的空长度行为；
- `tests/ops/test_base_length_protocol.py`（新文件）：`pre_execute` 两条检查、`_infer_output_length` 默认实现的端口类型过滤、超发护栏、以及遍历 `REGISTERED_OP` 的 `VARIABLE_OUTPUT` 契约测试。

---

## 附录 A：D1 最小复现脚本

未入库的临时脚本，用于验证「超发帧 → 生产者挂死」。可直接复制到仓库根目录运行。

```python
import asyncio, importlib, pkgutil, sys
from typing import Any, Awaitable, Mapping

sys.path.insert(0, ".")
import hoshicore.ops as ops_pkg
for _m in pkgutil.iter_modules(ops_pkg.__path__):
    importlib.import_module(f"hoshicore.ops.{_m.name}")

from hoshicore.component.queue import RichContextQueue, StreamExhausted
from hoshicore.ops.base import BaseOp, ParallelBaseOp


class OverEmitter(BaseOp):
    """声明长度 N（继承自输入），实际广播 N+1 帧。"""
    INPUTS = {"src": {"type": "sequence"}}
    OUTPUTS = {"out": {"type": "sequence"}}

    async def _async_execute(self, configs):
        for i in range(self.length + 1):        # 多发一帧
            await self._broadcast_outputs({"out": f"item{i}"})


class Consumer(ParallelBaseOp):
    INPUTS = {"src": {"type": "sequence"}}
    OUTPUTS = {"out": {"type": "sequence"}}
    got = 0

    async def _async_execute_single(self, data: Mapping[str, Awaitable[Any]], configs):
        value = await data["src"]
        type(self).got += 1
        return {"out": value}


async def main():
    prod, cons = OverEmitter(name="prod"), Consumer(name="cons")
    prod.outputs["out"].append(cons.inputs["src"])
    await prod.inputs["src"].set_length(2)      # 声明长度 2

    sink = RichContextQueue(maxsize=1)
    cons.outputs["out"].append(sink)

    async def drain_sink():
        try:
            while True:
                await sink.get()
        except (StreamExhausted, asyncio.CancelledError):
            pass

    tasks = [asyncio.create_task(prod.execute(), name="producer"),
             asyncio.create_task(cons.execute(), name="consumer"),
             asyncio.create_task(drain_sink(), name="sink_drain")]
    done, pending = await asyncio.wait(tasks, timeout=0.5)
    print("done:", sorted(t.get_name() for t in done),
          "| stuck:", sorted(t.get_name() for t in pending),
          "| consumer consumed:", Consumer.got)
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


asyncio.run(main())
```

预期输出：

```
done: ['consumer', 'sink_drain'] | stuck: ['producer'] | consumer consumed: 2
```

`producer` 卡在 `_send_sentinel()` 的 `await queue.put(SENTINEL)`——第 3 帧占满了 `maxsize=1` 的队列，且消费者已按声明长度 2 退出。
