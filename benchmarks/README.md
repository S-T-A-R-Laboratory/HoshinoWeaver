# benchmarks

本目录包含两类追踪的基准组件：

- **工作流性能基线**（`benchmark_workflow.py` / `workflow_bench.py`）：进程级基准，逐例通过 `launcher.py` 启动真实 CLI 工作流，记录本机上下文、耗时与资源占用，用于与同类软件端到端比较。见下文"工作流性能基线"。
- **Norma 对齐基准**（`benchmark_norma_alignment.py` / `norma_benchmark.py`）：直接调用 `solve_staged_alignment()` 的组件级回归基准。见下文"Norma 对齐基准"。

本地数据与产物都不进入版本控制：`benchmarks/local/`、`benchmark_results/`。

## 工作流性能基线

### 用途与边界

- 只把工作流当作黑盒：每例是一条由 JSON 描述的完整 `launcher.py` 命令行，在独立进程中顺序执行；不导入 DAG 引擎去执行任何流水线。
- 与 `bench/` 的分工：`bench/`（含 `pipeline.*`）是进程内 microbenchmark，用于定位内核热点；本工具测的是"从命令行启动到产物落盘"的真实用户口径，并额外采集 CPU/RAM/SWAP 与机器上下文，便于跨软件、跨机器对标。
- 当前暂不测定节点级"阶段耗时"：流式 DAG 中节点会提前启动并被上游背压阻塞，单节点墙钟时长不构成可比较的阶段成本，因此只记录进程级与管线级粗粒度边界。

### 快速开始

```powershell
# 1) 校验 schema / 配置键 / 输入清单，并打印每例将执行的完整命令（不创建任何目录）
python -m benchmarks.benchmark_workflow --dry-run

# 2) 打印本机与后端上下文（CPU / GPU 与显存 / RAM / swap / 编译后端）
python -m benchmarks.benchmark_workflow --hardware-only

# 3) 只跑某一例；可重复 --repeat 采样
python -m benchmarks.benchmark_workflow --case stack_sigma_clip_disk_50f --repeat 3

# 4) 跑整个套件
python -m benchmarks.benchmark_workflow

# 5) 与上一次结果对比
python -m benchmarks.benchmark_workflow --compare benchmark_results/workflow/<run>/results.json
```

默认套件路径是 `benchmarks/local/workflow_baseline.json`（被 git 忽略）。仓库内追踪的示例是 `benchmarks/workflow_bench.example.json`，其输入路径是占位符，`--dry-run` 会把缺失输入报告为警告而不是错误，方便先校验结构再补数据。

常用参数：`--case ID`（可重复）、`--repeat N`、`--output-dir PATH`、`--dry-run`、`--hardware-only`、`--compare RESULTS_JSON`。其余运行环境与采样参数见 `--help`，作用于整个套件；运行环境路径不支持逐例覆盖。

退出码：全部用例达到期望 → `0`；任一无期望结果 → `1`；用法/schema 错误 → `2`。

### 手跑 launcher 时的 `@` 清单文件

套件里的输入会由 harness 自动冻结成清单（见下文）；如果手动调用 `launcher.py`，同一个 `@` 约定可以直接使用：

```powershell
python launcher.py hoshicore/dag/stack.meta.yaml `
  --input "fnames=@D:\bench\lights_50.txt" `
  --route stacker=mean `
  --config output_filename=D:\out\stack.tif `
  --config buffer_mode=memory
```

`lights_50.txt` 内容即"一行一个路径"（UTF-8，空行与 `#` 注释被忽略）：

```
# 50 帧亮场
D:\data\lights\light_0001.cr3
D:\data\lights\light_0002.cr3
```

要点：

- `--input KEY=VALUE` 的取值优先级为 `@清单文件` → 已存在目录（展开为排序后的受支持图像）→ JSON 数组 `["a.tif","b.tif"]` → 其他原样字符串；`KEY` 是任意全局 sequence 输入名（如校准链的 `light_fnames`/`dark_fnames`）。
- 清单文件按行顺序投递：不排序、不去重、不展开通配符、不递归子目录。要固定"前 N 帧"或规避 Windows 命令行长度上限时用它。
- 相对路径按**启动 launcher 的当前目录**解析（不是清单文件所在目录）；harness 自己生成的清单写的是绝对路径，因此套件里的相对路径不受此影响。
- 是否生效可核对日志 `[Feeder] Global input '<name>': N items → M queue(s)`，对应结果里的 `phases.feeder_items` 与 `config.input_count`。

### 套件格式（`schema_version: 1`）

未知键会直接报错（防拼写错误）。

```json
{
  "schema_version": 1,
  "suite_id": "workflow_baseline",
  "defaults": {
    "preflight": "ignore",
    "timeout_seconds": 7200,
    "repeats": 1
  },
  "cases": [
    {
      "id": "stack_sigma_clip_disk_50f",
      "pipeline": "hoshicore/dag/stack.meta.yaml",
      "inputs": { "fnames": "D:/lights" },
      "input_limit": 50,
      "routes": { "stacker": "sigma_clip" },
      "configs": { "int_weight": true, "buffer_mode": "disk" },
      "description": "50-frame disk-buffer stack"
    }
  ]
}
```

- `pipeline`：相对路径按仓库根解析，绝对路径直接使用。解释器、launcher、工作目录可在套件 `defaults` 或命令行设置；输出/缓存根目录只可在 `defaults` 设置。这些运行环境选项均不支持逐例覆盖。
- `inputs`：目录路径、`@列表文件`（UTF-8、逐行、支持 `#` 注释）或路径数组三种写法等价（详见上文"手跑 launcher 时的 `@` 清单文件"）。三种写法都会在**计划阶段**被展开或读取成显式清单（仅目录输入按文件名排序，列表文件与数组保留原顺序），`input_limit` 在展开后截断，随后冻结写入`cases/<case_id>/run<k>/inputs_<name>.txt`，并以 `--input <name>=@<该文件>` 启动；清单指纹写入 `config.input_list_sha1`，包含输入名和实际投递顺序；它用于核对输入，不单独决定两次耗时能否比较。因此用例**不需要自己写 `@`**，除非希望清单由外部文件维护。
- 注入的配置：未给 `output_filename` 时注入`{输出根}/{case_id}.tif`（并预建目录，`save_img` 不会自建）；当生效的 `buffer_mode` 为 `disk`/`replay` 且未给 `temp_path` 时注入并预建独立的 cache 目录。
- 未指定 `expected_outputs` 时，默认检查生效的 `output_filename` 是否生成。
- 占位符：`{case_id}`、`{run_index}`、`{output_root}`、`{cache_root}`、`{run_dir}`、`{output_dir}`、`{working_dir}`；替换后仍含 `{` 视为错误。
- `configs` 值为 `null` 视为错误（请直接省略该键）；`dict`/`list` 会按 JSON 传递。
- 运行前会用 `inspect_yaml` 校验 route/config 名称与必需输入，配置类错误在启动前暴露。

### 指标口径

| 字段 | 含义 |
|------|------|
| `wall_seconds` | 外部 `perf_counter` 计的 launcher 进程端到端墙钟，**主对标口径** |
| `phases.pipeline_seconds` | 日志中 `run_from_yaml time cost`，进程内管线总时长 |
| `phases.execution_seconds` | 日志时间戳 `DAG execution starting` → `DAG execution completed. Results collected.` |
| `phases.feeder_items` | 每个全局输入实际投递的帧数（核对输入清单是否如预期） |
| `resources.cpu_percent_avg_normalized` | 默认展示的平均 CPU 占用：进程口径除以逻辑核数，100% 表示全机满载 |
| `resources.cpu_time_seconds` | 最后一次采样到的进程 user+system CPU 累计时间；短任务或退出瞬间可能低估 |
| `resources.rss_peak_bytes` | 进程 RSS 的采样峰值，默认内存口径；短暂峰值可能漏采 |
| `resources.peak_wset_bytes` | 操作系统报的峰值工作集（Windows 原生峰值，不依赖采样） |
| `resources.cpu_percent_avg/peak`、`rss_delta_*`、`system_mem_*`、`swap_used_*` | 仅保留在详细 JSON 中作诊断；系统 swap 变化不能归因于当前进程，启动时 RSS 也不是稳定基线 |
| `resources.pagefile_avg/peak_bytes` | Windows 进程提交量，不能当作实际换页量；仅保留在详细 JSON 中 |
| `storage.*.free_bytes` | 输入/缓存/输出位置运行前后的可用空间（路径不存在时取最近存在的父目录） |
| `declared_storage` | 用例声明的介质（`kind: declared` 或 `undeclared`），不自动探测 |
| `backend` | 该例 env 下由 `probe_runtime_components()` 得到的结构化后端报告 |
| `config.input_count` / `input_list_sha1` | 实际输入帧数 / 按输入名及投递顺序计算的清单指纹 |

采样间隔默认 0.5s（`--measure-interval`）。`--gpu-sample SECONDS>0` 时用
`nvidia-smi` 采样 GPU 显存与利用率，默认关闭，因为它的进程开销会影响 CPU 测量。

### 产物

```
benchmark_results/workflow/<suite_id>_<YYYYmmdd_HHMMSS>/
  results.json                         # 机器上下文 + 每例 runs/summary
  summary.csv                          # 每例每轮一行（对标用表）
  cases/<case_id>/run<k>/
    launcher.log                       # --log-path 目标（原始证据）
    stdout.txt / stderr.txt            # 子进程输出（含打印的日志路径）
    inputs_<name>.txt                  # 冻结的输入清单（以 --input k=@file 传入）
    output/                            # 默认输出目录（output_filename）
    cache/                             # disk 缓冲策略的 temp_path
```

`results.json` 顶层记录 `suite_path`、`cli_args`、生成时间、`environment`
（hardware / software / backend）；每例记录各轮 `runs` 与聚合 `summary`
（wall 的 min/mean/median/max、归一化 CPU 与峰值 RSS 汇总）。

### 与同类软件对标

1. 同一组输入帧、同一输出格式（建议无损 16-bit TIFF）、同一台机器。
2. 记录本工具给出的 `environment` 块；对标时同时记录对方软件的版本与后端设置。
3. 主口径用 `wall_seconds`；`pipeline_seconds` 只用于解释两者差异（例如同类软件的缓存/解码策略不同）。
4. 标注冷热缓存：每例 `run1` 近似冷缓存，后续轮次为热缓存；磁盘缓冲策略、`temp_path` 所在介质会显著影响结果，务必在同介质上比较。
5. 输入清单变化后不要直接比较：先核对 `config.input_list_sha1` 与`phases.feeder_items`。


## Norma 对齐基准

该目录包含 Norma 双帧相机模型对齐的本地开发基准。它直接调用 `solve_staged_alignment()` ，用于检查默认求解路线的匹配覆盖、收敛误差，以及提供 mask 时的图像域残余位移。

### Norma 边界

- 算法编排由 `hoshicore.component.norma` 提供；benchmark 不重复实现检测、匹配、优化或 refine。
- benchmark 只接受 `solve_staged_alignment()` 真实支持的 bootstrap 路径和参数。
- 本地数据集放在 `benchmarks/local/`，不进入版本控制。
- 根目录 `debug_*.py` 和 notebook 是独立诊断工具，不属于 benchmark API，也不应被正式测试导入。

### Norma 运行

```powershell
python -m benchmarks.benchmark_norma_alignment `
  benchmarks/local/norma_alignment_local.json `
  --output-dir benchmark_results/local_current `
  --write-remap
```

可使用 `--case ID`、`--tag TAG` 筛选样本，使用 `--seed` 固定 RANSAC 随机种子。默认 seed 为 `0`。

### Norma 数据集

数据集包含可选的 `defaults` 和必需的 `cases`。每个 case 至少包含：

- `id`
- `reference`
- `source`

`alignment` 中可设置生产 solver 已公开的参数，例如 `matching_path`、 `same_camera`、`bootstrap_scales`、相机初始化参数以及 `guided_refine`。当前回归基线应保持 `matching_path: "asterism"` 和 `guided_refine: false`。 未公开或已移除的实验参数会被拒绝，而不是被静默忽略。

提供 `mask`，或在 `evaluation` 中提供 `reference_mask` / `source_mask` 后， 会评价 remap 图像的局部 residual-shift P90。没有 mask 时只评价匹配点收敛。

### Norma 核心指标

- `final_pairs`：最终参与评价的匹配数量。
- `coverage_ratio`、`outer_pairs`：防止匹配集中在图像中心。
- `p90_px`：最终匹配点投影残差，仅表示 solver 内部收敛。
- `remap_residual_shift_p90_px`：图像域残余位移。
- `remap_evaluated_tiles`、`remap_evaluated_coverage`：图像指标支持度。

输出包括：

- `results.json`：单一、紧凑的 case 结果。
- `summary.csv`：核心指标投影。
- `<case>/src_aligned.tif` 与 `<case>/tgt_reference.tif`：启用
  `--write-remap` 时输出的人工检查图。

运行信息会记录 Git revision、Python、NumPy、OpenCV、平台和每个 case 的随机 seed。
