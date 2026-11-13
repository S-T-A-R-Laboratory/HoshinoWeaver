# 统一打包入口

`make_package.py` 管理 PyInstaller（默认）和 Nuitka 两个后端的公共发行流程。

## 使用方式

安装项目及开发依赖，使用同一 Python 构建 custom-op 和应用：

```sh
python -m pip install -r requirements.txt -r requirements-dev.txt
python make_package.py --no-build --verify
python make_package.py --backend nuitka --no-build --jobs=4 --verify
python make_package.py --backend nuitka --no-build --dry-run
python make_package.py --backend nuitka --verify-only
```

默认在 `_C` 不可导入时自动执行 `csrc/build_ops.py`，构建后重新探测导入；
若需 CUDA，先运行 `python csrc/build_ops.py --cuda`。平台要求见
[custom-op 构建说明](../csrc/README.md)。完整参数以 `python make_package.py --help` 为准。

| 参数 | 行为 |
|---|---|
| `--backend` | `pyinstaller` 或 `nuitka`，默认 `pyinstaller` |
| `--no-build` | 跳过 custom-op 自动编译，仍构建应用；缺少 `_C` 时失败 |
| `--allow-numpy-only` | 允许缺少 `_C`，使用 NumPy/contour 回退；不会禁用已可用的 `_C` |
| `--debug-gui` | GUI 保留控制台，发行目录追加 `-debug`；CLI 始终保留控制台 |
| `--jobs N` | 仅控制 Nuitka 的 C 编译并发；必须为正整数，默认 2，以降低 MSVC 内存压力 |
| `--build-dir PATH` | 指定后端中间目录；不得为仓库根目录或位于 `dist/` 内，不改变发行位置 |
| `--dry-run` | 探测 native、生成配置和命令，不编译 custom-op/应用、不发布 |
| `--overwrite` | 允许成功构建后替换同版本发行目录 |
| `--verify` | 发布前验证暂存包；验证失败保留既有发行目录 |
| `--verify-only` | 验证所选后端的当前版本发行目录，不探测或编译源码 native |
| `--apply-upx` | 压缩可处理的二进制，排除 `_C`、`_metal` 和 VC runtime |
| `--apply-zip` | 生成发行 ZIP，根目录直接包含发行文件 |

`--dry-run` 仍需要 native 探测通过或指定 `--allow-numpy-only`；
验证 NumPy-only 包时也需指定 `--allow-numpy-only`。

## 模块与公共流程

```text
make_package.py                    参数、后端分流、退出码
hoshicore/packaging/
    __init__.py                   轻量包入口
    common.py                     资源探测、native 准备、验证、发布和 ZIP
    pyinstaller.py                spec / MERGE / PyInstaller 调用
    nuitka.py                     package configuration / 编译 / 产物合并
```

这些模块仅用于构建，应用不导入它们，两个后端也排除 `hoshicore.packaging`。
`__init__.py` 保持轻量，打包器通过子进程调用。

项目存在 `external/` 时，公共流程在后端后处理之后、验证之前将其原样复制到发行
目录的 `external/`，供 FFmpeg 等外部可执行文件使用，不参与编译器的模块依赖分析。
macOS 非 debug GUI 另外复制一份到 `.app/Contents/MacOS/external/`，与 GUI 可执行文件
相邻，因此单独移动 `.app` 仍可发现 FFmpeg。复制后重新进行 ad-hoc 签名并验证运行；
这不是 Developer ID 签名或 Apple 公证。目录外的副本供 CLI 使用。

- `BuildOptions`：公共参数、仓库路径及后端选择。
- `PackageManifest`：DAG、settings、LICENSE、native 状态、包内 DLL、可选 turbojpeg 和 Metal shader；DLL 与数据资源分开描述。
- `BuildResult`：发行内容目录、CLI/GUI 可执行路径、CLI 资源目录和 GUI 日志目录，供公共验证器适配不同后端布局。

后端实现 `build(options, manifest, stage)`、`postprocess(result, options)`、
`describe_release(directory, options)`，公共流程为：

```text
native 准备 → 资源清单 → 后端构建和组装 → 后处理 → 按需验证 → 发布 → 按需 ZIP
```

后端各自维护资源清单到构建配置的转换及依赖排除策略。UPX 排除规则在公共层维护：
PyInstaller 在 EXE/COLLECT 阶段压缩；Nuitka 在组装后使用 `--best --lzma`，
与其 UPX 插件的压缩设置对齐，UPX 缺失或拒绝某文件时记录警告。
UPX 默认关闭，压缩收益和启动开销需实测。

## 模块排除

对于Nuitka，NumPy、SciPy、Astropy、tifffile、Pillow 的 Python 模块及 `ui.resource` 使用`--noinclude-custom-mode=<package>:bytecode` 收集。仍跟踪导入并收集 DLL/数据，现有 `.pyd` 直接收集；`hoshicore` 和其余 UI 代码继续编译。这减少科学库包装层生成的 C 文件及 MSVC 内存压力。

两种入口均排除 matplotlib、pandas、IPython、tkinter，以及公共排除项TensorFlow/Keras/PyTorch/PyQt。anti-bloat 禁用可选 pytest、setuptools 和Jupyter/IPython 组件；构建报告暴露的 SymPy、mpmath、Arrow、Narwhals 也被排除，对应的符号多项式和表格/统计功能不属于当前工作流。FITS 图像 I/O、NumPy/SciPy 数值计算和 NetworkX DAG 验证仍保留。
修改依赖或引入新工作流时，应重新核对这些排除项。

## 资源、发布与诊断

逻辑资源包括 `hoshicore/dag/`、`hoshicore/default_settings.yaml`、`LICENSE`
及可用的 native 依赖。

模型资源包含 `models/skyseg_v2/` 中的 ONNX 权重、运行时读取的 `metadata.json` 和许可证。

PyInstaller 的 CLI 资源位于 `_internal/`，Nuitka 位于发行根目录；
应用资源定位须独立于启动目录和开发源码。Windows 入口名为
`HoshinoWeaver desktop.exe` / `launcher.exe`；macOS 非 debug GUI 使用
`HoshinoWeaver desktop.app`，CLI 位于发行目录。

发行目录为 `dist/HoshinoWeaver desktop_<平台>_<VERSION>[-debug]/`，
ZIP 为 `HoshinoWeaver_<平台>_<VERSION>.zip`。两个后端共用命名，
切换后端构建同版本也需 `--overwrite`。

两种后端先在独立中间目录构建，再组装到 `dist/.package-*`。
合并时共享内容一致的同名文件，内容冲突直接报错。
默认拒绝覆盖已有发行目录；启用 `--verify` 时暂存包验证通过后才发布。
发布期间临时保留旧目录，新目录改名失败则恢复旧目录；ZIP 也先生成临时文件。

配置和命令分别保存在 `build/pyinstaller/`、`build/nuitka/`；
这些是构建产物。构建子进程过滤与构建无关的认证环境变量，避免诊断文件持久化这些值。

## 验证范围

`--verify` 与 `--verify-only` 共用验证器，清理开发环境的 Python/DLL/Qt 搜索路径，
在项目外的临时目录运行，固定使用 CPU 后端：

- CLI：检查 `--help`、内置 DAG `--inspect`、`_C`（native 包）及 pyexiv2 能力报告。
- 图像处理：真实叠加三张 uint16 PNG，逐像素比较最大值结果，覆盖 EXIF 读写路径。
- GUI：使用 Qt offscreen，等待启动能力日志（最长 90 秒），再检查 8 秒存活并关闭；Windows 用户配置隔离到临时目录，检查后移除新增 GUI 日志。

发布前仍需在真实显示器上检查主窗口、默认参数面板、图标和字体。
Linux/macOS、GPU、UPX、debug GUI 和 NumPy-only 配置需分别构建验证；
一次默认 Windows 验证不覆盖这些组合。

比较两种后端的启动时间、体积和处理速度时，应使用同版本、同依赖环境。
Nuitka standalone 仍保留 Python runtime 与第三方 native 库，性能收益以实测为准。

## 手动构建 macOS arm64（Nuitka）

工作流为 `.github/workflows/package-macos.yaml`，只接受 `workflow_dispatch`，不自动发布
Release。在 GitHub 的 Actions 中选择
**Package macOS arm64 (Nuitka)** → **Run workflow**，选定构建分支并填写：

- `ffmpeg_url`：独立 macOS arm64 FFmpeg 的 HTTPS ZIP 下载地址，ZIP 中必须恰有一个
  名为 `ffmpeg` 的文件，可位于子目录。
- `ffmpeg_sha256`：该 ZIP 的 64 位十六进制 SHA256，应从可信来源核对。

工作流固定 `macos-15` 和 arm64 Python 3.11，安装 OpenMP、inih、jpeg-turbo 及项目依赖，
构建 OpenMP/Metal 扩展，再调用 `make_package.py --backend nuitka --no-build --jobs=2 --verify`。
Python 依赖仍按仓库范围解析，并将实际版本写入发行目录的 `build-requirements.txt`；
当前模板不承诺完全可复现的构建或 macOS 13 最低系统兼容性，需在目标系统另测。

Homebrew 官方 `ffmpeg` formula 使用 shared 构建，不能将 `brew install ffmpeg` 等同于
获得独立静态程序。此模板通过 `lipo` 检查 arm64，通过 `otool -L` 拒绝非系统动态库依赖；
macOS 独立程序仍可依赖系统库。FFmpeg 必须支持 `libx264`、MP4 及 AVI/rawvideo。
ZIP 内的 LICENSE/COPYING/NOTICE 会一并保存，也记录下载地址、摘要、版本和 FFmpeg
自身的许可信息。公开再分发前仍需核对该构建对应的源码提供方式与许可要求。

构建完成后检查 CLI 和 `.app` 内各自的 FFmpeg，在收紧 PATH 的环境中编码短 MP4 和
无压缩 AVI，并检查 app 签名。完整发行目录（CLI、`.app`、外部程序及资源）打为
带 `macos-arm64` 标识的 `.tar.gz`，保留符号链接和执行权限，上传至运行页的 Artifacts，
保留 14 天；另附归档 SHA256。下载 artifact 后还需解开其中的 tar 包。
这里不使用普通文件 ZIP 或直接上传 `.app`，也不创建只包含 GUI 的 DMG。
首次运行仍需下载到无 Homebrew 的 Mac 上确认 GUI、视频导出及系统安全提示。
