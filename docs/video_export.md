# 序列工作流的视频导出

以序列作为输出的工作流（目前覆盖 `timelapse_startrail.meta.yaml` 和 `norma_bundle_window.meta.yaml` ）支持独立选择图像序列与 MP4/AVI 视频输出。默认保存图像序列、关闭视频；至少启用一项。视频分支直接消费最终图像流，不创建中间图片文件。

## 使用方式

GUI 输出面板分别勾选「输出序列」和「导出视频」。关闭的区域保留控件值，不提交路径或要求填写路径。视频完成后可通过「打开已导出视频」调用系统播放器。输出参数复用配置页的行模板和输入样式；数值框直接输入，也可用键盘上下键调整。长选项可在下拉列表或悬停提示中查看完整文字。视频「文件格式」可选 MP4 或 AVI，切换时保留编码参数并按格式记住路径；手工输入已支持的扩展名也会同步格式选择。

`ui/output_presets.py` 的 `VIDEO_FORMAT_PRESETS` 声明扩展名与允许的编码方式，`VIDEO_CODEC_PRESETS` 声明编码说明和适用参数；工作流的 `outputs.formats` 限制可选格式。格式与编码预设共同决定可用参数，不适用的行禁用、收起且不提交。质量模式继续控制 CRF/码率的条件显示，切回 H.264 时恢复此前的参数值。

CLI 只导出视频的例子：

```sh
python launcher.py hoshicore/dag/timelapse_startrail.meta.yaml C:/images --config enable_sequence_export=false --config enable_video_export=true --config video_output_filename=C:/output/timelapse.mp4 --config video_fps=25 --config video_width=1920
```

导出 AVI 时将 `video_output_filename` 改为 `.avi`。后端根据扩展名选择封装，临时文件使用对应后缀；MP4 启用 faststart，AVI 不传该 MP4 封装选项。默认编码仍为 H.264，两种格式均支持它。AVI 还可选择「无压缩 RGB」，CLI 增加`--config video_codec=raw_rgb`；MP4 与 raw_rgb 的组合会在处理图片前报错。无压缩使用 `rawvideo/bgr24`，固定 8 位、完整 RGB 数值范围，不进行 YCbCr 转换或色度采样；CRF、目标码率、编码速度、色度采样和 limited 设置均不用于该模式。输入色彩转换、缩放和帧率仍然有效。封装参数参见 [FFmpeg Formats](https://ffmpeg.org/ffmpeg-formats.html#mov_002c-mp4_002c-ismv)。

同时导出图片时设置 `enable_sequence_export=true` 并提供 `output_dir`。原有全局输出 `result` 仍是图片路径序列；关闭图片导出时该键缺席。`video` 为单个完成回执，包含 `path`、`frame_count`、`fps`、`width`、`height`、`codec`、`container`、`pixel_format`、`color_space`、`color_range` 和`color_metadata_embedded`。后者区分色彩约定与实际是否写入标记； `raw_rgb` 为 `false` 。

## FFmpeg 与视频参数

将 FFmpeg 放在项目或发行目录的 `external/ffmpeg.exe`（其他系统使用 `ffmpeg`）。也可在全局设置选择 `ffmpeg_path`，或通过 CLI 显式覆盖。查找顺序为显式配置、应用资源/可执行文件附近的 `external`、PATH。H.264 使用 CPU `libx264`，无压缩 RGB 使用 `rawvideo`。执行前检查所选编码器与封装器，不要求无压缩模式下安装 libx264。关闭视频导出时不要求安装 FFmpeg。

导出配置使用不可变的 `VideoExportOptions` 结构体。组件默认值由该结构体定义，Op 参数 schema从中提取默认值；YAML 和运行时配置仍可按既有优先级覆盖。字段缺席时使用默认值；显式 `ffmpeg_path=None` 表示自动查找，宽高 0 表示自动尺寸，其他不合法的 None、数值或未知字段会在开始编码前报错。

| 配置                           | 默认值    | 语义                                                              |
|--------------------------------|-----------|-------------------------------------------------------------------|
| `video_output_filename`        | null      | MP4 或 AVI 目标文件，启用视频时必填                               |
| `video_codec`                  | `h264`    | `h264` 或仅限 AVI 的 `raw_rgb`                                    |
| `video_fps`                    | 25        | 播放帧率，与实际计算速度无关                                      |
| `video_width` / `video_height` | 0 / 0     | 0 为自动；同时指定时按边界等比例缩放                              |
| `video_chroma`                 | `"420"`   | `"420"`、`"422"`、`"444"`，均为 8 位                              |
| `video_rate_control`           | `crf`     | `crf` 或 `bitrate`                                                |
| `video_crf`                    | 18        | 恒定质量参数，0–51，越小质量越高                                  |
| `video_bitrate_kbps`           | 12000     | 目标平均码率，仅 bitrate 模式生效                                 |
| `video_preset`                 | `medium`  | x264 速度/压缩效率预设                                            |
| `video_color_range`            | `limited` | H.264 可选 limited/full；raw_rgb 规范化为 full                    |
| `video_input_color`            | `auto`    | 使用传入 ICC，或明确指定 srgb、linear_srgb、bt709、rawpy_prophoto |

H.264 宽度在 420/422 下向上调整到偶数，高度在 420 下同样调整；无压缩 RGB允许奇数宽高，不应用这些约束。输出设置方形像素。帧尺寸必须在整段视频内一致。空序列报错，不生成空视频。

## 色彩约定

输出采用 Rec.709 SDR 原色与 BT.709 传递函数。`VideoColorConverter` 先处理输入曲线和色域。H.264 分支由 FFmpeg 执行缩放、BT.709 RGB→YCbCr、范围映射、色度采样与编码，并写入相应色彩标记。

无压缩 AVI 分支保存 Rec.709 编码的完整范围 RGB8，按约定解释，**不内嵌色彩标签**或 ICC。该分支仅在 FFmpeg 中执行缩放和 RGB 写入；播放器或下游编辑软件需要按Rec.709 解释像素。回执的 `color_space=bt709` 描述像素约定，不代表 AVI 内存在该标记。

- 普通图片的 ICC 通过已有 EXIF 分支传入，使用 Pillow zImageCms/LittleCMS 转换至 sRGB，再转换至 BT.709 曲线；按第一版约定，ICC 处理前量化至 8 位。
- 没有 ICC 的普通图片假定 sRGB，并记录警告；可通过 `video_input_color` 显式指定。
- `auto` 只使用传入的 ICC，不根据源文件路径、后缀或解析后端推断当前像素空间。源 ICC 是否仍与解码后像素匹配，由调用方保证。
- 解码改变色彩编码时，应显式指定实际输入空间。`rawpy_prophoto` 是手动选择的ProPhoto 原色＋BT.709 近似曲线转换；仅适用于匹配这一约定的像素，不代表所有 RAW。线性 sRGB 使用 `linear_srgb`；PS 生成 TIFF 时应使用生成文件中匹配像素的 ICC。

## 执行和取消

`VideoSaveOp` 继承 `BaseOp`，接受 `data` 和可选的同步 `exifs` 序列，支持固定长度及 `length=None`/sentinel。单个 FFmpeg 进程按顺序接收 BGR8 字节；每帧管道写入完成后才读取下一帧。编码较慢时通过有界队列向上游传递背压。

管道和进程等待运行在工作线程，stderr 由独立线程持续读取且仅保留有界尾部。结束时关闭 stdin，等待编码退出与视频封装完成，再原子替换正式目标文件。同名既有视频仅在成功时被替换；失败或取消会终止进程、等待写入线程退出并清理本次临时视频。FFmpeg 的编码缓冲内存不等同于 DAG 队列内存，资源估算中的转换缓冲不包含外部编码器的全部工作内存。
