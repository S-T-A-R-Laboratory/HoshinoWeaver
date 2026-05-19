# Benchmark Data

这个目录用于放 benchmark 可复用的输入数据。

目录约定：

- `bench/data/cache/`
  推荐的 raw cache 输入目录。适合重复 benchmark，避免重复图片解码。
- `bench/data/input/`
  手工准备的真实图片目录。
- `bench/data/generated/`
  用 `python -m bench.data_tools.generate_dataset` 生成的测试图片目录。

当前 benchmark 脚本的输入策略：

1. 如果显式传了 `--input-dir`，优先扫描该路径。
2. 否则先扫描 `bench/data/cache/`。
3. 再扫描 `bench/data/input/`。
4. 再扫描 `bench/data/generated/`。
5. 若以上都没有足够输入，再回退到合成随机数据。

说明：

- raw cache 是当前默认的性能 benchmark 输入。
- raw cache 可以通过 synthetic 生成，也可以由现有图片目录转换得到。
- 图片目录输入主要保留给 smoke test 和输入链路验证，建议只保留小规模图片集。
- 当前 benchmark 的输入准备发生在计时开始之前，不把 cache 加载或图片解码混入核心算子时间。
