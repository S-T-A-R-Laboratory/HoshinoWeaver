# Bundle Adjustment 与序列增稳

## 1. 坐标系定义

```
像素 (u,v)
  ↓  K⁻¹, undistort
相机系 v_cam
  ↓  R_cam(t)
世界系 v_world (地平坐标: Az/Alt)
  ↓  R_sid(t)
天球系 v_cel (RA/Dec, 恒星固定)
```

- **R_cam(t)**: 相机在世界系中的朝向 (3 DOF)
- **R_sid(t)**: 地球自转引起的世界系→天球系旋转，绕极轴以 ω=15°/h 匀速旋转

像素到天球的完整变换:

```
v_cel = R_sid(t) · R_cam(t) · K⁻¹ · undistort(pixel)
```

## 2. 星点对齐给出的信息

对帧 i 和参考帧 ref，星点对齐求解出旋转 R_i，使得:

```
v_ref_cam = R_i · v_i_cam
```

即 R_i 将帧 i 相机系下的星方向变换到参考帧相机系。展开天球约束:

```
R_i = R_cam(ref)ᵀ · R_sid(t_ref)ᵀ · R_sid(t_i) · R_cam(i)
    = R_cam(ref)ᵀ · ΔS_i · R_cam(i)
```

其中 **ΔS_i = R_sid(t_ref)ᵀ · R_sid(t_i)** 是绕极轴旋转 ω·(t_i - t_ref) 的矩阵。

**这是所有后续推导的基本方程。**

## 3. 三种增稳场景

### 3.1 地面延时增稳

对齐点在地面 (世界系固定)，无 R_sid 介入:

```
R_i = R_cam(ref)ᵀ · R_cam(i)
```

增稳: 对每帧施加 R_i⁻¹ 即可消除相机抖动。

### 3.2 星空延时增稳 (固定机位)

相机在三脚架上，R_cam(i) ≈ R_cam(ref) + 小扰动 (抖动)。

#### 有先验 (纬度 + 时间戳)

极轴方向在世界系中已知:

```
polar_axis_world = [cos(lat)·cos(0), cos(lat)·sin(0), sin(lat)]
                 = [cos(lat), 0, sin(lat)]  (取北为 x 轴)
```

ΔS_i 完全确定，直接解:

```
R_cam(i) = ΔS_i⁻¹ · R_cam(ref) · R_i
```

抖动 = R_cam(i) 相对于常数的偏差，warp 修正。

#### 无地理先验 (仅时间戳)

当相机近似不动时:

```
R_i ≈ R_cam(ref)ᵀ · ΔS_i · R_cam(ref)
```

所有 R_i 是**同一轴 p 在相机系下的共轭旋转**: p_cam = R_cam(ref)ᵀ · polar_axis_world。

估计极轴方向:

```python
rvecs = [cv2.Rodrigues(R_i)[0].flatten() for R_i in rotations]
axes = [rv / np.linalg.norm(rv) for rv in rvecs if np.linalg.norm(rv) > 1e-6]
# SVD 拟合共同方向
A = np.stack(axes)
_, _, Vt = np.linalg.svd(A)
polar_axis_cam = Vt[0]  # 第一主成分
```

分解出地球自转后，残差即为抖动:

```python
theta_i = np.dot(rvecs[i], polar_axis_cam)  # 自转角
ΔS_i = rotation_about_axis(polar_axis_cam, theta_i)
R_shake_i = ΔS_i_inv @ R_i
# 增稳: warp by R_shake_i⁻¹
```

### 3.3 移动星空延时增稳

相机有意 pan/tilt，R_cam(i) 非恒定。

#### 求解每帧世界朝向

若极轴已知 (先验或 BA 联合估计，见 §4):

```
R_cam(i) = ΔS_i⁻¹ · R_cam(ref) · R_i
```

每帧的世界朝向可完整计算。

#### 分离平滑意图与抖动

将 R_cam(i) 分解为:

```
R_cam(i) = R_smooth(t_i) · R_shake(t_i)
```

- **R_smooth(t)**: SO(3) 上的平滑曲线 (四元数 B-spline / Squad 插值)
- **R_shake(t)**: 高频小扰动

```python
quats = [mat_to_quat(R_cam(i)) for i in range(N)]
# 平滑拟合 (加权最小二乘或 Savitzky-Golay on quaternion manifold)
smooth_quats = fit_quaternion_spline(timestamps, quats, smoothness=λ)
# 抖动
shake_i = quat_inv(smooth_quats[i]) * quats[i]
# 增稳: warp 到 smooth 轨迹
```

#### 无极轴先验时的估计

相机在动时不能用"共同轴"简化，但可在 BA 中将极轴方向作为待估参数 (仅 2 DOF: 球面坐标 θ, φ)，由全序列观测联合约束。见 §4.2。

### 可行性总结

| 场景 | 可行 | 必需先验 | 可选先验 |
|------|------|---------|---------|
| 地面增稳 | 直接 | 无 | 无 |
| 星空增稳 (固定) | 可行 | 时间戳 (EXIF) | 纬度 (否则从数据估计极轴) |
| 星空增稳 (移动) | 可行 | 时间戳 (EXIF) | 纬度; 首尾帧辅助极轴估计 |

## 4. Bundle Adjustment 实现草案

### 4.1 当前架构的问题

当前 `StarAlignmentOp` 的流程:

```
对每帧 i:
  detect_stars(frame_i) → positions_i
  match_star_pairs(ref ↔ frame_i) → matched_pairs_i
  optimize_alignment(ref_camera, src_camera_i)
    → 独立优化: R_i, focal_scale, distortion (~11 参数)
    → camera1_refined_i ≠ camera1_refined_j (帧间不共享)
  warp_image_by_remap(frame_i)
```

问题:
1. 内参 (focal, distortion) 每帧独立估计，N 帧得到 N 组不同的值
2. 由于内参不固定，`unproject(dst_pixels)` 无法跨帧缓存
3. 少星点帧容易过拟合

### 4.2 BA 参数化

#### 基础 BA (叠加增强)

适用于深空叠加，目标是精确内参 + 帧间旋转。

```
参数向量 x:
  [focal_scale(1), distortion(4), rvec_1(3), rvec_2(3), ..., rvec_N(3)]
  总维度: 5 + 3N

残差 (对每帧 i 的每对匹配点 j):
  v_ref_j = unproject(p_ref_j, K(focal), dist)     -- 参考帧星点方向
  v_src_j = unproject(p_src_j, K(focal), dist)     -- 帧 i 星点方向
  R_i = Rodrigues(rvec_i)
  error_ij = arccos(clamp(dot(R_i · v_ref_j, v_src_j)))

总残差数: Σ_i |matches_i| (典型: 100帧 × 200点 = 20000)
```

#### 扩展 BA (增稳)

在基础 BA 上增加极轴方向估计:

```
参数向量 x:
  [focal_scale(1), distortion(4), polar_axis(2), R_cam_0(3), ..., R_cam_N(3)]
  总维度: 7 + 3N

残差:
  ΔS_i = Rot(polar_axis, ω · Δt_i)
  R_model_i = R_cam(ref)ᵀ · ΔS_i · R_cam(i)  -- 模型预测的星场旋转
  v_predicted = R_model_i · v_src_j
  error_ij = arccos(dot(v_ref_j, v_predicted))
```

### 4.3 实现流程 (两阶段)

```
阶段 1 — 逐帧匹配 (复用现有代码)
  matches: list[MatchResult] = []
  init_rvecs: list[ndarray] = []
  for frame_i in frames:
      geo_i = make_geometry(frame_i)
      match_i = match_star_pairs(ref_geo, geo_i)
      result_i = optimize_alignment(match_i, ref_camera, src_camera)  # 初值
      matches.append(match_i)
      init_rvecs.append(cv2.Rodrigues(result_i.rotation)[0].flatten())

阶段 2 — 全局 BA
  x0 = pack([focal_scale_init, dist_init, *init_rvecs])
  result = least_squares(
      bundle_residuals, x0,
      args=(matches, ref_camera.intrinsics, timestamps),
      method='trf',
      jac_sparsity=build_sparsity_pattern(N, n_matches),
  )
  focal_ba, dist_ba, rvecs_ba = unpack(result.x)

阶段 3 — 应用 (含缓存)
  camera_global = ref_camera.with_focal(focal_ba).with_distortion(dist_ba)
  # 全局内参固定 → unproject 只需做一次
  world_vecs = camera_global.unproject(dst_pixel_grid)  # 缓存

  for i, frame_i in enumerate(frames):
      R_i = Rodrigues(rvecs_ba[i])
      src_cam_i = camera_global.with_pointing_rotation(R_i)
      map_xy = src_cam_i.project(world_vecs)  # 仅 project 逐帧计算
      result_i = cv2.remap(frame_i, map_xy)
```

### 4.4 Jacobian 稀疏结构

BA 的 Jacobian 天然稀疏: 帧 i 的观测只依赖全局内参 + rvec_i:

```
        [focal, dist | rvec_1 | rvec_2 | ... | rvec_N]
obs_1:  [  ×    ×   |   ×    |        |     |       ]
obs_2:  [  ×    ×   |        |   ×    |     |       ]
...
obs_N:  [  ×    ×   |        |        |     |   ×   ]
```

可用 `jac_sparsity` 参数传递给 `least_squares`，显著加速大规模问题。

### 4.5 代码改动清单 (初步)

| 文件 | 改动 |
|------|------|
| `norma/bundle.py` | **新建** — `bundle_residuals()`, `bundle_adjust()`, 稀疏模式构建 |
| `norma/stabilization.py` | **新建** — 极轴估计, SO(3)平滑, 增稳 warp 计算 |
| `norma/types.py` | `project_image_from_camera` 增加 `cached_world_vecs` 参数 |
| `norma/alignment.py` | 导出 `bundle_adjust` |
| `ops/alignment_ops.py` | `StarAlignmentOp` 增加 `use_bundle_adjust` 配置，两阶段逻辑 |

### 4.6 性能预估

基于 `docs/time_cost.md` 中 1000 帧 24MP 的基线:

| 阶段 | 当前耗时 | BA 后预估 |
|------|---------|----------|
| optimization (逐帧) | 148.74 s | ~150 s (初值阶段不变) |
| BA 全局优化 | — | ~5-15 s (20k 观测, 3005 参数, 稀疏 LM) |
| remap (逐帧) | 243.32 s | ~130 s (unproject 缓存, 仅 project 逐帧) |
| **总计** | 398.46 s | ~300 s (估计节省 ~25%) |

主要收益来自 remap 阶段: 消除了 N 次 `unproject`（含 `cv2.undistortPoints` 的迭代求解），仅保留 N 次 `project`（解析公式，快 ~2x）。若进一步引入降采样插值，remap 可再降至 ~30 s。

### 4.7 增稳扩展的额外模块

```python
# norma/stabilization.py 草案结构

def estimate_polar_axis(rvecs: list[ndarray]) -> ndarray:
    """从 BA 旋转向量估计极轴方向 (相机系)。"""
    ...

def decompose_world_orientations(
    rvecs: list[ndarray],
    polar_axis: ndarray,
    timestamps: list[float],
    sidereal_rate: float = 7.2921e-5,
) -> list[ndarray]:
    """从星场旋转 + 极轴 → 每帧世界朝向 R_cam(i)。"""
    ...

def smooth_rotation_trajectory(
    orientations: list[ndarray],
    timestamps: list[float],
    smoothness: float = 1.0,
) -> list[ndarray]:
    """SO(3) 平滑拟合 (四元数 B-spline)。"""
    ...

def compute_stabilization_warps(
    orientations: list[ndarray],
    smooth_orientations: list[ndarray],
) -> list[ndarray]:
    """计算每帧的增稳修正旋转: R_correction = R_smooth · R_actual⁻¹。"""
    ...
```
