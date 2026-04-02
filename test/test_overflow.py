import numpy as np
from hoshicore.component.utils import FastGaussianParam

print("=== FastGaussianParam 溢出修复测试 ===\n")

# 测试 1: sum_mu 溢出检查
print("测试 1: sum_mu 自动 upscale")
img = np.full((100, 100), 255, dtype=np.uint8)
param = FastGaussianParam(img, source_dtype=np.dtype('uint8'))
print(f"初始化后: sum_mu.dtype={param.sum_mu.dtype}, square_sum.dtype={param.square_sum.dtype}")
print(f"安全叠加数量: {param._safe_add_count()}")

for i in range(300):
    param = param + FastGaussianParam(img, source_dtype=np.dtype('uint8'))
    if i == 256:
        print(f"累加 257 张后: sum_mu.dtype={param.sum_mu.dtype}, n.max={param.n.max()}")

print(f"累加 300 张后: sum_mu.dtype={param.sum_mu.dtype}, square_sum.dtype={param.square_sum.dtype}, n.max={param.n.max()}")
print(f"新的安全叠加数量: {param._safe_add_count()}\n")

# 测试 2: mask 方法
print("测试 2: mask 方法保持 n 值")
param2 = FastGaussianParam(img, source_dtype=np.dtype('uint8'))
param2.n[:] = 100
mask = np.ones((100, 100), dtype=bool)
mask[50:, :] = False
param2.mask(mask)
print(f"mask 后: n[0,0]={param2.n[0,0]}, n[50,0]={param2.n[50,0]}")
print(f"预期: n[0,0]=100 (保持), n[50,0]=0 (mask 掉)\n")

# 测试 3: var 计算
print("测试 3: var 计算")
print(f"方差形状: {param.var.shape}")
print(f"方差计算成功\n")

print("=== 所有测试完成 ===")
