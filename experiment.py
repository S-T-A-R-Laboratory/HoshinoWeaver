# 用于测试，拟合一些函数
import torch
from math import log
import matplotlib.pyplot as plt
# 对于N(0,1)分布，n次最大值的期望？
if False:
    # 采样1-1000, 采样1000次
    y=[torch.mean(torch.max(torch.randn((1000,i+1)),dim=1).values).numpy() for i in range(2000)]
    plt.plot(y)
    # N in [80,2000]时该函数拟合性质很不错
    y0=[1/2*log(x+1)**(0.84)+0.71 for x in range(2000)]
    plt.plot(y0,'r')
    plt.show()

# 采样1-1000, 采样1000次
y=[torch.mean(torch.max(torch.randn((1000,i+1)),dim=1).values).numpy() for i in range(80)]
plt.plot(y)
# N in [0,80]时该函数拟合性质挺好
y0=[0]+[1/2*log(x)**(0.91)+0.52 for x in range(1,80)]
plt.plot(y0,'r')
plt.show()
