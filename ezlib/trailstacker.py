import time

import cv2
import numpy as np
from .imgloader import ImgSeriesLoader
import multiprocessing as mp

MIN_IMG_PER_PROCESSOR=2

def StarTrailMaster(fname_list:list[str], fin_ratio: float, fout_ratio: float, mp_num:int)->np.ndarray:
    """星轨叠加主进程

    Args:
        fname_list (list[str]): 图片名称序列
        fin_ratio (float): 渐入效果比值
        fout_ratio (float): 渐出效果比值
        mp_num (int): 多进程数目

    Returns:
        np.ndarray: 叠加完成的图像
    """
    tot_length=len(fname_list)
    weight_list = generate_weight(tot_length, fin_ratio, fout_ratio)
    sub_flist,sub_wlist=[],[]
    results=mp.Queue()
    
    # 图像过少时减少处理器依赖数目
    # 每个处理器至少叠加2张图像
    mp_num = min((tot_length+MIN_IMG_PER_PROCESSOR-1)//MIN_IMG_PER_PROCESSOR, mp_num)
    sub_length=tot_length/mp_num
    for i in range(mp_num):
        l,r = int(i*sub_length),int((i+1)*sub_length)
        sub_flist.append(fname_list[l:r])
        sub_wlist.append(weight_list[l:r])

    
    mp.Pool(processes=StarTrailStacker)
    
def StarTrailStacker(fname_list:list[str], weight_list:list[float])->np.ndarray:
    img_num=len(fname_list)
    img_loader = ImgSeriesLoader(fname_list, max_poolsize=8)
    try:
        img_loader.start()
        base_img = img_loader.pop().img
        for i in range(img_num-1):
            cur_img = img_loader.pop().img
            base_img = np.max(cur_img, axis=0)

    finally:
        img_loader.stop()
        
        

def generate_weight(length, fin, fout):
    assert fin + fout <= 1
    in_len = int(length * fin)
    out_len = int(length * fout)
    ret_weight = np.ones((length, ), dtype=np.float16)
    if in_len>0:
        l = np.arange(1, 100, 100 / in_len) / 100
        ret_weight[:in_len] = l
    if out_len>0:
        r = np.arange(1, 100, 100 / out_len)[::-1] / 100
        ret_weight[-out_len:] = r
    return ret_weight