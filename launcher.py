import argparse
import os
import time

import cv2
import numpy as np
import progressbar
from progressbar import Timer, Percentage, Bar, AdaptiveETA, AnimatedMarker, SimpleProgress

from ezlib.imgloader import load_single_img, ImgSeriesLoader
from ezlib.utils import generate_weight

args = argparse.ArgumentParser()
args.add_argument("dirname", help="dir of images")
args.add_argument("--fade-in", default=0.1)
args.add_argument("--fade-out", default=0.1)
p = args.parse_args()


dir_name = p.dirname
fin_ratio, fout_ratio = float(p.fade_in), float(p.fade_out)

# get filename list in the directory
img_files = os.listdir(dir_name)
img_files.sort()
img_files = [os.path.join(dir_name, x) for x in img_files]


# 最好自适应确定是8还是16，应该是不能单纯根据后缀确定的
cv2.imwrite("1.tif", base_img.astype(np.uint16))