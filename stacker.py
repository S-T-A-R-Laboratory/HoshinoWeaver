import argparse
import os

import cv2
import numpy as np
import tqdm

from easystacker.imgloader import load_single_img
from easystacker.utils import generate_weight

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

# select base image
base_img_filename = img_files[len(img_files) // 2]
base_img = load_single_img(base_img_filename, with_exif=True).img

# calculate weight series
weight_series = generate_weight(len(img_files), fin_ratio, fout_ratio)

# use simple max to stack images
for (img_filename, w) in tqdm.tqdm(zip(img_files, weight_series),
                                   total=len(img_files),
                                   ncols=100):
    img = (w * load_single_img(img_filename, with_exif=False).img)
    base_img = np.max(np.concatenate((base_img[None, ...], img[None, ...]),
                                     axis=0),
                      axis=0)

cv2.imwrite("1.jpg", base_img.astype(np.uint8))
