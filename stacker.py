import argparse
import os
import time

import cv2
import numpy as np
import progressbar
from progressbar import Timer, Percentage, Bar, AdaptiveETA, AnimatedMarker, SimpleProgress

from easystacker.imgloader import load_single_img, BatchImgLoader
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

# calculate weight series
weight_series = generate_weight(len(img_files), fin_ratio, fout_ratio)

img_loader = BatchImgLoader(img_files, weight_series, max_poolsize=8)

# select base image
base_img_filename = img_files[len(img_files) // 2]
base_img = load_single_img(
    base_img_filename, with_exif=True).img * weight_series[len(img_files) // 2]

widgets = [
    'Progress: ',
    Percentage(), ' (',
    SimpleProgress("/"), ') ',
    Bar(marker=AnimatedMarker()), ' ',
    Timer(), ' ',
    AdaptiveETA(), ' '
]

pb = progressbar.ProgressBar(maxval=len(img_files)-1, widgets=widgets)

try:
    pb.start()
    img_loader.start()

    # use simple max to stack images
    while img_loader.prog < len(img_files) - 1:
        # not true end!
        pb.update(value=img_loader.prog)
        t0 = time.time()
        img_series = img_loader.pop(1)
        t1 = time.time()
        img_series.append(base_img)
        base_img = np.max(np.array(img_series), axis=0)
        t2 = time.time()
        #print("||| Loading img time cost: %.2f ; Merging img time cost: %.2f |||" %
        #      (t1 - t0, t2 - t1))

finally:
    img_loader.stop()
    print("writing files...")
    cv2.imwrite("1.tif", base_img.astype(np.uint16))
    print("done.")