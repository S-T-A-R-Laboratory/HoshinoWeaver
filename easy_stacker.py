import argparse
import os
import tqdm
import cv2
import numpy as np


def generate_weight(length, fin, fout):
    assert fin <= 0.5 and fout <= 0.5
    in_len = int(length * fin)
    out_len = int(length * fout)
    in_step, out_step = 100 / in_len, 100 / out_len
    l = np.arange(1, 100, in_step) / 100
    r = np.arange(1, 100, out_step)[::-1] / 100
    ret_weight = np.ones((length, ), dtype=np.float16)
    ret_weight[:in_len] = l
    ret_weight[-out_len:] = r
    return ret_weight


# support jpg only now.
def load_img(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8),
                        cv2.IMREAD_UNCHANGED)


args = argparse.ArgumentParser()
args.add_argument("dirname", help="dir of images")
args.add_argument("--fade-in", default=0.1)
args.add_argument("--fade-out", default=0.1)
p = args.parse_args()

dir_name = p.dirname
fin_ratio, fout_ratio = p.fade_in, p.fade_out

img_files = os.listdir(dir_name)
img_files.sort()
img_files = [os.path.join(dir_name, x) for x in img_files]

base_img_filename = img_files[len(img_files) // 2]
base_img = load_img(base_img_filename)

weight_series = generate_weight(len(img_files), fin_ratio, fout_ratio)
# simple max
for (img_filename, w) in tqdm.tqdm(zip(img_files, weight_series),
                                   total=len(img_files),
                                   ncols=100):
    img = (w * load_img(img_filename)).astype(np.uint8)
    base_img = np.max(np.concatenate((base_img[None, ...], img[None, ...]),
                                     axis=0),
                      axis=0)

cv2.imwrite("1.jpg", base_img)
