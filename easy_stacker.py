import PIL.Image
import numpy as np
import argparse
import os
import cv2

args = argparse.ArgumentParser()
args.add_argument("dirname", help="dir of images")
p = args.parse_args()

dir_name = p.dirname

img_files = os.listdir(dir_name)
img_files.sort()
img_files = [os.path.join(dir_name, x) for x in img_files]

base_img_filename = img_files[len(img_files) // 2]
with open(base_img_filename, mode='rb') as f:
    base_img = np.array(PIL.Image.open(f))

# simple max
for img_filename in img_files:
    with open(img_filename, mode='rb') as f:
        img = np.array(PIL.Image.open(f))
    base_img = np.max(np.concatenate((base_img[None,...], img[None,...]), axis=0), axis=0)

cv2.imwrite("1.jpg",base_img[...,[2,1,0]])