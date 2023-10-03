import argparse
import os

from ezlib.trailstacker import StarTrailMaster
from ezlib.imgfio import save_img

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("dirname", help="dir of images")
    args.add_argument("--fade-in", type=float, default=0.1)
    args.add_argument("--fade-out", type=float, default=0.1)
    p = args.parse_args()

    dir_name = p.dirname
    fin_ratio, fout_ratio = float(p.fade_in), float(p.fade_out)

    # get filename list in the directory
    img_files = os.listdir(dir_name)
    img_files.sort()
    img_files = [os.path.join(dir_name, x) for x in img_files]
    res = StarTrailMaster(img_files, fin_ratio, fout_ratio, int_weight=True)
    save_img("20231004.jpg", res, 90)