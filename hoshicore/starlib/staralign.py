"""
This module is based on the original code from: https://github.com/LoveDaisy/star_alignment .
Modified and optimized to fit HoshinoWeaver project.

Author: Jiajie Zhang
Email: zhangjiajie043@gmail.com

Updated: Sean Liu
Email: sean.liu.2004@gmail.com
"""

import os
from math import log
from typing import Optional, Union

import cv2
import numpy as np
import numpy.linalg as la
import pywt
from loguru import logger
from scipy.spatial import distance as spd

from ..ezlib.imgfio import load_img, load_info
from ..ezlib.utils import time_cost_warpper

class ImageProcessing(object):

    def __init__(self):
        super(ImageProcessing, self).__init__()

    @staticmethod
    def wavelet_dec_rec(img_blr: np.ndarray, resize_factor: float = 0.25):
        """
        Take a picture, does a wavelet decompsition, remove the low frequency (approximation) and highest details (noises)
        and return the recomposed picture

        Args:
            img_blr (np.ndarray): _description_
            resize_factor (float, optional): _description_. Defaults to 0.25.

        Returns:
            _type_: _description_
        """
        img_shape = img_blr.shape

        need_resize = abs(resize_factor - 1) > 0.001
        level = int(6 - log(1 / resize_factor, 2))

        if need_resize:
            img_blr_resize = cv2.resize(img_blr,
                                        None,
                                        fx=resize_factor,
                                        fy=resize_factor)
        else:
            img_blr_resize = img_blr
        coeffs = pywt.wavedec2(img_blr_resize, "db8", level=level)
        #remove the low freq (approximation)
        coeffs[0].fill(0)
        #remove the highest details (noise??)
        coeffs[-1][0].fill(0)
        coeffs[-1][1].fill(0)
        coeffs[-1][2].fill(0)

        img_rec_resize = pywt.waverec2(coeffs, "db8")
        if need_resize:
            img_rec = cv2.resize(img_rec_resize, (img_shape[1], img_shape[0]))
        else:
            img_rec = img_rec_resize

        return img_rec

    @staticmethod
    @time_cost_warpper
    def detect_star_points(img: np.ndarray,
                           mask: Optional[np.ndarray] = None,
                           resize_length: int = 2200,
                           ksize: int = 9,
                           sigma: int = 2) -> tuple:
        """基于小波变换的星点检测算法。
        
        Modified from the original code: https://github.com/LoveDaisy/star_alignment.

        Optimized to fit HoshinoWeaver project.

        Args:
            img_gray (np.ndarray): _description_
            mask (Optional[np.ndarray], optional): _description_. Defaults to None.
            resize_length (int, optional): _description_. Defaults to 2200.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            tuple: _description_
        """
        # NOTE: 此处默认输入均为归一化float。
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # GaussianBlur and normalize
        # TODO: 需要验证uint8是否因为精度损失可能影响效果
        img_shape = img.shape
        img_blr = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)
        img_blr_mean = np.mean(img_blr)
        img_blr_range = np.max(img_blr) - np.min(img_blr)
        img_blr = (img_blr - img_blr_mean) / img_blr_range

        # 2x resize to limited scale
        resize_factor = 1
        while max(img_shape) * resize_factor > resize_length:
            resize_factor /= 2

        # mask 90% dark pixel (or 0.15 as threshold, lower one)
        tmp_mask = cv2.resize(img, None, fx=resize_factor, fy=resize_factor)
        tmp_mask_10percent = np.percentile(tmp_mask, 10)
        tmp_mask = (tmp_mask < min(tmp_mask_10percent, 0.15)).astype(
            np.uint8) * 255

        dilate_size = int(max(img_shape) * 0.02 * resize_factor * 5)
        tmp_mask = 255 - cv2.dilate(
            tmp_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (dilate_size, dilate_size)))
        tmp_mask = cv2.resize(tmp_mask, (img_shape[1], img_shape[0]))

        # combined with input mask
        if mask is None:
            mask = tmp_mask > 127
        else:
            mask = np.logical_and(tmp_mask > 127, mask > 0)

        mask_rate = np.sum(mask) * 100.0 / np.prod(mask.shape)
        logger.debug(f"mask rate: {mask_rate:.2f}%")
        if mask_rate < 50:
            mask = np.ones(tmp_mask.shape, dtype="bool")

        while True:
            try:
                img_rec = ImageProcessing.wavelet_dec_rec(
                    img_blr, resize_factor=resize_factor) * mask

                bw = ((img_rec > np.percentile(img_rec[mask], 99.5)) *
                      mask).astype(np.uint8) * 255
                bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                                      np.ones((3, 3), np.uint8))

                contours, _ = cv2.findContours(np.copy(bw), cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_NONE)
                contours = [
                    contour for contour in contours if len(contour) > 5
                ]
                logger.debug("%d star points detected", len(contours))

                if len(contours) > 400:
                    break
                else:
                    raise ValueError("No enough points")
            except ValueError as e:
                if resize_factor >= 1:
                    raise ValueError("Cannot detect enough star points:" +
                                     str(e))
                else:
                    resize_factor *= 2
        logger.debug(f"resize factor = {resize_factor}")

        #elps - match contours to an ellipse. Return a Box2D - coordinates of a rotated rectangle - 3x tuples
        #first tuple is the center of the box, the second gives the width and the height and the last is the angle.
        elps = [cv2.fitEllipse(contour) for contour in contours]
        # centroids - the "center" of the ellipses
        centroids = np.array([e[0] for e in elps])
        #areas - the areas of the contours, but 0.5*len(contour)?
        areas = np.array([
            cv2.contourArea(contour) + 0.5 * len(contour)
            for contour in contours
        ])
        #eccentricities - how irregular the ellipse is.
        eccentricities = np.sqrt(
            np.array([1 - (elp[1][0] / elp[1][1])**2 for elp in elps]))

        # Calculate "intensity"

        mask = np.zeros(bw.shape, np.uint8)
        intensities = np.zeros(areas.shape)
        for i in range(len(contours)):
            cv2.drawContours(mask, contours[i], 0, 255, -1)
            #It is a straight rectangle, it doesn't consider the rotation of the object. .
            #Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
            #x,y,w,h = cv2.boundingRect(cnt)
            rect = cv2.boundingRect(contours[i])
            val = cv2.mean(
                img_rec[rect[1]:rect[1] + rect[3] + 1,
                        rect[0]:rect[0] + rect[2] + 1],
                mask[rect[1]:rect[1] + rect[3] + 1,
                     rect[0]:rect[0] + rect[2] + 1])
            mask[rect[1]:rect[1] + rect[3] + 1,
                 rect[0]:rect[0] + rect[2] + 1] = 0
            intensities[i] = val[0]

        # filter valid stars
        valid_stars = np.logical_and(areas > 20, areas < 200, eccentricities
                                     < .8)
        valid_stars = np.logical_and(
            valid_stars, areas > np.percentile(areas, 20), intensities
            > np.percentile(intensities, 20))

        star_pts = centroids[valid_stars]  # [x, y]
        logger.info(f"Final star points = {len(star_pts)}")

        areas = areas[valid_stars]
        intensities = intensities[valid_stars]

        return star_pts, areas * intensities

    @staticmethod
    def convert_to_spherical_coord(star_pts: np.ndarray,
                                   img_size: np.ndarray,
                                   focal_length: Union[float, int],
                                   crop_factor: Union[float, int] = 1.0):
        """convert standard pixel coordinates to spherical coordinates.

        Args:
            star_pts (np.ndarray):  np array of start points in (x,y) coodinates
            img_size (np.ndarray): image size in pixels, (h, w) order.
            focal_length (Union[float, int]): focal length, the "real focal length" before crop factor. In real life no real effect observed.
            crop_factor (Union[float, int], optional): sensor crop factor. In real life no real effect is observed. Defaults to 1.0.

        Returns:
            np.ndarray: theta and phi in spherical corrdinates.
        """
        logger.debug(
            "convert_coord_img_sph(Focal length {0}, crop_factor {1})".format(
                focal_length, crop_factor))
        FullFrameSensorWidth = 36  #Full frame sensor width is 36mm
        sensorSize = FullFrameSensorWidth / crop_factor  #Actual sensor size
        PPMM = np.max(img_size) / sensorSize  #Pixels per mm

        p0 = (star_pts - img_size / 2.0
              )  #Adjust start x,y coords to the middle of lens
        p = p0 / PPMM  #Convert coordinates to mm

        theta = np.arctan2(p[:, 0], focal_length)
        phi = np.arcsin(p[:, 1] /
                        np.sqrt(np.sum(p**2, axis=1) + focal_length**2))
        return np.stack((theta, phi), axis=-1)

    @staticmethod
    def extract_point_features(sph, vol, k=15):
        '''
            extract_point_features
            Calculate the "features", or signatures of each starpoint
            Identified by the angles (theta) and distances (ro) to K "neighbors"
            input: sph: Spherical coordinates, theta([:0]) and phi ([:1])
            vol: "Volume", i.e. the product of area and intensity/average luminosity
            k: number of "neighbors"
            output: Array of "features" of each star point derived from relationship between K neighbors:
            theta: angle from the star point
            rho: distance from the star point
            vol: volume
        '''
        logger.debug("extract_point_features()")
        pts_num = len(sph)
        #convert theta,phi to x,y,z, assuming ro is 1
        vec = np.stack(
            (np.cos(sph[:, 1]) * np.cos(sph[:, 0]),
             np.cos(sph[:, 1]) * np.sin(sph[:, 0]), np.sin(sph[:, 1])),
            axis=-1)
        #Caclulate cosine distance among vectors
        dist_mat = 1 - spd.cdist(vec, vec, "cosine")
        #Sort by cosine distance and store the index to vec_dist_ind
        #since cosine ranges from 1 (itself) to -1 (opposite), the order
        #of -dist_mat sorts from self to opposite
        vec_dist_ind = np.argsort(-dist_mat)
        #Make sure cosine is in range [-1,1]
        dist_mat = np.where(dist_mat < -1, -1,
                            np.where(dist_mat > 1, 1, dist_mat))
        #dist_mat = np.clip(dist_mat, -1, 1)

        # Calculate the angle to the closest 2*k (30) neighbors
        dist_mat = np.arccos(dist_mat[np.array(range(pts_num))[:, np.newaxis],
                                      vec_dist_ind[:, :2 * k]])
        ##vol: the "volume" of the stars, e.g intensity*area
        ##Find the volume of the closes 2k (30 neighbors)
        vol = vol[vec_dist_ind[:, :2 * k]]
        ##Create index of sorted product of vol and angle in descending order
        vol_ind = np.argsort(-vol * dist_mat)

        def make_cross_matrix(v):
            #Make a matrix for further angle calculation
            # Input v (x,y,z)
            # Return [0,-z,y]
            #       [z,0,-x]
            #       [-y,x,0]
            return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]],
                             [-v[1], v[0], 0]])

        theta_feature = np.zeros((pts_num, k))
        rho_feature = np.zeros((pts_num, k))
        vol_feature = np.zeros((pts_num, k))

        for i in range(pts_num):
            v0 = vec[i]  #current point vector, x,y,z
            vs = vec[vec_dist_ind[i, vol_ind[
                i, :k]]]  #coordinates[x,y,z] of k neighbours sorted by vol*dist
            angles = np.inner(vs, make_cross_matrix(
                v0))  #[-y1*z0+z1*y0,x1*z0-z1*x0,-x1*y0+y1*x0]
            #la.norm by default is Frobenius as in sqrt(sum(angles **2),axis=1))
            angles = angles / la.norm(angles, axis=1)[:, np.newaxis]
            cr = np.inner(angles, make_cross_matrix(angles[0]))
            s = la.norm(cr, axis=1) * np.sign(np.inner(cr, v0))
            c = np.inner(angles, angles[0])
            theta_feature[i] = np.arctan2(s, c)
            rho_feature[i] = dist_mat[i, vol_ind[i, :k]]
            vol_feature[i] = vol[i, vol_ind[i, :k]]

        fx = np.arange(-np.pi, np.pi, 3 * np.pi / 180)
        features = np.zeros((pts_num, len(fx)))
        for i in range(k):
            sigma = 2.5 * np.exp(-rho_feature[:, i] * 100) + .04
            tmp = np.exp(-np.subtract.outer(theta_feature[:, i], fx)**2 / 2 /
                         sigma[:, np.newaxis]**2)
            tmp = tmp * (vol_feature[:, i] * rho_feature[:, i]**2 /
                         sigma)[:, np.newaxis]
            features += tmp

        features = features / np.sqrt(np.sum(features**2, axis=1)).reshape(
            (pts_num, 1))
        return features

    @staticmethod
    def find_initial_match(feature1, feature2):
        logger.debug("find_initial_match()")
        measure_dist_mat = spd.cdist(feature1["feature"], feature2["feature"],
                                     "cosine")
        pts1, pts2 = feature1["pts"], feature2["pts"]
        pts_stack = np.vstack((pts1, pts2))
        pts_mean = np.mean(pts_stack, axis=0)
        pts_min = np.min(pts_stack, axis=0)
        pts_max = np.max(pts_stack, axis=0)
        pts_dist_mat = spd.cdist((pts1 - pts_mean) / (pts_max - pts_min),
                                 (pts2 - pts_mean) / (pts_max - pts_min),
                                 "euclidean")
        alpha = 0.00
        dist_mat = measure_dist_mat * (1 - alpha) + pts_dist_mat * alpha
        num1, num2 = dist_mat.shape

        # For a given point p1 in image1, find the most similar point p12 in image2,
        # then find the point p21 in image1 that most similar to p12, check the
        # distance between p1 and p21.

        idx12 = np.argsort(dist_mat, axis=1)
        idx21 = np.argsort(dist_mat, axis=0)
        ind = idx21[0, idx12[:, 0]] == range(num1)

        # Check Euclidean distance between the nearest pair
        d_th = min(np.percentile(dist_mat[range(num1), idx12[:, 0]], 30),
                   np.percentile(dist_mat[idx21[0, :],
                                          range(num2)], 30))
        ind = np.logical_and(ind, dist_mat[range(num1), idx12[:, 0]] < d_th)

        pair_idx = np.stack((np.where(ind)[0], idx12[ind, 0]), axis=-1)

        # Check angular distance between the nearest pair
        xyz1 = np.stack(
            (np.cos(feature1["sph"][:, 1]) * np.cos(feature1["sph"][:, 0]),
             np.cos(feature1["sph"][:, 1]) * np.sin(feature1["sph"][:, 0]),
             np.sin(feature1["sph"][:, 1])),
            axis=-1)
        xyz2 = np.stack(
            (np.cos(feature2["sph"][:, 1]) * np.cos(feature2["sph"][:, 0]),
             np.cos(feature2["sph"][:, 1]) * np.sin(feature2["sph"][:, 0]),
             np.sin(feature2["sph"][:, 1])),
            axis=-1)
        theta = np.arccos(
            np.sum(xyz1[pair_idx[:, 0]] * xyz2[pair_idx[:, 1]], axis=1))
        theta_th = min(np.percentile(theta, 75), np.pi / 6)

        pts_dist = la.norm(feature1["pts"][pair_idx[:, 0]] -
                           feature2["pts"][pair_idx[:, 1]],
                           axis=1)
        dist_th = max(np.max(feature1["pts"]), np.max(feature2["pts"])) * 0.3
        pair_idx = pair_idx[np.logical_and(theta < theta_th, pts_dist
                                           < dist_th)]

        logger.debug(f"Found {len(pair_idx)} initial pairs.")
        return pair_idx

    @staticmethod
    def fine_tune_transform(feature1, feature2, init_pair_idx):
        ind = []
        k = 1
        while len(ind) < 0.6 * min(len(feature1["pts"]), len(
                feature2["pts"])) and k <= 10:
            if k == 10:
                raise ValueError("Optimal alignment cannot be achieved.")
            # Step 1. Randomly choose 20 points evenly distributed on the image
            rand_pts = np.random.rand(20, 2) * (np.amax(feature1["pts"], axis=0) - np.amin(feature1["pts"], axis=0)) * \
                       np.array([1, 0.8]) + np.amin(feature1["pts"], axis=0)
            # Step 2. Find nearest points from feature1
            dist_mat = spd.cdist(rand_pts, feature1["pts"][init_pair_idx[:,
                                                                         0]])
            tmp_ind = np.argmin(dist_mat, axis=1)
            # Step 3. Use these points to find a homography
            tf = cv2.findHomography(feature1["pts"][init_pair_idx[tmp_ind, 0]],
                                    feature2["pts"][init_pair_idx[tmp_ind, 1]],
                                    method=cv2.RANSAC,
                                    ransacReprojThreshold=5)

            # Then use the transform find more matched points
            pts12 = cv2.perspectiveTransform(
                np.array([[p] for p in feature1["pts"]], dtype="float32"),
                tf[0])[:, 0, :]
            dist_mat = spd.cdist(pts12, feature2["pts"])
            num1, num2 = dist_mat.shape

            idx12 = np.argsort(dist_mat, axis=1)
            tmp_ind = np.argwhere(
                np.array([dist_mat[i, idx12[i, 0]] for i in range(num1)]) < 5)
            if len(tmp_ind) > len(ind):
                ind = tmp_ind
            logger.debug("len(ind) = %d, len(feature) = %d", len(ind),
                         min(len(feature1["pts"]), len(feature2["pts"])))
            k += 1

        pair_idx = np.hstack((ind, idx12[ind, 0]))

        tf = cv2.findHomography(feature1["pts"][pair_idx[:, 0]],
                                feature2["pts"][pair_idx[:, 1]],
                                method=cv2.RANSAC,
                                ransacReprojThreshold=5)
        return tf, pair_idx

    @staticmethod
    def convert_to_float(np_image: np.ndarray) -> np.ndarray:
        """将图像转换为浮点数[0,1]格式。

        Args:
            np_image (np.ndarray): input image
        Returns:
            _type_: _description_
        """
        if np_image.dtype == np.float32 or np_image.dtype == np.float64:
            return np.copy(np_image)
        else:
            return np_image.astype("float32") / np.iinfo(np_image.dtype).max
