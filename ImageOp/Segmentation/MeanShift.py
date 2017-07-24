import cv2
import numpy as np
import ImageOp.Kernel as Kernel
from PIL import Image
import ImageOp.Segmentation.KMeansHelper as KMeansHelper

'''uses KMeans to reduce a mean shift filter to a number of clusters. Also returns the
kmeans color cluster centers'''
def cluster_mean_shift(image, radius, color_dist, num_clusters, kmeans_max_iter = 30, kmeans_downscale_factor = 0.35):
    mean_shift_image = cv2.pyrMeanShiftFiltering(image, radius, color_dist)

    downscaled_mean_shift = cv2.resize(mean_shift_image, (int(mean_shift_image.shape[1] * kmeans_downscale_factor), int(mean_shift_image.shape[0] * kmeans_downscale_factor)))

    mean_shift_colors = downscaled_mean_shift.reshape((downscaled_mean_shift.shape[0] * downscaled_mean_shift.shape[1], -1)).astype(np.float32)
    kmeans_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, kmeans_max_iter, 1.0)
    ret, labels, centers = cv2.kmeans(mean_shift_colors, num_clusters, None, kmeans_criteria, kmeans_max_iter, cv2.KMEANS_RANDOM_CENTERS)

    kmeans_rounded_image = np.uint8(KMeansHelper.round_image_to_clusters(mean_shift_image, centers))
    return kmeans_rounded_image, centers
