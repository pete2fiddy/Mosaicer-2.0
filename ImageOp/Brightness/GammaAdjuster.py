import cv2
import numpy as np
from math import log

def gamma_correct_fit_stitch_to_base(base_stitch, fit_stitch):

    thresh_base_stitch = cv2.cvtColor(base_stitch, cv2.COLOR_RGB2GRAY)
    thresh_fit_stitch = cv2.cvtColor(fit_stitch, cv2.COLOR_RGB2GRAY)
    thresh_base_stitch[thresh_base_stitch > 0] = 1.0
    thresh_fit_stitch[thresh_fit_stitch > 0] = 1.0
    thresh_base_stitch = thresh_base_stitch.astype(np.bool)
    thresh_fit_stitch = thresh_fit_stitch.astype(np.bool)

    '''stitch_union isa thresholded image that is white where the two stitches
    overlap'''
    stitch_union = np.uint8(255*np.logical_and(thresh_base_stitch, thresh_fit_stitch).astype(np.int))

    '''the below two images hold the parts of the fit and base image that intersect
    each other, with the rest of the image removed. They are converted to grayscale'''
    fit_intersection_image = cv2.cvtColor(fit_stitch, cv2.COLOR_RGB2GRAY)
    base_intersection_image = cv2.cvtColor(base_stitch, cv2.COLOR_RGB2GRAY)
    fit_intersection_image[stitch_union < 1] = 0
    base_intersection_image[stitch_union < 1] = 0
    fit_intersection_image = fit_intersection_image.astype(np.float32)/255.0
    base_intersection_image = base_intersection_image.astype(np.float32)/255.0

    '''both take the mean intensity of all non-black pixels'''
    mean_fit_intersection_color = np.average(fit_intersection_image, axis = (0,1)) * (fit_intersection_image.shape[0] * fit_intersection_image.shape[1])/float(np.count_nonzero(stitch_union))
    mean_base_intersection_color = np.average(base_intersection_image, axis = (0,1)) * (base_intersection_image.shape[0] * base_intersection_image.shape[1])/float(np.count_nonzero(stitch_union))

    gamma_adjust = log(mean_base_intersection_color, mean_fit_intersection_color)
    return (255*(fit_stitch.astype(np.float32)/255.0)**gamma_adjust).astype(np.uint8)
