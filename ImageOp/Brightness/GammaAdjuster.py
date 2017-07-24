import cv2
import numpy as np
from math import log
from PIL import Image
import timeit
import ImageOp.Crop as Crop
import ImageOp.Paste as Paste

def gamma_correct_fit_stitch_to_base(base_stitch, fit_stitch):
    bw_base_stitch = cv2.cvtColor(base_stitch, cv2.COLOR_RGB2GRAY)
    bw_fit_stitch = cv2.cvtColor(fit_stitch, cv2.COLOR_RGB2GRAY)
    thresh_base_stitch = cv2.threshold(bw_base_stitch, 0, 255, cv2.THRESH_BINARY)[1]
    thresh_fit_stitch = cv2.threshold(bw_fit_stitch, 0, 255, cv2.THRESH_BINARY)[1]
    stitch_union = np.uint8(255*np.logical_and(thresh_base_stitch.astype(np.bool), thresh_fit_stitch.astype(np.bool)).astype(np.int))

    stitch_union_contours = cv2.findContours(stitch_union, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
    merged_stitch_union_contour = np.zeros((0,1,2), dtype = np.int)
    for i in range(0, len(stitch_union_contours)):
        merged_stitch_union_contour = np.concatenate((merged_stitch_union_contour, stitch_union_contours[i]), axis = 0)
    merged_stitch_union_contour = merged_stitch_union_contour[:,0,:]

    '''takes the average non-black gray value for both images'''
    avg_base_grayval = np.average(bw_base_stitch[merged_stitch_union_contour[:, 1], merged_stitch_union_contour[:, 0]])
    avg_fit_grayval = np.average(bw_fit_stitch[merged_stitch_union_contour[:, 1], merged_stitch_union_contour[:, 0]])
    gamma_adjust = log(avg_base_grayval, avg_fit_grayval)

    thresh_fit_stitch_contours = cv2.findContours(thresh_fit_stitch, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
    merged_fit_contour = np.zeros((0,1,2), dtype = np.int)
    for i in range(0, len(thresh_fit_stitch_contours)):
        merged_fit_contour = np.concatenate((merged_fit_contour, thresh_fit_stitch_contours[i]), axis = 0)

    fit_stitch_bbox = np.asarray(cv2.boundingRect(merged_fit_contour))
    fit_stitch_crop = Crop.crop_image_to_bbox(fit_stitch, fit_stitch_bbox)
    gamma_adjusted_fit_stitch_crop = (255.0*(fit_stitch_crop.astype(np.float32)/255.0)**gamma_adjust).astype(np.uint8)
    gamma_adjusted_fit_stitch = np.zeros(fit_stitch.shape, dtype = np.uint8)
    gamma_adjusted_fit_stitch = Paste.paste_image_onto_image_at_bbox(gamma_adjusted_fit_stitch, gamma_adjusted_fit_stitch_crop, fit_stitch_bbox)
    return gamma_adjusted_fit_stitch
