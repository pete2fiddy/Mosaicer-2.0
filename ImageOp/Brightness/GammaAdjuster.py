import cv2
import numpy as np
from math import log
from PIL import Image
import timeit

def gamma_correct_fit_stitch_to_base(base_stitch, fit_stitch):
    '''needs to be cleaned up'''
    '''takes about .4 seconds to do the below:'''
    bw_base_stitch = cv2.cvtColor(base_stitch, cv2.COLOR_RGB2GRAY)
    bw_fit_stitch = cv2.cvtColor(fit_stitch, cv2.COLOR_RGB2GRAY)
    thresh_base_stitch = np.zeros(bw_base_stitch.shape)
    thresh_fit_stitch = np.zeros(bw_fit_stitch.shape)

    thresh_base_stitch = cv2.threshold(bw_base_stitch, 0, 255, cv2.THRESH_BINARY)[1]
    thresh_fit_stitch = cv2.threshold(bw_fit_stitch, 0, 255, cv2.THRESH_BINARY)[1]

    fit_stitch_contour = cv2.findContours(thresh_fit_stitch.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
    fit_stitch_contour.sort(key = lambda contour: len(contour), reverse = True)
    fit_stitch_contour = fit_stitch_contour[0]

    thresh_base_stitch = thresh_base_stitch.astype(np.bool)
    thresh_fit_stitch = thresh_fit_stitch.astype(np.bool)

    '''stitch_union isa thresholded image that is white where the two stitches
    overlap'''
    '''runs a little slow but likely not the issue'''
    stitch_union = np.uint8(255*np.logical_and(thresh_base_stitch, thresh_fit_stitch).astype(np.int))\

    union_contours = cv2.findContours(stitch_union, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]

    '''runs fast'''
    merged_union_contour = np.zeros((0, 1, 2), dtype = np.int)
    #print("single contour shape: ", union_contours[0].shape)
    for i in range(0, len(union_contours)):
        merged_union_contour = np.concatenate((merged_union_contour, union_contours[i]), axis = 0)

    stitch_union_bbox = cv2.boundingRect(merged_union_contour)
    #print("merged union contour: ", merged_union_contour)
    #print("merged union contour shape: ", merged_union_contour.shape)
    #print("union contours: ", union_contours)
    #print("union contours shape: ", len(union_contours))
    '''runs fast'''
    old_fit_stitch = fit_stitch.copy()
    old_base_stitch = base_stitch.copy()


    '''runs fast'''
    fit_stitch = fit_stitch[stitch_union_bbox[1]: stitch_union_bbox[1] + stitch_union_bbox[3], stitch_union_bbox[0] : stitch_union_bbox[0] + stitch_union_bbox[2]]
    base_stitch = base_stitch[stitch_union_bbox[1]: stitch_union_bbox[1] + stitch_union_bbox[3], stitch_union_bbox[0] : stitch_union_bbox[0] + stitch_union_bbox[2]]
    stitch_union = stitch_union[stitch_union_bbox[1]: stitch_union_bbox[1] + stitch_union_bbox[3], stitch_union_bbox[0] : stitch_union_bbox[0] + stitch_union_bbox[2]]
    bw_base_stitch = bw_base_stitch[stitch_union_bbox[1]: stitch_union_bbox[1] + stitch_union_bbox[3], stitch_union_bbox[0] : stitch_union_bbox[0] + stitch_union_bbox[2]]
    bw_fit_stitch = bw_fit_stitch[stitch_union_bbox[1]: stitch_union_bbox[1] + stitch_union_bbox[3], stitch_union_bbox[0] : stitch_union_bbox[0] + stitch_union_bbox[2]]


    '''the below two images hold the parts of the fit and base image that intersect
    each other, with the rest of the image removed. They are converted to grayscale'''
    fit_intersection_image = bw_fit_stitch#cv2.cvtColor(fit_stitch, cv2.COLOR_RGB2GRAY)
    base_intersection_image = bw_base_stitch#cv2.cvtColor(base_stitch, cv2.COLOR_RGB2GRAY)
    fit_intersection_image[stitch_union < 1] = 0
    base_intersection_image[stitch_union < 1] = 0
    fit_intersection_image = fit_intersection_image.astype(np.float32)/255.0
    base_intersection_image = base_intersection_image.astype(np.float32)/255.0

    '''both take the mean intensity of all non-black pixels'''
    '''runs fast'''

    mean_fit_intersection_color = np.average(fit_intersection_image, axis = (0,1)) * (fit_intersection_image.shape[0] * fit_intersection_image.shape[1])/float(merged_union_contour.shape[0])
    mean_base_intersection_color = np.average(base_intersection_image, axis = (0,1)) * (base_intersection_image.shape[0] * base_intersection_image.shape[1])/float(merged_union_contour.shape[0])

    gamma_adjust = log(mean_base_intersection_color, mean_fit_intersection_color)
    #out_image = (255*(old_fit_stitch.astype(np.float32)/255.0)**gamma_adjust).astype(np.uint8)

    '''won't work if somehow there is more than one contour in the thresholded image. (Would
    only occur as a result of bad thresholding, doubt it will ever happen)'''

    fit_stitch_bbox = cv2.boundingRect(fit_stitch_contour)
    paste_fit_image = old_fit_stitch[fit_stitch_bbox[1] : fit_stitch_bbox[1] + fit_stitch_bbox[3], fit_stitch_bbox[0] : fit_stitch_bbox[0] + fit_stitch_bbox[2]]
    paste_fit_image =  (255*(paste_fit_image.astype(np.float32)/255.0)**gamma_adjust).astype(np.uint8)
    out_image = np.zeros(old_fit_stitch.shape, dtype = np.uint8)
    out_image[fit_stitch_bbox[1] : fit_stitch_bbox[1] + fit_stitch_bbox[3], fit_stitch_bbox[0] : fit_stitch_bbox[0] + fit_stitch_bbox[2]] = paste_fit_image

    return out_image
