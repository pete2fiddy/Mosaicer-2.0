import numpy as np
import cv2

'''returns two images that, when laid on top of each other, create a correct
mosaic. (must be blended, etc.)'''
def stitch_image(base_image, trans_fit_image, fit_shift):
    fit_shift = fit_shift
    stitch_image_size = get_stitch_image_size(trans_fit_image, base_image, fit_shift)
    print("stitch image size: ", stitch_image_size)
    fit_stitch = np.zeros((stitch_image_size[0], stitch_image_size[1], base_image.shape[2]))
    base_stitch = fit_stitch.copy()

    '''logic here will not always work, sometimes the fit image will have to be moved'''
    base_xy, fit_xy = get_stitch_image_corners(fit_shift)
    print("base xy: ", base_xy)
    print("fit xy: ", fit_xy)
    base_stitch[base_xy[0]:base_xy[0] + base_image.shape[0], base_xy[1]:base_xy[1] + base_image.shape[1]] = base_image
    fit_stitch[fit_xy[0]:fit_xy[0] + trans_fit_image.shape[0], fit_xy[1]:fit_xy[1] + trans_fit_image.shape[1]] = trans_fit_image
    return np.uint8(base_stitch), np.uint8(fit_stitch)


def get_stitch_image_corners(fit_shift):
    '''
    for each element of fit shift, either:
    base image is translated by its negative or
    fit image is translated by its positive
    '''
    fit_x = 0 if fit_shift[0] < 0 else fit_shift[0]
    fit_y = 0 if fit_shift[1] < 0 else fit_shift[1]
    base_x = 0 if fit_shift[0] > 0 else abs(fit_shift[0])
    base_y = 0 if fit_shift[1] > 0 else abs(fit_shift[1])
    fit_xy = np.array([fit_x, fit_y])
    base_xy = np.array([base_x, base_y])
    return base_xy, fit_xy

def get_stitch_image_size(trans_fit_image, base_image, fit_shift):
    trans_fit_image_corners = np.array([np.array([0,0]), np.array([trans_fit_image.shape[0], 0]), np.array([trans_fit_image.shape[0], trans_fit_image.shape[1]]), np.array([0, trans_fit_image.shape[1]])])
    base_image_corners = np.array([np.array([0,0]), np.array([base_image.shape[0], 0]), np.array([base_image.shape[0], base_image.shape[1]]), np.array([0, base_image.shape[1]])])
    trans_fit_image_corners += fit_shift
    all_image_corners = np.concatenate((trans_fit_image_corners, base_image_corners)).astype(np.int)
    all_corners_bbox = cv2.boundingRect(all_image_corners)
    return all_corners_bbox[2:4]
