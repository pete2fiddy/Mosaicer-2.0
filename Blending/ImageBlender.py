import cv2
import numpy as np
import timeit
from PIL import Image
import ImageOp.Crop as Crop
import ImageOp.Paste as Paste

'''
issue with all image blends:
    when images are transformed, the edge pixels "spill" over into space that should be black when
    it rounds to pixels. As a result, when thresholding the image, there are some very dark pixels
    (that aren't black, but close to it) around the edges of the transformed image. These false
    black edges can appear in the final mosaic and and show where images were blended
'''

'''takes two image stitches and fits them to one image by pasting
the fit stitch over the base. Takes no NamedArgs parameters, but
the field is still provided so that the functions can be passed
and will have the same number of arguments as the other blend
types'''
def paste_blend(base_stitch, fit_stitch, blend_params = None):
    '''needs to be made more efficient with large base images'''
    thresh_fit_stitch = cv2.cvtColor(fit_stitch, cv2.COLOR_RGB2GRAY)
    thresh_base_stitch = cv2.cvtColor(base_stitch, cv2.COLOR_RGB2GRAY)

    thresh_fit_stitch[thresh_fit_stitch > 0] = 1.0
    thresh_base_stitch[thresh_base_stitch > 0] = 1.0
    '''ignore_space is an image that is white (1.0) in any place outside of the
    bounds of fit_stitch'''
    ignore_space = np.logical_not(np.logical_and(thresh_fit_stitch, thresh_base_stitch)).astype(np.float32)
    out_image = base_stitch.copy().astype(np.float32)
    out_image += ignore_space[:,:,np.newaxis] * fit_stitch
    return np.uint8(out_image)


'''
blend_params arguments:
    ["window_size"]: the size of the blur window
'''
def feather_blend(base_stitch, fit_stitch, blend_params):
    window_size = blend_params["window_size"]

    thresh_base = cv2.cvtColor(base_stitch, cv2.COLOR_RGB2GRAY)
    thresh_fit = cv2.cvtColor(fit_stitch, cv2.COLOR_RGB2GRAY)
    thresh_base = cv2.threshold(thresh_base, 0, 1, cv2.THRESH_BINARY)[1].astype(np.bool)
    thresh_fit = cv2.threshold(thresh_fit, 0, 1, cv2.THRESH_BINARY)[1].astype(np.bool)

    and_thresh = np.uint8(255*np.logical_and(thresh_base, thresh_fit).astype(np.int))
    and_contours = cv2.findContours(and_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]

    merged_and_contour = np.zeros((0, 1, 2), dtype = np.int)
    for i in range(0, len(and_contours)):
        merged_and_contour = np.concatenate((merged_and_contour, and_contours[i]), axis = 0)

    fit_contours = cv2.findContours(np.uint8(255*thresh_fit.astype(np.int)), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
    merged_fit_contour = np.zeros((0,1,2), dtype = np.int)
    for i in range(0, len(fit_contours)):
        merged_fit_contour = np.concatenate((merged_fit_contour, fit_contours[i]), axis = 0)

    fit_bbox = np.asarray(cv2.boundingRect(merged_fit_contour))
    margin_fit_bbox = fit_bbox - (window_size-1)//2
    margin_fit_bbox[2:] += 3*(window_size-1)//2
    margin_fit_bbox[margin_fit_bbox < 0] = 0
    if margin_fit_bbox[1] + margin_fit_bbox[3] > fit_stitch.shape[0]:
        margin_fit_bbox[3] = fit_stitch.shape[0] - margin_fit_bbox[1]
    if margin_fit_bbox[0] + margin_fit_bbox[2] > fit_stitch.shape[1]:
        margin_fit_bbox[2] = fit_stitch.shape[1] - margin_fit_bbox[0]

    fit_to_margin_fit_bounds_response = Crop.crop_image_to_bbox(np.uint8(255*thresh_fit.astype(np.int)), margin_fit_bbox)

    '''median blur removes some small holes that are sometimes present in the thresholded fit image'''
    fit_to_margin_fit_bounds_response = cv2.medianBlur(fit_to_margin_fit_bounds_response, 3)
    fit_to_margin_fit_bounds_response = cv2.blur(fit_to_margin_fit_bounds_response, (window_size, window_size))
    fit_to_margin_fit_bounds_response = cv2.threshold(fit_to_margin_fit_bounds_response, 254, 255, cv2.THRESH_BINARY)[1]
    fit_to_margin_fit_bounds_response = cv2.blur(fit_to_margin_fit_bounds_response, (window_size, window_size))


    fit_to_fit_bounds_response = Crop.crop_image_to_bbox(fit_to_margin_fit_bounds_response, np.array([(window_size-1)//2, (window_size-1)//2, fit_bbox[2], fit_bbox[3]]))
    and_thresh_to_fit_bounds = Crop.crop_image_to_bbox(and_thresh, fit_bbox)
    base_to_fit_bounds = Crop.crop_image_to_bbox(base_stitch, fit_bbox)

    '''removes the fuzzy edges at places where the images do not intersect.
    Sometimes removes fuzziness where it is wanted. (not good)'''
    thresh_base_to_fit_bounds = Crop.crop_image_to_bbox(thresh_base, fit_bbox)
    not_and_thresh_to_fit_bounds = np.logical_and(np.logical_not(and_thresh_to_fit_bounds.astype(np.bool)), np.logical_not(thresh_base_to_fit_bounds))
    fit_to_fit_bounds_response[not_and_thresh_to_fit_bounds] = 255

    normed_fit_to_fit_bounds_response = (fit_to_fit_bounds_response.astype(np.float32))/255.0

    out_image = base_stitch.copy()
    fit_to_fit_bounds = fit_stitch[fit_bbox[1] : fit_bbox[1] + fit_bbox[3], fit_bbox[0] : fit_bbox[0] + fit_bbox[2]]
    thresh_fit_to_fit_bounds = Crop.crop_image_to_bbox(thresh_fit, fit_bbox)


    weighted_fit_to_fit_bounds = np.uint8(normed_fit_to_fit_bounds_response[:,:,np.newaxis] * fit_to_fit_bounds)
    weighted_base_to_fit_bounds = np.uint8( (1.0 - normed_fit_to_fit_bounds_response[:,:,np.newaxis]) * base_to_fit_bounds)
    out_image = Paste.paste_image_onto_image_at_bbox(out_image, np.uint8( (weighted_fit_to_fit_bounds + weighted_base_to_fit_bounds)), fit_bbox)
    return out_image
