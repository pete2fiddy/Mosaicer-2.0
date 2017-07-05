import cv2
import numpy as np
import timeit
from PIL import Image

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


def feather_blend2(base_stitch, fit_stitch, blend_params):
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
    '''
    or_thresh = np.uint8(255*np.logical_or(thresh_base, thresh_fit).astype(np.int))
    or_contours = cv2.findContours(or_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]

    merged_or_contour = np.zeros((0,1,2), dtype = np.int)
    for i in range(0, len(or_contours)):
        merged_or_contour = np.concatenate((merged_or_contour, or_contours[i]), axis = 0)
    or_bbox = np.asarray(cv2.boundingRect(merged_or_contour))
    '''

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

    fit_to_margin_fit_bounds_response = np.uint8(255*thresh_fit.astype(np.int))[margin_fit_bbox[1] : margin_fit_bbox[1] + margin_fit_bbox[3], margin_fit_bbox[0] : margin_fit_bbox[0] + margin_fit_bbox[2]]
    '''median blur removes some small holes that are sometimes present in the thresholded fit image'''
    fit_to_margin_fit_bounds_response = cv2.medianBlur(fit_to_margin_fit_bounds_response, 3)
    fit_to_margin_fit_bounds_response = cv2.blur(fit_to_margin_fit_bounds_response, (window_size, window_size))
    fit_to_margin_fit_bounds_response = cv2.threshold(fit_to_margin_fit_bounds_response, 254, 255, cv2.THRESH_BINARY)[1]
    fit_to_margin_fit_bounds_response = cv2.blur(fit_to_margin_fit_bounds_response, (window_size, window_size))

    fit_to_fit_bounds_response = fit_to_margin_fit_bounds_response[(window_size-1)//2 : ((window_size-1)//2) + fit_bbox[3], (window_size - 1)//2 : ((window_size-1)//2) + fit_bbox[2]]
    normed_fit_to_fit_bounds_response = (fit_to_fit_bounds_response.astype(np.float32))/255.0


    '''
    and_bbox = np.asarray(cv2.boundingRect(merged_and_contour))

    print("and bbox: ", and_bbox)
    if and_bbox[2] == 0 or and_bbox[3] == 0:
        return fit_stitch

    margin_and_bbox = and_bbox.copy()
    margin_and_bbox[:2] -= (window_size-1)//2
    margin_and_bbox[2:] += (window_size-1)
    margin_and_bbox[margin_and_bbox < 0] = 0
    if margin_and_bbox[0] + margin_and_bbox[2] > fit_stitch.shape[1]:
        margin_and_bbox[2] = fit_stitch.shape[1] - margin_and_bbox[0]
    if margin_and_bbox[1] + margin_and_bbox[3] > fit_stitch.shape[0]:
        margin_and_bbox[3] = fit_stitch.shape[0] - margin_and_bbox[1]

    crop_and_thresh = and_thresh[margin_and_bbox[1] : margin_and_bbox[1] + margin_and_bbox[3], margin_and_bbox[0] : margin_and_bbox[0] + margin_and_bbox[2]]

    crop_fit_response = cv2.blur(crop_and_thresh, (window_size, window_size))

    crop_fit_response = cv2.threshold(crop_fit_response, 254, 255, cv2.THRESH_BINARY)[1]
    crop_fit_response = cv2.blur(crop_fit_response, (window_size, window_size))
    crop_fit_response = crop_fit_response.astype(np.float32)/255.0
    crop_base_response = 1.0 - crop_fit_response
    Image.fromarray(np.uint8(255*crop_fit_response)).show()
    fit_response_to_fit_bounds = np.zeros((fit_bbox[3], fit_bbox[2]))
    crop_fit_response_delta_to_fit_bounds = (and_bbox - fit_bbox)[:2]
    fit_response_to_fit_bounds[crop_fit_response_delta_to_fit_bounds[1] : crop_fit_response_delta_to_fit_bounds[1] + crop_fit_response.shape[0], crop_fit_response_delta_to_fit_bounds[0] : crop_fit_response_delta_to_fit_bounds[0] + crop_fit_response.shape[1]] = crop_fit_response
    Image.fromarray(np.uint8(255*fit_response_to_fit_bounds)).show()
    '''
    start_time = timeit.default_timer()
    out_image = base_stitch.copy()#(thresh_base.astype(np.int)[:,:,np.newaxis] * base_stitch)
    fit_to_fit_bounds = fit_stitch[fit_bbox[1] : fit_bbox[1] + fit_bbox[3], fit_bbox[0] : fit_bbox[0] + fit_bbox[2]]
    thresh_fit_to_fit_bounds = thresh_fit[fit_bbox[1] : fit_bbox[1] + fit_bbox[3], fit_bbox[0] : fit_bbox[0] + fit_bbox[2]]
    and_thresh_to_fit_bounds = and_thresh[fit_bbox[1] : fit_bbox[1] + fit_bbox[3], fit_bbox[0] : fit_bbox[0] + fit_bbox[2]]
    print("time taken to bound images: ", timeit.default_timer() - start_time)
    start_time = timeit.default_timer()
    base_to_fit_bounds = base_stitch[fit_bbox[1] : fit_bbox[1] + fit_bbox[3], fit_bbox[0] : fit_bbox[0] + fit_bbox[2]]
    weighted_fit_to_fit_bounds = np.uint8(normed_fit_to_fit_bounds_response[:,:,np.newaxis] * fit_to_fit_bounds)
    weighted_base_to_fit_bounds = np.uint8( (1.0 - normed_fit_to_fit_bounds_response[:,:,np.newaxis]) * base_to_fit_bounds)
    #Image.fromarray(np.uint8(weighted_fit_to_fit_bounds + weighted_base_to_fit_bounds)).show()
    print("time taken to weigh and bound: ", timeit.default_timer() - start_time)
    out_image[fit_bbox[1] : fit_bbox[1] + fit_bbox[3], fit_bbox[0] : fit_bbox[0] + fit_bbox[2]] = np.uint8( (weighted_fit_to_fit_bounds + weighted_base_to_fit_bounds))# += np.uint8(thresh_fit_to_fit_bounds.astype(np.int)[:,:,np.newaxis] * np.logical_not(and_thresh_to_fit_bounds.astype(np.bool)).astype(np.int)[:,:,np.newaxis] * normed_fit_to_fit_bounds_response[:,:,np.newaxis] * fit_to_fit_bounds)

    '''
    out_image = base_stitch.copy()
    normed_base_to_fit_bounds_response = 1.0-normed_fit_to_fit_bounds_response
    thresh_base_to_fit_bounds = thresh_base[fit_bbox[1] :fit_bbox[1] + fit_bbox[3], fit_bbox[0] : fit_bbox[0] + fit_bbox[2]]
    base_to_fit_bounds = base_stitch[fit_bbox[1] :fit_bbox[1] + fit_bbox[3], fit_bbox[0] : fit_bbox[0] + fit_bbox[2]]
    out_image[fit_bbox[1] : fit_bbox[1] + fit_bbox[3], fit_bbox[0] : fit_bbox[0] + fit_bbox[2]] += np.uint8(thresh_base_to_fit_bounds.astype(np.int)[:,:,np.newaxis] * np.logical_not(and_thresh_to_fit_bounds.astype(np.bool)).astype(np.int)[:,:,np.newaxis] * normed_base_to_fit_bounds_response[:,:,np.newaxis] * base_to_fit_bounds)
    '''
    return out_image


'''
blend_params arguments:
    ["window_size"]: the size of the blur window
'''
def feather_blend(base_stitch, fit_stitch, blend_params):
    window_size = blend_params["window_size"]
    '''creates two threshold images that are white wherever the image is present'''
    start_time = timeit.default_timer()
    base_thresh_image = cv2.cvtColor(base_stitch, cv2.COLOR_RGB2GRAY)
    fit_thresh_image = cv2.cvtColor(fit_stitch, cv2.COLOR_RGB2GRAY)
    base_thresh_image = cv2.threshold(base_thresh_image, 0, 255, cv2.THRESH_BINARY)[1]
    fit_thresh_image = cv2.threshold(fit_thresh_image, 0, 255, cv2.THRESH_BINARY)[1]

    #both_thresh_image = 255*np.logical_or(base_thresh_image.astype(np.bool), fit_thresh_image.astype(np.bool)).astype(np.int)
    #Image.fromarray(both_thresh_image).show()

    print("time taken to 1st threshold images: ", timeit.default_timer() - start_time)
    window = (window_size, window_size)
    '''
    holds the response of the fit image to the feather blend. Each pixel
    represents how much weight the "vote" from the fit image has in the
    color chosen for feathering.

    A similar image is created for the base image
    '''
    start_time = timeit.default_timer()
    fit_response_image = cv2.blur(fit_thresh_image, window)
    '''blurring the threshold image will just extend the bounds where the
    threshold image is white. What the below does is shrink the threshold
    image so that, when blurred once more, the edge of the blurred portions
    will align with the original threshold image'''
    fit_response_image = cv2.threshold(fit_response_image, 254, 255, cv2.THRESH_BINARY)[1]
    fit_response_image = cv2.blur(fit_response_image, window)
    base_response_image = 255.0 - fit_response_image
    print("time taken to create response images: ", timeit.default_timer() - start_time)
    '''normalizes the image for a wheighted sum (the response between)
    the two images at any given pixel will always sum to 1 (if at least one image
    is present at that pixel), or 0 (if neither image has a colored pixel at that
    point)
    '''
    start_time = timeit.default_timer()
    fit_response_image = np.float32(fit_response_image/255.0)
    '''
    The below presents "fuzzy edges" where non-intersecting edges would still
    be feather-blend weighted. However, the below occurs as a result:

    near corners, you can sometimes see an overlap point because of the
    thresholding process below.
    '''
    fit_response_image[base_thresh_image == 0] = 1.0
    base_response_image = np.float32(base_response_image/255.0)
    print("time taken to represent responses as floats: ", timeit.default_timer() - start_time)
    '''
    performs a weighted addition where, at each pixel, the responses of
    each image at each pixel are multiplied the color at that pixel and summed.
    '''
    start_time = timeit.default_timer()
    weighted_fit_image = fit_response_image[:,:,np.newaxis] * fit_stitch
    weighted_base_image = base_response_image[:,:,np.newaxis] * base_stitch
    out_image = np.uint8(weighted_fit_image + weighted_base_image)
    print("time taken to perform last part of algo: ", timeit.default_timer() - start_time)
    return out_image
