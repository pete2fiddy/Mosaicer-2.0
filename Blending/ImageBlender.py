import cv2
import numpy as np

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


'''
blend_params arguments:
    ["window_size"]: the size of the blur window
'''
def feather_blend(base_stitch, fit_stitch, blend_params):
    window_size = blend_params["window_size"]
    '''creates two threshold images that are white wherever the image is present'''
    base_thresh_image = cv2.cvtColor(base_stitch, cv2.COLOR_RGB2GRAY)
    fit_thresh_image = cv2.cvtColor(fit_stitch, cv2.COLOR_RGB2GRAY)
    base_thresh_image[base_thresh_image > 0] = 255.0
    fit_thresh_image[fit_thresh_image > 0] = 255.0

    window = (window_size, window_size)
    '''
    holds the response of the fit image to the feather blend. Each pixel
    represents how much weight the "vote" from the fit image has in the
    color chosen for feathering.

    A similar image is created for the base image
    '''
    fit_response_image = cv2.blur(fit_thresh_image, window)
    '''blurring the threshold image will just extend the bounds where the
    threshold image is white. What the below does is shrink the threshold
    image so that, when blurred once more, the edge of the blurred portions
    will align with the original threshold image'''
    fit_response_image[fit_response_image < 254] = 0
    fit_response_image = cv2.blur(fit_response_image, window)
    base_response_image = 255.0 - fit_response_image

    '''normalizes the image for a wheighted sum (the response between)
    the two images at any given pixel will always sum to 1 (if at least one image
    is present at that pixel), or 0 (if neither image has a colored pixel at that
    point)
    '''
    fit_response_image = np.float32(fit_response_image/255.0)
    '''
    The below presents "fuzzy edges" where non-intersecting edges would still
    be feather-blend weighted. However, the below occurs as a result:

    near corners, you can sometimes see an overlap point because of the
    thresholding process below.
    '''
    fit_response_image[base_thresh_image == 0] = 1.0
    base_response_image = np.float32(base_response_image/255.0)

    '''
    performs a weighted addition where, at each pixel, the responses of
    each image at each pixel are multiplied the color at that pixel and summed.
    '''
    weighted_fit_image = fit_response_image[:,:,np.newaxis] * fit_stitch
    weighted_base_image = base_response_image[:,:,np.newaxis] * base_stitch
    out_image = np.uint8(weighted_fit_image + weighted_base_image)
    return out_image
