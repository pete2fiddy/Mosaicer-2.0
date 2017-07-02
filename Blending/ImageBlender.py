import cv2
import numpy as np

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
