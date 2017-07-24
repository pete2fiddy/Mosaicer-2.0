import cv2
import numpy as np

def crop_image_to_bbox(image, bbox):
    return image[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]

'''takes an image with a black background and a
larger image pasted over it. Determines the mask
where the image is, then bounds based on this mask.

May want to split this function that finds the merged contour
bounding box into another class for thresholding???(or something)

'''
def crop_to_bounds(image):
    mask_image = image
    if len(image.shape) > 2:
        mask_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    mask_image = cv2.threshold(mask_image, 0, 255, cv2.THRESH_BINARY)[1]
    mask_contours = cv2.findContours(np.uint8(mask_image), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
    merged_mask_contour = np.zeros((0,1,2), dtype = np.int)
    for i in range(0, len(mask_contours)):
        merged_mask_contour = np.concatenate((merged_mask_contour, mask_contours[i]), axis = 0)
    merged_mask_bbox = cv2.boundingRect(merged_mask_contour)
    return crop_image_to_bbox(image, merged_mask_bbox)

def crop_with_margins_from_center(image, margins):
    bbox = (margins[0], margins[1], image.shape[1]-2*margins[0], image.shape[0]-2*margins[1])
    return crop_image_to_bbox(image, bbox)
