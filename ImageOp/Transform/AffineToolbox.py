import cv2
import numpy as np
import ImageOp.Crop as Crop

def bounded_cv_affine_transform_image(image, cv_affine_mat, crop_to_bounds = True):
    affine_mat = np.identity(3)
    affine_mat[:2, :3] = cv_affine_mat
    image_corners = np.array([[0,0,1],
                              [image.shape[1],0,1],
                              [image.shape[1],image.shape[0],1],
                              [0,image.shape[0],1]])
    trans_image_corners = (image_corners.dot(affine_mat.T)[:, :2])
    trans_image_corners = trans_image_corners[:, np.newaxis , :].astype(np.int)
    trans_corners_bounding_rect = cv2.boundingRect(trans_image_corners)
    cv_affine_mat = cv_affine_mat.copy()
    cv_affine_mat[:2, 2] -= trans_corners_bounding_rect[:2]
    trans_image_size = trans_corners_bounding_rect[2:4]
    affined_image = cv2.warpAffine(image, cv_affine_mat, trans_image_size)
    if crop_to_bounds:
         affined_image = Crop.crop_to_bounds(affined_image)
    return affined_image
