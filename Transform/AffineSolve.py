from Transform.AlignSolve import AlignSolve
import cv2
import numpy as np

class AffineSolve(AlignSolve):
    def __init__(self, match_subset, feature_matches, align_mat = None):
        AlignSolve.__init__(self, match_subset, feature_matches, align_mat)



    def solve_mat(self):
        base_xys_subset, fit_xys_subset = self.match_subset.as_points()
        '''OpenCV's getAffineTransform method is likely identical to the normal
        solution presented here:
        http://www.cs.cornell.edu/courses/cs4670/2016sp/lectures/lec16_alignment_web.pdf
        (but likely faster)
        Does have a maximum of three points for solving, however. If more precision
        is required, use the link above.
        '''
        cv_affine_mat = cv2.getAffineTransform(fit_xys_subset, base_xys_subset)
        align_mat = np.zeros((3,3))
        align_mat[0:2][0:3] = cv_affine_mat
        align_mat[2,2] = 1.0
        return align_mat

    def transform_image(self, image):
        transformed_image_bbox = cv2.boundingRect(self.get_transformed_image_corners(image))
        transformed_image_bounds = transformed_image_bbox[2:4]
        cv_affine_align_mat = self.align_mat[:2, :].copy()
        '''fit_shift is reversed because of the flipped X and Y of numpy images.
        (makes more sense to rectify) this here than to save it then do so in
        the stitching process.'''
        cv_affine_align_mat[0,2] -= transformed_image_bbox[0]
        cv_affine_align_mat[1,2] -= transformed_image_bbox[1]
        transformed_image = cv2.warpAffine(image, cv_affine_align_mat, transformed_image_bounds, flags = cv2.INTER_LINEAR)
        return transformed_image

    def transform_points(self, points):
        points_to_trans = np.concatenate((points, np.ones((points.shape[0], 1), dtype = np.float)), axis = 1)
        transformed_points = points_to_trans.dot(self.align_mat.T)
        transformed_points = transformed_points[:, :2]
        return transformed_points



    def MATCH_SUBSET_SIZE():
        return 3
