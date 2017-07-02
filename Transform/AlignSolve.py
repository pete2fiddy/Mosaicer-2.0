from abc import ABC, abstractmethod
import numpy as np
import cv2

'''
An abstract class that Mosaicing image transformation methods
extend.
'''
class AlignSolve(ABC):
    '''
    Arguments:
        match_subset: A subset of feature matches with which the
        transformation is computed. (For eample with RANSAC, is the
        random sample)

        feature_matches: All the feature matches of the two images
    '''
    def __init__(self, match_subset, feature_matches):
        self.match_subset = match_subset
        self.feature_matches = feature_matches
        self.align_mat = self.solve_mat()

    '''
    returns the matrix that transforms the fit image to fit the
    base image
    '''
    @abstractmethod
    def solve_mat(self):
        pass

    '''
    transforms a numpy array of 2d numpy arrays that represents a
    point in an image. Different matrix transformations compute this
    differently, so it is required for each class extending AlignSolve
    '''
    @abstractmethod
    def transform_points(self, points):
        pass

    '''
    transforms the inputted image using align mat and returns the shift vector
    that informs where the transformed image must be placed relative to the base
    image.
    '''
    @abstractmethod
    def transform_image(self, image):
        pass

    '''
    returns an integer representing the match subset size required
    for the transformation to be computed
    '''
    @staticmethod
    @abstractmethod
    def MATCH_SUBSET_SIZE():
        pass

    '''returns a numpy array of scalars where each index holds the
    distance between fit_xy and base_xy after fit_xy has been
    transformed with the matrix the AlignSolve has created.

    These distances can be used to gauge the quality of mosaic that
    align_mat provides'''
    def get_transformation_match_distances(self):
        base_xys, fit_xys = self.feature_matches.as_points()
        trans_fit_xys = self.transform_points(fit_xys)
        return np.linalg.norm(trans_fit_xys - base_xys, axis = 1)

    def get_transformed_image_corners(self, image):
        image_corners = np.array([np.array([0,0]), np.array([image.shape[1], 0]), np.array([image.shape[1], image.shape[0]]), np.array([0, image.shape[0]])])
        transformed_image_corners = self.transform_points(image_corners).astype(np.int)
        return transformed_image_corners

    def get_transformed_image_bounds(self, image):
        transformed_image_corners = self.get_transformed_image_corners(image)
        cv_bbox = cv2.boundingRect(transformed_image_corners)
        return cv_bbox[2:4]
