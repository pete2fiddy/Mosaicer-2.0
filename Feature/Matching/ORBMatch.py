from Feature.MatchType import MatchType
import cv2
import numpy as np
from Feature.Matching.FeatureMatch import FeatureMatch
from Feature.Matching.FeatureMatches import FeatureMatches
from PIL import Image

'''
uses OpenCV's ORB feature extraction and description method to
initialize a set of feature matches
'''
class ORBMatch(MatchType):

    def __init__(self, base_image, fit_image, extract_params = None):
        MatchType.__init__(self, base_image, fit_image, extract_params)

    def match_features(self):
        base_keypoints, base_descriptors, fit_keypoints, fit_descriptors = self.get_features_and_descriptors()
        '''
        parameters for brute force matcher were recommended for use with
        ORB
        '''
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        keypoint_matches = bf_matcher.match(fit_descriptors, base_descriptors)

        #matches_image = cv2.drawMatches(self.base_image, base_keypoints, self.fit_image, fit_keypoints, keypoint_matches, (2 * self.base_image.shape[0], self.base_image.shape[1]))
        #Image.fromarray(matches_image).show()
        return FeatureMatch.cv_matches_to_feature_matches(keypoint_matches, base_keypoints, fit_keypoints)

    def get_features_and_descriptors(self):
        orb = cv2.ORB_create()
        base_keypoints, base_descriptors = orb.detectAndCompute(self.base_image, mask = None)
        fit_keypoints, fit_descriptors = orb.detectAndCompute(self.fit_image, mask = None)
        return base_keypoints, base_descriptors, fit_keypoints, fit_descriptors
