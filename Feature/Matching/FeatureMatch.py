import Toolbox.CV.CVConverter as CVConverter
from Feature.Matching.FeatureMatches import FeatureMatches

class FeatureMatch:
    '''not created init method yet'''
    def __init__(self, base_xy, fit_xy):
        self.base_xy = base_xy
        self.fit_xy = fit_xy

    '''
    assumes that trainIdx matches base_features and queryIdx
    matches fit_features
    '''
    @staticmethod
    def cv_matches_to_feature_matches(cv_matches, base_features, fit_features):
        out_feature_matches = []
        for i in range(0, len(cv_matches)):
            base_feature_index = cv_matches[i].trainIdx
            fit_feature_index = cv_matches[i].queryIdx
            base_numpy_keypoint = CVConverter.keypoint_to_numpy(base_features[base_feature_index])[:2]
            fit_numpy_keypoint = CVConverter.keypoint_to_numpy(fit_features[fit_feature_index])[:2]
            append_feature_match = FeatureMatch(base_numpy_keypoint, fit_numpy_keypoint)
            out_feature_matches.append(append_feature_match)
        return FeatureMatches(out_feature_matches)

    def __repr__(self):
        return "base xy: {0}, fit xy: {1}".format(str(self.base_xy), str(self.fit_xy))
