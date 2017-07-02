from abc import ABC, abstractmethod


'''Abstract class for different types of feature extraction and matching
classes to extend. Allows new feature extraction methods to be easy to
create and usable by all classes that employ them'''
class MatchType(ABC):
    '''
    Arguments:
        base_image: the image to which the fit_image must be fit

        fit_image: the image that will be transformed to fit the base

        params: not required (can be None), dictates special parameters
        for the feature extraction algorithm. Is a NamedArgs class. Will
        not be accessed by the MatchType abstract class, but is init'd
        here so that it is accessible by anything that "asks" for it.
    '''
    def __init__(self, base_image, fit_image, extract_params):
        self.base_image = base_image
        self.fit_image = fit_image
        self.extract_params = extract_params
        self.feature_matches = self.match_features()

    '''
    instantiates a list of "FeatureMatch" objects using whatever
    feature extraction algorithm the MatchType employs
    '''
    @abstractmethod
    def match_features(self):
        pass
