import numpy as np
import random
'''
made to hold FeatureMatch instances and provide functionality
for getting informationf rom multiple FeatureMatch instances
'''
class FeatureMatches:
    def __init__(self, matches):
        self.matches = matches
        '''
        special methods should be set up in a way that allows
        the indexes of base_xys and fit_xys to be moved with the
        indexes of matches. Same with deletions'''
        self.init_as_points()

    def init_as_points(self):
        self.base_xys = np.zeros((len(self), 2), dtype = np.float32)
        self.fit_xys = np.zeros((len(self), 2), dtype = np.float32)
        for i in range(0, len(self)):
            self.base_xys[i] = self[i].base_xy
            self.fit_xys[i] = self[i].fit_xy

    '''returns a FeatureMatches object created from this object's feature
    matches. The list is comprised of the feature matches at each index in
    sample_indexes'''
    def sub_sample(self, sample_indexes):
        sample = [self[sample_indexes[i]] for i in range(0, len(sample_indexes))]
        return FeatureMatches(sample)

    '''returns a randomly selected sample of feature matches in this object
    of size sample_size.'''
    def random_sample(self, sample_size):
        random_sample = FeatureMatches(random.sample(self.matches, sample_size))
        return random_sample

    def as_points(self):
        return self.base_xys, self.fit_xys

    def __getitem__(self, index):
        return self.matches[index]

    def __len__(self):
        return len(self.matches)

    def __repr__(self):
        out_str = ""
        for i in range(0, len(self)):
            out_str += str(self[i]) + "\n"
        return out_str
