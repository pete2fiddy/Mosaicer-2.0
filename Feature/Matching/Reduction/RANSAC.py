from math import log

class RANSAC:
    '''
    Arguments:
        feature_matches: A FeatureMatches object
        align_solve_type: The AlignSolve class that is used to transform
        the feature matches
        inlier_params: A NamedArgs object that contains RANSAC parameters.
    '''
    def __init__(self, feature_matches, align_solve_type, ransac_params):
        self.feature_matches = feature_matches
        self.align_solve_type = align_solve_type
        self.init_inlier_params(ransac_params)
        self.init_num_iter()

    '''
    RANSAC Inlier params types:
        "inlier_distance": the maximum distance a set of transformed
        features can be away from each other for the set to be considered
        an inlier

        "ransac_confidence": the confidence that the outputted result will
        be close to ideal (all selected points are inliers). This is used
        to calculate the minimum number of times for ransac to run to acheive
        this confidence

        "inlier_proportion": the estimated proportion of inliers in the
        feature matches. Also used to calculate the minimum number of
        times for ransac to run to acheive ransac_confidence
    '''
    def init_inlier_params(self, ransac_params):
        self.inlier_distance = ransac_params["inlier_distance"]
        self.ransac_confidence = ransac_params["ransac_confidence"]
        self.inlier_proportion = ransac_params["inlier_proportion"]
        self.num_inlier_breakpoint = ransac_params["num_inlier_breakpoint"]

    '''
    initializes the number of iterations ransac needs to run in order to
    return and output of the inputted confidence.
    '''
    def init_num_iter(self):
        self.num_iter = int(log(1.0 - self.ransac_confidence)/log(1.0 - self.inlier_proportion**self.align_solve_type.MATCH_SUBSET_SIZE())) + 1
        print("RANSAC iterations is: ", self.num_iter)

    def fit(self):
        best_fit_align_solve = None
        best_fit_align_score = None
        for iter in range(0, self.num_iter):
            random_sample = self.feature_matches.random_sample(self.align_solve_type.MATCH_SUBSET_SIZE())
            iter_align_solve = self.align_solve_type(random_sample, self.feature_matches)
            iter_inlier_score = self.get_sample_inlier_score(iter_align_solve)
            if best_fit_align_score is None or iter_inlier_score > best_fit_align_score:
                print("Better inlier score found: ", iter_inlier_score)
                best_fit_align_score = iter_inlier_score
                best_fit_align_solve = iter_align_solve
                if best_fit_align_score >= self.num_inlier_breakpoint:
                    break
        return best_fit_align_solve

    '''counts the number of inliers that the inputted align_solve has when its
    transformation is applied to its feature matches'''
    def get_sample_inlier_score(self, align_solve):
        match_distances = align_solve.get_transformation_match_distances()
        inlier_score = (match_distances < self.inlier_distance).sum()
        return inlier_score
