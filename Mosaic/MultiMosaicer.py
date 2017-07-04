import os
import cv2
from Feature.Matching.Reduction.RANSAC import RANSAC
import numpy as np

class MultiMosaicer:

    def __init__(self, match_type_and_params, align_solve_type, ransac_params, mosaic_params):
        self.init_params(mosaic_params)
        self.match_type = match_type_and_params.class_type
        self.match_params = match_type_and_params
        self.align_solve_type = align_solve_type
        self.ransac_params = ransac_params
        self.init_mosaic_image_paths()
        self.init_save_dir()

    '''
    Possible mosaic params:
        ["start_index"]: the index of the image from which the mosaicing process begins
        ["num_images"]: the number of images from which the mosaic is comprised
        ["image_step"]: the number of images between two given images that are
                             skipped (e.g. every fourth image is mosaiced)
        ["image_path"]: the path to the images that are mosaiced
        ["image_extension"]: the image suffix (e.g. ".png")
        ["save_path"]: the path to save the images
    '''
    def init_params(self, mosaic_params):
        self.image_path = mosaic_params["image_path"]
        self.start_index = mosaic_params["start_index"]
        self.num_images = mosaic_params["num_images"]
        self.image_step = mosaic_params["image_step"]
        self.image_extension = mosaic_params["image_extension"]
        self.save_path = mosaic_params["save_path"]

    def init_mosaic_image_paths(self):
        self.image_names = os.listdir(self.image_path)
        image_index = 0
        while image_index < len(self.image_names):
            if self.image_extension not in self.image_names[image_index]:
                del self.image_names[image_index]
            else:
                image_index += 1

        self.image_names.sort(key = lambda name : int(name[:name.index(self.image_extension)]))
        end_index = self.start_index + self.image_step * self.num_images
        self.image_names = tuple(self.image_names[self.start_index: end_index: self.image_step])
        self.image_paths = tuple([self.image_path + self.image_names[i] for i in range(0, len(self.image_names))])

    def load_image_at_index(self, index):
        return cv2.imread(self.image_paths[index])

    def init_save_dir(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    '''takes every consecutive set of images and creates the transformation to fit one image
    to the other. In order to create a full mosaic, it keeps tracking of the running previous
    matrix through compounded matrix multiplication. The transformation for the image to fit
    the other image is multiplied by the compounded previous transformations'''
    def save_mosaic_transformations(self):
        first_trans_matrix = self.get_transformation_at_index(0)
        self.save_transformation_matrix_at_index(first_trans_matrix, 0)
        '''keeps track of previous transformations. The only transformation that is solved for is
        one that transforms a consecutive image to fit the base. However, the base itself will
        be transformed to fit the remainder of the mosaic. By dotting the matrix that transforms
        the fit image to the base with the compounded dot products of all previous transformations,
        it will fit the fit image to the rest of the mosaic rather than just an untransformed base.
        '''
        previous_transformation_compound = first_trans_matrix.copy()
        for i in range(1, len(self.image_paths)-1):
            print("---------------------------------")
            print("On image: ", i)
            trans_at_i = self.get_transformation_at_index(i)
            trans_at_i = previous_transformation_compound.dot(trans_at_i)
            '''Just dotting the compounds may not work for homographies. It also may be possible
            that dividing by W can be done all at the end rather than at each step, so that saving
            un-normalized transformations would not matter and can be rectified during the mosaic
            rebuild process'''
            self.save_transformation_matrix_at_index(trans_at_i, i)
            previous_transformation_compound = trans_at_i
        identity_mat = np.identity(previous_transformation_compound.shape[0])
        self.save_transformation_matrix_at_index(identity_mat, len(self.image_paths)-1)


    def get_transformation_at_index(self, base_image_index):
        base_image = self.load_image_at_index(base_image_index)
        fit_image = self.load_image_at_index(base_image_index + 1)
        print("base image path: ", self.image_paths[base_image_index])
        print("fit image path: ", self.image_paths[base_image_index + 1])
        match_type_instance = self.match_type(base_image, fit_image, self.match_params)
        feature_matches = match_type_instance.feature_matches
        ransac = RANSAC(feature_matches, self.align_solve_type, self.ransac_params)
        fit_align_solve = ransac.fit()
        fit_align_mat = fit_align_solve.align_mat
        return fit_align_mat

    def save_transformation_matrix_at_index(self, trans_matrix, index):
        save_name = self.image_names[index][:self.image_names[index].index(self.image_extension)]
        save_path = self.save_path + save_name
        np.save(save_path, trans_matrix)
