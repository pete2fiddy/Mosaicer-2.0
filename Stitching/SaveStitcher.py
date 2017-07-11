import os
import cv2
import numpy as np
from PIL import Image
import Stitching.ImageStitcher as ImageStitcher
import Blending.ImageBlender as ImageBlender
import ImageOp.Brightness.GammaAdjuster as GammaAdjuster
import timeit
import ImageOp.Resize as Resize
from Stitching.StitchType import StitchType
class SaveStitcher(StitchType):
    '''
    stitches saved images as well as their transformations to create a full image
    '''
    def __init__(self, image_path, image_extension, transformations_path, transform_type):
        self.image_path = image_path
        self.image_extension = image_extension
        self.transformations_path = transformations_path
        self.transform_type = transform_type
        self.init_transformation_paths()
        self.init_image_paths()

    def init_transformation_paths(self):
        self.transformation_names = os.listdir(self.transformations_path)
        transformation_index = 0
        while transformation_index < len(self.transformation_names):
            if ".npy" not in self.transformation_names[transformation_index]:
                del self.transformation_names[transformation_index]
            else:
                transformation_index += 1

        self.transformation_names.sort(key = lambda name : int(name[:name.index(".npy")]))
        self.transformation_paths = tuple([self.transformations_path + self.transformation_names[i] for i in range(0, len(self.transformation_names))])
        print("self.transformation_paths: ", len(self.transformation_paths))

    def init_image_paths(self):
        image_names = os.listdir(self.image_path)
        image_names_without_extension = [image_names[i][:image_names[i].index(self.image_extension)] for i in range(0, len(image_names))]
        image_index = 0
        transform_image_names = []
        for i in range(0, len(self.transformation_names)):
            transformation_name_without_extension = self.transformation_names[i][:len(self.transformation_names[i]) - len(".npy")]
            index_in_image_names = image_names_without_extension.index(transformation_name_without_extension)
            transform_image_names.append(image_names[index_in_image_names])

        self.image_paths = tuple([self.image_path + transform_image_names[i] for i in range(0, len(transform_image_names))])
        '''
        there are one fewer transformation files than there are images. However, in order
        for it to be simple to load as many pictures as the transformations allow, the last
        transformation (which will just be an empty, identity placeholder) is removed
        '''
        self.transformation_names = self.transformation_names[:len(self.transformation_names)-1]
        self.transformation_paths = self.transformation_paths[:len(self.transformation_names)]
        print("self.image_paths: ", len(self.image_paths))


    def blend(self, blend_func_and_params, show_creation_image_size = None):
        trans_mats = []
        for i in range(0, len(self.transformation_paths)):
            trans_mats.append(np.load(self.transformation_paths[i]))
        image_shapes = []
        for i in range(0, len(self.image_paths)):
            image_shapes.append(cv2.imread(self.image_paths[i]).shape)
        mosaic_shape, bounded_trans_image_bboxes = self.get_mosaic_image_shape_and_bounded_trans_image_bboxes(self.transform_type, trans_mats, image_shapes)
        mosaic_image = np.zeros(mosaic_shape, dtype = np.uint8)

        for i in range(1, len(self.transformation_paths)):
            print("------------------------")
            print("stitching image: ", i)
            trans_mat = trans_mats[i-1]
            fit_image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
            fit_stitch = self.get_fit_stitch(mosaic_image, fit_image, self.transform_type, trans_mat, bounded_trans_image_bboxes[i]) if i == 1 else self.get_gamma_adjusted_fit_stitch(mosaic_image, fit_image, self.transform_type, trans_mat, bounded_trans_image_bboxes[i])
            mosaic_image = np.uint8(blend_func_and_params.class_type(mosaic_image, fit_stitch, blend_func_and_params))
            if show_creation_image_size is not None:
                cv2.imshow("Mosaic Image", Resize.resize_image_to_constraints(cv2.cvtColor(mosaic_image, cv2.COLOR_RGB2BGR), show_creation_image_size))
                cv2.waitKey(1)
        return np.uint8(mosaic_image)
