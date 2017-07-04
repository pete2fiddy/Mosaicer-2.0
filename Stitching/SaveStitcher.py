import os
import cv2
import numpy as np
from PIL import Image
import Stitching.ImageStitcher as ImageStitcher
import Blending.ImageBlender as ImageBlender
import ImageOp.Brightness.GammaAdjuster as GammaAdjuster

class SaveStitcher:
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


    def blend(self, blend_func_and_params):
        trans_corners = self.get_image_corners()
        for i in range(1, trans_corners.shape[0]):
            trans_mat = np.load(self.transformation_paths[i-1])
            align_solve = self.transform_type.init_with_align_mat(trans_mat)
            trans_corners[i] = align_solve.transform_points(trans_corners[i])
        flattened_trans_corners = trans_corners.reshape((trans_corners.shape[0] * 4, 2))
        trans_corners_bbox = cv2.boundingRect(flattened_trans_corners)
        mosaic_image = np.zeros((trans_corners_bbox[3], trans_corners_bbox[2], 3), dtype = np.uint8)
        bounded_trans_corners = trans_corners - np.asarray(trans_corners_bbox[:2])

        bounded_trans_image_bboxes = []
        for i in range(0, bounded_trans_corners.shape[0]):
            bounded_trans_image_bboxes.append(cv2.boundingRect(bounded_trans_corners[i]))

        for i in range(1, len(self.transformation_paths)):
            print("------------------------")
            print("stitching image: ", i)
            trans_mat = np.load(self.transformation_paths[i-1])
            fit_image = cv2.imread(self.image_paths[i])



            align_solve = self.transform_type.init_with_align_mat(trans_mat)
            trans_fit_image= align_solve.transform_image(fit_image)
            fit_bbox_xy = bounded_trans_image_bboxes[i][:2][::-1]
            fit_stitch = np.zeros((mosaic_image.shape), dtype = np.uint8)
            fit_stitch[fit_bbox_xy[0] : fit_bbox_xy[0] + trans_fit_image.shape[0], fit_bbox_xy[1] : fit_bbox_xy[1] + trans_fit_image.shape[1]] = trans_fit_image
            if i != 1:
                '''gamma adjust the fit stitch so that its brightness matches the rest of the mosaic'''
                fit_stitch= GammaAdjuster.gamma_correct_fit_stitch_to_base(mosaic_image, fit_stitch)
            mosaic_image = np.uint8(blend_func_and_params.class_type(mosaic_image, fit_stitch, blend_func_and_params))#ImageBlender.paste_blend(np.uint8(mosaic_image),  np.uint8(fit_stitch))

        return np.uint8(mosaic_image)


        '''
        trans_origins = [np.array([0,0])]
        corners = self.get_image_corners()
        trans_corners = []
        for i in range(0, len(self.transformation_paths)):
            trans_mat = np.load(self.transformation_paths[i])
            align_solve = self.transform_type.init_with_align_mat(trans_mat)
            append_origin = align_solve.transform_points(np.array([np.array([0,0])]))[0]
            append_trans_corners = align_solve.transform_points(corners[i])
            trans_origins.append(append_origin)
            trans_corners.append(append_trans_corners)
        trans_origins = np.asarray(trans_origins).astype(np.int)
        print("origins: ", trans_origins)
        trans_corners = np.asarray(trans_corners)
        trans_flattened_corners = trans_corners.reshape((len(self.transformation_paths) * 4, 2)).astype(np.int)
        #corners_bbox = cv2.boundingRect(trans_flattened_corners)
        print("flattened corners: ", trans_flattened_corners)
        print("corners shape: ", corners.shape)
        #print("corners bbox: ", corners_bbox)
        #trans_origins -= np.array(corners_bbox[:2])

        bounded_origins = np.zeros(trans_origins.shape)
        for i in range(0, bounded_origins.shape[0]):
            print("corners sublist: ", trans_flattened_corners[:4*i])
            corners_bbox = cv2.boundingRect(trans_flattened_corners[:4*i])
            print("corners bbox: ", corners_bbox)
            bounded_origins[i] = trans_origins[i] - np.array(corners_bbox[:2])

        bounded_origins = bounded_origins.astype(np.int)

        print("bounded origins: ", bounded_origins)



        current_mosaic = cv2.imread(self.image_paths[0])
        for i in range(0, len(self.transformation_paths)):
            base_image = current_mosaic
            fit_image = cv2.imread(self.image_paths[i+1])
            trans_mat = np.load(self.transformation_paths[i])

            align_solve = self.transform_type.init_with_align_mat(trans_mat)

            trans_fit_image, fit_shift = align_solve.transform_image(fit_image)
            #Image.fromarray(trans_fit_image).show()
            fit_shift += bounded_origins[i]
            #fit_shift = np.array([-200,0])

            base_stitch, fit_stitch = ImageStitcher.stitch_image(base_image, trans_fit_image, fit_shift)
            current_mosaic = ImageBlender.paste_blend(base_stitch, fit_stitch)
        Image.fromarray(current_mosaic).show()
        '''

    def get_image_corners(self):
        corners = []
        for i in range(0, len(self.image_paths)):
            image_shape = cv2.imread(self.image_paths[i]).shape
            append_corners = np.array([np.array([0,0]), np.array([image_shape[1], 0]), np.array([image_shape[1], image_shape[0]]), np.array([0, image_shape[0]])])
            corners.append(append_corners)
        return np.asarray(corners)
