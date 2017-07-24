from abc import ABC, abstractmethod
import numpy as np
import ImageOp.Paste as Paste
import cv2
import ImageOp.Brightness.GammaAdjuster as GammaAdjuster
'''Is an abstract class that all stitch classes extend. Provides basic stitch
functionality so that any stitch class can "blend" images through whatever
means or input they take'''

class StitchType(ABC):
    '''
    Over-generality about the function of a stitch class makes this class clunky to use.
    As more stitch classes are added (or if stitch classes are rarely added or not at all),
    question how much can be offloaded to this class (such as saving the transformations,
    trans_image_corners bbox, etc.) so the class functions don't have to take so many
    convoluted arguments
    '''

    '''
    all StitchType must have a blend method that takes the blend type and
    the display size of the mosaic (if the user wishes to watch it being
    constructed)
    '''
    @abstractmethod
    def blend(self, blend_func_and_params, show_creation_image_size = None):
        pass

    def get_fit_stitch(self, mosaic_image, fit_image, align_solve_type, trans_mat, bounded_trans_image_bbox):
        align_solve = align_solve_type.init_with_align_mat(trans_mat)
        trans_fit_image = align_solve.transform_image(fit_image)
        fit_bbox_xy = bounded_trans_image_bbox[:2][::-1]
        fit_stitch = np.zeros((mosaic_image.shape), dtype = np.uint8)
        fit_stitch_bbox = np.array([fit_bbox_xy[1], fit_bbox_xy[0], trans_fit_image.shape[1], trans_fit_image.shape[0]])
        fit_stitch = Paste.paste_image_onto_image_at_bbox(fit_stitch, trans_fit_image, fit_stitch_bbox)
        return fit_stitch

    def get_gamma_adjusted_fit_stitch(self, mosaic_image, fit_image, align_solve_type, trans_mat, bounded_trans_image_bbox):
        fit_stitch = self.get_fit_stitch(mosaic_image, fit_image, align_solve_type, trans_mat, bounded_trans_image_bbox)
        fit_stitch = GammaAdjuster.gamma_correct_fit_stitch_to_base(mosaic_image, fit_stitch)
        return fit_stitch

    '''returns the size of the mosaic image must be to fully
    contain every image from which it is constructed'''
    def get_mosaic_image_shape_and_bounded_trans_image_bboxes(self, align_solve_type, trans_mats, image_shapes):
        trans_corners = self.get_image_corners(image_shapes)
        for i in range(1, trans_corners.shape[0]):
            trans_mat = trans_mats[i-1]
            align_solve = align_solve_type.init_with_align_mat(trans_mat)
            trans_corners[i] = align_solve.transform_points(trans_corners[i])
        flattened_trans_corners = trans_corners.reshape((trans_corners.shape[0] * 4, 2))
        trans_corners_bbox = cv2.boundingRect(flattened_trans_corners)
        image_shape = (trans_corners_bbox[3], trans_corners_bbox[2], 3)
        bounded_trans_corners = trans_corners - np.asarray(trans_corners_bbox[:2])

        bounded_trans_image_bboxes = np.zeros((bounded_trans_corners.shape[0], 4))
        for i in range(0, bounded_trans_image_bboxes.shape[0]):
            bounded_trans_image_bboxes[i] = cv2.boundingRect(bounded_trans_corners[i])
        bounded_trans_image_bboxes = bounded_trans_image_bboxes.astype(np.int)
        return image_shape, bounded_trans_image_bboxes, trans_corners_bbox

    def get_image_corners(self, image_shapes):
        corners = []
        for i in range(0, len(image_shapes)):
            image_shape = image_shapes[i]
            append_corners = np.array([np.array([0,0]), np.array([image_shape[1], 0]), np.array([image_shape[1], image_shape[0]]), np.array([0, image_shape[0]])])
            corners.append(append_corners)
        return np.asarray(corners)

    '''
    for each image, determines where the center point is using image_shapes,
    and places a GPS location at that point on the image. Then, the vector at
    that point is transformed the same way
    '''
    def get_image_midpoint_locations_on_mosaic(self, trans_corners_bbox, image_shapes, align_solve_type, trans_mats):
        trans_midpoints = np.zeros((len(image_shapes), 2))
        for i in range(1, trans_midpoints.shape[0]):
            center_xy = np.array([image_shapes[i][1]//2, image_shapes[i][0]//2])
            align_solve = align_solve_type.init_with_align_mat(trans_mats[i-1])
            trans_center_xy = align_solve.transform_points(np.array([center_xy]))[0]
            trans_midpoints[i] = trans_center_xy
        trans_midpoints[0] = np.array([image_shapes[0][1], image_shapes[0][0]])/2.0
        trans_midpoints -= np.array(trans_corners_bbox[:2])
        return trans_midpoints

    '''all stitch types must be able to return their transformed image midpoints
    sorted by order of their corresponding frame's appearance in the video'''
    @abstractmethod
    def get_midpoints(self):
        pass
