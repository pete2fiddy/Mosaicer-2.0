from Stitching.StitchType import StitchType
import numpy as np
import ImageOp.Resize as Resize
import cv2

class VideoStitcher(StitchType):

    def __init__(self, video_mosaicer):
        self.video_mosaicer = video_mosaicer

    def blend(self, blend_func_and_params, show_creation_image_size = None):
        mosaic_shape, bounded_trans_image_bboxes, trans_corners_bbox = self.get_mosaic_image_shape_and_bounded_trans_image_bboxes(self.video_mosaicer.align_solve_type, self.video_mosaicer.trans_mats, self.video_mosaicer.image_shapes)
        mosaic_image = np.zeros(mosaic_shape, dtype = np.uint8)

        vidcap = cv2.VideoCapture(self.video_mosaicer.video_path)
        frame_count = 0
        num_images_mosaiced = 1
        cap_success, cap_image = vidcap.read()
        base_image = None
        while cap_success:
            cap_success, cap_image = vidcap.read()
            if frame_count in self.video_mosaicer.image_frame_indexes:
                if base_image is None:
                    base_image = cv2.cvtColor(cap_image, cv2.COLOR_BGR2RGB)
                else:
                    fit_image = cv2.cvtColor(cap_image, cv2.COLOR_BGR2RGB)
                    trans_mat = self.video_mosaicer.trans_mats[num_images_mosaiced-1]
                    fit_stitch = self.get_fit_stitch(mosaic_image, fit_image, self.video_mosaicer.align_solve_type, trans_mat, bounded_trans_image_bboxes[num_images_mosaiced]) if num_images_mosaiced == 1 else self.get_gamma_adjusted_fit_stitch(mosaic_image, fit_image, self.video_mosaicer.align_solve_type, trans_mat, bounded_trans_image_bboxes[num_images_mosaiced])
                    mosaic_image = np.uint8(blend_func_and_params.class_type(mosaic_image, fit_stitch, blend_func_and_params))
                    num_images_mosaiced += 1
                    base_image = fit_image
                if show_creation_image_size is not None:
                    print("resized image shape: ", Resize.resize_image_to_constraints(cv2.cvtColor(mosaic_image, cv2.COLOR_RGB2BGR), show_creation_image_size).shape)
                    cv2.imshow("Mosaic Image", Resize.resize_image_to_constraints(cv2.cvtColor(mosaic_image, cv2.COLOR_RGB2BGR), show_creation_image_size))
                    cv2.waitKey(1)

            frame_count += 1
            if frame_count > self.video_mosaicer.end_frame:
                cap_success = False
        self.midpoints = self.get_image_midpoint_locations_on_mosaic(trans_corners_bbox, image_shapes, self.transform_type, trans_mats)
        return np.uint8(mosaic_image)

    def get_midpoints(self):
        return self.midpoints
