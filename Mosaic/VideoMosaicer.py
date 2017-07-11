import cv2
from Feature.Matching.Reduction.RANSAC import RANSAC
import numpy as np
import exifread

'''takes a video and solves for mosaic transformations using
the frame of the video. Would be nice if VideoMosaicer and
MultiMosaicer had some form of shared inheritance... '''
class VideoMosaicer:

    def __init__(self, match_type_and_params, align_solve_type, ransac_params, mosaic_params):
        self.init_params(mosaic_params)
        self.match_type = match_type_and_params.class_type
        self.match_params = match_type_and_params
        self.align_solve_type = align_solve_type
        self.ransac_params = ransac_params


    '''
    Possible VideoMosaicer params:
        ["video_path"]: the path to the video
        ["video_frame_rate"]: the frame rate of the video (FPS) (can probably
        pull from metadata somehow but not sure how)
        ["start_time_seconds"]: the time stamp (in seconds) of the first image
        in the mosaic (can be a decimal)
        ["mosaic_run_time"]: the number of seconds that the mosaic encompasses
        ["seconds_between_frames"]: the number of seconds between frames (can be
        a decimal)
    '''
    def init_params(self, mosaic_params):
        self.video_path = mosaic_params["video_path"]
        self.frame_rate = mosaic_params["video_frame_rate"]
        self.start_frame = int(mosaic_params["start_time_seconds"] * self.frame_rate)
        self.frame_step = mosaic_params["seconds_between_frames"] * self.frame_rate
        self.num_frames = int(float(mosaic_params["mosaic_run_time"])/float(mosaic_params["seconds_between_frames"]))
        self.end_frame = self.start_frame + self.num_frames * self.frame_step
        print("start frame: ", self.start_frame)
        print("end frame: ", self.end_frame)
        print("num frames: ", self.num_frames)
        print("frame step: ", self.frame_step)

    def set_trans_mats(self):
        vidcap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        cap_success, cap_image = vidcap.read()
        self.trans_mats = []
        self.image_frame_indexes = []
        self.image_shapes = []
        previous_frame = None
        previous_transformation_compound = None
        while cap_success:
            cap_success, cap_image = vidcap.read()
            print("read a new frame: ", frame_count, ", success: ", cap_success)

            if frame_count > self.start_frame:

                if (frame_count - self.start_frame)%self.frame_step == 0:
                    self.image_frame_indexes.append(frame_count)
                    self.image_shapes.append(cap_image.shape)
                    if previous_frame is None:
                        previous_frame = cap_image
                    else:
                        trans_mat = self.get_trans_mat(previous_frame, cap_image)
                        if previous_transformation_compound is not None:
                            trans_mat = previous_transformation_compound.dot(trans_mat)
                        self.trans_mats.append(trans_mat)
                        previous_transformation_compound = trans_mat
                        previous_frame = cap_image
            frame_count += 1
            if frame_count > self.end_frame:
                cap_success = False

        self.trans_mats.append(np.identity(self.trans_mats[len(self.trans_mats)-1].shape[0]))


    def get_trans_mat(self, base_image, fit_image):
        match_type_instance = self.match_type(base_image, fit_image, self.match_params)
        feature_matches = match_type_instance.feature_matches
        ransac = RANSAC(feature_matches, self.align_solve_type, self.ransac_params)
        fit_align_solve = ransac.fit()
        fit_align_mat = fit_align_solve.align_mat
        return fit_align_mat
