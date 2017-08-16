from Feature.Matching.ORBMatch import ORBMatch
from Feature.Matching.FeatureMatches import FeatureMatches
from Transform.AffineSolve import AffineSolve
from Toolbox.NamedArgs import NamedArgs
from Toolbox.ClassArgs import ClassArgs
from Feature.Matching.Reduction.RANSAC import RANSAC
import cv2
from PIL import Image
import os
import Stitching.ImageStitcher as ImageStitcher
import Blending.ImageBlender as ImageBlender
from Mosaic.MultiMosaicer import MultiMosaicer
from Stitching.SaveStitcher import SaveStitcher
from Mosaic.VideoMosaicer import VideoMosaicer
from Stitching.VideoStitcher import VideoStitcher
import timeit
from MetaData.LogExtractor import LogExtractor
import numpy as np
from random import random
from matplotlib import pyplot as plt
from Test.AffineRegressTest import AffineRegressTest
from math import sin, cos, atan2, pi
from GPS.GPS import GPS
import Angle.AngleConverter as AngleConverter
from GPS.GEOFitter import GEOFitter
from Reconstruct.StitchReconstructor import StitchReconstructor
from Reconstruct.DistanceTransformReconstructor import DistanceTransformReconstructor
from Feature.Matching.WindowedPCAReduction import WindowedPCAReduction
from Feature.Matching.FeatureMatch import FeatureMatch
from Regression.LinearLOESS import LinearLOESS
#from MotionTrack.OpticalFlow import MultiResLucasKanade2
from MotionTrack.OpticalFlow.HornSchunck import TwoFrameHornSchunck2
#from MotionTrack.OpticalFlow.TVL1Flow import TwoFrameTVL1Three
from MotionTrack.OpticalFlow.TVL1Flow3 import TVL1Flow
import MotionTrack.OpticalFlow.FlowSmooth as FlowSmooth
import MotionTrack.OpticalFlow.FlowHelper as FlowHelper
from MotionTrack.OpticalFlow.SlidingWindowOpticalFlow import SlidingWindowOpticalFlow
import ImageOp.Denoising.SmoothMapper as SmoothMapper
'''
To Do:
    Fix: Gamma Adjustment seems to drift off course as mosaic gets bigger (especially when
    camera rotates)
'''

start_time = timeit.default_timer()

frame_start_num = 4500
frame_skip = 10
num_frames = 10

mosaic_images_path = "E:/Big DZYNE Files/Mosaicing Data/Mosaic Video/XTEK0025 frames/"#"E:/Big DZYNE Files/Mosaicing Data/Naplate West/"#
save_path = "E:/Big DZYNE Files/Mosaicing Data/Saved Mosaics/GEO Video Mosaics/Video Mosaic 5.png"
transformation_save_path = "E:/Big DZYNE Files/Mosaicing Data/Saved Mosaics/Saved Transformations/" + str(frame_start_num) + "-" + "20" + "-" + "5" + "/"
start_time = timeit.default_timer()

#multi_mosaic_params = NamedArgs(image_path = mosaic_images_path, save_path = transformation_save_path, start_index = 2000, num_images = 200, image_step = 10, image_extension = ".png")
video_mosaic_params = NamedArgs(video_path = "E:/Big DZYNE Files/Mosaicing Data/Mosaic Video/XTEK0025.ts", video_frame_rate = 25, start_time_seconds = 200.0, mosaic_run_time = 60.0, seconds_between_frames = 15.0/25.0)
ransac_params = NamedArgs(inlier_distance = 5, ransac_confidence = .99999999, inlier_proportion = .2, num_inlier_breakpoint = 300)
align_type_and_params = ClassArgs(ORBMatch)

#video_mosaicer = VideoMosaicer(align_type_and_params, AffineSolve, ransac_params, video_mosaic_params)
#video_mosaicer.set_trans_mats()
#print("video trans mats: ", video_mosaicer.trans_mats)
#print("video trans mats shape: ", len(video_mosaicer.trans_mats))
#multi_mosaicer = MultiMosaicer(align_type_and_params, AffineSolve, ransac_params, multi_mosaic_params)
#multi_mosaicer.save_mosaic_transformations()

transform_run_time = timeit.default_timer() - start_time


log_extractor = LogExtractor("E:/Big DZYNE Files/Mosaicing Data/Mosaic Video/XTEK0025_Log.txt", frame_start_num, num_frames, frame_skip)#4500, 4500 + 10 * 10, 10)
frame_geos = log_extractor.frame_geos

'''
new_frame_geos = []
for i in range(0, len(frame_geos)):
    new_frame_geos.append(frame_geos[i])
    new_frame_geos.append(frame_geos[i])
frame_geos = new_frame_geos
'''

print("Frame geos: ", frame_geos)
#video_stitcher = VideoStitcher(video_mosaicer)


save_stitcher = SaveStitcher(mosaic_images_path, ".png", transformation_save_path, AffineSolve)
#mosaic_image = save_stitcher.blend(ClassArgs(ImageBlender.feather_blend, window_size = 21), show_creation_image_size = (1920, 820))
#mosaic_midpoints = save_stitcher.get_midpoints()

'''
temp_mosaic_midpoints = []
for i in range(0, mosaic_midpoints.shape[0], 2):
    temp_mosaic_midpoints.append(mosaic_midpoints[i])
mosaic_midpoints = np.asarray(temp_mosaic_midpoints)
'''

'''print("Len frame geos: ", len(frame_geos))
for i in range(0, mosaic_midpoints.shape[0]):
    cv2.putText(mosaic_image, str(frame_geos[i]), tuple(mosaic_midpoints[i].astype(np.int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
'''
#print("mosaic midpoints: ", mosaic_midpoints)

print("total time elapsed: ", timeit.default_timer() - start_time)
print("transform run time: ", transform_run_time)
print("stitching run time: ", timeit.default_timer() - start_time - transform_run_time)

'''
Can't just do flow with magnitude of flows for reconstruction -- must be baseline rectified first
'''

#geo_fitter = GEOFitter(mosaic_image, mosaic_midpoints, frame_geos)
#mosaic_ppm = geo_fitter.ppm


frames_path = "C:/Users/Peter/Desktop/DZYNE/Git Repos/Mosaicer 2.0/Mosaicer 2.0/Test/Flow Data Sequences/Backyard/"
image_names = os.listdir(frames_path)
resize_factor = 1.0
bw_frames = []
for i in range(0, len(image_names)):
    frame = cv2.imread(frames_path + image_names[i])
    bw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resize_dims = tuple((np.array(bw_frame.shape[::-1]) * resize_factor).astype(np.int))
    resized_bw_frame = cv2.resize(bw_frame, resize_dims)
    bw_frames.append(resized_bw_frame)
bw_frames = np.array(bw_frames, dtype = np.float32)
window_size = 7
bw_frames = bw_frames[:window_size, :, :]
flow_class_and_params = ClassArgs(TVL1Flow, flow_smooth_func = FlowSmooth.median_blur, flow_smooth_args = NamedArgs(k_size = 3, num_iter = 1), smooth_weight = 0.5, max_iter_per_warp = 20, theta = 0.3, time_step = .001, pyr_scale_factor = 0.5, num_scales = 5, num_warps = 10, convergence_thresh = 0.000001)
sliding_window_op_flow = SlidingWindowOpticalFlow(bw_frames, window_size, flow_class_and_params)
frame_flows = sliding_window_op_flow.frame_flows

for i in range(0, 1):
    Image.fromarray(FlowHelper.calc_flow_angle_image(frame_flows[i])).show()



'''
tvl1 = TVL1Flow(bw_base_image, bw_fit_image, NamedArgs(flow_smooth_func = FlowSmooth.median_blur, flow_smooth_args = NamedArgs(k_size = 3, num_iter = 1), smooth_weight = 0.5, max_iter_per_warp = 20, theta = 0.3, time_step = .001, pyr_scale_factor = 0.5, num_scales = 5, num_warps = 10, convergence_thresh = 0.000000000001))#TwoFrameTVL1Three(bw_base_image, bw_fit_image)
Image.fromarray(FlowHelper.calc_flow_angle_image(tvl1.flows)).show()
'''



'''
def image_to_function(image):
    X_out = []
    y_out = []
    for x_image in range(0, image.shape[1]):
        for y_image in range(0, image.shape[0]):
            if image[y_image, x_image] != 0:
                X_out.append(np.array([x_image, y_image]))
                y_out.append(image[y_image, x_image])
    X_out = np.array(X_out)
    y_out = np.array(y_out)
    return X_out, y_out

image_func_X, image_func_y = image_to_function(base_disparity_image)
loess_window = 5
lin_loess = LinearLOESS(image_func_X, image_func_y, loess_window)

predict_disparity_image = np.zeros(base_disparity_image.shape)
for x in range(0, predict_disparity_image.shape[1]):
    for y in range(0, predict_disparity_image.shape[0]):
        x_point = np.array([x,y])
        disparity_predict = lin_loess.predict(x_point)
        #print("disparity predict: ", disparity_predict)
        predict_disparity_image[y,x] = disparity_predict
    #print("On x: ", x)
print("min predict disparity: ", np.amin(predict_disparity_image))
print("max predict disparity: ", np.amax(predict_disparity_image))
Image.fromarray(np.uint8(255 * (predict_disparity_image - 0)/(np.amax(predict_disparity_image - 0)))).show()
base_disparity_image = predict_disparity_image


base_depth_image = 1.0/(base_disparity_image + 1.0)
base_depth_image[base_disparity_image == 0] == 0

sqrt_base_depth_image = np.sqrt(base_depth_image)

#Image.fromarray(np.uint8(255 * base_disparity_image/np.amax(base_disparity_image))).show()
#Image.fromarray(np.uint8(255 * base_depth_image/np.amax(base_depth_image))).show()
#Image.fromarray(np.uint8(255 * sqrt_base_depth_image/np.amax(sqrt_base_depth_image))).show()
'''
