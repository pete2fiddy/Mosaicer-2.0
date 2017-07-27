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
from MotionTrack.OpticalFlow.TVL1Flow2 import TVL1Flow2

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
#Image.fromarray(mosaic_image).show()
'''
mosaic_midpoint_framenums = [frame_start_num + frame_skip*i for i in range(0, num_frames)]
temp_mosaic_midpoints = []
for i in range(0, len(mosaic_midpoint_framenums)):
    if not mosaic_midpoint_framenums[i] in log_extractor.missing_frame_nums:
        temp_mosaic_midpoints.append(mosaic_midpoints[i])

mosaic_midpoints = np.array(temp_mosaic_midpoints)
'''
'''optical flow improvements:
Check out openCV's LTV Dual (something like that)
look at stereo algorithms.
Can't just do flow with magnitude of flows for reconstruction -- must be baseline rectified first
'''

#geo_fitter = GEOFitter(mosaic_image, mosaic_midpoints, frame_geos)
#mosaic_ppm = geo_fitter.ppm

frames_base_path = "C:/Users/Peter/Desktop/DZYNE/Git Repos/Mosaicer 2.0/Mosaicer 2.0/Test/Flow Data 2/Urban3/"#"C:/Users/Peter/Desktop/DZYNE/Git Repos/Mosaicer 2.0/Mosaicer 2.0/Test/StereoReconstruction/TwoView/Middlebury/Bicycle/"
scale_factor = 1.0

image_size = cv2.imread(frames_base_path + "frame10.png").shape[:2][::-1]


base_image = cv2.imread(frames_base_path + "frame10.png")
fit_image = cv2.imread(frames_base_path + "frame11.png")

#base_image = cv2.imread("E:/Big DZYNE Files/Mosaicing Data/Mosaic Video/XTEK0025 frames/4580.png")
#fit_image = cv2.imread("E:/Big DZYNE Files/Mosaicing Data/Mosaic Video/XTEK0025 frames/4585.png")

#base_image = cv2.GaussianBlur(base_image, (7,7), 2.0)
#fit_image = cv2.GaussianBlur(fit_image, (7,7), 2.0)

base_image = cv2.resize(base_image, (int(image_size[0]*scale_factor), int(image_size[1]*scale_factor)))
fit_image = cv2.resize(fit_image, (int(image_size[0]*scale_factor), int(image_size[1]*scale_factor)))

bw_base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
bw_fit_image = cv2.cvtColor(fit_image, cv2.COLOR_BGR2GRAY)

#bw_base_image = cv2.GaussianBlur(bw_base_image, (7,7), 2.0)
#bw_fit_image = cv2.GaussianBlur(bw_fit_image, (7,7), 2.0)

#op_flow = MultiResLucasKanade2(np.array([bw_base_image, bw_fit_image]), 5, scale_factor = 2)
#Image.fromarray(fit_image).show()

image_path = "C:/Users/Peter/Desktop/DZYNE/Git Repos/Mosaicer 2.0/Mosaicer 2.0/Test/Flow Data 2/ToyData/BallBlankBackground/"#"C:/Users/Peter/Desktop/DZYNE/Git Repos/Mosaicer 2.0/Mosaicer 2.0/Test/Flow Data 2/Beanbags/"#
image_names = os.listdir(image_path)

flow_images = []
for i in range(0, len(image_names)):
    flow_images.append(cv2.cvtColor(cv2.imread(image_path + image_names[i]), cv2.COLOR_BGR2GRAY))
flow_images = np.asarray(flow_images)

start_time = timeit.default_timer()

tvl1 = TVL1Flow2(bw_base_image, bw_fit_image, smooth_weight = .05, theta = 0.3, time_step = .15)#TwoFrameTVL1Three(bw_base_image, bw_fit_image)

'''
horn_schunck = TwoFrameHornSchunck2(bw_base_image, bw_fit_image, 100.0 , num_iter = 1000)
print("time elapsed: ", timeit.default_timer() - start_time)
Image.fromarray(horn_schunck.get_flow_angle_image()).show()
Image.fromarray(horn_schunck.get_flow_vector_image(5)).show()
'''

'''
#dist_transform_reconstructor = DistanceTransformReconstructor(base_image, fit_image, 85.6, 1.0)



window_pca_reduc = WindowedPCAReduction(bw_base_image, bw_fit_image, NamedArgs(window_size = 9, window_step = 5))


num_eigenvecs = 20

base_image_projection = window_pca_reduc.project_image(bw_base_image, num_eigenvecs)
fit_image_projection = window_pca_reduc.project_image(bw_fit_image, num_eigenvecs)

base_kps, base_des = window_pca_reduc.projected_image_to_keypoints_and_descriptors(base_image_projection)
fit_kps, fit_des = window_pca_reduc.projected_image_to_keypoints_and_descriptors(fit_image_projection)

bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
matches = bf_matcher.match(base_des, fit_des)
matches.sort(key = lambda match: match.distance)
match_image = np.zeros((base_image.shape[0], base_image.shape[1] + fit_image.shape[1], 3))
match_image[:, :base_image.shape[1], :] = base_image
match_image[:, base_image.shape[1]:, :] = fit_image
num_best_matches_to_use = 500#len(matches)#60000
best_matches_subset = matches[:num_best_matches_to_use]
match_image = cv2.drawMatches(base_image, base_kps, fit_image, fit_kps, best_matches_subset, match_image, flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
Image.fromarray(match_image).show()

feature_matches = FeatureMatch.cv_matches_to_feature_matches(best_matches_subset, base_kps, fit_kps)
#print("feature matches: ", feature_matches)
print("Len feature matches: ", len(feature_matches))
print("Len kps: ", len(fit_kps))

base_disparity_image = np.zeros(bw_fit_image.shape)
print("base disparity image shape: ", base_disparity_image.shape)
base_xys, fit_xys = feature_matches.as_points()
for i in range(0, len(feature_matches)):
    dist = np.linalg.norm(base_xys[i] - fit_xys[i])
    base_pixel = base_xys[i].astype(np.int)
    base_disparity_image[base_pixel[1], base_pixel[0]] = dist
#base_disparity_image[base_disparity_image == 0] = np.average(base_disparity_image, axis = None)
print("min base disparity: ", np.amin(base_disparity_image))
print("max base disparity: ", np.amax(base_disparity_image))
Image.fromarray(np.uint8(255*base_disparity_image/np.amax(base_disparity_image))).show()

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
