from Feature.Matching.ORBMatch import ORBMatch
from Feature.Matching.FeatureMatches import FeatureMatches
from Transform.AffineSolve import AffineSolve
from Toolbox.NamedArgs import NamedArgs
from Toolbox.ClassArgs import ClassArgs
from Feature.Matching.Reduction.RANSAC import RANSAC
import cv2
from PIL import Image
import Stitching.ImageStitcher as ImageStitcher
import Blending.ImageBlender as ImageBlender
from Mosaic.MultiMosaicer import MultiMosaicer
from Stitching.SaveStitcher import SaveStitcher
from Mosaic.VideoMosaicer import VideoMosaicer
from Stitching.VideoStitcher import VideoStitcher
import timeit
from MetaData.LogExtractor import LogExtractor


'''
To Do:
    Fix: Gamma Adjustment seems to drift off course as mosaic gets bigger (especially when
    camera rotates)
'''


start_time = timeit.default_timer()
mosaic_images_path = "E:/Big DZYNE Files/Mosaicing Data/Mosaic Video/XTEK0025 frames/"#"E:/Big DZYNE Files/Mosaicing Data/Naplate West/"#
save_path = "E:/Big DZYNE Files/Mosaicing Data/Saved Mosaics/GEO Video Mosaics/Video Mosaic 4.png"
transformation_save_path = "E:/Big DZYNE Files/Mosaicing Data/Saved Mosaics/Saved Transformations/2000-200-10/"
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


log_extractor = LogExtractor("E:/Big DZYNE Files/Mosaicing Data/Mosaic Video/XTEK0025_Log.txt", 2000, 2000 + 200*10, 10)

#video_stitcher = VideoStitcher(video_mosaicer)


#save_stitcher = SaveStitcher(mosaic_images_path, ".png", transformation_save_path, AffineSolve)
#mosaic_image = save_stitcher.blend(ClassArgs(ImageBlender.feather_blend, window_size = 21), show_creation_image_size = (1920, 820))


print("total time elapsed: ", timeit.default_timer() - start_time)
print("transform run time: ", transform_run_time)
print("stitching run time: ", timeit.default_timer() - start_time - transform_run_time)
#Image.fromarray(mosaic_image).show()

#cv2.imwrite(save_path, mosaic_image)
