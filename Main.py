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
import timeit


'''
To Do:
    SaveStitcher seems to run very slow. Fix speed issues.

'''



mosaic_images_path = "E:/Big DZYNE Files/Mosaicing Data/Mosaic Video/XTEK0025 frames/"#"E:/Big DZYNE Files/Mosaicing Data/Naplate West/"#
save_path = "E:/Big DZYNE Files/Mosaicing Data/Saved Mosaics/GEO Video Mosaics/2000-200-10.png"
transformation_save_path = "E:/Big DZYNE Files/Mosaicing Data/Saved Mosaics/Saved Transformations/2000-200-10/"
start_time = timeit.default_timer()

multi_mosaic_params = NamedArgs(image_path = mosaic_images_path, save_path = transformation_save_path, start_index = 2000, num_images = 200, image_step = 10, image_extension = ".png")
ransac_params = NamedArgs(inlier_distance = 5, ransac_confidence = .99999999, inlier_proportion = .2, num_inlier_breakpoint = 300)
align_type_and_params = ClassArgs(ORBMatch)
#multi_mosaicer = MultiMosaicer(align_type_and_params, AffineSolve, ransac_params, multi_mosaic_params)
#multi_mosaicer.save_mosaic_transformations()

save_stitcher = SaveStitcher(mosaic_images_path, ".png", transformation_save_path, AffineSolve)
mosaic_image = save_stitcher.blend(ClassArgs(ImageBlender.feather_blend, window_size = 21))
Image.fromarray(mosaic_image).show()
cv2.imwrite(save_path, mosaic_image)
