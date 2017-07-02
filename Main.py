from Feature.Matching.ORBMatch import ORBMatch
from Feature.Matching.FeatureMatches import FeatureMatches
from Transform.AffineSolve import AffineSolve
from Toolbox.NamedArgs import NamedArgs
from Feature.Matching.Reduction.RANSAC import RANSAC
import cv2
from PIL import Image
import Stitching.ImageStitcher as ImageStitcher
import Blending.ImageBlender as ImageBlender
import timeit


mosaic_images_path = "E:/Big DZYNE Files/Mosaicing Data/Mosaic Video/XTEK0025 frames/"#"E:/Big DZYNE Files/Mosaicing Data/Naplate West/"#
base_image = cv2.resize(cv2.imread(mosaic_images_path + "4497.png"), (1280, 720))
fit_image = cv2.resize(cv2.imread(mosaic_images_path + "4502.png"), (1280, 720))
start_time = timeit.default_timer()
orb_match = ORBMatch(base_image, fit_image)
feature_matches = orb_match.feature_matches

ransac = RANSAC(feature_matches, AffineSolve, NamedArgs(inlier_distance = 5, ransac_confidence = .99999999, inlier_proportion = .2, num_inlier_breakpoint = 100))
best_align_solve = ransac.fit()
print("time elapsed: ", timeit.default_timer() - start_time)
cv_affine_mat = best_align_solve.align_mat[:2, :]

transformed_image, fit_shift = best_align_solve.transform_image(fit_image)
base_stitch, fit_stitch = ImageStitcher.stitch_image(base_image, transformed_image, fit_shift)
blended_image = ImageBlender.paste_blend(base_stitch, fit_stitch)
Image.fromarray(blended_image).show()
