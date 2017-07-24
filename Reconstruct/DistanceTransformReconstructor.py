import ImageOp.Segmentation.MeanShift as MeanShift
import ImageOp.Segmentation.KMeansHelper as KMeansHelper
import cv2
import numpy as np
from PIL import Image
import ImageOp.Transform.DistanceTransform as DistanceTransform
import timeit
from Regression.LinearRegression import LinearRegression


class DistanceTransformReconstructor:
    '''need to add in gamma adjustment normalization'''
    def __init__(self, base_image, fit_image, ppm, focal_length_meters):
        self.base_image = base_image
        self.fit_image = fit_image
        self.ppm = ppm
        self.focal_length_meters = focal_length_meters
        self.init_depth_map()

    def init_depth_map(self):
        MEAN_SHIFT_DIAMETER = 20.0#5.0#11.0
        MEAN_SHIFT_MAX_COLOR_DIST = 20.0#20.0
        MEAN_SHIFT_NUM_CLUSTERS = 30


        GAUSSIAN_BLUR_STD_DEV = 2.0
        GAUSSIAN_BLUR_WINDOW = (7,7)
        blur_base_image = cv2.GaussianBlur(self.base_image, GAUSSIAN_BLUR_WINDOW, GAUSSIAN_BLUR_STD_DEV)
        blur_fit_image = cv2.GaussianBlur(self.fit_image, GAUSSIAN_BLUR_WINDOW, GAUSSIAN_BLUR_STD_DEV)
        segmented_fit_image, fit_image_cluster_centers = MeanShift.cluster_mean_shift(blur_fit_image, MEAN_SHIFT_DIAMETER, MEAN_SHIFT_MAX_COLOR_DIST, MEAN_SHIFT_NUM_CLUSTERS)


        mean_shifted_base_image = cv2.pyrMeanShiftFiltering(blur_base_image, MEAN_SHIFT_DIAMETER, MEAN_SHIFT_MAX_COLOR_DIST)#may need num clusters???
        segmented_base_image = np.uint8(KMeansHelper.round_image_to_clusters(mean_shifted_base_image, fit_image_cluster_centers))

        #Image.fromarray(segmented_base_image).show()
        #Image.fromarray(segmented_fit_image).show()

        fit_masks = KMeansHelper.get_cluster_masks(segmented_fit_image, fit_image_cluster_centers).astype(np.uint8)
        base_masks = KMeansHelper.get_cluster_masks(segmented_base_image, fit_image_cluster_centers).astype(np.uint8)

        start_time = timeit.default_timer()
        fit_masks_distance_transforms = np.zeros(fit_masks.shape)
        fit_masks_contours = []
        base_masks_contours = []
        fit_masks_centroids = []
        base_masks_centroids = []
        for i in range(0, fit_masks.shape[0]):
            #fit_masks_distance_transforms[i] = cv2.distanceTransform(255-cv2.Canny(fit_masks[i], 1, 1, 3), cv2.DIST_L2, 5)
            fit_masks_distance_transforms[i] = DistanceTransform.distance_transform_across_vector(255-cv2.Canny(fit_masks[i], 1, 1, 3), np.array([1.0, 0.0]))#self.vertical_distance_transform(255-cv2.Canny(fit_masks[i], 1, 1, 3))
            #Image.fromarray(np.uint8(255*fit_masks_distance_transforms[i]/np.amax(fit_masks_distance_transforms))).show()
            base_mask_contour = cv2.findContours(base_masks[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
            fit_mask_contour = cv2.findContours(fit_masks[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
            append_base_mask_centroid = []
            for j in range(0, len(base_mask_contour)):
                base_mask_contour[j] = base_mask_contour[j][:,0,:]
                append_base_mask_centroid.append(np.average(base_mask_contour[j], axis = 0))
            base_masks_centroids.append(append_base_mask_centroid)

            append_fit_mask_centroid = []
            for j in range(0, len(fit_mask_contour)):
                fit_mask_contour[j] = fit_mask_contour[j][:,0,:]
                append_fit_mask_centroid.append(np.average(fit_mask_contour[j], axis = 0))
            fit_masks_centroids.append(append_fit_mask_centroid)


            fit_masks_contours.append(fit_mask_contour)
            base_masks_contours.append(base_mask_contour)


        base_mean_centroid = np.zeros((2))
        fit_mean_centroid = np.zeros((2))
        for i in range(0, len(base_masks_centroids)):
            base_mean_centroid += np.average(base_masks_centroids[i], axis = 0)
            fit_mean_centroid += np.average(fit_masks_centroids[i], axis = 0)
        base_mean_centroid /= float(len(base_masks_centroids))
        fit_mean_centroid /= float(len(fit_masks_centroids))

        translation_direction = (fit_mean_centroid - base_mean_centroid)/np.linalg.norm(fit_mean_centroid - base_mean_centroid)

        '''for testing'''
        translation_direction = np.array([1, 0])
        print("translation direction: ", translation_direction)


        disparity_image = np.zeros(self.fit_image.shape[:2], dtype = np.float32)
        DRAWCONTOUR_DISPARITY_UPSCALE_FACTOR = 100000000000.0


        '''TO DO:
        HAVE TO WRITE A DISTANCE TRANSFORM THAT ONLY GIVES NEAREST PIXEL DISTANCE
        ACROSS THE TRANSLATION VECTOR. REPLACE THIS WITH NORMAL DISTANCE TRANSFORM'''

        for i in range(0, fit_masks_distance_transforms.shape[0]):
            dist_transform_at_i = fit_masks_distance_transforms[i]
            for j in range(0, len(base_masks_contours[i])):
                dist_transform_vals_on_contour = dist_transform_at_i[base_masks_contours[i][j][:, 1], base_masks_contours[i][j][:, 0]]
                contour_disparity_val = np.average(dist_transform_vals_on_contour)
                MIN_INTERP_CONTOUR_SIZE = 20
                if base_masks_contours[i][j].shape[0] > MIN_INTERP_CONTOUR_SIZE:
                    try:
                        lin_regress = LinearRegression(base_masks_contours[i][j], dist_transform_vals_on_contour)
                        lin_regress.train()


                        #print("contour: ", base_masks_contours[i][j])

                        '''should use some better statistics than just taking the mean'''


                        #disparity_image = cv2.drawContours(disparity_image, base_masks_contours[i], j, int(DRAWCONTOUR_DISPARITY_UPSCALE_FACTOR * contour_disparity_val), thickness = -1)

                        contour_disparity_mask = cv2.drawContours(np.zeros(disparity_image.shape), base_masks_contours[i], j, 1, thickness = -1)

                        contour_fill_points = np.where(contour_disparity_mask != 0)
                        #print("first contour fill points: ", contour_fill_points)
                        contour_fill_points = np.dstack((contour_fill_points[1], contour_fill_points[0]))[0]
                        #print("contour fill points: ",contour_fill_points)
                        #print("contour fill points shape: ", contour_fill_points.shape)

                        fill_predictions = lin_regress.predict_set(contour_fill_points)
                        fill_predictions_avg = np.average(fill_predictions)
                        print("avg contour val: ", contour_disparity_val)
                        print("avg fill prediction: ", fill_predictions_avg)
                        print("min fill prediction: ", np.amin(fill_predictions))
                        print("min dist_transform val: ", np.amin(dist_transform_vals_on_contour))
                        for k in range(0, contour_fill_points.shape[0]):
                            if fill_predictions[k] > 0:
                                disparity_image[contour_fill_points[k][1], contour_fill_points[k][0]] = fill_predictions[k] * DRAWCONTOUR_DISPARITY_UPSCALE_FACTOR

                        #for k in range(0, dist_transform_vals_on_contour.shape[0]):
                        #    disparity_image[base_masks_contours[i][j][k, 1], base_masks_contours[i][j][k, 0]] = dist_transform_vals_on_contour[k] * DRAWCONTOUR_DISPARITY_UPSCALE_FACTOR

                        '''min_disparity_on_contour = np.amin(dist_transform_vals_on_contour)
                        max_disparity_on_contour = np.amax(dist_transform_vals_on_contour)
                        filled_contour_mask = cv2.drawContours(np.zeros(disparity_image.shape), base_masks_contours[i], j, 1, thickness = -1)

                        contour_fill_indices = np.where(filled_contour_mask != 0)
                        '''
                    except:
                        print("Likely crashed due to singular matrix error")
                        disparity_image = cv2.drawContours(disparity_image, base_masks_contours[i], j, int(DRAWCONTOUR_DISPARITY_UPSCALE_FACTOR * contour_disparity_val), thickness = -1)
                else:
                    disparity_image = cv2.drawContours(disparity_image, base_masks_contours[i], j, int(DRAWCONTOUR_DISPARITY_UPSCALE_FACTOR * contour_disparity_val), thickness = -1)

            print("On cluster", i)
        disparity_image /= DRAWCONTOUR_DISPARITY_UPSCALE_FACTOR


        #disparity_image *= self.ppm
        #Image.fromarray(np.uint8(255*disparity_image/np.amax(disparity_image))).show()

        baseline_meters = 1.0
        depth_image = (baseline_meters * self.focal_length_meters)/(disparity_image + 1)
        depth_image[disparity_image == 0] = 0
        #print("max of depth image: ", np.amax(depth_image))

        view_distributed_depth_image = np.uint8(255*np.sqrt(depth_image)/np.amax(np.sqrt(depth_image)))

        canny_debug_map = np.dstack((view_distributed_depth_image, np.zeros(disparity_image.shape), np.zeros(disparity_image.shape)))
        canny_debug_map = np.uint8(255*canny_debug_map/np.amax(canny_debug_map))
        fit_canny = cv2.Canny(segmented_fit_image, 1, 1, 3)
        base_canny = cv2.Canny(segmented_base_image, 1, 1, 3)

        #Image.fromarray(canny_debug_map).show()
        canny_debug_map[:,:,2][fit_canny != 0] = 255
        #Image.fromarray(canny_debug_map).show()
        print("time elapsed: ", timeit.default_timer() - start_time)
        Image.fromarray(view_distributed_depth_image).show()
