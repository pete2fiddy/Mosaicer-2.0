import numpy as np
import cv2
import ImageOp.Crop as Crop
from PIL import Image
import ImageOp.Kernel as Kernel
import ImageOp.Segmentation.MeanShift as MeanShift
import ImageOp.Segmentation.KMeansHelper as KMeansHelper
from math import sqrt

class StitchReconstructor:

    def __init__(self, base_stitch, fit_stitch, ppm, focal_length_meters):
        self.base_stitch = base_stitch
        self.fit_stitch = fit_stitch
        self.ppm = ppm
        self.focal_length_meters = focal_length_meters
        self.init_stitch_union_mask()
        self.init_stitch_bboxes()
        self.init_baseline_meters()
        self.init_depth_map()

    def init_stitch_union_mask(self):
        bw_base_stitch = cv2.cvtColor(self.base_stitch, cv2.COLOR_RGB2GRAY)
        bw_fit_stitch = cv2.cvtColor(self.fit_stitch, cv2.COLOR_RGB2GRAY)
        self.stitch_union_mask = np.logical_and(bw_base_stitch.astype(np.bool), bw_fit_stitch.astype(np.bool)).astype(np.uint8)
        stitch_union_contours = cv2.findContours(self.stitch_union_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
        merged_stitch_union_contour = np.zeros((0,1,2), dtype = np.int)
        for i in range(0, len(stitch_union_contours)):
            merged_stitch_union_contour = np.concatenate((merged_stitch_union_contour, stitch_union_contours[i]), axis = 0)
        self.stitch_union_bbox = cv2.boundingRect(merged_stitch_union_contour)

    def init_stitch_bboxes(self):
        base_stitch_thresh = cv2.cvtColor(self.base_stitch, cv2.COLOR_RGB2GRAY).astype(np.bool).astype(np.uint8)
        base_stitch_contours = cv2.findContours(base_stitch_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
        self.merged_base_stitch_contour = np.zeros((0,1,2), dtype = np.int)
        for i in range(0, len(base_stitch_contours)):
            self.merged_base_stitch_contour = np.concatenate((self.merged_base_stitch_contour, base_stitch_contours[i]), axis = 0)
        self.base_stitch_bbox = cv2.boundingRect(self.merged_base_stitch_contour)


        fit_stitch_thresh = cv2.cvtColor(self.fit_stitch, cv2.COLOR_RGB2GRAY).astype(np.bool).astype(np.uint8)
        fit_stitch_contours = cv2.findContours(fit_stitch_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
        self.merged_fit_stitch_contour = np.zeros((0,1,2), dtype = np.int)
        for i in range(0, len(fit_stitch_contours)):
            self.merged_fit_stitch_contour = np.concatenate((self.merged_fit_stitch_contour, fit_stitch_contours[i]), axis = 0)
        self.fit_stitch_bbox = cv2.boundingRect(self.merged_fit_stitch_contour)
        '''just cropping to the stitch will still leave some parts of the fit and
        base stitch that never intersect in the image. Mask out non-intersections'''

    def init_baseline_meters(self):
        pixel_baseline_d1 = self.stitch_union_bbox[0]
        pixel_baseline_d2 = self.fit_stitch.shape[1] - (self.stitch_union_bbox[0] + self.stitch_union_bbox[2])
        self.baseline_pixels = pixel_baseline_d1 + pixel_baseline_d2
        self.baseline_meters = self.baseline_pixels/self.ppm



    def init_depth_map(self):
        '''
        IDEA:
        FORGO MOSAICING, JUST USE TWO FRAMES OF VIDEO. BECAUSE THE FRAMES ARE SO SIMILAR, THE DISTANCE MAP APPROACH
        LIKELY STILL HOLDS.


        TO DO:
        Segment out the floor (find color of floor or something... Should be easy, is very big). Determine a correct
        depth to set the segmented floor to. Make the floor constant
        '''



        self.fit_stitch_crop = Crop.crop_image_to_bbox(self.fit_stitch, self.stitch_union_bbox)
        self.base_stitch_crop = Crop.crop_image_to_bbox(self.base_stitch, self.stitch_union_bbox)
        Image.fromarray(self.fit_stitch_crop).show()
        MEAN_SHIFT_DIAMETER = 11.0
        MEAN_SHIFT_MAX_COLOR_DIST = 20.0#40.0
        MEAN_SHIFT_NUM_CLUSTERS = 20#50

        '''
        may want to flip the fit and base stitch around, for some reason it makes
        more sense to me to pick clusters off of the base image not the fit. (probably
        doesn't matter)
        '''

        segmented_fit_stitch, fit_stitch_cluster_centers = MeanShift.cluster_mean_shift(self.fit_stitch_crop, MEAN_SHIFT_DIAMETER, MEAN_SHIFT_MAX_COLOR_DIST, MEAN_SHIFT_NUM_CLUSTERS)

        mean_shifted_base_stitch = cv2.pyrMeanShiftFiltering(self.base_stitch_crop, MEAN_SHIFT_DIAMETER, MEAN_SHIFT_MAX_COLOR_DIST, MEAN_SHIFT_NUM_CLUSTERS)#shouldn't need num clusters???
        segmented_base_stitch = np.uint8(KMeansHelper.round_image_to_clusters(mean_shifted_base_stitch, fit_stitch_cluster_centers))#, base_stitch_cluster_centers = MeanShift.cluster_mean_shift(self.base_stitch_crop, MEAN_SHIFT_DIAMETER, MEAN_SHIFT_MAX_COLOR_DIST, MEAN_SHIFT_NUM_CLUSTERS)

        fit_masks = KMeansHelper.get_cluster_masks(segmented_fit_stitch, fit_stitch_cluster_centers).astype(np.uint8)
        base_masks = KMeansHelper.get_cluster_masks(segmented_base_stitch, fit_stitch_cluster_centers).astype(np.uint8)

        fit_masks_distance_transforms = np.zeros(fit_masks.shape)
        base_masks_contours = []
        for i in range(0, fit_masks.shape[0]):
            '''not sure if the distance transform should take 255-cv2.Canny(fit_masks[i],1,1,3)
            of fit_masks[i] as the image to distance transform

            (Makes more sense intuitively to use 255-cv2.Canny(fit_masks[i],1,1,3) because it
            gives distance maps inside and outside of the mask)'''
            fit_masks_distance_transforms[i] = cv2.distanceTransform(255-cv2.Canny(fit_masks[i], 1, 1, 3), cv2.DIST_L2, 5)#fit_masks[i]
            #Image.fromarray(np.uint8(255*fit_masks_distance_transforms[i]/np.amax(fit_masks_distance_transforms[i]))).show()
            base_mask_contour = cv2.findContours(base_masks[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
            for j in range(0, len(base_mask_contour)):
                base_mask_contour[j] = base_mask_contour[j][:, 0, :]
            base_masks_contours.append(base_mask_contour)

        '''try to see if I can move contours as much as possible into numpy arrays
        instead of a list of list of numpy arrays'''

        disparity_image = np.zeros(self.fit_stitch_crop.shape[:2], dtype = np.float32)
        DRAWCONTOUR_DISPARITY_UPSCALE_FACTOR = 100000000.0
        MIN_CONTOUR_SIZE = 20
        for i in range(0, fit_masks_distance_transforms.shape[0]):
            dist_transform_at_i = fit_masks_distance_transforms[i]

            for j in range(0, len(base_masks_contours[i])):
                if len(base_masks_contours[i][j]) > MIN_CONTOUR_SIZE:
                    dist_transform_vals_on_contour = dist_transform_at_i[base_masks_contours[i][j][:, 1], base_masks_contours[i][j][:, 0]]
                    '''rather than just bootstrap mean, try to use confidence and statistics
                    to pick the most likely value to be the true disparity across the mask'''
                    contour_disparity_val = self.bootstrap_mean(dist_transform_vals_on_contour)#np.average(dist_transform_vals_on_contour)
                    '''would be best to rasterize instead of filling the whole contour with one depth color. Could use the distance to near
                    points and their depth, or use the distance only to the lowest depth point and place it in a range between minimum and
                    maximum disparities in the contour using that distance'''
                    disparity_image = cv2.drawContours(disparity_image, base_masks_contours[i], j, int(DRAWCONTOUR_DISPARITY_UPSCALE_FACTOR * contour_disparity_val), thickness = -1)
        disparity_image /= DRAWCONTOUR_DISPARITY_UPSCALE_FACTOR
        #Image.fromarray(np.uint8(255*disparity_image/np.amax(disparity_image))).show()

        depth_image = (self.baseline_meters * self.focal_length_meters)/(disparity_image)
        '''looks like items touching the edge of the crop union are negatively
        affecting the contour disparities. Threshold contours so that any points
        outside of the union mask are removed. '''


        '''circles appear to have bad disparities (probably because of the way the edges line up when moved,
        many disparity values will not be the true disparity value and will be much closer)'''

        depth_image[disparity_image == 0] = 0
        Image.fromarray(np.uint8(255*np.sqrt(depth_image/np.amax(depth_image))/np.amax(sqrt(np.amax(depth_image))))).show()
        #Image.fromarray(np.uint8(255*depth_image/np.amax(depth_image))).show()


    def bootstrap_mean(self, values, sample_proportion = 0.4, num_subsets = 50):
        mean_sum = 0
        subset_size = int(values.shape[0] * sample_proportion)
        for i in range(0, num_subsets):
            subset = np.random.choice(values, size = subset_size, replace = True)
            mean_sum += np.average(subset)
        return mean_sum/float(num_subsets)


    def init_depth_map2(self):
        self.union_crop_fit_stitch = Crop.crop_image_to_bbox(self.fit_stitch, self.stitch_union_bbox)
        self.union_crop_base_stitch = Crop.crop_image_to_bbox(self.base_stitch, self.stitch_union_bbox)

        GAUSSIAN_WINDOW = (7,7)
        GAUSSIAN_STD_DEV = 2.0

        BILAT_THRESHES = (-1, 20.0, 20.0)
        #self.union_crop_fit_stitch = cv2.bilateralFilter(self.union_crop_fit_stitch, BILAT_THRESHES[0], BILAT_THRESHES[1], BILAT_THRESHES[2])
        #self.union_crop_base_stitch = cv2.bilateralFilter(self.union_crop_base_stitch, BILAT_THRESHES[0], BILAT_THRESHES[1], BILAT_THRESHES[2])
        Image.fromarray(self.union_crop_fit_stitch).show()
        Image.fromarray(self.union_crop_base_stitch).show()

        blur_union_crop_fit_stitch = cv2.GaussianBlur(cv2.cvtColor(self.union_crop_fit_stitch, cv2.COLOR_RGB2GRAY), GAUSSIAN_WINDOW, GAUSSIAN_STD_DEV)
        blur_union_crop_base_stitch = cv2.GaussianBlur(cv2.cvtColor(self.union_crop_base_stitch, cv2.COLOR_RGB2GRAY), GAUSSIAN_WINDOW, GAUSSIAN_STD_DEV)

        CANNY_THRESH = (30, 60)
        union_crop_canny_fit_stitch = cv2.Canny(blur_union_crop_fit_stitch, CANNY_THRESH[0], CANNY_THRESH[1], 3)
        union_crop_canny_base_stitch = cv2.Canny(blur_union_crop_base_stitch, CANNY_THRESH[0], CANNY_THRESH[1], 3)

        '''assumes that contours are fairly similar -- the distance transform for each point
        in the base image to its nearest edge is used everywhere that the canny of the fit image
        is white, and it assumes that an edge in the fit image will be very close to an edge in
        the base image. Thus, taking the value at the distance transform of an edge in fit image
        is likely the distance to the nearest edge in the base image'''

        base_thresh_images = []
        fit_thresh_images = []
        fit_thresh_contours = []
        unedited_fit_thresh_contours = []
        base_dist_transforms = []
        THRESH_RANGE = 35
        for lower_color_bound in range(0, 255 - THRESH_RANGE, THRESH_RANGE):
            upper_color_bound = lower_color_bound + THRESH_RANGE
            base_thresh_mask = np.zeros(blur_union_crop_base_stitch.shape[:2], dtype = np.uint8)
            base_thresh_mask[blur_union_crop_base_stitch > lower_color_bound] = 255
            base_thresh_mask[blur_union_crop_base_stitch > upper_color_bound] = 0
            base_thresh_images.append(base_thresh_mask)

            fit_thresh_mask = np.zeros(blur_union_crop_fit_stitch.shape[:2], dtype = np.uint8)
            fit_thresh_mask[blur_union_crop_fit_stitch > lower_color_bound] = 255
            fit_thresh_mask[blur_union_crop_base_stitch > upper_color_bound] = 0
            fit_thresh_images.append(fit_thresh_mask)



            iter_fit_thresh_contours = cv2.findContours(fit_thresh_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
            unedited_fit_thresh_contours.append(list(iter_fit_thresh_contours))
            for i in range(0, len(iter_fit_thresh_contours)):
                iter_fit_thresh_contours[i] = iter_fit_thresh_contours[i][:, 0, :]

            fit_thresh_contours.append(iter_fit_thresh_contours)


            base_mask_dist_transform = cv2.distanceTransform(base_thresh_mask, cv2.DIST_L2, 5)
            #Image.fromarray(np.uint8(255*base_mask_dist_transform/np.amax(base_mask_dist_transform))).show()
            base_dist_transforms.append(base_mask_dist_transform)

        disparity_map = np.zeros(blur_union_crop_base_stitch.shape[:2], dtype = np.float32)
        CONTOUR_FILLING_UPSCALE_FACTOR = 1000000.0
        for i in range(0, len(fit_thresh_contours)):
            thresh_contours = fit_thresh_contours[i]
            base_dist_transform = base_dist_transforms[i]
            for j in range(0, len(thresh_contours)):
                '''
                may need to unflip x and y in thresh contours (pretty sure findCountours returns x,y)
                '''
                avg_disparity_of_contour = np.average(base_dist_transform[thresh_contours[j][:, 1], thresh_contours[j][:, 0]])
                #avg_disparity_of_contour = np.amax(base_dist_transform[thresh_contours[j][:,1], thresh_contours[j][:,0]])
                '''could instead make each individual pixel a gradient between highest and lowest disparity,
                probably either:
                1) That pixel's distance to all pixels in the contour and the disparity values of those pixels in the contour or
                2) The pixel's distance only to the highest and lowest disparity pixels in the contour, and weight the result that
                way'''
                print("avg disparity of contour: ", avg_disparity_of_contour)
                cv2.drawContours(disparity_map, thresh_contours, j, int(CONTOUR_FILLING_UPSCALE_FACTOR * avg_disparity_of_contour), thickness = -1)


        disparity_map /= CONTOUR_FILLING_UPSCALE_FACTOR
        Image.fromarray(np.uint8(255*disparity_map/np.amax(disparity_map))).show()

        '''
        +1 added to divisor so no divide by zeros
        '''
        depth_map = (self.baseline_meters * self.focal_length_meters)/(disparity_map + 1)
        depth_map[disparity_map == 0] = 0
        Image.fromarray(np.uint8(255*depth_map/np.amax(depth_map))).show()

        '''base_dist_transform = cv2.distanceTransform(255 - union_crop_canny_base_stitch, cv2.DIST_L2, 5)

        Image.fromarray(np.uint8(255*base_dist_transform/np.amax(base_dist_transform))).show()

        fit_disparity_map = cv2.threshold(union_crop_canny_fit_stitch, 0, 1, cv2.THRESH_BINARY)[1].astype(np.float32)

        fit_disparity_map = np.multiply(fit_disparity_map, base_dist_transform)
        Image.fromarray(np.uint8(255*np.sqrt(fit_disparity_map/np.amax(fit_disparity_map)))).show()
        '''




        '''fit_dist_transform = cv2.distanceTransform(255 - union_crop_canny_fit_stitch, cv2.DIST_L2, 5)
        print("max fit dist transform: ", np.amax(fit_dist_transform))
        base_dist_transform = cv2.distanceTransform(255 - union_crop_canny_base_stitch, cv2.DIST_L2, 5)
        Image.fromarray(np.uint8(255*(fit_dist_transform/np.amax(fit_dist_transform)))).show()'''

        #Image.fromarray(union_crop_canny_fit_stitch).show()
        #Image.fromarray(union_crop_canny_base_stitch).show()
