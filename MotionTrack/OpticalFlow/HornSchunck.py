import numpy as np
import cv2
from PIL import Image
import time
from math import cos, sin
import ImageOp.Kernel as Kernel


class TwoFrameHornSchunck2:
    WEIGHTED_AVG_FLOW_KERNEL = np.array([[1/12, 1/6, 1/12],
                                         [1/6, 0, 1/6],
                                         [1/12, 1/6, 1/12]])
    def __init__(self, frame1, frame2, smooth_weight, num_iter = 100):
        self.frame1 = frame1
        self.frame2 = frame2
        self.smooth_weight = smooth_weight
        self.num_iter = num_iter
        self.init_flows()

    def init_flows(self):
        self.x_flows = np.zeros(self.frame1.shape[:2], dtype = np.float32)
        self.y_flows = np.zeros(self.frame1.shape[:2], dtype = np.float32)
        x_grad_kernel = np.array([[-1,1],
                                  [-1,1]])*.25
        y_grad_kernel = np.array([[-1,-1],
                                  [1, 1]])*.25
        time_partial_kernel = np.ones((2,2))*.25
        frame2_grad_x = cv2.filter2D(self.frame2, cv2.CV_32F, x_grad_kernel)#cv2.Sobel(self.frame2, cv2.CV_32F, 1, 0)
        frame2_grad_y = cv2.filter2D(self.frame2, cv2.CV_32F, y_grad_kernel)#cv2.Sobel(self.frame2, cv2.CV_32F, 0, 1)
        frame2_grad_x2 = np.square(frame2_grad_x)
        frame2_grad_y2 = np.square(frame2_grad_y)
        partial_frame2_partial_time = cv2.filter2D(self.frame2, cv2.CV_32F, time_partial_kernel) - cv2.filter2D(self.frame1, cv2.CV_32F, time_partial_kernel)#(self.frame2.astype(np.float32) - self.frame1.astype(np.float32))#

        warp_frame = self.frame1.copy()

        for iter in range(0, self.num_iter):
            weighted_avg_x_flows = self.get_weighted_avg_flows(self.x_flows)
            weighted_avg_y_flows = self.get_weighted_avg_flows(self.y_flows)

            '''could speed up more by precalculating as much as possible'''

            flow_updates_shared_terms = ((frame2_grad_x * weighted_avg_x_flows) + (frame2_grad_y * weighted_avg_y_flows) + partial_frame2_partial_time)/(self.smooth_weight + frame2_grad_x2 + frame2_grad_y2)

            self.x_flows = weighted_avg_x_flows - frame2_grad_x * flow_updates_shared_terms
            self.y_flows = weighted_avg_y_flows - frame2_grad_y * flow_updates_shared_terms


            print("max x flows: ", np.amax(self.x_flows))

            flow_stack = np.dstack((self.y_flows, self.x_flows))
            disparity_mags = np.linalg.norm(flow_stack, axis = 2)
            normed_disparity_mags = np.uint8(255*disparity_mags/np.amax(disparity_mags))

            cv2.imshow("Warp frame1: ", self.warp_frame1_with_flows())
            #cv2.imshow("Flow angle image: ", self.get_flow_angle_image())
            cv2.imshow("Flow vector image: ", self.get_flow_vector_image())
            cv2.imshow("Disparity mags: ", normed_disparity_mags)
            cv2.waitKey(1)





    def get_weighted_avg_flows(self, flows):
        return cv2.filter2D(flows, cv2.CV_32F, TwoFrameHornSchunck2.WEIGHTED_AVG_FLOW_KERNEL)

    def warp_frame1_with_flows(self):
        x_indices_mat = np.array([np.arange(self.frame1.shape[1]) for j in range(0, self.frame1.shape[0])])
        y_indices_mat = np.array([np.arange(self.frame1.shape[0]) for j in range(0, self.frame1.shape[1])]).T

        x_map = (x_indices_mat + self.x_flows).astype(np.float32)
        y_map = (y_indices_mat + self.y_flows).astype(np.float32)
        warp_image = cv2.remap(self.frame1, x_map, y_map, cv2.INTER_LINEAR)
        return warp_image

    def get_flow_angle_image(self):
        flow_stacks = np.dstack((self.x_flows, self.y_flows))

        hsv_angles = np.arctan2(self.y_flows, self.x_flows)
        hsv_angles = np.rad2deg(hsv_angles)%360

        hsv_saturations = (np.linalg.norm(flow_stacks, axis = 2))
        hsv_saturations /= np.amax(hsv_saturations)

        flow_hsv_image = np.dstack((hsv_angles, hsv_saturations, np.ones((self.x_flows.shape[0], self.x_flows.shape[1])))).astype(np.float32)
        flow_rgb_image = np.uint8(255 * cv2.cvtColor(flow_hsv_image, cv2.COLOR_HSV2RGB))
        return flow_rgb_image

    def get_flow_vector_image(self, max_vec_mag = 40, min_mag = 2, step = 5, color = (255,0,0)):

        vector_image = self.frame1.copy()
        if len(self.frame1.shape) == 2:
            vector_image = cv2.cvtColor(self.frame1, cv2.COLOR_GRAY2RGB)

        vec_angles = np.arctan2(self.y_flows, self.x_flows)
        stacked_flows = np.dstack((self.x_flows, self.y_flows))
        flow_mags = np.linalg.norm(stacked_flows, axis = 2)
        max_mag = np.amax(flow_mags)

        for x in range(0, self.x_flows.shape[1], step):
            for y in range(0, self.x_flows.shape[0], step):
                if flow_mags[y,x] > min_mag:
                    start_point = np.array([x,y])
                    normalized_vec_mag = (max_vec_mag * flow_mags[y,x]/max_mag)
                    end_point = (start_point + (np.array([cos(vec_angles[y,x]), sin(vec_angles[y,x])]) * normalized_vec_mag)).astype(np.int)
                    vector_image = cv2.arrowedLine(vector_image, tuple(start_point), tuple(end_point), color)
        return vector_image


class TwoFrameHornSchunck:
    '''WEIGHTED_AVG_FLOW_KERNEL = np.array([[1/12, 1/6, 1/12],
                                         [1/6, 0, 1/6],
                                         [1/12, 1/6, 1/12]])'''
    DERIV_KERNEL_SIZE = 3
    WEIGHTED_AVG_FLOW_KERNEL = np.array([[1/12, 1/6, 1/12],
                                         [1/6, -1, 1/6],
                                         [1/12, 1/6, 1/12]])
    #WEIGHTED_AVG_FLOW_KERNEL = Kernel.get_gaussian_kernel((DERIV_KERNEL_SIZE, DERIV_KERNEL_SIZE), 1.0)#cv2.getGaussianKernel(DERIV_KERNEL_SIZE, 2.0, cv2.CV_32F)
    #WEIGHTED_AVG_FLOW_KERNEL[(DERIV_KERNEL_SIZE-1)//2, (DERIV_KERNEL_SIZE-1)//2] = 0
    #WEIGHTED_AVG_FLOW_KERNEL /= np.sum(WEIGHTED_AVG_FLOW_KERNEL)


    def __init__(self, images, smooth_weight, num_iter = 2):
        self.images = images
        self.smooth_weight = smooth_weight
        self.num_iter = num_iter
        self.init_flows()

    '''to test: create synthetic images in paint or something of a ball that moves --
    create a histogram of flow magnitudes, see how it differs from ground truth
    Rather than updating over and over for only one pair of images, refine the
    velocities by using later frames and treating them as iterations.
    '''
    def init_flows(self):

        '''
        PROBLEM: NEED TO WARP IMAGE EVERY ITERATION. DOING SO WILL UPDATE
        PARTIAL_FRAME2_PARTIAL_TIME
        '''
        self.x_flows = np.zeros(self.images[0].shape, dtype = np.float32)
        self.y_flows = np.zeros(self.images[0].shape, dtype = np.float32)

        #smooth_weight_sqrd = self.smooth_weight ** 2

        for i in range(0, self.images.shape[0]-1):
            frame1 = self.images[i]
            frame2 = self.images[i+1]
            grad_x_frame2 = cv2.Sobel(frame2, cv2.CV_32F, 1, 0)#, ksize = TwoFrameHornSchunck.DERIV_KERNEL_SIZE)

            grad_x2_frame2 = np.square(grad_x_frame2)
            grad_y_frame2 = cv2.Sobel(frame2, cv2.CV_32F, 0, 1)#, ksize = TwoFrameHornSchunck.DERIV_KERNEL_SIZE)

            grad_y2_frame2 = np.square(grad_y_frame2)
            partial_frame2_partial_time = frame2 - frame1

            for iter in range(0, self.num_iter):
                neighborhood_avg_x_flows = self.get_neighborhood_avg_flow_image(self.x_flows)
                neighborhood_avg_y_flows = self.get_neighborhood_avg_flow_image(self.y_flows)
                print("neighborhood avg x flows: ", neighborhood_avg_x_flows)
                updates_shared_terms = (grad_x_frame2 * neighborhood_avg_x_flows + grad_y_frame2 * neighborhood_avg_y_flows + partial_frame2_partial_time)/(self.smooth_weight + grad_x2_frame2 + grad_y2_frame2)
                self.x_flows = neighborhood_avg_x_flows - grad_x_frame2 * updates_shared_terms
                self.y_flows = neighborhood_avg_y_flows - grad_y_frame2 * updates_shared_terms

                max_x_flow = np.amax(np.abs(self.x_flows))
                max_y_flow = np.amax(np.abs(self.y_flows))
                print("mag biggest flow: ", np.linalg.norm(np.array([max_x_flow, max_y_flow])))

                stacked_flows = np.dstack((self.x_flows, self.y_flows))
                flow_mags = np.linalg.norm(stacked_flows, axis = 2)
                disparity_map = np.uint8(255*flow_mags/np.amax(flow_mags))

                cv2.imshow("Flow angle image", self.get_flow_angle_image())
                cv2.imshow("Flow vector image", self.get_flow_vector_image(0))
                #cv2.imshow("Magnitude disparity map", disparity_map)

                cv2.waitKey(1)
                #self.x_flows /= grad_x_frame2
                #self.y_flows /= grad_y_frame2
                time.sleep(1)

    def get_flow_angle_image(self):
        flow_stacks = np.dstack((self.x_flows, self.y_flows))

        hsv_angles = np.arctan2(self.y_flows, self.x_flows)
        hsv_angles = np.rad2deg(hsv_angles)%360

        hsv_saturations = (np.linalg.norm(flow_stacks, axis = 2))
        hsv_saturations /= np.amax(hsv_saturations)

        flow_hsv_image = np.dstack((hsv_angles, hsv_saturations, np.ones((self.x_flows.shape[0], self.x_flows.shape[1])))).astype(np.float32)
        flow_rgb_image = np.uint8(255 * cv2.cvtColor(flow_hsv_image, cv2.COLOR_HSV2RGB))
        return flow_rgb_image

    def get_flow_vector_image(self, mag_thresh, max_vec_mag = 40, step = 7, color = (255,0,0)):

        vector_image = cv2.cvtColor(self.images[0], cv2.COLOR_GRAY2RGB)

        vec_angles = np.arctan2(self.y_flows, self.x_flows)
        stacked_flows = np.dstack((self.x_flows, self.y_flows))
        flow_mags = np.linalg.norm(stacked_flows, axis = 2)
        max_mag = np.amax(flow_mags)

        for x in range(0, self.x_flows.shape[1], step):
            for y in range(0, self.x_flows.shape[0], step):

                start_point = np.array([x,y])
                normalized_vec_mag = (max_vec_mag * flow_mags[y,x]/max_mag)
                if normalized_vec_mag > 1:
                    end_point = (start_point + (np.array([cos(vec_angles[y,x]), sin(vec_angles[y,x])]) * normalized_vec_mag)).astype(np.int)
                    vector_image = cv2.arrowedLine(vector_image, tuple(start_point), tuple(end_point), color)
        return vector_image


    '''for each pixel in flows, sets to the weighted average around the neibhborhood
    of that pixel'''
    def get_neighborhood_avg_flow_image(self, flows):
        return cv2.filter2D(flows, cv2.CV_32F, TwoFrameHornSchunck.WEIGHTED_AVG_FLOW_KERNEL)
