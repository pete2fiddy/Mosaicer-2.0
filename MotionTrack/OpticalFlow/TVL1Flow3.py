import numpy as np
import cv2
from PIL import Image
import MotionTrack.OpticalFlow.FlowHelper as FlowHelper

class TVL1Flow:

    IMAGE_GRAD_X_KERNEL = np.array([[-1.0, 0.0, 1.0]], dtype = np.float32)*0.5
    IMAGE_GRAD_Y_KERNEL = np.array([[-1.0],
                                   [0.0],
                                   [1.0]], dtype = np.float32)*0.5
    DIV_P_KERNEL1 = np.array([[-1.0, 1.0]], dtype = np.float32)
    DIV_P_KERNEL2 = np.array([[-1.0],
                              [1.0]], dtype = np.float32)
    DIV_P_KERNEL1_ANCHOR = (1,0)
    DIV_P_KERNEL2_ANCHOR = (0,1)

    FLOW_GRAD_KERNEL_X = np.array([[-1.0, 1.0]], dtype = np.float32)
    FLOW_GRAD_KERNEL_Y = np.array([[-1.0],
                                   [1.0]], dtype = np.float32)
    FLOW_GRAD_KERNEL_X_ANCHOR = (0, 0)
    FLOW_GRAD_KERNEL_Y_ANCHOR = (0, 0)

    GAUSSIAN_BLUR_KSIZE = (7,7)
    GAUSSIAN_BLUR_STD_DEV = 0.8


    '''try using Linear LOESS for smoothing'''

    '''use notes:
    Flows may have outliers, false vectors with VERY high magnitudes can appear, causing
    normalized vector images and flow angle maps to not look correct because, in relation
    to the very large outlier, the true flows fade away.

    Smoothing U serves to dampen the impact of these outliers, but as far as I can tell
    do not greatly affect the actual output of the algorithm save for dampening those
    large flows.

    Lower time steps makes the algorithm mroe resilient to creating outliers. Causes it
    to converge slower

    The less agressive the flow_smooth_func is, the lower the time_step must be
    in order to keep the flows from going crazy

    May want to use another convergence stop thresholding metric, because lower
    time steps cause it to trigger much easier (as it dictates how much U steps
    between iterations)...
    '''
    '''params is a NamedArgs object'''
    def __init__(self, frame1, frame2, params):
        self.frame1 = frame1.astype(np.float32)
        self.frame2 = frame2.astype(np.float32)
        self.init_params(params)

        self.calculate_flows()


    def init_params(self, params):
        '''may want to find a way to pass flow smooth func and flow smooth args in a
        functionArgs or ClassArgs object'''
        self.flow_smooth_func = params["flow_smooth_func"]
        self.flow_smooth_args = params["flow_smooth_args"]
        self.flow_smooth_func = params["flow_smooth_func"]
        self.flow_smooth_args = params["flow_smooth_args"]
        self.smooth_weight = params["smooth_weight"]
        self.time_step = params["time_step"]
        self.theta = params["theta"]
        self.convergence_thresh = params["convergence_thresh"]
        self.pyr_scale_factor = params["pyr_scale_factor"]
        self.num_scales = params["num_scales"]
        self.num_warps = params["num_warps"]
        self.max_iter_per_warp = params["max_iter_per_warp"]

    def calculate_flows(self):
        frame1_pyr = self.build_pyramid(self.frame1)
        frame2_pyr = self.build_pyramid(self.frame2)
        U = np.zeros(self.frame1.shape[:2] + (2,), dtype = np.float32)

        for pyr_index in range(0, len(frame1_pyr)):
            downscale_factor = self.pyr_scale_factor**(len(frame1_pyr) - pyr_index - 1)
            upscale_factor = 1.0/downscale_factor

            U_downscaled = cv2.resize(U, frame1_pyr[pyr_index].shape[:2][::-1], cv2.INTER_CUBIC)*downscale_factor

            U_downscaled = self.calc_tvl1_flows(frame1_pyr[pyr_index], frame2_pyr[pyr_index], U_downscaled)

            U = cv2.resize(U_downscaled, self.frame1.shape[:2][::-1], cv2.INTER_CUBIC) * upscale_factor
        self.flows = U


    def calc_tvl1_flows(self, frame1, frame2, U_in):
        P1 = np.zeros(U_in.shape, dtype = np.float32)
        P2 = np.zeros(U_in.shape, dtype = np.float32)
        U = U_in.copy()

        for warp_iter in range(0, self.num_warps):
            U_0 = U.copy()
            '''not sure to use L1 or L2 for gradient magnitudes'''
            warp_image, warp_grad_xy, warp_grad_mags = self.calc_warp_image_and_gradients_and_mags(frame2, U_0, 2)
            #cv2.imshow("warp image: ", np.uint8(warp_image))
            for iter in range(0, self.max_iter_per_warp):
                V = self.iterate_V(U, U_0, warp_image, warp_grad_xy, warp_grad_mags, frame1)
                U_old = U.copy()
                U = self.iterate_U(V, P1, P2)
                U = self.flow_smooth_func(U, self.flow_smooth_args)

                convergence_crit = self.calc_convergence_criteria(U, U_old)
                if convergence_crit < self.convergence_thresh:
                    break

                P1, P2 = self.iterate_P(P1, P2, U)
                cv2.imshow("Flow vector image:", FlowHelper.calc_flow_vector_image(frame1, U))
                cv2.imshow("Flow angle image: ", FlowHelper.calc_flow_angle_image(U, use_mag = True))
                cv2.waitKey(1)
        return U


    def build_pyramid(self, image):
        pyramid = [image]
        for i in range(0, self.num_scales - 1):
            prev_image = pyramid[len(pyramid)-1]
            append_image = cv2.GaussianBlur(prev_image, TVL1Flow.GAUSSIAN_BLUR_KSIZE, TVL1Flow.GAUSSIAN_BLUR_STD_DEV)
            resize_dims = tuple((np.asarray(append_image.shape[:2][::-1])*self.pyr_scale_factor).astype(np.int))
            append_image = cv2.resize(append_image, resize_dims, cv2.INTER_CUBIC)
            pyramid.append(append_image)
        return list(reversed(pyramid))



    def calc_convergence_criteria(self, U, U_old):
        sqrd_U_sub_U_0 = np.square(U - U_old)
        criteria = sqrd_U_sub_U_0[:,:,0] + sqrd_U_sub_U_0[:,:,1]
        return np.average(criteria)

    def iterate_P(self, P1, P2, U):
        U_x_grad_xy, U_y_grad_xy = self.calc_2D_flow_gradients(U)
        '''try replacing the denominator with maximum(1, other term)
        if not working correctly'''
        P1_new = (P1 + (self.time_step/self.theta) * U_x_grad_xy)/(((self.time_step/self.theta) * U_x_grad_xy) + 1.0)
        P2_new = (P2 + (self.time_step/self.theta) * U_y_grad_xy)/(((self.time_step/self.theta) * U_y_grad_xy) + 1.0)
        return P1_new, P2_new


    def calc_2D_flow_gradients(self, flows):
        flows_x_grad_xy = self.calc_1D_flow_gradients(flows[:,:,0])
        flows_y_grad_xy = self.calc_1D_flow_gradients(flows[:,:,1])
        return flows_x_grad_xy, flows_y_grad_xy

    def calc_1D_flow_gradients(self, flow):
        '''check if border assignments are correct'''
        flow_grad_x = cv2.filter2D(flow, cv2.CV_32F, TVL1Flow.FLOW_GRAD_KERNEL_X, anchor = TVL1Flow.FLOW_GRAD_KERNEL_X_ANCHOR)
        flow_grad_y = cv2.filter2D(flow, cv2.CV_32F, TVL1Flow.FLOW_GRAD_KERNEL_Y, anchor = TVL1Flow.FLOW_GRAD_KERNEL_Y_ANCHOR)

        flow_grad_x[flow_grad_x.shape[0]-1, :] = 0
        flow_grad_y[:, flow_grad_y.shape[0]-1] = 0

        return np.dstack((flow_grad_x, flow_grad_y))

    def iterate_U(self, V, P1, P2):
        U_new = V.copy()
        U_new[:,:,0] += self.theta*self.div_P(P1)
        U_new[:,:,1] += self.theta*self.div_P(P2)
        return U_new

    def div_P(self, P):
        '''https://gyazo.com/eaabba5060bd043785fc1dd041224cb1'''
        responses1 = cv2.filter2D(P[:,:,0], cv2.CV_32F, TVL1Flow.DIV_P_KERNEL1, anchor = TVL1Flow.DIV_P_KERNEL1_ANCHOR)
        responses2 = cv2.filter2D(P[:,:,1], cv2.CV_32F, TVL1Flow.DIV_P_KERNEL2, anchor = TVL1Flow.DIV_P_KERNEL2_ANCHOR)

        responses1[0, :] = P[0, :, 0]
        responses1[P.shape[0]-1, :] = -P[P.shape[0]-1, :, 0]

        responses2[:, 0] = P[:, 0, 0]
        responses2[:, P.shape[1]-1] = -P[:, P.shape[1]-1, 0]
        return responses1 + responses2

    def iterate_V(self, U, U_0, warp_image, warp_grad_xy, warp_grad_mags, base_image):
        '''https://gyazo.com/d18a993f1c831fd6f834b5a2652ae2a9. Not sure if
        add U into or not. Previously adding U caused to fail.'''
        p_of_U = self.calc_P_of_flows(U, U_0, warp_image, warp_grad_xy, base_image)

        '''not sure if using the L1 or L2 norm here... I assume L2 because
        that makes more sense when used with image gradients...
        If not, look here for something to change'''
        indices_condition1 = np.where(p_of_U < -self.smooth_weight*self.theta*warp_grad_mags**2)
        indices_condition2 = np.where(p_of_U > self.smooth_weight*self.theta*warp_grad_mags**2)
        indices_condition3 = np.where(np.abs(p_of_U) <= self.smooth_weight*self.theta*warp_grad_mags**2)

        V_new = np.zeros(U.shape, dtype = np.float32)
        '''condition must be assigned backwards so that it satisfies an "if else" chain'''
        '''still may crash on zero gradient???'''

        condition3_value_mat = (-p_of_U[:,:,np.newaxis]*warp_grad_xy/(warp_grad_mags**2)[:,:,np.newaxis])
        condition3_value_mat[warp_grad_mags == 0] = 0
        V_new[indices_condition3[0], indices_condition3[1], :] = condition3_value_mat[indices_condition3[0], indices_condition3[1], :]
        V_new[indices_condition2[0], indices_condition2[1], :] = (-self.smooth_weight*self.theta*warp_grad_xy)[indices_condition2[0], indices_condition2[1], :]
        V_new[indices_condition1[0], indices_condition1[1], :] = (self.smooth_weight*self.theta*warp_grad_xy)[indices_condition1[0], indices_condition1[1], :]
        '''not sure if add U to V_new here'''
        return V_new + U


    def calc_P_of_flows(self, U, U_0, warp_image, warp_grad_xy, base_image):
        U_sub_U_0 = U - U_0
        grad_x_comp = warp_grad_xy[:,:,0] * U_sub_U_0[:,:,0]
        grad_y_comp = warp_grad_xy[:,:,1] * U_sub_U_0[:,:,1]
        grad_dots = grad_x_comp + grad_y_comp
        return warp_image + grad_dots - base_image

    '''
    uses Bicubic interpolation to calculate I1(x+U_0)
    and grad I1(x+U_0)
    '''
    def calc_warp_image_and_gradients_and_mags(self, image, U_0, norm_type):
        warp_image = FlowHelper.warp_image_with_flows(image, U_0)
        warp_grad_xy = self.calc_image_gradients(image)
        warp_grad_x = FlowHelper.warp_image_with_flows(warp_grad_xy[:,:,0], U_0)
        warp_grad_y = FlowHelper.warp_image_with_flows(warp_grad_xy[:,:,1], U_0)
        warp_grad_xy = np.dstack((warp_grad_x, warp_grad_y))
        warp_grad_mags = np.linalg.norm(warp_grad_xy, axis = 2, ord = norm_type)
        return warp_image, warp_grad_xy, warp_grad_mags

    def calc_image_gradients(self, image):
        grad_x = cv2.filter2D(image, cv2.CV_32F, TVL1Flow.IMAGE_GRAD_X_KERNEL, borderType = cv2.BORDER_CONSTANT)
        grad_y = cv2.filter2D(image, cv2.CV_32F, TVL1Flow.IMAGE_GRAD_Y_KERNEL, borderType = cv2.BORDER_CONSTANT)

        #grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        #grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)

        '''check if border cases are correct'''


        grad_x[0, :] = 0
        grad_x[grad_x.shape[0]-1, :] = 0

        grad_y[:, 0] = 0
        grad_y[:, grad_x.shape[0]-1] = 0

        grad_xy = np.dstack((grad_x, grad_y))
        return grad_xy
