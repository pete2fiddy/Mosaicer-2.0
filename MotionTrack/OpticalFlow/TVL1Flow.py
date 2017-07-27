import numpy as np
import cv2
from math import cos, sin
from PIL import Image

class TwoFrameTVL1Three:

    GAUSSIAN_STD_DEV = 0.8

    def __init__(self, frame1, frame2, smooth_weight = 0.15, time_step = .1, theta = .3, pyr_scale_factor = 0.5, convergence_thresh = 0.01, num_scales = 6, num_warps = 5, iters_per_warp = 50):
        self.frame1 = frame1.astype(np.float32)
        self.frame2 = frame2.astype(np.float32)
        self.smooth_weight = smooth_weight
        self.theta = theta
        self.time_step = time_step
        self.pyr_scale_factor = pyr_scale_factor
        self.convergence_thresh = convergence_thresh
        self.num_scales = num_scales
        self.num_warps = num_warps
        self.iters_per_warp = iters_per_warp

        self.frame1_pyr = self.build_pyramid(self.frame1)
        self.frame2_pyr = self.build_pyramid(self.frame2)

        self.U = np.zeros(self.frame1.shape[:2] + (2,))
        self.init_flows()

        Image.fromarray(np.uint8(self.warp_image_with_flows(self.frame2, self.U))).show()
        Image.fromarray(np.uint8(self.get_flow_vector_image(self.frame2, self.U))).show()


    def init_flows(self):


        for scale_index in range(0, len(self.frame2_pyr)):

            frame1_at_scale = self.frame1_pyr[scale_index]
            frame2_at_scale = self.frame2_pyr[scale_index]

            shrunk_U = cv2.resize(self.U, frame1_at_scale.shape[:2][::-1], cv2.INTER_CUBIC).astype(np.float32)
            shrunk_U *= frame1_at_scale.shape[0]/self.frame1.shape[0]

            scale_add_flows = self.calc_tvl1_flows(frame1_at_scale, frame2_at_scale, shrunk_U)

            to_scale_add_flows = cv2.resize(scale_add_flows, self.U.shape[:2][::-1], cv2.INTER_CUBIC)
            upscale_factor = to_scale_add_flows.shape[0]/scale_add_flows.shape[0]
            to_scale_multiplied_add_flows = to_scale_add_flows * upscale_factor
            self.U += to_scale_multiplied_add_flows


    def calc_tvl1_flows(self, frame1, frame2, U_0_in):
        U = U_0_in.copy()#np.zeros(frame1.shape[:2] + (2,), dtype = np.float32)
        V = np.zeros(U.shape, dtype = np.float32)
        P1 = np.zeros(U.shape, dtype = np.float32)
        P2 = np.zeros(U.shape, dtype = np.float32)
        #U_0 = U_0_in.copy()
        for warp_count in range(0, self.num_warps):
            warp_frame2 = self.warp_image_with_flows(frame2, U)
            warp_frame2_grads = self.calc_image_gradients(warp_frame2)
            U_0 = U.copy()
            for iter_count in range(0, self.iters_per_warp):
                V = self.iterate_V(V, U, U_0, warp_frame2, warp_frame2_grads, frame1)
                U_old = U.copy()
                U = self.iterate_U(V, P1, P2)
                P1, P2 = self.iterate_P(P1, P2, U)
                #cv2.imshow("Flow angle image: ", self.get_flow_angle_image(U))
                #cv2.waitKey(1)
                print("convergence score: ", self.calc_U_convergence_score(U, U_old))
                if self.calc_U_convergence_score(U, U_old) <= self.convergence_thresh:
                    print("=========================================")
                    print("broken due to convergence")
                    print("=========================================")
                    break
                print("----------------------------------------")

        return U

    def iterate_V(self, V, U, U_0, warp_frame2, warp_frame2_grads, frame1):
        flow_errors = self.calc_op_flow_errors(warp_frame2, warp_frame2_grads, frame1, U, U_0)
        mag_warp_frame2_grads = np.linalg.norm(warp_frame2_grads, axis = 2)
        V_new = np.zeros(warp_frame2_grads.shape, dtype = np.float32)
        indices_condition1 = np.where(flow_errors < -self.smooth_weight*self.theta*mag_warp_frame2_grads**2)
        indices_condition2 = np.where(flow_errors > self.smooth_weight*self.theta*mag_warp_frame2_grads**2)
        indices_condition3 = np.where(np.abs(flow_errors) <= self.smooth_weight*self.theta*mag_warp_frame2_grads**2)
        V_new[indices_condition1[0], indices_condition1[1], :] = (self.smooth_weight*self.theta*warp_frame2_grads)[indices_condition1[0], indices_condition1[1], :]
        V_new[indices_condition2[0], indices_condition2[1], :] = (-self.smooth_weight*self.theta*warp_frame2_grads)[indices_condition2[0], indices_condition2[1], :]

        condition3_value_mat = (-flow_errors[:,:,np.newaxis] * (warp_frame2_grads/(mag_warp_frame2_grads**2)[:,:,np.newaxis]))
        condition3_value_mat[mag_warp_frame2_grads == 0 ] = 0
        V_new[indices_condition3[0], indices_condition3[1], :] = condition3_value_mat[indices_condition3[0], indices_condition3[1], :]#(-flow_errors[:,:,np.newaxis] * (warp_frame2_grads/(mag_warp_frame2_grads**2)[:,:,np.newaxis]))[indices_condition3[0], indices_condition3[1], :]
        #V_new = U.astype(np.float32) + V_new
        return V_new

    def iterate_U(self, V, P1, P2):
        U_new = np.zeros(V.shape, dtype = np.float32)
        U_new[:,:,0] = self.theta*self.div_P(P1)
        U_new[:,:,1] = self.theta*self.div_P(P2)
        U_new = V + U_new
        return U_new

    def iterate_P(self, P1, P2, U):
        U_x_grads, U_y_grads = self.calc_flow_gradients(U)
        P1_new = (P1 + (self.time_step/self.theta) * U_x_grads)/(1.0 + (self.time_step/self.theta) * np.linalg.norm(U_x_grads, axis = 2))[:,:,np.newaxis]
        P2_new = (P2 + (self.time_step/self.theta) * U_y_grads)/(1.0 + (self.time_step/self.theta) * np.linalg.norm(U_y_grads, axis = 2))[:,:,np.newaxis]
        return P1_new.astype(np.float32), P2_new.astype(np.float32)

    def div_P(self, P):
        '''need to handle edge cases where the filter can't touch'''
        Px_kernel = np.array([[-1.0, 1.0]], dtype = np.float32)
        Py_kernel = np.array([[-1.0],
                              [1.0]], dtype = np.float32)
        Px_component = cv2.filter2D(P[:,:,0], cv2.CV_32F, Px_kernel)
        Px_component[0,:] = P[0,:,0]
        Px_component[Px_component.shape[0]-1, :] = P[P.shape[0]-1,:,0]

        Py_component = cv2.filter2D(P[:,:,1], cv2.CV_32F, Py_kernel)
        Py_component[:,0] = P[:,0,1]
        Py_component[:,Py_component.shape[1]-1] = P[:,P.shape[1]-1, 1]
        return (Px_component + Py_component).astype(np.float32)

    '''is p(u) in math notation'''
    def calc_op_flow_errors(self, warp_frame2, warp_frame2_grads, frame1, U, U_0):
        U_sub_U_0 = U - U_0
        grad_dot_x_component = warp_frame2_grads[:,:,0] * U_sub_U_0[:,:,0]
        grad_dot_y_component = warp_frame2_grads[:,:,1] * U_sub_U_0[:,:,1]
        grad_dot = grad_dot_x_component + grad_dot_y_component
        op_flow_errors = grad_dot + warp_frame2 - frame1
        return op_flow_errors

    '''calculates using U at iter k+1 and U at iter k a score. The
    lower this score, the more U^k+1 and U are alike, and if this score
    is below a threshold, the algorithm has likely converged'''
    def calc_U_convergence_score(self, U_new, U_old):
        U_new_x = U_new[:,:,0]
        U_new_y = U_new[:,:,1]
        U_old_x = U_old[:,:,0]
        U_old_y = U_old[:,:,1]

        U_x_sub_sqrd = (U_new_x - U_old_x)**2
        U_y_sub_sqrd = (U_new_y - U_old_y)**2
        U_convergence_scores = U_x_sub_sqrd + U_y_sub_sqrd
        avg_convergence_score = np.average(U_convergence_scores)
        return avg_convergence_score

    def calc_image_gradients(self, image):
        '''x_grad_kernel = np.array([[-1, 1],
                                  [-1, 1]], dtype = np.float32)*.25
        y_grad_kernel = np.array([[-1, -1],
                                  [1, 1]], dtype = np.float32)*.25'''
        x_grad_kernel = np.array([[-1.0, 0.0, 1.0]], dtype = np.float32)*.5
        y_grad_kernel = np.array([[-1.0],
                                  [0.0],
                                  [1.0]], dtype = np.float32)*.5
        image_grad_x = cv2.filter2D(image, cv2.CV_32F, x_grad_kernel)
        image_grad_y = cv2.filter2D(image, cv2.CV_32F, y_grad_kernel)
        return np.dstack((image_grad_x, image_grad_y))

    def calc_flow_gradients(self, flows_xy):
        '''need to add Neumann boundary conditions'''
        x_grad_kernel = np.array([[-1.0, 1.0]], dtype = np.float32)
        y_grad_kernel = np.array([[-1.0],
                                  [1.0]], dtype = np.float32)
        x_flows_x_grad = cv2.filter2D(flows_xy[:,:,0], cv2.CV_32F, x_grad_kernel)
        x_flows_y_grad = cv2.filter2D(flows_xy[:,:,0], cv2.CV_32F, y_grad_kernel)

        y_flows_x_grad = cv2.filter2D(flows_xy[:,:,1], cv2.CV_32F, x_grad_kernel)
        y_flows_y_grad = cv2.filter2D(flows_xy[:,:,1], cv2.CV_32F, y_grad_kernel)

        x_flow_grads = np.dstack((x_flows_x_grad, x_flows_y_grad))
        y_flow_grads = np.dstack((y_flows_x_grad, y_flows_y_grad))
        return x_flow_grads, y_flow_grads

    def warp_image_with_flows(self, image, U):
        x_flows = U[:,:,0]
        y_flows = U[:,:,1]
        x_indices_mat = np.array([np.arange(image.shape[1]) for j in range(0, image.shape[0])])
        y_indices_mat = np.array([np.arange(image.shape[0]) for j in range(0, image.shape[1])]).T

        x_map = (x_indices_mat + x_flows).astype(np.float32)
        y_map = (y_indices_mat + y_flows).astype(np.float32)
        warp_image = cv2.remap(image, x_map, y_map, cv2.INTER_CUBIC)
        return warp_image.astype(np.float32)

    def build_pyramid(self, image):
        pyramid = [image]
        for i in range(0, self.num_scales-1):
            last_image = pyramid[len(pyramid)-1]
            append_image = cv2.GaussianBlur(last_image, (7,7), TwoFrameTVL1Three.GAUSSIAN_STD_DEV)
            resize_dims = np.asarray(append_image.shape[:2][::-1])
            resize_dims = self.pyr_scale_factor * resize_dims
            resize_dims = tuple(resize_dims.astype(np.int))
            append_image = cv2.resize(append_image, resize_dims, cv2.INTER_CUBIC).astype(np.float32)
            pyramid.append(append_image)
        return list(reversed(pyramid))

    def get_flow_vector_image(self, image, U, max_vec_mag = 40, min_mag = 2, step = 5, color = (255,0,0)):
        vector_image = image.copy()
        if len(self.frame1.shape) == 2:
            vector_image = cv2.cvtColor(self.frame1, cv2.COLOR_GRAY2RGB)

        vec_angles = np.arctan2(U[:,:,1], U[:,:,0])
        stacked_flows = U#np.dstack((self.U_x, self.U_y))
        flow_mags = np.linalg.norm(stacked_flows, axis = 2)
        max_mag = np.amax(flow_mags)

        for x in range(0, U.shape[1], step):
            for y in range(0, U.shape[0], step):
                if flow_mags[y,x] > min_mag:
                    start_point = np.array([x,y])
                    normalized_vec_mag = (max_vec_mag * flow_mags[y,x]/max_mag)
                    end_point = (start_point + (np.array([cos(vec_angles[y,x]), sin(vec_angles[y,x])]) * normalized_vec_mag)).astype(np.int)
                    vector_image = cv2.arrowedLine(vector_image, tuple(start_point), tuple(end_point), color)
        return vector_image

    def get_flow_angle_image(self, U):
        U_upscaled = cv2.resize(U, self.U.shape[:2][::-1], cv2.INTER_CUBIC)
        U_upscaled *= self.U.shape[0]/U.shape[0]

        flow_angles = np.arctan2(U_upscaled[:,:,1], U_upscaled[:,:,0])
        flow_angles = np.rad2deg(flow_angles)%360.0
        flow_mags = np.linalg.norm(U_upscaled, axis = 2)
        max_flow_mag = np.amax(flow_mags)

        hsv_image = np.ones(self.U.shape[:2] + (3,), dtype = np.float32)
        hsv_image[:,:,0] = flow_angles
        hsv_image[:,:,1] = flow_mags/max_flow_mag
        flow_rgb_image = np.uint8(255 * cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB))
        return flow_rgb_image




class TwoFrameTVL1Two:
    DEFAULT_THETA = 0.3
    DEFAULT_TIME_STEP = 0.25

    def __init__(self, frame1, frame2, smooth_weight = 0.15, num_iter = 500):
        self.frame1 = frame1.astype(np.float32)
        self.frame2 = frame2.astype(np.float32)
        self.smooth_weight = smooth_weight
        self.theta = TwoFrameTVL1Two.DEFAULT_THETA
        self.time_step = TwoFrameTVL1Two.DEFAULT_TIME_STEP
        self.num_iter = num_iter

        self.gradient_kernel_x = np.array([[-1, 1],
                                           [-1, 1]], dtype = np.float32) * 0.25
        self.gradient_kernel_y = np.array([[-1, -1],
                                           [1, 1]], dtype = np.float32) * 0.25

        self.frame2_grad_x, self.frame2_grad_y = self.calc_gradients(self.frame2)
        self.frame2_grad_xy = np.dstack((self.frame2_grad_x, self.frame2_grad_y))
        self.frame2_grad_mags = np.linalg.norm(self.frame2_grad_xy, axis = 2)
        self.init_flows()

    def init_flows(self):
        self.U_x = np.zeros(self.frame2.shape[:2], dtype = np.float32)
        self.U_y = np.zeros(self.frame2.shape[:2], dtype = np.float32)
        self.V_x = np.zeros(self.frame2.shape[:2], dtype = np.float32)
        self.V_y = np.zeros(self.frame2.shape[:2], dtype = np.float32)

        self.P_x = np.zeros((self.frame2.shape[0], self.frame2.shape[1], 2), dtype = np.float32)
        self.P_y = np.zeros((self.frame2.shape[0], self.frame2.shape[1], 2), dtype = np.float32)
        for iter in range(0, self.num_iter * 2):
            if iter % 2 == 0:

                self.U_x = self.V_x - self.theta * self.div(self.P_x)
                self.U_y = self.V_y - self.theta * self.div(self.P_y)
                self.P_x, self.P_y = self.iterate_P(self.P_x, self.P_y, self.V_x, self.V_y)
                cv2.imshow("Flow vector image: ", np.uint8(self.get_flow_vector_image()))
                cv2.imshow("Warp frame1: ", np.uint8(self.warp_image_with_flows(self.frame1, self.U_x, self.U_y)))
                cv2.imshow("frame2 approx: ", np.uint8(self.approximate_frame2_with_flows(self.U_x, self.U_y)))
                cv2.waitKey(1)
            else:
                self.V_x, self.V_y = self.iterate_V(self.U_x, self.U_y)


    def iterate_P(self, P_x, P_y, V_x, V_y):
        '''replace reliance on U_x, U_y with V_x, V_y'''
        x_grad_x, x_grad_y = self.calc_gradients(V_x + self.theta*self.div(P_x))
        x_grad_xy = np.dstack((x_grad_x, x_grad_y))
        y_grad_x, y_grad_y = self.calc_gradients(V_y + self.theta*self.div(P_y))
        y_grad_xy = np.dstack((y_grad_x, y_grad_y))
        numerator_x = P_x + (self.time_step/self.theta) * x_grad_xy
        numerator_y = P_y + (self.time_step/self.theta) * y_grad_xy

        P_x_new = numerator_x/(1.0 + np.linalg.norm(numerator_x, axis = 2)[:,:,np.newaxis])#np.maximum(1.0, np.linalg.norm(numerator_x, axis = 2)[:,:,np.newaxis])
        P_y_new = numerator_y/(1.0 + np.linalg.norm(numerator_y, axis = 2)[:,:,np.newaxis])#np.maximum(1.0, np.linalg.norm(numerator_y, axis = 2)[:,:,np.newaxis])

        return P_x_new, P_y_new

    def iterate_V(self, U_x, U_y):
        flow_warp_score = self.get_flow_warp_score_image(U_x, U_y, self.warp_image_with_flows(self.frame1, U_x, U_y))

        V_add = np.zeros((self.frame2.shape[0], self.frame2.shape[1], 2), dtype = np.float32)
        V_add[flow_warp_score < -self.smooth_weight*self.theta*self.frame2_grad_mags**2] = (self.smooth_weight * self.theta * self.frame2_grad_xy)\
             [flow_warp_score < -self.smooth_weight*self.theta*self.frame2_grad_mags**2]

        V_add[flow_warp_score > self.smooth_weight*self.theta*self.frame2_grad_mags**2] = (-self.smooth_weight * self.theta * self.frame2_grad_xy)\
             [flow_warp_score > self.smooth_weight*self.theta*self.frame2_grad_mags**2]

        V_add[np.abs(flow_warp_score) <= self.smooth_weight*self.theta*self.frame2_grad_mags**2] = (-np.dstack((flow_warp_score, flow_warp_score)) * (self.frame2_grad_xy/np.dstack((self.frame2_grad_mags**2, self.frame2_grad_mags**2))))\
             [np.abs(flow_warp_score) <= self.smooth_weight*self.theta*self.frame2_grad_mags**2]

        V_x_new = U_x + V_add[:,:,0]
        V_y_new = U_y + V_add[:,:,1]
        return V_x_new, V_y_new


    def get_flow_warp_score_image(self, warp_image, flow_X, flow_Y):
        return self.approximate_frame2_with_flows(flow_X, flow_Y) - warp_image

    def approximate_frame2_with_flows(self, flow_X, flow_Y):
        x_portion = self.frame2_grad_x * flow_X
        y_portion = self.frame2_grad_y * flow_Y
        return self.frame2 + (x_portion + y_portion)

    def div(self, vec_field):
        grad_x1, grad_y1 = self.calc_gradients(vec_field[:,:,0])
        grad_x2, grad_y2 = self.calc_gradients(vec_field[:,:,1])
        return grad_x1 + grad_y2

    def calc_gradients(self, mat):
        x_grad = cv2.filter2D(mat, cv2.CV_32F, self.gradient_kernel_x)
        y_grad = cv2.filter2D(mat, cv2.CV_32F, self.gradient_kernel_y)
        return x_grad, y_grad

    def get_flow_vector_image(self, max_vec_mag = 40, min_mag = 2, step = 5, color = (255,0,0)):

        vector_image = self.frame1.copy()
        if len(self.frame1.shape) == 2:
            vector_image = cv2.cvtColor(self.frame1, cv2.COLOR_GRAY2RGB)

        vec_angles = np.arctan2(self.U_y, self.U_x)
        stacked_flows = np.dstack((self.U_x, self.U_y))
        flow_mags = np.linalg.norm(stacked_flows, axis = 2)
        max_mag = np.amax(flow_mags)

        for x in range(0, self.U_x.shape[1], step):
            for y in range(0, self.U_x.shape[0], step):
                if flow_mags[y,x] > min_mag:
                    start_point = np.array([x,y])
                    normalized_vec_mag = (max_vec_mag * flow_mags[y,x]/max_mag)
                    end_point = (start_point + (np.array([cos(vec_angles[y,x]), sin(vec_angles[y,x])]) * normalized_vec_mag)).astype(np.int)
                    vector_image = cv2.arrowedLine(vector_image, tuple(start_point), tuple(end_point), color)
        return vector_image

    def warp_image_with_flows(self, image, x_flows, y_flows):
        x_indices_mat = np.array([np.arange(image.shape[1]) for j in range(0, image.shape[0])])
        y_indices_mat = np.array([np.arange(image.shape[0]) for j in range(0, image.shape[1])]).T

        x_map = (x_indices_mat + x_flows).astype(np.float32)
        y_map = (y_indices_mat + y_flows).astype(np.float32)
        warp_image = cv2.remap(image, x_map, y_map, cv2.INTER_LINEAR)
        return warp_image.astype(np.float32)

class TwoFrameTVL1:

    DEFAULT_THETA = 0.0000001
    DEFAULT_TIME_STEP = 0.1
    def __init__(self, frame1, frame2, smooth_weight = 1.0, num_iter = 100):
        self.frame1 = frame1.astype(np.float32)
        self.frame2 = frame2.astype(np.float32)
        self.smooth_weight = smooth_weight
        self.theta = TwoFrameTVL1.DEFAULT_THETA
        self.time_step = TwoFrameTVL1.DEFAULT_TIME_STEP
        self.num_iter = num_iter

        self.gradient_kernel_x = np.array([[-1,1],
                                  [-1,1]])*.25
        self.gradient_kernel_y =  np.array([[-1,-1],
                                  [1, 1]])*.25

        frame2_grad = self.calc_gradients_2d(self.frame2)
        self.frame2_grad_x = frame2_grad[:,:,0]
        self.frame2_grad_y = frame2_grad[:,:,1]
        self.frame2_grad_xy = np.dstack((self.frame2_grad_x, self.frame2_grad_y))
        self.frame2_grad_mags = np.linalg.norm(self.frame2_grad_xy, axis = 2)
        self.init_flows()

    def init_flows(self):
        U = np.zeros((self.frame1.shape[0], self.frame1.shape[1], 2), dtype = np.float32)
        V = np.zeros((self.frame1.shape[0], self.frame1.shape[1], 2), dtype = np.float32)
        dual_field_P = np.zeros((self.frame1.shape[0], self.frame1.shape[1], 2), dtype = np.float32)
        for iter in range(0, self.num_iter*2):
            if iter%2 == 0:
                U = V - self.theta * self.div(dual_field_P)
                print("U shape: ", U.shape)
                '''must update dual_field_P here'''
                dual_field_P = self.iterate_dual_field_P(dual_field_P, U)
                self.U = U
            else:
                flow_match_scores = self.calc_flow_match_scores(U)
                V_adds = np.zeros(flow_match_scores.shape)
                '''need to add case where if U pushes pixels off the image, set to just Theta(I think)'''
                V_adds[flow_match_scores < -self.smooth_weight*self.theta*self.frame2_grad_mags**2] = \
                        (self.theta * self.smooth_weight * self.frame2_grad_xy)[flow_match_scores < -self.smooth_weight*self.theta*self.frame2_grad_mags**2]
                V_adds[flow_match_scores > self.smooth_weight*self.theta*self.frame2_grad_mags**2] = \
                        (-self.smooth_weight*self.theta*self.frame2_grad_xy)[flow_match_scores > self.smooth_weight*self.theta*self.frame2_grad_mags**2]
                V_adds[np.abs(flow_match_scores) <= self.smooth_weight*self.theta*self.frame2_grad_mags**2] = \
                        (-flow_match_scores*self.frame2_grad_xy/(self.frame2_grad_mags**2))[np.abs(flow_match_scores) <= self.smooth_weight*self.theta*self.frame2_grad_mags**2]

                V = U + V_adds

    def calc_flow_match_scores(self, U):
        return self.approx_frame2_plus_u(U) - self.frame1

    def approx_frame2_plus_u(self, U):
        x_portion = self.frame2_grad_x * U[:,:,0]
        y_portion = self.frame2_grad_y * U[:,:,1]
        return self.frame2 + np.add(x_portion, y_portion)

    def iterate_dual_field_P(self, dual_field_P, U):
        print("self.calculate gradients (U) shape: ", self.calc_gradients_2d(U).shape)
        shared_terms = dual_field_P + (self.time_step/self.theta) * self.calc_gradients_2d(U)
        out_field_terms = np.full(shared_terms, dual_field_P.shape[:2])
        print("out field terms shape: ", out_field_terms.shape)
        out_field_terms[:,:,0] /= np.maximum(1.0, np.abs(shared_terms[:,:,0]))
        out_field_terms[:,:,1] /= np.maximum(1.0, np.abs(shared_terms[:,:,1]))
        return out_field_terms

    def div(self, vector_field):
        field_gradients_dim1 = self.calc_gradients_2d(vector_field[:,:,0])
        field_gradients_dim2 = self.calc_gradients_2d(vector_field[:,:,1])
        return np.dstack((np.sum(field_gradients_dim1, axis = 2), np.sum(field_gradients_dim2, axis = 2)))

    def calc_gradients_2d(self, flows):
        print("flow shape: ", flows.shape)
        flow_x_grad = cv2.filter2D(flows, cv2.CV_32F, self.gradient_kernel_x)
        flow_y_grad = cv2.filter2D(flows, cv2.CV_32F, self.gradient_kernel_y)
        return np.dstack((flow_x_grad, flow_y_grad))
