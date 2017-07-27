import numpy as np
import cv2
from PIL import Image
import Matrix.MatrixHelper as MatrixHelper
from math import atan2
import Angle.AngleConverter as AngleConverter


class MultiResLucasKanade2:

    def __init__(self, frames, num_pyramid_layers, scale_factor = 2):
        self.frames = frames
        self.scale_factor = scale_factor
        self.num_pyramid_layers = num_pyramid_layers
        self.calculate_flows()

    def calculate_flows(self):
        '''function doesn't "save" any flows calculated right now -- just shows the "flow
        image" of the flows it calculates'''
        for frame_index in range(1, self.frames.shape[0]):
            base_frame1 = self.frames[frame_index - 1]
            base_frame2 = self.frames[frame_index]
            frame1_pyr = self.build_pyramid(base_frame1)
            frame2_pyr = self.build_pyramid(base_frame2)
            flow_x_pyr = self.build_pyramid(np.zeros(base_frame1.shape))
            flow_y_pyr = self.build_pyramid(np.zeros(base_frame2.shape))

            flow_x_pyr[0], flow_y_pyr[0] = self.get_flows_between_frames2(frame1_pyr[0], frame2_pyr[0])
            upscaled_previous_flow_x = cv2.resize(flow_x_pyr[0], frame1_pyr[1].shape[::-1])*self.scale_factor
            upscaled_previous_flow_y = cv2.resize(flow_y_pyr[0], frame1_pyr[1].shape[::-1])*self.scale_factor

            flow_x_sum = cv2.resize(upscaled_previous_flow_x, base_frame1.shape[::-1], cv2.INTER_LINEAR)
            flow_y_sum = cv2.resize(upscaled_previous_flow_y, base_frame1.shape[::-1], cv2.INTER_LINEAR)

            for pyr_index in range(1, len(frame1_pyr)):
                frame2 = frame2_pyr[pyr_index]
                frame1 = self.flow_interpolate_image(frame1_pyr[pyr_index], upscaled_previous_flow_x, upscaled_previous_flow_y)

                flow_x_pyr[pyr_index], flow_y_pyr[pyr_index] = self.get_flows_between_frames2(frame1, frame2)
                if pyr_index < len(frame1_pyr) - 1:
                    layer_scale_factor = (self.scale_factor)**(len(frame1_pyr) - 1 - pyr_index)
                    upscaled_x_pyr = cv2.resize(flow_x_pyr[pyr_index], base_frame1.shape[::-1], cv2.INTER_LINEAR) * layer_scale_factor
                    upscaled_y_pyr = cv2.resize(flow_y_pyr[pyr_index], base_frame1.shape[::-1], cv2.INTER_LINEAR) * layer_scale_factor
                    flow_x_sum += upscaled_x_pyr
                    flow_y_sum += upscaled_y_pyr

                    downscale_to_next_layer_factor = 1.0/((self.scale_factor)**(len(frame1_pyr) - 2 - pyr_index))

                    upscaled_previous_flow_x = cv2.resize(flow_x_sum, frame1_pyr[pyr_index+1].shape[::-1], cv2.INTER_LINEAR)*downscale_to_next_layer_factor
                    upscaled_previous_flow_y = cv2.resize(flow_y_sum, frame1_pyr[pyr_index+1].shape[::-1], cv2.INTER_LINEAR)*downscale_to_next_layer_factor

            full_flow_image = self.draw_flow_arrows(base_frame1, flow_x_sum, flow_y_sum, color = (255,0,0))#self.get_flow_image(flow_x_sum, flow_y_sum)
            Image.fromarray(full_flow_image).show()
            Image.fromarray(self.get_flow_image(flow_x_sum, flow_y_sum)).show()
            Image.fromarray(self.flow_interpolate_image(frame1, flow_x_sum, flow_y_sum)).show()


    def build_pyramid(self, image):
        pyramid = [image]
        for i in range(0, self.num_pyramid_layers - 1):
            prev_pyr_image = pyramid[len(pyramid)-1]
            #append_pyr_image = prev_pyr_image
            append_pyr_image = cv2.GaussianBlur(prev_pyr_image, (5,5), 1.0)
            append_pyr_image = cv2.resize(append_pyr_image, (int((prev_pyr_image.shape[1])/self.scale_factor), int((prev_pyr_image.shape[0])/self.scale_factor)))
            pyramid.append(append_pyr_image)
        pyramid = list(reversed(pyramid))
        return pyramid

    '''interpolates the first frame to fit the second'''
    def flow_interpolate_image(self, image, x_flows, y_flows):
        x_indices_mat = np.array([np.arange(image.shape[1]) for j in range(0, image.shape[0])])
        y_indices_mat = np.array([np.arange(image.shape[0]) for j in range(0, image.shape[1])]).T

        x_map = (x_indices_mat - x_flows).astype(np.float32)
        y_map = (y_indices_mat - y_flows).astype(np.float32)
        warp_image = cv2.remap(image, x_map, y_map, cv2.INTER_LINEAR)
        return warp_image

    def get_flows_between_frames(self, frame1, frame2):

        WINDOW_SIZE = 3
        WINDOW_MARGIN = (WINDOW_SIZE - 1)//2

        x_flows = np.zeros((frame1.shape[0], frame1.shape[1]))
        y_flows = np.zeros((frame1.shape[0], frame1.shape[1]))

        frame2_grad_x = cv2.Scharr(frame2, cv2.CV_32F, 1, 0)
        frame2_grad_y = cv2.Scharr(frame2, cv2.CV_32F, 0, 1)
        partial_frame2_partial_time = frame2.astype(np.float32) - frame1.astype(np.float32)

        for x in range(WINDOW_MARGIN, x_flows.shape[1] - WINDOW_MARGIN):
            for y in range(WINDOW_MARGIN, x_flows.shape[0] - WINDOW_MARGIN):
                frame2_gradx_window = frame2_grad_x[y-WINDOW_MARGIN : y+WINDOW_MARGIN+1, x-WINDOW_MARGIN : x+WINDOW_MARGIN+1].flatten()
                frame2_grady_window = frame2_grad_y[y-WINDOW_MARGIN : y+WINDOW_MARGIN+1, x-WINDOW_MARGIN : x+WINDOW_MARGIN+1].flatten()
                frame2_gradients_window = np.zeros((frame2_gradx_window.shape[0], 2))
                frame2_gradients_window[:, 0] = frame2_gradx_window
                frame2_gradients_window[:, 1] = frame2_grady_window
                frame2_partial_time_window = partial_frame2_partial_time[y-WINDOW_MARGIN : y+WINDOW_MARGIN+1, x-WINDOW_MARGIN : x+WINDOW_MARGIN+1].flatten()

                covar_mat = frame2_gradients_window.T.dot(frame2_gradients_window)

                if MatrixHelper.is_invertible(covar_mat):
                    mat_product = frame2_gradients_window.T.dot(-frame2_partial_time_window)
                    flow_steps = np.linalg.inv(covar_mat).dot(mat_product)
                    x_flows[y, x] = flow_steps[0]
                    y_flows[y, x] = flow_steps[1]
        x_flows = cv2.resize(x_flows, frame1.shape[::-1], cv2.INTER_LINEAR)
        y_flows = cv2.resize(y_flows, frame1.shape[::-1], cv2.INTER_LINEAR)
        return x_flows, y_flows

    def get_flows_between_frames2(self, frame1, frame2):
        WINDOW_SIZE = 5
        WINDOW_MARGIN = (WINDOW_SIZE - 1)//2

        #weight_kernel = cv2.getGaussianKernel((WINDOW_SIZE, WINDOW_SIZE), 1.0, cv2.CV_32F)

        x_flows = np.zeros(frame1.shape)
        y_flows = np.zeros(frame1.shape)

        frame2_grad_x = cv2.Scharr(frame2, cv2.CV_32F, 1, 0)
        frame2_grad_y = cv2.Scharr(frame2, cv2.CV_32F, 0, 1)
        frame2_grad_x2 = frame2_grad_x * frame2_grad_x
        frame2_grad_xy = frame2_grad_x * frame2_grad_y
        frame2_grad_y2 = frame2_grad_y * frame2_grad_y
        partial_frame2_partial_time = frame2-frame1

        frame2_grad_x_partial_time = frame2_grad_x * partial_frame2_partial_time
        frame2_grad_y_partial_time = frame2_grad_y * partial_frame2_partial_time

        sum_filter = np.ones((WINDOW_SIZE, WINDOW_SIZE))#cv2.getGaussianKernel(WINDOW_SIZE, 1.0, cv2.CV_32F)#

        sum_frame2_grad_x2 = cv2.filter2D(frame2_grad_x2, cv2.CV_32F, sum_filter)
        sum_frame2_grad_xy = cv2.filter2D(frame2_grad_xy, cv2.CV_32F, sum_filter)
        sum_frame2_grad_y2 = cv2.filter2D(frame2_grad_y2, cv2.CV_32F, sum_filter)
        sum_frame2_grad_x_partial_time = cv2.filter2D(frame2_grad_x_partial_time, cv2.CV_32F, sum_filter)
        sum_frame2_grad_y_partial_time = cv2.filter2D(frame2_grad_y_partial_time, cv2.CV_32F, sum_filter)

        second_moments = np.zeros((frame1.shape[0], frame1.shape[1], 2, 2))
        second_moments[:,:,0,0] = sum_frame2_grad_x2
        second_moments[:,:,0,1] = sum_frame2_grad_xy
        second_moments[:,:,1,0] = sum_frame2_grad_xy
        second_moments[:,:,1,1] = sum_frame2_grad_y2

        flat_second_moments = second_moments.reshape(second_moments.shape[0] * second_moments.shape[1], 2, 2)
        #flat_second_moments_inv = np.zeros(flat_second_moments.shape)

        target_vectors = np.zeros((frame1.shape[0], frame1.shape[1], 2))
        target_vectors[:,:,0] = -sum_frame2_grad_x_partial_time
        target_vectors[:,:,1] = -sum_frame2_grad_y_partial_time

        flat_target_vectors = target_vectors.reshape((target_vectors.shape[0] * target_vectors.shape[1], 2))
        flat_all_flows = np.zeros(flat_target_vectors.shape)

        for i in range(0, flat_second_moments.shape[0]):
            try:
                flat_all_flows[i] = np.linalg.solve(flat_second_moments[i], flat_target_vectors[i])
            except:
                '''matrix was not invertible'''
        print("flat all flows: ", flat_all_flows)
        all_flows = flat_all_flows.reshape((frame1.shape[0], frame1.shape[1], -1))
        print("all flows shape: ", all_flows.shape)
        x_flows = all_flows[:,:,0]
        y_flows = all_flows[:,:,1]
        return x_flows, y_flows


    def draw_flow_arrows(self, frame, x_flows, y_flows, step = 10, vec_mag = 10, color = (255,0,0)):
        vector_image = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        for x in range(0, x_flows.shape[1], step):
            for y in range(0, x_flows.shape[0], step):
                xy_point = np.array([x,y])
                flow_vec = np.array([x_flows[y,x], y_flows[y,x]])
                flow_vec /= np.linalg.norm(flow_vec)
                end_point = (xy_point - flow_vec*vec_mag).astype(np.int)
                vector_image = cv2.arrowedLine(vector_image, tuple(xy_point), tuple(end_point), color)
        return vector_image

    def get_flow_image(self, x_flows, y_flows):
        flow_stacks = np.dstack((x_flows, y_flows))

        hsv_angles = np.arctan2(y_flows, x_flows)
        hsv_angles = np.rad2deg(hsv_angles)%360

        hsv_saturations = np.ones(x_flows.shape)#(np.linalg.norm(flow_stacks, axis = 2))
        hsv_saturations /= np.amax(hsv_saturations)

        flow_hsv_image = np.dstack((hsv_angles, hsv_saturations, np.ones((x_flows.shape[0], x_flows.shape[1])))).astype(np.float32)
        flow_rgb_image = np.uint8(255 * cv2.cvtColor(flow_hsv_image, cv2.COLOR_HSV2RGB))
        return flow_rgb_image


class MultiResLucasKanade:
    WINDOW_SIZE = 3
    WINDOW_MARGIN = (WINDOW_SIZE - 1)//2
    def __init__(self, frames, num_pyramid_layers):
        self.frames = frames
        self.num_pyramid_layers = num_pyramid_layers
        self.calculate_flows()

    def calculate_flows(self):
        '''
        holds the flows for the image in frames i+1 to frames i
        '''
        self.frame_flows = np.zeros(self.frames[:self.frames.shape[0]-1, :, :].shape)
        for i in range(1, self.frames.shape[0]):

            frame1 = self.frames[i-1]
            frame2 = self.frames[i]
            frame_x_flows, frame_y_flows = self.calculate_image_flows(frame1, frame2, 0)

            x_flow_indices = np.array([np.arange(frame_x_flows.shape[1]) for j in range(0, frame_x_flows.shape[0])])
            y_flow_indices = np.array([np.arange(frame_y_flows.shape[0]) for j in range(0, frame_y_flows.shape[1])]).T

            remap_x_from_flows = x_flow_indices - frame_x_flows
            remap_y_from_flows = y_flow_indices - frame_y_flows

            warp_frame1 = cv2.remap(frame2, remap_x_from_flows.astype(np.float32), remap_y_from_flows.astype(np.float32), cv2.INTER_LINEAR)
            Image.fromarray(warp_frame1).show()
            for flow_iter in range(0, 18):
                step_x_flows, step_y_flows = self.calculate_image_flows(warp_frame1, frame2, 0)

                print("mean step x flows: ", np.average(np.abs(step_x_flows)))
                print("mean step y flows: ", np.average(np.abs(step_y_flows)))
                #step_x_flows = step_x_flows.astype(np.int)
                #step_y_flows = step_y_flows.astype(np.int)

                print("max step flow: ", np.amax(np.abs(step_x_flows)))

                frame_x_flows += step_x_flows
                frame_y_flows += step_y_flows


                remap_x_from_flows = x_flow_indices - frame_x_flows
                remap_y_from_flows = y_flow_indices - frame_y_flows

                rgb_flow_image = self.get_flow_image(frame1, frame_x_flows, frame_y_flows)
                cv2.imshow("RGB flow image: ", rgb_flow_image)



                warp_frame1 = cv2.remap(frame2, remap_x_from_flows.astype(np.float32), remap_y_from_flows.astype(np.float32), cv2.INTER_LINEAR)
                cv2.imshow("warp frame 1: ", warp_frame1)
                cv2.waitKey(1)
            print("frame x flows: ", frame_x_flows)
            #Image.fromarray(np.uint8(255*(frame_x_flows - np.amin(frame_x_flows))/(np.amax(frame_x_flows) - np.amin(frame_x_flows)))).show()
            #Image.fromarray(np.uint8(255*(frame_y_flows - np.amin(frame_y_flows))/(np.amax(frame_y_flows) - np.amin(frame_y_flows)))).show()
            Image.fromarray(warp_frame1).show()
            Image.fromarray(self.get_flow_image(frame1, frame_x_flows, frame_y_flows)).show()
            #flow_stacks = np.dstack((frame_x_flows, np.zeros(frame_x_flows.shape), frame_y_flows))
            '''
            flow_color_image = np.zeros((frame1.shape[0], frame1.shape[1], 3))
            for x in range(0, flow_color_image.shape[1]):
                for y in range(0, flow_color_image.shape[0]):
                    flow_at_xy = np.array([frame_x_flows[y,x], frame_y_flows[y,x]])
                    flow_color_image[y,x] = np.array([atan2(flow_at_xy[1], flow_at_xy[0]), np.linalg.norm(flow_at_xy), 255])

            flow_color_image[:,:,1] /= np.amax(flow_color_image[:,:,1])/255.0
            flow_color_image = flow_color_image.astype(np.float32)

            rgb_flow_color_image = cv2.cvtColor(flow_color_image, cv2.COLOR_HSV2RGB)
            Image.fromarray(np.uint8(rgb_flow_color_image)).show()
            #Image.fromarray(np.uint8(255*(flow_stacks - np.amin(flow_stacks))/(np.amax(flow_stacks) - np.amin(flow_stacks)))).show()
            '''
            '''frame1_pyramid = [frame1]
            frame2_pyramid = [frame2]
            for j in range(0, self.num_pyramid_layers-1):
                frame1_pyramid.append(cv2.pyrDown(frame1_pyramid[len(frame1_pyramid)-1]))
                frame2_pyramid.append(cv2.pyrDown(frame2_pyramid[len(frame2_pyramid)-1]))
            frame1_pyramid = list(reversed(frame1_pyramid))
            frame2_pyramid = list(reversed(frame2_pyramid))

            total_flows = np.array(self.frame_flows[i-1].shape)
            displaced_frame1 = frame1_pyramid[0]
            for pyramid_index in range(0, len(frame2_pyramid)):
                x_flows, y_flows = self.calculate_image_flows(displaced_frame1, frame2_pyramid[pyramid_index], pyramid_index)
                #Image.fromarray(np.uint8(255*(x_flows - np.amin(x_flows))/(np.amax(x_flows) - np.amin(x_flows)))).show()
                print("max x flow: ", np.amax(x_flows))
                print("min x flow: ", np.amin(x_flows))
                print("max y flow: ", np.amax(y_flows))
                print("min y flow: ", np.amin(y_flows))
                print("-------------------------------------")
                if pyramid_index + 1 < len(frame1_pyramid):
                    displaced_frame1 = self.warp_first_frame_with_flows(frame1_pyramid[pyramid_index+1], frame2_pyramid[pyramid_index], x_flows*2, y_flows*2, pyramid_index)
                    Image.fromarray(displaced_frame1).show()
                #print("x flows: ", x_flows)
                #print("y flows: ", y_flows)'''

    def get_flow_image(self, frame1, frame_x_flows, frame_y_flows):
        flow_color_image = np.zeros((frame1.shape[0], frame1.shape[1], 3))
        for x in range(0, flow_color_image.shape[1]):
            for y in range(0, flow_color_image.shape[0]):
                flow_at_xy = np.array([frame_x_flows[y,x], frame_y_flows[y,x]])
                #print("HSV Angle: ", AngleConverter.truncate_degrees(AngleConverter.radians2deg(atan2(flow_at_xy[1], flow_at_xy[0])))/2.0)
                flow_color_image[y,x] = np.array([AngleConverter.truncate_degrees(AngleConverter.radians2deg(atan2(flow_at_xy[1], flow_at_xy[0]))), np.linalg.norm(flow_at_xy), 1])

        flow_color_image[:,:,1] /= np.amax(flow_color_image[:,:,1])
        #flow_color_image[:,:,1] *= 255
        flow_color_image = flow_color_image.astype(np.float32)

        rgb_flow_color_image = 255*cv2.cvtColor(flow_color_image, cv2.COLOR_HSV2RGB)
        print("max of rgb flow color image: ", np.amax(rgb_flow_color_image))
        return np.uint8(rgb_flow_color_image)

    def calculate_image_flows(self, frame1, frame2, pyramid_index):
        step = 2**pyramid_index
        x_flows = np.zeros((frame1.shape[0]//step, frame1.shape[1]//step))
        y_flows = np.zeros((frame1.shape[0]//step, frame1.shape[1]//step))

        frame2_grad_x = cv2.Scharr(frame2, cv2.CV_32F, 1, 0)
        frame2_grad_y = cv2.Scharr(frame2, cv2.CV_32F, 0, 1)
        partial_frame2_partial_time = frame2 - frame1
        for x in range(MultiResLucasKanade.WINDOW_MARGIN, x_flows.shape[1] - MultiResLucasKanade.WINDOW_MARGIN, 2**pyramid_index):
            for y in range(MultiResLucasKanade.WINDOW_MARGIN, x_flows.shape[0] - MultiResLucasKanade.WINDOW_MARGIN, 2**pyramid_index):
                frame2_window = frame2[y - MultiResLucasKanade.WINDOW_MARGIN : y + MultiResLucasKanade.WINDOW_MARGIN + 1, x-MultiResLucasKanade.WINDOW_MARGIN: x + MultiResLucasKanade.WINDOW_MARGIN + 1].flatten()
                frame2_grad_x_window = frame2_grad_x[y - MultiResLucasKanade.WINDOW_MARGIN : y + MultiResLucasKanade.WINDOW_MARGIN + 1, x-MultiResLucasKanade.WINDOW_MARGIN: x + MultiResLucasKanade.WINDOW_MARGIN + 1].flatten()
                frame2_grad_y_window = frame2_grad_y[y - MultiResLucasKanade.WINDOW_MARGIN : y + MultiResLucasKanade.WINDOW_MARGIN + 1, x-MultiResLucasKanade.WINDOW_MARGIN: x + MultiResLucasKanade.WINDOW_MARGIN + 1].flatten()
                frame2_partial_time_window = partial_frame2_partial_time[y - MultiResLucasKanade.WINDOW_MARGIN : y + MultiResLucasKanade.WINDOW_MARGIN + 1, x-MultiResLucasKanade.WINDOW_MARGIN: x + MultiResLucasKanade.WINDOW_MARGIN + 1].flatten()
                '''covar_mat = np.array([[np.sum(frame2_grad_x_window * frame2_grad_x_window), np.sum(frame2_grad_x_window * frame2_grad_y_window)],
                                     [np.sum(frame2_grad_x_window * frame2_grad_y_window), np.sum(frame2_grad_y_window * frame2_grad_y_window)]])'''

                frame2_grad_xy_window = np.zeros((frame2_grad_x_window.shape[0], 2))
                frame2_grad_xy_window[:, 0] = frame2_grad_x_window
                frame2_grad_xy_window[:, 1] = frame2_grad_y_window

                covar_mat = frame2_grad_xy_window.T.dot(frame2_grad_xy_window)

                if MatrixHelper.is_invertible(covar_mat):
                    mat_product = -np.array([np.sum(frame2_grad_x_window * frame2_partial_time_window), np.sum(frame2_grad_y_window * frame2_partial_time_window)])
                    #print("mat product 1: ", mat_product)

                    mat_product = frame2_grad_xy_window.T.dot(-frame2_partial_time_window)
                    #print("mat product 2: ", mat_product)

                    flow_steps = np.linalg.inv(covar_mat).dot(mat_product)
                    x_flows[y//step,x//step] = flow_steps[0]
                    y_flows[y//step,x//step] = flow_steps[1]

        return x_flows, y_flows

    '''
    def warp_first_frame_with_flows(self, frame1, frame2, x_flows, y_flows, pyramid_index):
        step = 2**(pyramid_index+1)
        out_image = np.zeros(frame1.shape)
        rescale_factor = int(frame1.shape[0]/frame2.shape[0])

        for x_flow_index in range(0, x_flows.shape[1]):
            for y_flow_index in range(0, x_flows.shape[0]):
                patch_flow = np.array([x_flows[y_flow_index, x_flow_index], y_flows[y_flow_index, x_flow_index]]).astype(np.int)
                #print("patch flow: ", patch_flow)
                frame2_flow_displaced_patch = frame2[(y_flow_index*step)+patch_flow[1] : ((y_flow_index+1)*step)+patch_flow[1], (x_flow_index*step)+patch_flow[0] : ((x_flow_index+1)*step)+patch_flow[0]]
                #print("patch bounds: ", ((y_flow_index*step)+patch_flow[1], ((y_flow_index+1)*step)+patch_flow[1], (x_flow_index*step)+patch_flow[0], ((y_flow_index+1)*step)+patch_flow[0]))
                #print("frame1_flow_displaced_patch: ", frame2_flow_displaced_patch)
                #try:
                out_image[rescale_factor * y_flow_index*step : rescale_factor * (y_flow_index+1)*step, rescale_factor * x_flow_index*step : rescale_factor * (x_flow_index+1)*step] = frame2_flow_displaced_patch
                #except:
                #    print("tried to do impossible displacement")
        return out_image'''
