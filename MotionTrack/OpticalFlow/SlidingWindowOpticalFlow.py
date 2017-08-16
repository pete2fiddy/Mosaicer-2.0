import numpy as np
import cv2
from PIL import Image
import MotionTrack.OpticalFlow.FlowHelper as FlowHelper
from math import sqrt, pi
class SlidingWindowOpticalFlow:

    '''all dual frame optical flows passed to this algorithm must be instantiated
    in the form: Frame1, Frame2, **params(NamedArgs)'''
    '''frames: The images from which to construct optical flows
    frame_window: The number of frames to use per single set of flows. For cases
    where the number of frames required exceeds the bounds of the frames passed
    (<0 or greater than frames.shape[0]), as many frames as possible are used.
    flow_class_and_params: A ClassArgs object containing the parameters for
    dual frame optical flow and the class to be used to construct the flows'''
    '''
    You can't just average the flows, because the locations shift from frame to frame...
    You need to track pixels across frames using the velocity, or somehow find the change
    in location of velocity between frames in order to rectify...
    '''
    TEST_DUO_FLOW_SAVE_PATH = "C:/Users/Peter/Desktop/DZYNE/Git Repos/Mosaicer 2.0/Mosaicer 2.0/Test/Saved Flows/Backyard.npy"
    def __init__(self, frames, frame_window, flow_class_and_params):
        self.frames = frames
        self.frame_window = frame_window
        self.flow_class_and_params = flow_class_and_params
        #self.init_dual_frame_flows()
        self.init_dual_frame_flows_from_path()
        self.init_flows()

    def init_dual_frame_flows(self):
        self.duo_flows = np.zeros((self.frames.shape[0]-1,) + self.frames.shape[1:] + (2,), dtype = np.float32)
        for i in range(0, self.duo_flows.shape[0]):
            frame1 = self.frames[i]
            frame2 = self.frames[i+1]
            op_flow = self.flow_class_and_params.class_type(frame1, frame2, self.flow_class_and_params)
            self.duo_flows[i] = op_flow.flows
        np.save(SlidingWindowOpticalFlow.TEST_DUO_FLOW_SAVE_PATH, self.duo_flows)

    def init_dual_frame_flows_from_path(self):
        self.duo_flows = np.load(SlidingWindowOpticalFlow.TEST_DUO_FLOW_SAVE_PATH)
        self.duo_flows = self.duo_flows[:7,:,:,:]


    def init_flows(self):
        self.frame_flows = np.zeros((self.frames.shape[0]-1, ) + self.frames.shape[1:] + (2,), dtype = np.float32)
        '''bulk of optical flows already calculated when switching frames, as frame_window - 1 will
        be relevant in calcuating the flows of the next frame.'''
        for frame_num in range(0, self.frame_flows.shape[0]):
            self.frame_flows[frame_num] = self.get_flows_at_index2(frame_num)

    def get_flows_at_index(self, index):
        flow_subset = self.get_windowed_flow_subset(index)
        warped_flow_sums = flow_subset[flow_subset.shape[0]-1]#np.zeros(flow_subset.shape[1:], dtype = np.float32)
        for i in range(flow_subset.shape[0]-1, 1, -1):
            x_new_warped_flow_sums = FlowHelper.warp_image_with_flows(warped_flow_sums[:,:,0], flow_subset[i])
            y_new_warped_flow_sums = FlowHelper.warp_image_with_flows(warped_flow_sums[:,:,1], flow_subset[i])
            warped_flow_sums = np.dstack((x_new_warped_flow_sums, y_new_warped_flow_sums)) + flow_subset[i-1]
        '''would be ideal if this would divide each pixel by the number of times its flow
        actually contributes to the average. Occlusions can cause pixels to drop out, etc,
        that this would count in the mean despite them not being there'''
        warped_flow_sums /= float(flow_subset.shape[0])
        #return np.mean(flow_subset, axis = 0)
        '''could try to use baye's theorem to estimate the probability of the pixel
        being the flow it is (low confidence = lots of contestion for flow of the pixel)
        '''
        return warped_flow_sums


    def get_rectified_flows(self, index):
        flow_subset = self.get_windowed_flow_subset(index)
        rectified_flows = flow_subset.copy()
        for i in range(0, rectified_flows.shape[0]):
            for j in range(i, 0,-1):
                x_flow_warped = FlowHelper.warp_image_with_flows(rectified_flows[i,:,:,0], flow_subset[j])
                y_flow_warped = FlowHelper.warp_image_with_flows(rectified_flows[i,:,:,1], flow_subset[j])
                rectified_flows[i] = np.dstack((x_flow_warped, y_flow_warped))
            cv2.waitKey(1)
        return rectified_flows


    def get_flows_at_index2(self, index):
        rectified_flows = self.get_rectified_flows(index)
        '''could use thresholding rather than normal distribution? Or threshold
        based on the probability that the distribution at pixel(or neighborhood)
        would would go outside of the angle threshold bounds'''
        angle_variance = 10.0**2

        prob_image = self.calc_outlier_probabilities(rectified_flows, angle_variance)
        normed_prob_image = prob_image/prob_image.max()
        print("max normed prob image: ", normed_prob_image.max())
        print("min normed prob image: ", normed_prob_image.min())
        print("max prob image: ", prob_image.max())
        to_view_prob_image = np.uint8(255*prob_image/np.amax(prob_image))
        #Image.fromarray(to_view_prob_image).show()
        out_flows = np.mean(rectified_flows, axis = 0)
        probability_thresh = 0.6
        out_flows[:,:][normed_prob_image < probability_thresh] = rectified_flows[0][:,:][normed_prob_image < probability_thresh]
        return out_flows

    def calc_outlier_probabilities(self, rect_flows, variance):
        prob_image = np.ones(rect_flows.shape[1:3], np.float32)
        for i in range(1, rect_flows.shape[0]):
            likelihoods = self.get_likelihood_image(rect_flows[i-1], rect_flows[i], variance)
            priors = self.get_gaussian_mixture_probabilities_image(rect_flows[i], rect_flows[:i], variance)
            evidences = self.get_gaussian_mixture_probabilities_image(rect_flows[i-1], rect_flows[:i], variance)
            prob_image *= (likelihoods*priors)/evidences

        return prob_image

    def get_angles_from_vec(self, vecs, base_vec):
        base_vec_mag = np.linalg.norm(base_vec)
        vecs_mags = np.linalg.norm(vecs, axis = 1)
        dot_prods = (base_vec[0]*vecs[:,0]) + (base_vec[1]*vecs[:,1])
        angles_between = np.arccos(dot_prods/(base_vec_mag*vecs_mags))
        angles_between = np.rad2deg(angles_between)%180
        angles_between[np.isnan(angles_between)] = 0
        return angles_between

    '''calculates the likelihood of the flows to fit their mean using the
    normal distribution'''
    def get_likelihood_image(self, flows, flow_means, variance):
        likelihood_image = self.get_normal_distribution_image(flows, flow_means, variance)
        return likelihood_image

    '''variance in terms of angle between the flow means and input flows'''
    def get_normal_distribution_image(self, input_flows, flow_means, variance):
        dot_prods = (flow_means[:,:,0]*input_flows[:,:,0]) + (flow_means[:,:,1]*input_flows[:,:,1])

        input_flow_mags = np.linalg.norm(input_flows, axis = 2)
        flow_means_mags = np.linalg.norm(flow_means, axis = 2)
        angles_between = np.arccos(dot_prods/(input_flow_mags*flow_means_mags))

        angles_between = np.rad2deg(angles_between)%180
        #angles_between[(flow_means_mags==0)] = 0
        #angles_between[(input_flow_mags==0)] = 0
        #print("angles between: ", angles_between[:5, :5])
        #print("num instances of NaN in angles between: ", np.where(np.isnan(angles_between)))
        angles_between[np.isnan(angles_between)] = 0
        exps = np.exp(np.ones(angles_between.shape, dtype = np.float32)*-(angles_between/(2.0*variance)))
        probabilities = (1.0/(sqrt(2.0*pi*variance))) * exps
        return probabilities

    def get_gaussian_mixture_probabilities_image(self, input_flows, param_flows, variance):
        gaussian_mix_image = np.zeros(input_flows.shape[:2], dtype = np.float32)
        for i in range(0, param_flows.shape[0]):
            gaussian_mix_image += self.get_normal_distribution_image(input_flows, param_flows[i], variance)
        gaussian_mix_image /= float(param_flows.shape[0])
        return gaussian_mix_image

    def get_windowed_flow_subset(self, index):
        end_frame_index = index + self.frame_window
        if end_frame_index >= self.duo_flows.shape[0]:
            end_frame_index = self.duo_flows.shape[0]
        window_flows = self.duo_flows[index : end_frame_index, :, :]
        return window_flows

    '''
    def get_windowed_frame_subset(self, index):
        end_frame_index = index + self.frame_window
        if end_frame_index >= self.frames.shape[0]:
            end_frame_index = self.frames.shape[0]-1-index
        window_frames = self.frames[index : end_frame_index, :, :]
        return window_frames

    def get_subset_flows(self, window_frames):
        window_flows = np.zeros((window_frames.shape[0]-1,) + window_frames.shape[1:] + (2,), dtype = np.float32)
        for i in range(0, window_frames.shape[0]-1):
            frame1 = window_frames[i]
            frame2 = window_frames[i+1]
            op_flow = self.flow_class_and_params.class_type(frame1, frame2, self.flow_class_and_params)
            window_flows[i] = op_flow.flows
        return window_flows
    '''
