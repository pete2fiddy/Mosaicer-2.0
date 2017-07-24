import numpy as np
import cv2


class TwoFrameHornSchunck:

    '''for weighted average of u and v, take a 3x3 neighborhood around
    (x,y). Weight all instances where (x_index != x and y_index != y) as having
    1/12 weight, and all instances where(x_index != x and y_index = y) or
    (x_index = x and y_index != y) as having weight of 1/6. Do not count u or v
    at (x,y) in the average'''

    def __init__(self, frame1, frame2, smooth_weight):
        assert (frame1.shape == frame2.shape and len(frame1) == 2)

        self.frame1 = frame1
        self.frame2 = frame2
        self.smooth_weight = smooth_weight
        self.init_flows()

    def init_flows(self):
        self.x_flows = np.zeros(self.frame1.shape)
        self.y_flows = np.zeros(self.frame1.shape)

        grad_x_frame2 = cv2.Scharr(self.frame2, cv2.CV_32F, 1, 0)
        grad_y_frame2 = cv2.Scharr(self.frame2, cv2.CV_32F, 0, 1)
        partial_frame2_partial_time = self.frame2 - self.frame1
        
