import numpy as np
import cv2

'''a static class made to convert OpenCV objects into versions of that
object that this project employs (custom)'''

'''
creates a numpy array that corresponds with the x, y, and size of the
keypoint
'''
def keypoint_to_numpy(keypoint):
    return np.array([keypoint.pt[0], keypoint.pt[1], keypoint.size])

'''
creates a keypoint from a numpy array in the same format as
keypoint_to_numpy (mostly so it can be drawn using OpenCV methods)
'''
def numpy_to_keypoint(numpy_keypoint):
    if numpy_keypoint.shape[0] == 2:
        return cv2.KeyPoint(tuple(numpy_keypoint), 1)
    return cv2.KeyPoint(tuple(numpy_keypoint[:2]), numpy_keypoint[2])
