import cv2
import numpy as np

def median_blur(U, func_args):
    k_size = func_args["k_size"]
    num_iter = func_args["num_iter"]
    U_new = U.copy()
    for i in range(0, num_iter):
        U_blurred_channel1 = cv2.medianBlur(U_new[:,:,0], k_size)
        U_blurred_channel2 = cv2.medianBlur(U_new[:,:,1], k_size)
        U_new = np.dstack((U_blurred_channel1, U_blurred_channel2))
    return U_new

def mean_blur(U, func_args):
    k_size = (func_args["k_size"], func_args["k_size"])

    U_blurred_channel1 = cv2.blur(U[:,:,0], k_size)
    U_blurred_channel2 = cv2.blur(U[:,:,1], k_size)
    return np.dstack((U_blurred_channel1, U_blurred_channel2))
