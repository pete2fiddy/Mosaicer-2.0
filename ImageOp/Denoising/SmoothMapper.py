import numpy as np
import cv2
from PIL import Image

def smooth_map_image(image, kernel_size):
    '''does not look in an area a radius away on the tangent plane of the
    function, as this seems very expensive and annoying to implement'''
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    grad_mags = np.linalg.norm(np.dstack((grad_x, grad_y)), axis = 2)
    unit_grad_x = grad_x/grad_mags
    unit_grad_y = grad_y/grad_mags
    unit_grad_x[grad_mags == 0] = 0
    unit_grad_y[grad_mags == 0] = 0
    smooth_unit_grad_x = cv2.blur(unit_grad_x, (kernel_size, kernel_size))
    smooth_unit_grad_y = cv2.blur(unit_grad_y, (kernel_size, kernel_size))
    '''uses a square weight kernel but maybe could use a circular one instead'''
    weight_kernel = get_weight_kernel(kernel_size)
    weighted_smooth_grad_x = cv2.filter2D(smooth_unit_grad_x, cv2.CV_32F, weight_kernel)
    weighted_smooth_grad_y = cv2.filter2D(smooth_unit_grad_y, cv2.CV_32F, weight_kernel)
    smooth_responses_x_comp = unit_grad_x*weighted_smooth_grad_x
    smooth_responses_y_comp = unit_grad_y*weighted_smooth_grad_y
    smooth_responses = smooth_responses_x_comp + smooth_responses_y_comp
    '''test results against hard coded version'''
    return smooth_responses


'''is slow and basically for testing'''
def smooth_map_image2(image, kernel_size):
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    mag_grads = np.linalg.norm(np.dstack((grad_x, grad_y)), axis = 2)
    unit_grad_x = grad_x/mag_grads
    unit_grad_y = grad_y/mag_grads
    unit_grad_x[mag_grads == 0] = 0
    unit_grad_y[mag_grads == 0] = 0
    kernel_margin = (kernel_size - 1)//2
    weight_kernel = get_weight_kernel(kernel_size)
    out_image = np.zeros(image.shape, dtype = np.float32)
    for x in range(kernel_margin, image.shape[1]-kernel_margin):
        for y in range(kernel_margin, image.shape[0]-kernel_margin):
            base_grad_x = unit_grad_x[y,x]
            base_grad_y = unit_grad_y[y,x]

            sum = 0
            for i in range(x-kernel_margin, x+kernel_margin+1):
                for j in range(y-kernel_margin, y+kernel_margin+1):
                    sum += base_grad_x*unit_grad_x[j,i]*weight_kernel[y+kernel_margin-j, x+kernel_margin-i]
                    sum += base_grad_y*unit_grad_y[j,i]*weight_kernel[y+kernel_margin-j, x+kernel_margin-i]
            out_image[y,x] = sum
    return out_image




def get_weight_kernel(kernel_size):
    kernel = np.zeros((kernel_size, kernel_size), dtype = np.float32)
    middle_pixel = np.array(((kernel_size-1)/2, (kernel_size-1)/2))
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            kernel[i,j] = np.linalg.norm(np.array((i,j)) - middle_pixel)
    max_dist = np.amax(kernel)
    kernel /= max_dist
    kernel = (1.0 - np.abs(kernel)**3)**3
    return kernel
