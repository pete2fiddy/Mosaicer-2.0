import numpy as np
from math import sqrt
import numpy as np

'''returns a kernel that is filled with value wherever the kernel
is and 0.0 everywhere else.
Kernel is bounded so that the circle is centered and touches the
edges of the kernel'''
def get_circle_kernel(diameter, value = 1.0):
    dist_kernel = np.zeros((diameter, diameter))
    radius = diameter/2.0
    center_point = np.array([diameter/2.0, diameter/2.0])
    for i in range(0, dist_kernel.shape[0]):
        for j in range(0, dist_kernel.shape[0]):
            dist_from_center = np.linalg.norm(np.array([j+0.5,i+0.5], dtype = np.float32) - center_point)
            dist_kernel[i,j] = dist_from_center
    kernel = np.zeros((diameter, diameter))
    kernel[dist_kernel < radius] = value
    return kernel
