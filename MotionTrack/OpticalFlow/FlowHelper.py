import numpy as np
import cv2

def calc_flow_angle_image(flows):
    flow_angles = np.arctan2(flows[:,:,1], flows[:,:,0])
    flow_angles = np.rad2deg(flow_angles)+180.0
    flow_mags = np.linalg.norm(flows, axis = 2)
    max_mag = np.amax(flow_mags)
    normed_flow_mags = flow_mags/max_mag
    #normed_flow_mags = np.ones(flows.shape[:2])
    flow_hsv_image = np.dstack((flow_angles, normed_flow_mags, np.ones(flows.shape[:2]))).astype(np.float32)
    flow_rgb_image = np.uint8(255 * cv2.cvtColor(flow_hsv_image, cv2.COLOR_HSV2RGB))
    return flow_rgb_image

def calc_flow_vector_image(image, flows, step = 6, max_mag = 30.0, color = (255,0,0), min_flow_mag = 0.01, thickness = 1):
    vec_image = image.copy()
    if len(image.shape) == 2:
        vec_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    flow_mags = np.linalg.norm(flows, axis = 2)
    max_flow_mag = np.amax(flow_mags)

    for x in range(0, flows.shape[1], step):
        for y in range(0, flows.shape[0], step):
            if flow_mags[y,x] > min_flow_mag:
                start_point = np.array([x,y])
                normed_vec = max_mag*flows[y,x]/max_flow_mag
                end_point = start_point + normed_vec
                vec_image = cv2.arrowedLine(vec_image, tuple(start_point.astype(np.int)), tuple(end_point.astype(np.int)), color, thickness = thickness)
    return np.uint8(vec_image)
