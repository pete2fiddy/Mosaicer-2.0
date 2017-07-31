import numpy as np
import cv2

def calc_flow_angle_image(flows, use_mag = True):
    flow_angles = np.arctan2(flows[:,:,1], flows[:,:,0])
    flow_angles = np.rad2deg(flow_angles)+180.0
    flow_mags = np.linalg.norm(flows, axis = 2)
    max_mag = np.amax(flow_mags)
    normed_flow_mags = np.ones(flows.shape[:2])
    if use_mag:
        normed_flow_mags = (flow_mags/max_mag)
    flow_hsv_image = np.dstack((flow_angles, normed_flow_mags, np.ones(flows.shape[:2]))).astype(np.float32)
    flow_rgb_image = np.uint8(255 * cv2.cvtColor(flow_hsv_image, cv2.COLOR_HSV2RGB))
    return flow_rgb_image

def calc_flow_vector_image(image, flows, step = 6, max_mag = 30.0, color = (255,0,0), min_flow_mag = 1.0, thickness = 1):
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

def warp_image_with_scaled_flows(image, scaled_flows):
    to_scale_factor = image.shape[0]/scaled_flows.shape[0]
    to_scale_flows = cv2.resize(scaled_flows, image.shape[:2][::-1], cv2.INTER_CUBIC).astype(np.float32) * to_scale_factor *10.0

    x_indices_mat = np.array([np.arange(image.shape[1]) for j in range(0, image.shape[0])]).astype(np.float32)
    y_indices_mat = np.array([np.arange(image.shape[0]) for j in range(0, image.shape[1])]).T.astype(np.float32)
    x_warp_map = (x_indices_mat + to_scale_flows[:,:,0]).astype(np.float32)
    y_warp_map = (y_indices_mat + to_scale_flows[:,:,1]).astype(np.float32)
    warp_image = cv2.remap(image, x_warp_map, y_warp_map, cv2.INTER_CUBIC)
    return warp_image.astype(np.float32)

def warp_image_with_flows(image, flows):
    '''will not allow a warp outside of the bounds of the image. If this is attempted,
    keeps that pixel the same'''
    x_indices_mat = np.array([np.arange(image.shape[1]) for j in range(0, image.shape[0])]).astype(np.float32)
    y_indices_mat = np.array([np.arange(image.shape[0]) for j in range(0, image.shape[1])]).T.astype(np.float32)
    x_warp_map= (x_indices_mat + flows[:,:,0]).astype(np.float32)
    #x_warp_map[x_warp_map < 0] = x_indices_mat[x_warp_map < 0]
    #x_warp_map[x_warp_map > flows.shape[1]-1] = x_indices_mat[x_warp_map > flows.shape[1]-1]

    y_warp_map = np.zeros(flows.shape[:2], dtype = np.float32)
    y_warp_map = (y_indices_mat + flows[:,:,1]).astype(np.float32)
    #y_warp_map[y_warp_map < 0] = y_indices_mat[y_warp_map < 0]
    #y_warp_map[y_warp_map > flows.shape[0]-1] = y_indices_mat[y_warp_map > flows.shape[0]-1]

    warp_image = cv2.remap(image, x_warp_map, y_warp_map, cv2.INTER_CUBIC)
    return warp_image.astype(np.float32)
