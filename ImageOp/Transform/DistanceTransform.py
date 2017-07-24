import numpy as np
from math import cos, sin, atan2, tan, pi
import cv2
import ImageOp.Transform.AffineToolbox as AffineToolbox
from PIL import Image
import ImageOp.Crop as Crop
import Angle.AngleConverter as AngleConverter

def distance_transform_across_vector(mask, vector):
    '''first rectifies the image so that the vector is oriented
    upward'''
    '''a lot of the below is sloppy so that it works with existing code for affine transform
    bounding by thresholding to find the image (never thought of using it with binary images whose
    background may be black)'''
    mask = mask.copy()
    mask[mask == 0] = 128
    vector_theta_radians = atan2(vector[1], vector[0])
    rotate_amount_radians = vector_theta_radians - pi/2.0
    rotate_amount_deg = AngleConverter.radians2deg(rotate_amount_radians)
    cv_rot_mat = cv2.getRotationMatrix2D((mask.shape[1]//2, mask.shape[0]//2), rotate_amount_deg, 1.0)

    rot_mask = AffineToolbox.bounded_cv_affine_transform_image(mask, cv_rot_mat, crop_to_bounds = True)

    bounding_mask = rot_mask.copy()
    bounding_mask[bounding_mask > 0] = 1
    vert_dist_transform = vertical_distance_transform(rot_mask, bounding_mask = bounding_mask).astype(np.float32)
    vert_dist_transform[vert_dist_transform == 0] = 0.001
    vert_dist_transform[bounding_mask == 0] = 0
    cv_unrot_mat = cv2.getRotationMatrix2D((rot_mask.shape[1]//2, rot_mask.shape[0]//2), -rotate_amount_deg, 1.0)
    unrot_dist_transform = AffineToolbox.bounded_cv_affine_transform_image(vert_dist_transform, cv_unrot_mat, crop_to_bounds = True)
    predicted_crop_margin = ((unrot_dist_transform.shape[1] - mask.shape[1])//2, (unrot_dist_transform.shape[0] - mask.shape[0])//2)
    unrot_dist_transform = Crop.crop_with_margins_from_center(unrot_dist_transform, predicted_crop_margin)
    #Image.fromarray(np.uint8(255*unrot_dist_transform/np.amax(unrot_dist_transform))).show()
    return unrot_dist_transform

def vertical_distance_transform(mask, bounding_mask = None, mask_val = 255):
    if bounding_mask is None:
        bounding_mask = np.ones(mask.shape)
    vert_dist_map = np.full(mask.shape, mask.shape[1])
    for x in range(0, vert_dist_map.shape[1]):
        try:
            mask_slice = mask[:, x]
            edge_ys_at_x = np.where(mask_slice != mask_val)[0]#np.extract(mask[:, 0] == x, mask[:, 1]))
            #print("edge ys at x: ", edge_ys_at_x)
            slice_ys_at_x = np.arange(0, mask_slice.shape[0], 1)
            #print("slice ys at x: ", slice_ys_at_x)
            #print("slice ys at x shape: ", slice_ys_at_x.shape)
            down_distance_indices = np.searchsorted(edge_ys_at_x, slice_ys_at_x)
            down_distance_indices[down_distance_indices >= edge_ys_at_x.shape[0]] = edge_ys_at_x.shape[0]-1
            down_distance_indices[down_distance_indices < 0] = 0
            #print("down distance indices: ", down_distance_indices)
            up_distance_indices = down_distance_indices - 1
            down_nearest_edges = edge_ys_at_x.take(down_distance_indices)
            up_nearest_edges = edge_ys_at_x.take(up_distance_indices)

            down_edge_distances = np.abs(slice_ys_at_x - down_nearest_edges)
            up_edge_distances = np.abs(slice_ys_at_x - up_nearest_edges)
            indices_where_upper_edge_closer = np.where(up_edge_distances < down_edge_distances)

            slice_distances = down_edge_distances
            #slice_distances[indices_where_upper_edge_closer] = up_edge_distances[indices_where_upper_edge_closer]

            vert_dist_map[slice_ys_at_x, x] = slice_distances
        except:
            '''Sloppy, but try except is used to prevent taking from empty arrays(
            mask not present on that scan line)'''
    vert_dist_map[bounding_mask == False] = 0
    return vert_dist_map
