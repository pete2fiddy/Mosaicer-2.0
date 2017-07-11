import cv2

'''returns the bounds required to fit the image without warping
into a box of the size constraint_bounds'''
def get_resize_bounds_to_fit_constraint(image, constraint_bounds):
    constraint_width_to_image_width_ratio = float(constraint_bounds[0]) / float(image.shape[1])
    constraint_height_to_image_height_ratio = float(constraint_bounds[1]) / float(image.shape[0])
    resize_factor = constraint_width_to_image_width_ratio if constraint_width_to_image_width_ratio < constraint_height_to_image_height_ratio else constraint_height_to_image_height_ratio
    return (int(image.shape[1] * resize_factor), int(image.shape[0] * resize_factor))

def resize_image_to_constraints(image, constraint_bounds):
    resized_image_dims = get_resize_bounds_to_fit_constraint(image, constraint_bounds)
    return cv2.resize(image, resized_image_dims)
