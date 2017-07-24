import numpy as np

'''returns a list of binary images, each of which are white (value)
when the center at i is the one to which the pixel belongs'''
def get_cluster_masks(image, centers, value = 255):
    center_subtraction_mag_images = np.zeros((centers.shape[0], image.shape[0], image.shape[1]))
    for i in range(0, centers.shape[0]):
        center_subtraction_mag_images[i] = np.linalg.norm(image - centers[i], axis = 2)



    min_center_mag_indexes = np.argmin(center_subtraction_mag_images, axis = 0)
    cluster_masks = np.zeros((centers.shape[0], image.shape[0], image.shape[1]))
    for i in range(0, centers.shape[0]):
        cluster_masks[i][min_center_mag_indexes == i] = value
    return cluster_masks


def round_image_to_clusters(image, centers):
    cluster_masks = get_cluster_masks(image, centers, value = 1.0)
    kmeans_rounded_image = np.zeros(image.shape)
    for i in range(0, cluster_masks.shape[0]):
        kmeans_rounded_image[cluster_masks[i] == 1.0] = centers[i]
    return kmeans_rounded_image
