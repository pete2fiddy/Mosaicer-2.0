import numpy as np
import cv2
from PIL import Image
from sklearn.decomposition import PCA

class WindowedPCAReduction:
    '''does not extend MatchType because it should not be used for mosaicing. Is strongly
    rotation variant. (also only trains on one image, and doesn't really need(but I suppose
    could use) a fit image to train on)'''
    def __init__(self, image, fit_image, reduction_params):
        self.image = image
        self.fit_image = fit_image
        self.init_reduction_params(reduction_params)
        self.init_pca()

    def init_reduction_params(self, reduction_params):
        self.window_size = reduction_params["window_size"]
        self.window_step = 1 if reduction_params["window_step"] is None else reduction_params["window_step"]
        self.window_margin = (self.window_size-1)//2

    def init_pca(self):
        self.window_vector_image = self.image_to_window_vector_image(self.image, self.window_step)
        self.fit_window_vector_image = self.image_to_window_vector_image(self.fit_image, self.window_step)
        self.window_vectors = self.window_vector_image_to_vectors(self.window_vector_image)
        self.fit_window_vectors = self.window_vector_image_to_vectors(self.fit_window_vector_image)
        self.window_vectors = np.concatenate((self.window_vectors, self.fit_window_vectors), axis = 0)
        print("window vectors shape: ", self.window_vectors.shape)
        self.avg_window_vector = np.average(self.window_vectors, axis = 0)
        self.pca = PCA()
        self.pca.fit(self.window_vectors)

    def image_to_window_vector_image(self, image, step):
        window_vectors = np.zeros(((image.shape[0] - self.window_size)//step, (image.shape[1] - self.window_size)//step, self.window_size**2))
        kernel_margin = (self.window_size - 1)//2

        for x_window in range(0, window_vectors.shape[1]):
            x_image = (x_window * step) + kernel_margin
            for y_window in range(0, window_vectors.shape[0]):
                y_image = (y_window * step) + kernel_margin
                window_vectors[y_window,x_window] = image[y_image - kernel_margin : y_image + kernel_margin + 1, x_image - kernel_margin : x_image + kernel_margin + 1].flatten()
        return window_vectors

    def window_vector_image_to_vectors(self, window_vector_image):
        return np.reshape(window_vector_image, (-1, window_vector_image.shape[2]))

    def project_image(self, image, num_components, step = 1):
        window_vector_image = self.image_to_window_vector_image(image, step)
        window_vectors = self.window_vector_image_to_vectors(window_vector_image)
        window_vectors -= self.avg_window_vector
        transformed_window_vectors = self.pca.transform(window_vectors)[:, :num_components]
        reshape_dims = ((image.shape[0] - self.window_size)//step, (image.shape[1] - self.window_size)//step, num_components)
        pca_image = np.reshape(transformed_window_vectors, reshape_dims)
        return pca_image

    def projected_image_to_keypoints_and_descriptors(self, project_image, step = 1):
        kps = []
        descriptors = []
        for x in range(0, project_image.shape[1], step):
            for y in range(0, project_image.shape[0], step):
                kps.append(cv2.KeyPoint(x, y, self.window_step))
                descriptors.append(project_image[y,x])
        '''descriptors may need to be an int'''
        '''for OPENCV matching, make sure that a correct norm is used
        for the descriptor type (tried using NORM_HAMMING and that only works
        with binary features)
        '''
        descriptors = np.asarray(descriptors).astype(np.float32)
        return kps, descriptors
