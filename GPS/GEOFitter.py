import numpy as np
from math import cos, sin, atan2, pi, sqrt
import cv2
import Angle.AngleConverter as AngleConverter
from PIL import Image
import ImageOp.Transform.AffineToolbox as AffineToolbox
'''
Important: GPS Logs are not accurate per frame. To remedy this,
the mosaic should be built using frames that correspond with GPS updates (
doesn't matter about the frames between, so long as A frame corresponds with
each GPS update that occurs within the mosaic. (e.g. a starting frame that is
a multiple of the GPS refresh rate must be chosen, and a frame step that the
GPS refresh rate (in frames) is divisible by must be chosen)


takes a stitched mosaic image along with midpoint locations on the image
and FrameGPS's and determines the proper transformation to apply
so that north points upward in the image, and the mosaic fits as best as
possible onto the world when properly "pasted"
'''
class GEOFitter:
    '''technically mosaic_image is not needed, but useful for debugging
    since you can draw on it'''
    '''
    PROBLEMS: Appears that GPS is not updated every frame(or even every COUPLE of frames).
    This causes a few issues...
    1): Some GPS points(when projected onto the image) are copied.
    2): When choosing basis points, or at any time where a pair between an image midpoint
    and GPS is specified, there is no guarantee that the image midpoint actually lines up
    with the corresponding frame geo at that index (the frame geo could just be a duplicate
    of a prior GPS point). As a result, it is possible that the ppsm and ppm calculations
    could be incorrect as they rely on finding the ratio between image distance between two
    midpoints and the distance between their corresponding GPS points

    Potential solutions:

    1) remove image midpoints that do not correspond with a change in GPS
    from the frame prior. Problem with this solution: Because mosaics are not created frame
    by frame, that means that even if this approach is used, it is unlikely that GPS
    coordinates were updated on the same frame as selected for the image taken. (i.e. if
    you start at frame 7 and pull a frame every 10 frames, and GPS coordinates are only
    updated every 10 frames starting from 0, you will always have GPS coordinates that
    are 7 frames off with this approach).

    2) GPS likely updates every second/fraction of a second. Keep this in mind and calculate
    frame numbers where GPS will change. User will be forced to use a frame step that allows
    for frames to be captured so that every time interval that a GPS updates on, a frame will
    be present and will accurately represent the GPS point.

    3)Do #1 and force user to round frame steps and starting frames so that all new GPS
    info is captured on some of the frames accurately.
    '''
    def __init__(self, mosaic_image, midpoints, frame_geos):
        self.mosaic_image = mosaic_image
        self.midpoints = midpoints
        self.frame_geos = frame_geos
        self.remove_duplicate_geos_and_corresponding_midpoints()
        self.init_basis_midpoints_and_frame_geos()
        self.init_ppsi()
        self.init_compass_basises()
        self.init_geo_transform()
        print("ppm: ", self.ppm)
        print("ppsm: ", self.ppsm)
        print("north vector: ", self.north_vec)
        print("east vector: ", self.east_vec)
        print("geo transform: ", self.geo_transform)

        compass_vec_mag = 500
        east_vec_line = (int(self.basis_points[0][0]), int(self.basis_points[0][1]), int(self.basis_points[0][0] + compass_vec_mag * self.east_vec[0]), int(self.basis_points[0][1] + compass_vec_mag * -self.east_vec[1]))
        self.mosaic_image = cv2.arrowedLine(self.mosaic_image, east_vec_line[:2], east_vec_line[2:4], (0,255,0), thickness = 2)

        north_vec_line = (int(self.basis_points[0][0]), int(self.basis_points[0][1]), int(self.basis_points[0][0] + compass_vec_mag * self.north_vec[0]), int(self.basis_points[0][1] + compass_vec_mag * -self.north_vec[1]))
        self.mosaic_image = cv2.arrowedLine(self.mosaic_image, north_vec_line[:2], north_vec_line[2:4], (0,0,255), thickness = 2)

        rot_mat = cv2.getRotationMatrix2D(tuple((np.array(self.mosaic_image.shape[:2][::-1])/2.0).astype(np.int)), AngleConverter.radians2deg(atan2(-self.east_vec[1], self.east_vec[0])), 1)
        self.mosaic_image = AffineToolbox.bounded_cv_affine_transform_image(self.mosaic_image, rot_mat)#cv2.warpAffine(self.mosaic_image, rot_mat, self.mosaic_image.shape[:2])
        Image.fromarray(self.mosaic_image).show()



    def remove_duplicate_geos_and_corresponding_midpoints(self):
        i = 1
        while i < len(self.frame_geos):
            if self.frame_geos[i] == self.frame_geos[i-1]:
                del[self.frame_geos[i]]
                self.midpoints = np.delete(self.midpoints, i, axis = 0)
            else:
                i+=1



    '''
    picks two common midpoints and corresponding frame geos
    to use for calculating compass directions and the ppsi of
    the mosaic
    '''
    def init_basis_midpoints_and_frame_geos(self):
        basis_indices = (0, self.midpoints.shape[0]-1)
        self.basis_points = np.array([self.midpoints[basis_indices[0]], self.midpoints[basis_indices[1]]])
        self.basis_geos = np.array([self.frame_geos[basis_indices[0]], self.frame_geos[basis_indices[1]]])

    '''
    uses the pixel euclidian distance and haversine distance between
    midpoints and frame geos to estimate the mosaic's pixels per square meter
    (initalizes ppm, pixels per m (if you have distance in pixels and want
    to convert to meters, and ppsm, pixels per square meter))
    '''
    def init_ppsi(self):
        dist_meters_between_basises = self.basis_geos[0].haversine_distance(self.basis_geos[1])
        dist_pixels_between_basises = np.linalg.norm(self.basis_points[1] - self.basis_points[0])
        self.ppm = dist_pixels_between_basises/dist_meters_between_basises
        self.ppsm = self.ppm**2


    '''
    initializes the compass basis vectors, labeled:
    north_vec and east_vec, which are unit vectors that point in the direction
    of north and east in the image (with y axis flipped so that points appear
    correctly in the image).
    '''
    def init_compass_basises(self):
        basis_compass_bearing = self.basis_geos[0].bearing_toward(self.basis_geos[1], radians = True)
        angle_through_basis_points = atan2(-(self.basis_points[1] - self.basis_points[0])[1], (self.basis_points[1] - self.basis_points[0])[0])
        north_angle_in_image = angle_through_basis_points + basis_compass_bearing
        east_angle_in_image = north_angle_in_image - 0.5*pi

        self.north_vec = np.array([cos(north_angle_in_image), sin(north_angle_in_image)])
        self.east_vec = np.array([cos(east_angle_in_image), sin(east_angle_in_image)])

    '''
    Initializes a matrix called geo_transform that, when
    multiplied with a column GPS Vector (in meters) correctly
    transforms the point to its location on the image
    '''
    def init_geo_transform(self):
        self.geo_transform = np.array([self.east_vec,
                                       self.north_vec])
        self.geo_transform *= self.ppm

    '''
    initializes the target and training points based on the GPS positions of
    the midpoints
    '''
    def init_set(self):
        '''
        adds an extra vector index with value 1 because
        it uses an affine transform
        '''
        self.X = np.ones((self.midpoints.shape[0], 3))
        self.X[:, :2] = self.midpoints.copy()
        self.Y = np.ones((self.midpoints.shape[0], 3))
        '''
        gps_vecs is a list of GPS vectors (in meters) relative to
        basis_geos[0]
        '''
        self.gps_vecs = np.zeros((self.midpoints.shape[0], 3))

        for i in range(0, self.gps_vecs.shape[0]):
            self.gps_vecs[i] = self.basis_geos[0].vectorize(self.frame_geos[i])

        self.Y[:, :2] = self.gps_vecs[:, :2] * 0.33*self.ppm + self.basis_points[0] #.dot(self.geo_transform.T) + self.basis_points[0]
        print("X: ", self.X)
        print("Y: ", self.Y)


        for i in range(0, self.Y.shape[0]):
            self.mosaic_image = cv2.circle(self.mosaic_image, tuple(self.X[i][:2].astype(np.int)), 30, (255,0,0), thickness = 2)
            self.mosaic_image = cv2.circle(self.mosaic_image, tuple(self.Y[i][:2].astype(np.int)), 30, (0,255,0), thickness = 2)



        compass_vec_mag = 200
        east_vec_line = (int(self.basis_points[0][0]), int(self.basis_points[0][1]), int(self.basis_points[0][0] + compass_vec_mag * self.east_vec[0]), int(self.basis_points[0][1] + compass_vec_mag * -self.east_vec[1]))
        self.mosaic_image = cv2.arrowedLine(self.mosaic_image, east_vec_line[:2], east_vec_line[2:4], (0,255,0), thickness = 2)

        north_vec_line = (int(self.basis_points[0][0]), int(self.basis_points[0][1]), int(self.basis_points[0][0] + compass_vec_mag * self.north_vec[0]), int(self.basis_points[0][1] + compass_vec_mag * -self.north_vec[1]))
        self.mosaic_image = cv2.arrowedLine(self.mosaic_image, north_vec_line[:2], north_vec_line[2:4], (0,0,255), thickness = 2)
        print("GPS Vecs: ", self.gps_vecs)
        print("Basis geos: ", self.basis_geos)
        print("Midpoints: ", self.midpoints)
        print("Geos: ", self.frame_geos)
        #Image.fromarray(self.mosaic_image).show()
