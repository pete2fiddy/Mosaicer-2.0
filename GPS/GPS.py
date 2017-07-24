from math import cos, sin, atan2, pi, sqrt
import numpy as np
import Angle.AngleConverter as AngleConverter

class GPS:
    EARTH_RADIUS_METERS = 6.371 * 1000000.0
    def __init__(self, lat, lon, alt):
        self.lat = lat
        self.lon = lon
        self.alt = alt

    @classmethod
    def init_with_frame_exif(cls, frame_exif):
        return GPS(frame_exif['Frame Center Latitude'], frame_exif['Frame Center Longitude'], frame_exif['Frame Center Altitude'])

    '''
    --------------------------------------------------------------------------
    all below GPS tool functions assume a latitude in the direction of North
    and a longitude in the direction of West. (May not be the correct format?).
    Need to check that these work -- can get results similar to calculator at
    http://www.movable-type.co.uk/scripts/latlong.html
    but some numbers do not yield similar/identical answers(perhaps because they
    are eggregiously disparate)
    --------------------------------------------------------------------------
    '''

    '''
    returns a 3d vector specifying the change in meters across East,
    change in meters across North, and change in altitude between this
    GPS toward gps2
    '''
    def vectorize(self, gps2):
        bearing_radians = self.bearing_toward(gps2, radians = True)
        bearing_from_east_ccw_radians = (pi/2.0) - bearing_radians
        distance_meters = self.haversine_distance(gps2)
        dx = cos(bearing_from_east_ccw_radians) * distance_meters
        dy = sin(bearing_from_east_ccw_radians) * distance_meters
        dz = gps2.alt - self.alt
        return np.array([dx, dy, dz])

    '''
    returns the haversine distance (from earth radius) between this GPS and
    gps2 in meters
    '''
    def haversine_distance(self, gps2):
        lat1_radians = AngleConverter.deg2radians(self.lat)
        lat2_radians = AngleConverter.deg2radians(gps2.lat)
        lon1_radians = AngleConverter.deg2radians(self.lon)
        lon2_radians = AngleConverter.deg2radians(gps2.lon)

        dlat_radians = AngleConverter.deg2radians(gps2.lat - self.lat)
        dlon_radians = AngleConverter.deg2radians(gps2.lon - self.lon)

        a = sin(dlat_radians/2.0) * sin(dlat_radians/2.0) + \
            cos(lat1_radians) * cos(lat2_radians) * \
            sin(dlon_radians/2.0) * sin(dlon_radians/2.0)

        c = 2.0 * atan2(sqrt(a), sqrt(1.0-a))


        distance_meters = GPS.EARTH_RADIUS_METERS * c
        return distance_meters

    '''
    returns the bearing (counterclockwise from East)
    from this GPS to gps2 in radians
    '''
    def bearing_toward(self, gps2, radians = True):
        dlon = gps2.lon - self.lon
        dlon_radians = AngleConverter.deg2radians(dlon)
        gps2_lat_radians = AngleConverter.deg2radians(gps2.lat)
        my_lat_radians = AngleConverter.deg2radians(self.lat)

        dy = sin(dlon_radians) * cos(gps2_lat_radians)
        dx = cos(my_lat_radians) * sin(gps2_lat_radians) - sin(my_lat_radians) * cos(gps2_lat_radians) * cos(dlon_radians)
        bearing = atan2(dy, dx)
        if not radians:
            bearing = AngleConverter.radians2deg(bearing)
        return bearing

    def __eq__(self, gps2):
        return (self.lat == gps2.lat and self.lon == gps2.lon and self.alt == gps2.alt)

    def __repr__(self):
        return "Lat: " + str(self.lat) + ", Lon: " + str(self.lon) + ", Alt: " + str(self.alt)
