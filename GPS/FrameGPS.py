from GPS.GPS import GPS

class FrameGPS(GPS):
    def __init__(self, frame_num, lat, lon, alt):
        GPS.__init__(self, lat, lon, alt)
        self.frame_num = frame_num

    @classmethod
    def init_with_frame_exif(cls, frame_exif):
        return FrameGPS(frame_exif.frame_num, frame_exif['Frame Center Latitude'], frame_exif['Frame Center Longitude'], frame_exif['Frame Center Altitude'])

    def __repr__(self):
        return "Frame Num: " + str(self.frame_num) + ", " + GPS.__repr__(self)
