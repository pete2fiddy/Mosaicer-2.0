from MetaData.FrameEXIF import FrameEXIF
from GPS.GPS import GPS
from GPS.FrameGPS import FrameGPS

class LogExtractor:
    '''
    Assumes the log file is in the style of Nick's video extraction:
    Frame Number:XXX
    Aircraft Altitude:XXX
    Aircraft Latitude:XXX
    Aircraft Longitude:XXX
    Aircraft Pitch:XXX
    Aircraft Roll:XXX
    Aircraft Yaw:XXX
    Camera Roll:XXX
    Date Time:XXX
    Frame Center Altitude:XXX
    Frame Center Latitude:XXX
    Frame Center Longitude:XXX
    Gimbal Azimuth:XXX
    Gimbal Elevation:XXX
    Hor. FOV:XXX
    Ver. FOV:XXX
    (new line between entries)
    '''
    '''
    Issue: Log seems to save only even frames, so asking for metadata for
    odd-numbered frames will cause a crash
    '''
    def __init__(self, log_path, start_frame, num_frames, frame_step):
        self.log_path = log_path
        self.start_frame = start_frame
        self.frame_step = frame_step
        self.end_frame = start_frame + num_frames * self.frame_step
        self.extract_frame_exifs()
        self.init_frame_geos()

    def extract_frame_exifs(self):
        self.frame_exifs = []
        self.missing_frame_nums = []
        with open(self.log_path, 'r') as log_file:
            log = log_file.read()
            for frame_num in range(self.start_frame, self.end_frame, self.frame_step):
                print("Looking for: ", "Frame Number:" + str(int(frame_num)))
                '''"\n" is added to the index string because otherwise a number that
                starts with the same digits as specified could be returned in place
                of the actual number required (e.g. 4500 when searching for only 450)'''
                framenum_index = log.find("Frame Number:" + str(int(frame_num)) + "\n")
                if framenum_index == -1:
                    self.missing_frame_nums.append(frame_num)
                else:
                    end_framenum_index = log.index("\n\n", framenum_index)
                    frame_sublog = log[framenum_index : end_framenum_index]
                    self.frame_exifs.append(FrameEXIF(frame_sublog))
        print("Missing frame nums: ", self.missing_frame_nums)

    def init_frame_geos(self):
        self.frame_geos = [FrameGPS.init_with_frame_exif(self.frame_exifs[i]) for i in range(0, len(self.frame_exifs))]

    '''returns the number of frames between GPS updates
    (estimates if frame step is not minimum frame count between log frames)'''
    '''
    def get_geo_refresh_frame_step(self):
        update_frame_counts = []
        for i in range(1, len(self.frame_geos)):
            if self.frame_geos[i-1] != self.frame_geos[i]:
                print("triggered on: ", self.frame_geos[i-1], ", ", self.frame_geos[i])
                update_frame_counts.append(self.frame_exifs[i].frame_num)
        print("update frame counts: ", update_frame_counts)'''
