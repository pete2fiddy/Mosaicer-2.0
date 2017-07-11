from MetaData.FrameEXIF import FrameEXIF


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
    Gimbal Aximuth:XXX
    Gimbal Elevation:XXX
    Hor. FOV:XXX
    Ver. FOV:XXX
    (new line between entries)
    '''
    def __init__(self, log_path, start_frame, end_frame, frame_step):
        self.log_path = log_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frame_step = frame_step
        self.extract_frame_exifs()

    def extract_frame_exifs(self):
        self.frame_exifs = []
        with open(self.log_path, 'r') as log_file:
            log = log_file.read()
            for frame_num in range(self.start_frame, self.end_frame, self.frame_step):
                framenum_index = log.index("Frame Number:" + str(frame_num))
                end_framenum_index = log.index("\n\n", framenum_index)
                frame_sublog = log[framenum_index : end_framenum_index]
                self.frame_exifs.append(FrameEXIF(frame_sublog))
