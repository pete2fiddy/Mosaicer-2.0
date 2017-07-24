

'''takes the log file for a frame and converts it into a dictionary that
can access the information at that frame'''
class FrameEXIF:
    '''
    log is the string that makes up the log for that image. Is in the form:
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
    '''
    def __init__(self, log):
        self.init_log_to_dict(log)

    '''indices of the log correspond exactly to the wording in the log (e.g.
    Ver. FOV will be in the dictionary as Ver. FOV)'''
    def init_log_to_dict(self, log):
        last_newline_index = log.index("\n")
        '''frame num is also saved as a local variable because it seemed important
        enough to be easily accessible without referring to the dictionary'''
        frame_num_key, self.frame_num = self.extract_index_and_value(log[:last_newline_index], cast_type = int)
        self.exif_dict = {frame_num_key : self.frame_num}
        while(True):
            next_newline_index = log.find("\n", last_newline_index + 1)
            if next_newline_index != -1:
                line_sublog = log[last_newline_index + 1: next_newline_index]
                index, val = self.extract_index_and_value(line_sublog)
                self.exif_dict[index] = val
                last_newline_index = next_newline_index
            else:
                line_sublog = log[last_newline_index + 1 : ]
                index, val = self.extract_index_and_value(line_sublog)
                self.exif_dict[index] = val
                break
        self.exif_dict['Date Time'] = int(self.exif_dict['Date Time'])


    '''
    takes a line of the log file and extracts the index (name of the field
    on that line) and value (number corresponding to that field) and returns
    both separately.

    Gives the option of specifying the type of variable(e.g. float, int, etc)
    by passing the type's cast function as an argument (as cast_type)
    '''
    def extract_index_and_value(self, sublog, cast_type = float):
        colon_index = sublog.index(":")
        index_str = sublog[:colon_index]
        val_str = sublog[colon_index+1:]
        num_val = cast_type(val_str)
        return index_str, num_val

    def __getitem__(self, index):
        return self.exif_dict[index]
