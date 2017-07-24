from math import pi

def deg2radians(deg):
    return pi*float(deg)/180.0

def radians2deg(radians):
    return 180.0 * (radians/pi)

def truncate_radians(radians):
    return radians % (2.0*pi)

def truncate_degrees(deg):
    return deg % (360.0)
