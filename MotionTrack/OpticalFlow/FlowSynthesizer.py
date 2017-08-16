from Stat.Gaussian.UnivariateGaussian import UnivariateGaussian
import numpy as np
'''using the UnivariateGaussian class is easy, but likely very slow because
it involves for looping through the entire image and calculating probabilities
individually...'''


'''takes a set of flows warped so that each pixel aligns correctly,
then uses recursive Bayesian filtering to interpolate the "hidden value"
for the flows at each pixel'''
def synth_flows(rect_flows, Z_var):

def calc_1d_flow_posteriors(flows, from_x_var, from_z_var):

 
def calc_full_posterior_variances(flows, from_x_var, from_z_var):
