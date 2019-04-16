"""This module contains the default values and constants used in pvfactors """
from future.utils import iteritems
import numpy as np

# Geometry params
DEFAULT_NORMAL_VEC = None
TOL_COLLINEAR = 1e-5

# Ground params
MAX_X_GROUND = 1e2
MIN_X_GROUND = - MAX_X_GROUND
Y_GROUND = 0.

# PV rows parameters
X_ORIGIN_PVROWS = 0.

# Define colors used for plotting the 2D arrays
COLOR_DIC = {
    'i': '#FFBB33',
    's': '#A7A49D',
    't': '#6699cc',
    'pvrow_illum': '#6699cc',
    'pvrow_shaded': '#ff0000',
    'ground_shaded': '#A7A49D',
    'ground_illum': '#FFBB33'
}
PLOT_FONTSIZE = 20
ALPHA_TEXT = 0.20

# Tolerance and thresholds to use from experience getting errors with shapely
DISTANCE_TOLERANCE = 1e-8
THRESHOLD_DISTANCE_TOO_CLOSE = 1e-10


# The view dictionaries associate integer indices to 'types' of views. For
# instance "ground_sky" would be the view between a ground surface and the
# sky, and it would use the integer index ``1``.
VIEW_DICT = {
    None: 0,
    "ground_sky": 1,
    "back_gnd": 2,
    "gnd_back": 3,
    "front_sky": 4,
    "back_sky": 5,
    "back_gnd_obst": 6,
    "gnd_back_obst": 7,
    "front_gnd_obst": 8,
    "gnd_front_obst": 9,
    "pvrows": 10}
REVERSE_VIEW_DICT = {v: k for (k, v) in iteritems(VIEW_DICT)}
THRESHOLD_VF_12 = 5e-5


# Gaussian shading default parameters: TOTAL_GAUSSIAN_AREA dependent on these
SIGMA = 1. / np.sqrt(2.)
N_SIGMA = 3.
GAUSSIAN_DIAMETER_CIRCUMSOLAR = 2. * N_SIGMA * SIGMA
RADIUS_CIRCUMSOLAR = 1.
DEFAULT_CIRCUMSOLAR_ANGLE = 30.

# Horizon band shading
DEFAULT_HORIZON_BAND_ANGLE = 6.5

SKY_REFLECTIVITY_DUMMY = 1.
