"""This module contains the default values and constants used in pvfactors """

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
