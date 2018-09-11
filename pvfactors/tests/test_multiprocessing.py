# -*- coding: utf-8 -*-

"""
Testing the multiprocessing calculation from the ``tools`` module
"""

from pvfactors.tools import (calculate_radiosities_parallel_perez,
                             breakup_df_inputs)
import os
import pandas as pd

TEST_DIR = os.path.dirname(__file__)
TEST_DATA = os.path.join(TEST_DIR, 'test_files')


def test_calculate_radiosities_parallel_perez():
    """
    Check that the parallel calculation using Perez diffuse model is able to
    run. The consistency is not tested here, because it is already tested for
    the serial calculation (which it relies on)
    """
    # Inputs to the calculations
    filename = "file_test_multiprocessing_inputs.csv"
    subset_idx = 100
    arguments = {
        'n_pvrows': 2,
        'pvrow_height': 3.,
        'pvrow_width': 1.,
        'array_azimuth': 270.,
        'array_tilt': -20.,
        'gcr': 0.3,
        'solar_zenith': 20.,
        'solar_azimuth': 90.,
        'rho_ground': 0.2,
        'rho_front_pvrow': 0.01,
        'rho_back_pvrow': 0.03
    }
    # Import inputs
    df_inputs_simulation = pd.read_csv(os.path.join(TEST_DATA, filename),
                                       index_col=0)
    df_inputs_simulation.index = pd.DatetimeIndex(
        df_inputs_simulation.index)
    # Reduce number of inputs
    df_inputs_simulation = df_inputs_simulation.iloc[:subset_idx, :]
    # Select number of processes
    n_processes = None
    # Break up inputs
    (timestamps, array_tilt, array_azimuth,
     solar_zenith, solar_azimuth, dni, dhi) = breakup_df_inputs(df_inputs_simulation)
    # Run calculation
    results = calculate_radiosities_parallel_perez(
        arguments, timestamps, array_tilt, array_azimuth,
        solar_zenith, solar_azimuth, dni, dhi, n_processes=n_processes)
