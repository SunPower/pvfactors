# -*- coding: utf-8 -*-

"""
Test that the serial calculation (using the Perez diffuse sky model) outputs
consistent results
"""

from pvfactors.tools import calculate_radiosities_serially_perez
import pandas as pd
import numpy as np
import os

TEST_DIR = os.path.dirname(__file__)
TEST_DATA = os.path.join(TEST_DIR, 'test_files')


def test_serial_calculation():
    """
    Make sure that the calculations using the Perez model stay consistent for
    all the modeled surfaces. Also testing that there is no unexpected NaN.
    """
    # Create array
    arguments = {
        'n_pvrows': 2,
        'pvrow_height': 1.5,
        'pvrow_width': 1.,
        'array_azimuth': 270,
        'array_tilt': -20.,
        'gcr': 0.3,
        'solar_zenith': 30.,
        'solar_azimuth': 90.,
        'rho_ground': 0.22,
        'rho_pvrow_front': 0.01,
        'rho_pvrow_back': 0.03
    }

    # Import simulation inputs for calculation
    filename = "file_test_multiprocessing_inputs.csv"
    df_inputs_simulation = pd.read_csv(os.path.join(TEST_DATA, filename),
                                       index_col=0)
    df_inputs_simulation.index = pd.DatetimeIndex(df_inputs_simulation.index)
    idx_subset = 10
    df_inputs_simulation = df_inputs_simulation.iloc[0:idx_subset, :]

    # Run calculation in 1 process only
    (df_outputs, df_bifacial, _, _) = (
        calculate_radiosities_serially_perez((arguments, df_inputs_simulation))
    )

    # Did the outputs remain consistent?
    assert values_are_consistent(df_outputs)


def values_are_consistent(df_outputs):
    """
    Helper function to check consistency with data from file

    :param df_outputs: output values from calculation as dataframe
    :return: result of comparison to data in test file
    """
    # Read file
    filepath = os.path.join(TEST_DATA,
                            'file_test_serial_perez_calculations_outputs.csv')
    expected_df_outputs = pd.read_csv(filepath, header=[0, 1, 2], index_col=[0]
                                      )
    expected_outputs_values = expected_df_outputs.values

    # Compare to calculation outputs
    rtol = 0.
    atol = 1e-7
    return np.allclose(expected_outputs_values.astype(float),
                       df_outputs.values.astype(float),
                       rtol=rtol, atol=atol, equal_nan=False)
