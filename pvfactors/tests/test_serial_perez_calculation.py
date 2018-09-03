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
    test_results = values_are_consistent(df_outputs)
    for result in test_results:
        assert result['passed'], "test failed for %s" % result['irradiance_term']


def values_are_consistent(df_outputs):
    """
    Helper function to check consistency with data from file.
    Compares all irradiance terms together.

    :param df_outputs: output values from calculation as dataframe
    :return: result of comparison to data in test file
    """
    # Read file
    filepath = os.path.join(TEST_DATA,
                            'file_test_serial_perez.csv')

    expected_df_outputs = pd.read_csv(filepath, index_col=False, parse_dates=['timestamps'])
    expected_df_outputs['origin'] = 'expected'
    df_outputs = (df_outputs.assign(timestamps=lambda x: x.index)
                  .drop('array_is_shaded', axis=1)
                  .reset_index(drop=True)
                  .melt(id_vars=['timestamps']))
    df_outputs['origin'] = 'calculated'

    # Merge calculated and expected values into same dataframe
    df_comparison = pd.concat([expected_df_outputs, df_outputs], axis=0, join='outer')
    df_comparison['value'] = df_comparison['value'].astype(float)

    # Group by irradiance term for clearer comparison
    grouped_comparison = df_comparison.groupby('term')
    atol = 0.
    rtol = 1e-7
    test_results = []
    for name, group in grouped_comparison:
        df_term = group.pivot_table(index=['timestamps', 'pvrow', 'side'],
                                    columns=['origin'], values='value')
        compare = ('calculated' in df_term.columns) & ('expected' in df_term.columns)
        # Only run the tests if the category exists in the expected data
        if compare:
            test_result = np.allclose(df_term.calculated, df_term.expected,
                                      atol=0, rtol=rtol)

        test_results.append({'irradiance_term': name, 'passed': test_result,
                             'df_term': df_term})

    return test_results
