# -*- coding: utf-8 -*-

"""
Test that the serial calculation (using the Perez diffuse sky model) outputs
consistent results
"""

from pvfactors.timeseries import (calculate_radiosities_serially_perez,
                                  get_average_pvrow_outputs,
                                  breakup_df_inputs)
import pandas as pd
import numpy as np
import os

TEST_DIR = os.path.dirname(__file__)
TEST_DATA = os.path.join(TEST_DIR, 'test_files')


def test_serial_calculation_with_skips(
    pvarray_parameters_serial_calc,
        df_inputs_serial_calculation_with_skips):
    """
    Make sure that the calculations using the Perez model stay consistent for
    all the modeled surfaces. Also testing that there is no unexpected NaN.
    """

    # Break up inputs
    (timestamps, tracker_theta, surface_azimuth,
     solar_zenith, solar_azimuth, dni, dhi) = breakup_df_inputs(
         df_inputs_serial_calculation_with_skips)

    # Run calculation in 1 process only
    df_registries, _ = calculate_radiosities_serially_perez(
        (pvarray_parameters_serial_calc, timestamps,
         solar_zenith, solar_azimuth,
         tracker_theta, surface_azimuth, dni, dhi))

    list_nan_idx = df_registries.index[
        df_registries.set_index('timestamps').count(axis=1) == 0]
    # There should be one line with only nan values
    assert len(list_nan_idx) == 1


def test_serial_calculation(pvarray_parameters_serial_calc,
                            df_inputs_serial_calculation):
    """
    Make sure that the calculations using the Perez model stay consistent for
    all the modeled surfaces. Also testing that there is no unexpected NaN.
    """

    # Break up inputs
    (timestamps, tracker_theta, surface_azimuth,
     solar_zenith, solar_azimuth, dni, dhi) = breakup_df_inputs(
         df_inputs_serial_calculation)

    # Run calculation in 1 process only
    df_registries, _ = calculate_radiosities_serially_perez(
        (pvarray_parameters_serial_calc, timestamps,
         solar_zenith, solar_azimuth,
         tracker_theta, surface_azimuth, dni, dhi))

    # Format df_registries to get outputs
    df_outputs = get_average_pvrow_outputs(df_registries,
                                           include_shading=False)

    # Did the outputs remain consistent?
    test_results = values_are_consistent(df_outputs)
    for result in test_results:
        assert result['passed'], ("test failed for %s" % result[
            'irradiance_term'])


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

    expected_df_outputs = pd.read_csv(filepath, index_col=False,
                                      parse_dates=['timestamps'])
    expected_df_outputs['origin'] = 'expected'
    df_outputs = (df_outputs.assign(timestamps=lambda x: x.index)
                  .reset_index(drop=True)
                  .melt(id_vars=['timestamps']))
    df_outputs['origin'] = 'calculated'

    # Merge calculated and expected values into same dataframe
    df_comparison = pd.concat([expected_df_outputs, df_outputs],
                              axis=0, join='outer', sort=True)
    df_comparison['value'] = df_comparison['value'].astype(float)

    # Group by irradiance term for clearer comparison
    grouped_comparison = df_comparison.groupby('term')
    rtol = 1e-4
    test_results = []
    for name, group in grouped_comparison:
        df_term = group.pivot_table(index=['timestamps', 'pvrow_index',
                                           'surface_side'],
                                    columns=['origin'], values='value')
        compare = (('calculated' in df_term.columns)
                   & ('expected' in df_term.columns))
        # Only run the tests if the category exists in the expected data
        if compare:
            test_result = np.allclose(df_term.calculated, df_term.expected,
                                      atol=0, rtol=rtol)

            test_results.append({'irradiance_term': name,
                                 'passed': test_result,
                                 'df_term': df_term})

    return test_results
