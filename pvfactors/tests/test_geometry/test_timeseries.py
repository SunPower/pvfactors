import os
from pvfactors.geometry.timeseries import TsPVRow
import pandas as pd
import numpy as np


def test_ts_pvrow():
    """Test timeseries pv row creation and shading cases.
    Note that shading must always be zero when pv rows are flat"""

    xy_center = (0, 2)
    width = 2.
    df_inputs = pd.DataFrame({
        'rotation_vec': [20., -30., 0.],
        'shaded_length_front': [1.3, 0., 1.9],
        'shaded_length_back': [0, 0.3, 0.6]})
    cut = {'front': 3, 'back': 4}

    ts_pvrow = TsPVRow.from_raw_inputs(
        xy_center, width, df_inputs.rotation_vec,
        cut, df_inputs.shaded_length_front,
        df_inputs.shaded_length_back)

    # Check timeseries length of front and back segments
    for seg in ts_pvrow.front.list_segments:
        np.testing.assert_allclose(width / cut['front'], seg.length)
    for seg in ts_pvrow.back.list_segments:
        np.testing.assert_allclose(width / cut['back'], seg.length)

    # Check shaded length on either sides of pv rows
    expected_front_shading = np.where(df_inputs.rotation_vec,
                                      df_inputs.shaded_length_front, 0.)
    expected_back_shading = np.where(df_inputs.rotation_vec,
                                     df_inputs.shaded_length_back, 0.)
    np.testing.assert_allclose(expected_front_shading,
                               ts_pvrow.front.shaded_length)
    np.testing.assert_allclose(expected_back_shading,
                               ts_pvrow.back.shaded_length)


def test_plot_ts_pvrow():

    is_ci = os.environ.get('CI', False)
    if not is_ci:
        import matplotlib.pyplot as plt

        # Create a PV row
        xy_center = (0, 2)
        width = 2.
        df_inputs = pd.DataFrame({
            'rotation_vec': [20., -30., 0.],
            'shaded_length_front': [1.3, 0., 1.9],
            'shaded_length_back': [0, 0.3, 0.6]})
        cut = {'front': 3, 'back': 4}

        ts_pvrow = TsPVRow.from_raw_inputs(
            xy_center, width, df_inputs.rotation_vec,
            cut, df_inputs.shaded_length_front,
            df_inputs.shaded_length_back)

        # Plot it at ts 0
        f, ax = plt.subplots()
        ts_pvrow.plot_at_idx(0, ax)
        plt.show()

        # Plot it at ts 1
        f, ax = plt.subplots()
        ts_pvrow.plot_at_idx(1, ax)
        plt.show()

        # Plot it at ts 2: flat case
        f, ax = plt.subplots()
        ts_pvrow.plot_at_idx(2, ax)
        plt.show()
