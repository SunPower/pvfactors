import os
from pvfactors.geometry.coords import TsPVRow
import pandas as pd
import numpy as np


def test_ts_pvrow():

    xy_center = (0, 2)
    width = 2.
    df_inputs = pd.DataFrame({
        'rotation_vec': [20., -30.],
        'shaded_length_front': [0.3, 0],
        'shaded_length_back': [0, 0.3]})
    cut = {'front': 3, 'back': 4}
    is_left_edge = False
    is_right_edge = False

    ts_pvrow = TsPVRow.from_raw_inputs(
        xy_center, width, df_inputs.rotation_vec,
        cut, df_inputs.shaded_length_front,
        df_inputs.shaded_length_back, is_left_edge,
        is_right_edge)

    # Check timeseries length of front and back segments
    for seg in ts_pvrow.front.list_segments:
        np.testing.assert_allclose(width / cut['front'], seg.length)
    for seg in ts_pvrow.back.list_segments:
        np.testing.assert_allclose(width / cut['back'], seg.length)

    # Check shaded length on either sides of pv rows
    np.testing.assert_allclose(0, ts_pvrow.front.shaded_length)
    np.testing.assert_allclose(0, ts_pvrow.back.shaded_length)


def test_plot_ts_pvrow():

    is_ci = os.environ.get('CI', False)
    if not is_ci:
        import matplotlib.pyplot as plt

        # Create a PV row
        xy_center = (0, 2)
        width = 2.
        df_inputs = pd.DataFrame({
            'rotation_vec': [20., -30.],
            'shaded_length_front': [0.3, 0],
            'shaded_length_back': [0, 0.3]})
        cut = {'front': 3, 'back': 4}
        is_left_edge = False
        is_right_edge = False

        ts_pvrow = TsPVRow.from_raw_inputs(
            xy_center, width, df_inputs.rotation_vec,
            cut, df_inputs.shaded_length_front,
            df_inputs.shaded_length_back, is_left_edge,
            is_right_edge)

        # Plot it at ts 0
        f, ax = plt.subplots()
        ts_pvrow.plot_at_idx(0, ax)
        plt.show()

        # Plot it at ts 1
        f, ax = plt.subplots()
        ts_pvrow.plot_at_idx(1, ax)
        plt.show()
