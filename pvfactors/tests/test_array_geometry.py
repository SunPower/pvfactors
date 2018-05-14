# -*- coding: utf-8 -*-

"""
Test the creation of ``shapely`` view factor PV arrays, and direct shading
conditions
"""

from pvfactors.pvarray import Array
from pvfactors.pvrow import PVRowLine
from pvfactors.pvcore import LinePVArray
import os
import numpy as np


def test_create_array():
    """
    Check that the pvrows know what's the index of their neighbors.
    """
    # PV array parameters
    arguments = {
        'n_pvrows': 3,
        'pvrow_height': 1.5,
        'pvrow_width': 1.,
        'gcr': 0.3,
        'array_tilt': 20.
    }
    # Create vf array
    array = Array(**arguments)

    # Run some sanity checks on the creation of the vf array
    assert len(array.pvrows) == 3
    assert isinstance(array.pvrows[0], PVRowLine)
    assert isinstance(array.pvrows[0].lines[0], LinePVArray)
    assert array.line_registry.shape[0] == 13

    # Check that the expected neighbors are correct
    tol = 1e-8
    expected_pvrow_neighbors = np.array(
        [np.nan, 0., 1., np.nan, np.nan, np.nan, np.nan, np.nan,
         np.nan, np.nan, np.nan, np.nan, np.nan, 1., 2., np.nan])
    assert np.allclose(array.surface_registry.index_pvrow_neighbor.values,
                       expected_pvrow_neighbors, atol=tol, rtol=0,
                       equal_nan=True)


def test_plotting():
    """
    Check that the plotting functions are functional (only on local machine)
    """

    is_ci = os.environ.get('CI', False)
    if not is_ci:
        import matplotlib.pyplot as plt
        from pvfactors.tools import plot_line_registry
        # Create array where sun vector is in the direction of the modules
        arguments = {
            'n_pvrows': 3,
            'pvrow_height': 1.5,
            'solar_zenith': 30,
            'solar_azimuth': 0.,
            'array_azimuth': 180.,
            'pvrow_width': 1.,
            'gcr': 0.3,
            'array_tilt': 0.
        }
        array = Array(**arguments)
        f, ax = plt.subplots(figsize=(10, 5))
        _ = plot_line_registry(ax, array)

        # Test with interrow forward shading
        arguments = {
            'n_pvrows': 5,
            'pvrow_height': 3.0,
            'solar_zenith': 30,
            'solar_azimuth': 180.,
            'array_azimuth': 180.,
            'pvrow_width': 3.0,
            'gcr': 0.9,
            'array_tilt': 20.
        }
        array = Array(**arguments)
        f, ax = plt.subplots()
        _ = plot_line_registry(ax, array)

        # Test with interrow backward shading
        arguments = {
            'n_pvrows': 5,
            'pvrow_height': 3.0,
            'solar_zenith': 60,
            'solar_azimuth': 0.,
            'array_azimuth': 180.,
            'pvrow_width': 3.0,
            'gcr': 0.9,
            'array_tilt': -20.
        }
        array = Array(**arguments)
        f, ax = plt.subplots()
        _ = plot_line_registry(ax, array)

    else:
        print("Not running 'test_plotting' in CI")


def test_merge_shadows():
    """
    When direct shading happens between pvrows, the shadow of the rows on the
    ground is supposed to form a continuous shadow (because of the overlap).
    Test that this functionally works
    """
    # Use specific vf array configuration leading to direct shading
    arguments = {
        'n_pvrows': 5,
        'pvrow_height': 2.,
        'solar_zenith': 30,
        'solar_azimuth': 0.,
        'array_azimuth': 180.,
        'pvrow_width': 3,
        'gcr': 0.9,
        'array_tilt': -20.
    }
    array = Array(**arguments)
    # There should be 1 continuous shadow on the groud, but 4 distinct ground
    #  shaded areas, delimited by what the front and the back of the pvrows
    # can see
    assert (array.line_registry.loc[(array.line_registry.line_type == 'ground')
                                    & array.line_registry.shaded]
            .shape[0] == 4)


def test_interrow_shading():
    """
    Testing the ability of the model to find direct shading between pvrows
    """
    # Forward direct shading of the pvrows
    arguments = {
        'n_pvrows': 5,
        'pvrow_height': 3.,
        'solar_zenith': 30,
        'solar_azimuth': 180.,
        'array_azimuth': 180.,
        'pvrow_width': 3.,
        'gcr': 0.9,
        'array_tilt': 20.
    }
    array = Array(**arguments)
    # There should be 4 pvrows with direct shading
    assert (array.line_registry.loc[
        (array.line_registry.line_type == 'pvrow')
        & array.line_registry.shaded]
        .shape[0] == 4)

    # Backward direct shading of the pvrows (sun in the back of the modules)
    arguments = {
        'n_pvrows': 5,
        'pvrow_height': 3.0,
        'solar_zenith': 60,
        'solar_azimuth': 0.,
        'array_azimuth': 180.,
        'pvrow_width': 3.0,
        'gcr': 0.9,
        'array_tilt': -20.
    }
    array = Array(**arguments)
    # There should still be 4 pvrows with direct shading
    assert (array.line_registry.loc[
        (array.line_registry.line_type == 'pvrow')
        & array.line_registry.shaded]
        .shape[0] == 4)

    print("Done.")
