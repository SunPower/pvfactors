import os
import numpy as np
import pandas as pd
from pvfactors.geometry.pvarray import OrderedPVArray
from pvfactors.config import MAX_X_GROUND, MIN_X_GROUND, DISTANCE_TOLERANCE
from pvfactors.geometry.pvground import TsGround
from pvfactors.geometry.timeseries import TsSurface


def test_ordered_pvarray_from_dict(params):
    """Test that can successfully create ordered pvarray from parameters dict,
    and that the axis azimuth convention works correctly (via normal vector)
    """
    pvarray = OrderedPVArray.fit_from_dict_of_scalars(params)

    # Test that ground is created successfully
    assert isinstance(pvarray.ts_ground, TsGround)
    # TODO: check why this is not matching exactly: hint = look at length
    # of ground shaded surfaces, some small tolerance may be chipped away
    np.testing.assert_allclose(pvarray.ts_ground.length,
                               MAX_X_GROUND - MIN_X_GROUND)

    # Test the front and back sides
    assert len(pvarray.ts_pvrows) == 3
    np.testing.assert_array_equal(
        pvarray.ts_pvrows[0].front.n_vector,
        -pvarray.ts_pvrows[0].back.n_vector)
    assert pvarray.ts_pvrows[0].front.shaded_length == 0
    assert pvarray.gcr == params['gcr']
    assert np.abs(pvarray.rotation_vec) == params['surface_tilt']
    assert pvarray.ts_pvrows[0].front.n_vector[0] > 0
    distance_between_pvrows = \
        pvarray.ts_pvrows[1].centroid.x - pvarray.ts_pvrows[0].centroid.x
    assert distance_between_pvrows == 5.0
    assert pvarray.n_ts_surfaces == 40

    # Orient the array the other way
    params.update({'surface_azimuth': 270.})
    pvarray = OrderedPVArray.fit_from_dict_of_scalars(params)
    assert pvarray.ts_pvrows[0].front.n_vector[0] < 0
    assert pvarray.n_ts_surfaces == 40


# def test_plot_ordered_pvarray():
#     """Test that ordered pv array plotting works correctly"""
#     is_ci = os.environ.get('CI', False)
#     if not is_ci:
#         import matplotlib.pyplot as plt

#         # Create base params
#         params = {
#             'n_pvrows': 3,
#             'pvrow_height': 2.5,
#             'pvrow_width': 2.,
#             'surface_azimuth': 90.,  # east oriented modules / point right
#             'axis_azimuth': 0.,  # axis of rotation towards North
#             'surface_tilt': 20.,
#             'gcr': 0.4,
#             'solar_zenith': 20.,
#             'solar_azimuth': 90.,  # sun located in the east
#             'rho_ground': 0.2,
#             'rho_front_pvrow': 0.01,
#             'rho_back_pvrow': 0.03
#         }

#         # Plot simple ordered pv array
#         ordered_pvarray = OrderedPVArray.fit_from_dict_of_scalars(params)
#         f, ax = plt.subplots()
#         ordered_pvarray.plot_at_idx(0, ax)
#         plt.show()

#         # Plot discretized ordered pv array
#         params.update({'cut': {0: {'front': 5}, 1: {'back': 3}},
#                        'surface_azimuth': 270.})  # point left
#         ordered_pvarray = OrderedPVArray.fit_from_dict_of_scalars(params)
#         f, ax = plt.subplots()
#         ordered_pvarray.plot_at_idx(0, ax)
#         plt.show()


def test_discretization_ordered_pvarray(discr_params):
    """Test that the number of segments and surfaces is correct
    when discretizing the PV rows"""
    pvarray = OrderedPVArray.fit_from_dict_of_scalars(discr_params)
    pvrows = pvarray.ts_pvrows

    # Check the transformed geometries
    assert len(pvrows[0].front.list_segments) == 5
    assert len(pvrows[0].back.list_segments) == 1
    assert len(pvrows[1].back.list_segments) == 3
    assert len(pvrows[1].front.list_segments) == 2
    # Check the timeseries geometries
    assert pvarray.ts_pvrows[0].n_ts_surfaces == 12
    assert pvarray.ts_pvrows[1].n_ts_surfaces == 10
    assert pvarray.ts_pvrows[2].n_ts_surfaces == 4
    assert pvarray.ts_ground.n_ts_surfaces == 28
    assert pvarray.n_ts_surfaces == 54
    # Check that the list of ts surfaces match
    assert len(set(pvarray.all_ts_surfaces)) == 54


def test_ts_surfaces_side_of_cut_point(params):
    """Check that can successfully call list ts surfaces on side
    of cut point"""
    pvarray = OrderedPVArray.fit_from_dict_of_scalars(params)
    # For first pvrow
    list_left = pvarray.ts_ground.ts_surfaces_side_of_cut_point(
        'left', 0)
    list_right = pvarray.ts_ground.ts_surfaces_side_of_cut_point(
        'right', 0)
    assert len(list_left) == 7
    assert len(list_right) == 21
    # For second pv row
    list_left = pvarray.ts_ground.ts_surfaces_side_of_cut_point(
        'left', 1)
    list_right = pvarray.ts_ground.ts_surfaces_side_of_cut_point(
        'right', 1)
    assert len(list_left) == 14
    assert len(list_right) == 14
    # For rightmost pv row
    list_left = pvarray.ts_ground.ts_surfaces_side_of_cut_point(
        'left', 2)
    list_right = pvarray.ts_ground.ts_surfaces_side_of_cut_point(
        'right', 2)
    assert len(list_left) == 21
    assert len(list_right) == 7

# TODO: see if can fix, or find way to replace with ts
# def test_ordered_pvarray_gnd_shadow_casting(params):
#     """Test shadow casting on ground, no inter-row shading"""

#     # Test front shading on right
#     ordered_pvarray = OrderedPVArray.fit_from_dict_of_scalars(params)
#     assert len(ordered_pvarray.ts_ground.non_point_shaded_surfaces_at(0)) == 3
#     assert len(ordered_pvarray.ts_ground.non_point_illum_surfaces_at(0)) == 7
#     assert ordered_pvarray.ts_ground.shaded_length == 6.385066634855473


# TODO: counting shadows on the ground will be important
# def _check_ground_surfaces(ts_ground, expected_n_shadow_surfaces,
#                            expected_n_illum_surfaces):
#     """Check the number of ground surfaces after merging the shadows when
#     possible"""
#     # Check shadow casting on ground when merging the shadow surfaces
#     non_pt_shadow_elements = [
#         shadow_el for shadow_el in ts_ground.shadow_elements
#         if shadow_el.coords.length[0] > DISTANCE_TOLERANCE]
#     list_shadow_surfaces = ts_ground._merge_shadow_surfaces(
#         0, non_pt_shadow_elements)
#     assert len(list_shadow_surfaces) == expected_n_shadow_surfaces
#     # Check illuminated surfaces
#     # assert len(ts_ground.non_point_illum_surfaces_at(0)
#     #            ) == expected_n_illum_surfaces
#     np.testing.assert_allclose(ts_ground.length,
#                                MAX_X_GROUND - MIN_X_GROUND)


def test_ordered_pvarray_gnd_pvrow_shadow_casting_right(params_direct_shading):
    """Front direct shading with the sun on the right side"""
    # Test front shading on right
    ordered_pvarray = OrderedPVArray.fit_from_dict_of_scalars(
        params_direct_shading)
    # _check_ground_surfaces(ordered_pvarray.ts_ground, 2, 4)

    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[0].front.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[1].front.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[2].front.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[0].back.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[1].back.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[2].back.shaded_length, 0.)


def test_ordered_pvarray_gnd_pvrow_shadow_casting_left(params_direct_shading):
    """Front direct shading with the sun on the left side"""
    params_direct_shading.update({'solar_azimuth': 270,
                                  'surface_azimuth': 270})
    # Test front shading on right
    ordered_pvarray = OrderedPVArray.fit_from_dict_of_scalars(
        params_direct_shading)
    # _check_ground_surfaces(ordered_pvarray.ts_ground, 2, 4)

    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[2].front.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[1].front.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[0].front.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[2].back.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[1].back.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[0].back.shaded_length, 0.)


def test_ordered_pvarray_gnd_pvrow_shadow_casting_back(params_direct_shading):
    """Back direct shading with the sun on the left side"""
    params_direct_shading.update({'solar_azimuth': 270,
                                  'surface_tilt': 120})

    # Test front shading on right
    ordered_pvarray = OrderedPVArray.fit_from_dict_of_scalars(
        params_direct_shading)
    # _check_ground_surfaces(ordered_pvarray.ts_ground, 2, 4)

    # Shading length should be identical as in previous test for front surface,
    # but now with back surface
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[2].back.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[1].back.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[0].back.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[2].front.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[1].front.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[0].front.shaded_length, 0.)


def test_ordered_pvarray_gnd_pvrow_shadow_casting_right_n_seg(
        params_direct_shading):
    """Front direct shading with the sun on the right side and discretized
    pv row sides"""
    params_direct_shading.update({'cut': {1: {'front': 7}}})
    # Test front shading on right
    ordered_pvarray = OrderedPVArray.fit_from_dict_of_scalars(
        params_direct_shading)
    # _check_ground_surfaces(ordered_pvarray.ts_ground, 2, 4)

    # Test pvrow sides: should be the same as without segments
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[0].front.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[1].front.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[2].front.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[0].back.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[1].back.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[2].back.shaded_length, 0.)

    # Test individual segments
    center_row = ordered_pvarray.ts_pvrows[1]
    list_pvsegments = center_row.front.list_segments
    fully_shaded_segment = list_pvsegments[-1]
    partial_shaded_segment = list_pvsegments[-2]
    np.testing.assert_allclose(fully_shaded_segment.illum.length, 0)
    np.testing.assert_almost_equal(
        fully_shaded_segment.shaded.length,
        list_pvsegments[0].length)
    assert partial_shaded_segment.shaded.length > 0
    assert partial_shaded_segment.illum.length > 0
    sum_lengths = (partial_shaded_segment.illum.length +
                   partial_shaded_segment.shaded.length)
    np.testing.assert_almost_equal(sum_lengths, list_pvsegments[0].length)


def test_ordered_pvarray_gnd_pvrow_shadow_casting_back_n_seg(
        params_direct_shading):
    """Back direct shading with discretized pv row sides"""
    params_direct_shading.update({'cut': {1: {'back': 7}},
                                  'solar_azimuth': 270,
                                  'surface_tilt': 120})
    # Test front shading on right
    ordered_pvarray = OrderedPVArray.fit_from_dict_of_scalars(
        params_direct_shading)
    # _check_ground_surfaces(ordered_pvarray.ts_ground, 2, 4)

    # Shading length should be identical as in previous test for front surface,
    # but now with back surface
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[2].back.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[1].back.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[0].back.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[2].front.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[1].front.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.ts_pvrows[0].front.shaded_length, 0.)

    # Test individual segments
    center_row = ordered_pvarray.ts_pvrows[1]
    list_pvsegments = center_row.back.list_segments
    fully_shaded_segment = list_pvsegments[-1]
    partial_shaded_segment = list_pvsegments[-2]
    np.testing.assert_allclose(fully_shaded_segment.illum.length, 0)
    np.testing.assert_almost_equal(
        fully_shaded_segment.shaded.length,
        list_pvsegments[0].length)
    assert partial_shaded_segment.shaded.length > 0
    assert partial_shaded_segment.illum.length > 0
    sum_lengths = (partial_shaded_segment.illum.length +
                   partial_shaded_segment.shaded.length)
    np.testing.assert_almost_equal(sum_lengths, list_pvsegments[0].length)

# FIXME
# def test_ordered_pvarray_cuts_for_pvrow_view(ordered_pvarray):
#     """Test that pvarray ground is cut correctly"""

#     n_surfaces_ground_before_cut = 7
#     ground_length = 200.0

#     n_surfaces_1 = ordered_pvarray.ts_ground.n_non_point_surfaces_at(0)
#     len_1 = ordered_pvarray.ts_ground.length

#     assert n_surfaces_1 == n_surfaces_ground_before_cut + 3
#     np.testing.assert_allclose(len_1, ground_length)


def test_ordered_pvarray_list_surfaces(ordered_pvarray):
    """Check that getting a correct list of surfaces"""
    n_surfaces = ordered_pvarray.n_ts_surfaces
    list_surfaces = ordered_pvarray.all_ts_surfaces

    assert isinstance(list_surfaces, list)
    assert len(list_surfaces) == n_surfaces
    assert isinstance(list_surfaces[0], TsSurface)


def test_ts_surface_indices_order(ordered_pvarray):
    """Check that the indices of the ts surfaces in the list
    is the same as the list ordering"""
    # Check surface indices after indexing
    surf_indices = ordered_pvarray.ts_surface_indices
    np.testing.assert_array_equal(surf_indices,
                                  range(ordered_pvarray.n_ts_surfaces))


def test_param_names(params):
    """Test that parameter names are passed correctly"""
    param_names = ['qinc']
    pvarray = OrderedPVArray.fit_from_dict_of_scalars(
        params, param_names=param_names)

    # Set all surfaces parameters to 1
    pvarray.update_params({'qinc': 1})

    # Check that all surfaces of the correct surface params
    all_ts_surfaces = pvarray.all_ts_surfaces
    for ts_surf in all_ts_surfaces:
        assert ts_surf.param_names == param_names
        assert ts_surf.get_param('qinc') == 1

    # Check weighted values
    np.testing.assert_almost_equal(
        pvarray.ts_ground.get_param_weighted('qinc'), 1)
    np.testing.assert_almost_equal(
        pvarray.ts_ground.get_param_ww('qinc'),
        pvarray.ts_ground.length)
    for ts_pvrow in pvarray.ts_pvrows:
        # Front
        np.testing.assert_almost_equal(
            ts_pvrow.front.get_param_weighted('qinc'), 1)
        np.testing.assert_almost_equal(
            ts_pvrow.front.get_param_ww('qinc'), ts_pvrow.front.length)
        # Back
        np.testing.assert_almost_equal(
            ts_pvrow.back.get_param_weighted('qinc'), 1)
        np.testing.assert_almost_equal(
            ts_pvrow.back.get_param_ww('qinc'), ts_pvrow.back.length)


def test_orderedpvarray_almost_flat():
    """Making sure that things are correct when the pvarray is almost flat
    and the sun is very low, which means that the shadows on the ground, and
    the edge points will be outside of ground range (since not infinite)"""

    params = {
        'n_pvrows': 3,
        'pvrow_height': 2.5,
        'pvrow_width': 2.,
        'surface_azimuth': 90.,  # east oriented modules
        'axis_azimuth': 0.,      # axis of rotation towards North
        'surface_tilt': 0.01,    # almost flat
        'gcr': 0.4,
        'solar_zenith': 89.9,    # sun super low
        'solar_azimuth': 90.,    # sun located in the east
    }

    pvarray = OrderedPVArray.fit_from_dict_of_scalars(params)
    ts_ground = pvarray.ts_ground

    # there should be 3 shadow elements on the ground
    assert len(ts_ground.shadow_elements) == 3
    # But their lengths should be = to zero
    np.testing.assert_allclose(ts_ground.shaded_length, 0)
    # all of the edge points should be outside of range of ground geometry
    for coords in ts_ground.cut_point_coords:
        assert (coords.x < MIN_X_GROUND) | (coords.x > MAX_X_GROUND)


def test_ordered_pvarray_gnd_shadow_casting_tolerance():
    """It seems that there are roundoff errors when running shadow casting
    on some computers, test that this case works."""

    params = {'axis_azimuth': 0,
              'gcr': 0.3,
              'n_pvrows': 3,
              'pvrow_height': 1.8,
              'pvrow_width': 1.98,
              'solar_azimuth': 263.99310644558074,
              'solar_zenith': 73.91658668648401,
              'surface_azimuth': 270.0,
              'surface_tilt': 51.98206680806641}
    pvarray_w_direct_shading = OrderedPVArray.fit_from_dict_of_scalars(
        params)

    # Check that 3 shadows on ground
    # FIXME
    # assert len(pvarray_w_direct_shading
    #            .ts_ground.non_point_shaded_surfaces_at(0)) == 5
    # Check that there is no shading on the center pv row
    ts_pvrow = pvarray_w_direct_shading.ts_pvrows[1]
    assert ts_pvrow.front.list_segments[0].shaded.length[0] \
        < DISTANCE_TOLERANCE


def test_coords_ground_shadows():
    """Check coords of timeseries ground shadows"""
    # Create base params
    params = {
        'axis_azimuth': 0,
        'n_pvrows': 2,
        'pvrow_height': 2.5,
        'pvrow_width': 2.,
        'gcr': 0.4,
        'cut': {0: {'front': 5}, 1: {'back': 3}}
    }

    # Timeseries parameters for testing
    solar_zenith = np.array([20., 45.])
    solar_azimuth = np.array([70., 200.])
    surface_tilt = np.array([10., 70.])
    surface_azimuth = np.array([90., 270.])

    # Plot simple ordered pv array
    ordered_pvarray = OrderedPVArray(**params)
    ordered_pvarray.fit(solar_zenith, solar_azimuth, surface_tilt,
                        surface_azimuth)

    expected_gnd_shadow_coords = [
        [([-1.89924929, 0.19163641], [0., 0.]),
         ([0.18914857, 1.51846431], [0., 0.])],
        [([3.10075071, 5.19163641], [0., 0.]),
         ([5.18914857, 6.51846431], [0., 0.])]
    ]
    gnd_shadow_coords = [shadow.coords.as_array for shadow
                         in ordered_pvarray.ts_ground.shadow_elements]

    np.testing.assert_almost_equal(
        expected_gnd_shadow_coords, gnd_shadow_coords)


def test_coords_cut_points():
    """Test timeseries coords of cut points"""
    # Create base params
    params = {
        'axis_azimuth': 0,
        'n_pvrows': 2,
        'pvrow_height': 2.5,
        'pvrow_width': 2.,
        'gcr': 0.4,
        'cut': {0: {'front': 5}, 1: {'back': 3}}
    }

    # Timeseries parameters for testing
    solar_zenith = np.array([20., 45.])
    solar_azimuth = np.array([70., 200.])
    surface_tilt = np.array([10., 70.])
    surface_azimuth = np.array([90., 270.])

    # Plot simple ordered pv array
    ordered_pvarray = OrderedPVArray(**params)
    ordered_pvarray.fit(solar_zenith, solar_azimuth, surface_tilt,
                        surface_azimuth)

    expected_cut_point_coords = [
        [[14.17820455, -0.90992559], [0., 0.]],
        [[19.17820455, 4.09007441], [0., 0.]]]

    cut_pt_coords = [cut_point.as_array
                     for cut_point in
                     ordered_pvarray.ts_ground.cut_point_coords]
    np.testing.assert_almost_equal(
        expected_cut_point_coords, cut_pt_coords)


def test_ordered_pvarray_from_dict_w_direct_shading():
    """Test that can successfully create ordered pvarray from parameters dict,
    and that the axis azimuth convention works correctly (via normal vector),
    and check that ground surfaces make sense.
    Came from direct shading case where ground shadows not correctly created
    """
    # Specify array parameters
    params = {
        'n_pvrows': 3,
        'pvrow_height': 1,
        'pvrow_width': 1,
        'axis_azimuth': 0.,
        'gcr': 0.4,
        'rho_front_pvrow': 0.01,
        'rho_back_pvrow': 0.03,
        'solar_zenith': 74,
        'solar_azimuth': 229,
        'surface_tilt': 50,
        'surface_azimuth': 270
    }
    pvarray = OrderedPVArray.fit_from_dict_of_scalars(params)

    # Test that ground is created successfully
    assert isinstance(pvarray.ts_ground, TsGround)
    np.testing.assert_equal(pvarray.ts_ground.length,
                            MAX_X_GROUND - MIN_X_GROUND)
    np.testing.assert_equal(pvarray.ts_pvrows[0].length, 2)
    np.testing.assert_equal(pvarray.ts_pvrows[1].length, 2)
    np.testing.assert_equal(pvarray.ts_pvrows[2].length, 2)

    # Test the front and back sides
    assert len(pvarray.ts_pvrows) == 3
    np.testing.assert_array_equal(
        pvarray.ts_pvrows[0].front.n_vector, -pvarray.ts_pvrows[0].back.n_vector)
    np.testing.assert_allclose(pvarray.ts_pvrows[1].front.shaded_length,
                               0.05979874)
    assert pvarray.gcr == params['gcr']
    assert np.abs(pvarray.rotation_vec) == params['surface_tilt']
    assert pvarray.ts_pvrows[0].front.n_vector[0] < 0
    distance_between_pvrows = \
        pvarray.ts_pvrows[1].centroid.x - pvarray.ts_pvrows[0].centroid.x
    assert distance_between_pvrows == 2.5


def test_ordered_pvarray_direct_shading():
    """Test that direct shading is calculated correctly in the following
    5 situations:
    - PV rows tilted to the left and front side shading
    - PV rows tilted to the right and front side shading
    - PV rows tilted to the left and back side shading
    - PV rows tilted to the right and back side shading
    - no shading
    """
    # Base params
    params = {
        'n_pvrows': 3,
        'pvrow_height': 1,
        'pvrow_width': 1,
        'axis_azimuth': 0.,
        'gcr': 0.5
    }
    # Timeseries inputs
    df_inputs = pd.DataFrame({
        'solar_zenith': [70., 80., 80., 70., 10.],
        'solar_azimuth': [270., 90., 270., 90., 90.],
        'surface_tilt': [45., 45., 45., 45., 45.],
        'surface_azimuth': [270., 270., 90., 90., 90.]})

    # Initialize and fit pv array
    pvarray = OrderedPVArray.init_from_dict(params)
    # Fit pv array to timeseries data
    pvarray.fit(df_inputs.solar_zenith, df_inputs.solar_azimuth,
                df_inputs.surface_tilt, df_inputs.surface_azimuth)

    expected_ts_front_shading = [0.24524505, 0., 0., 0.24524505, 0.]
    expected_ts_back_shading = [0., 0.39450728, 0.39450728, 0., 0.]

    # Test that timeseries shading calculated correctly
    np.testing.assert_allclose(expected_ts_front_shading,
                               pvarray.shaded_length_front)
    np.testing.assert_allclose(expected_ts_back_shading,
                               pvarray.shaded_length_back)
