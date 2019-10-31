import os
import numpy as np
import pandas as pd
from pvfactors.geometry import OrderedPVArray, PVGround, PVSurface
from pvfactors.geometry.utils import contains
from pvfactors.config import MAX_X_GROUND, MIN_X_GROUND


def test_ordered_pvarray_from_dict(params):
    """Test that can successfully create ordered pvarray from parameters dict,
    and that the axis azimuth convention works correctly (via normal vector)
    """
    pvarray = OrderedPVArray.transform_from_dict_of_scalars(params)

    # Test that ground is created successfully
    assert isinstance(pvarray.ground, PVGround)
    # TODO: check why this is not matching exactly: hint = look at length
    # of ground shaded surfaces, some small tolerance may be chipped away
    np.testing.assert_allclose(pvarray.ground.length,
                               MAX_X_GROUND - MIN_X_GROUND)

    # Test the front and back sides
    assert len(pvarray.pvrows) == 3
    np.testing.assert_array_equal(
        pvarray.pvrows[0].front.n_vector, -pvarray.pvrows[0].back.n_vector)
    assert pvarray.pvrows[0].front.shaded_length == 0
    assert pvarray.gcr == params['gcr']
    assert np.abs(pvarray.rotation_vec) == params['surface_tilt']
    assert pvarray.pvrows[0].front.n_vector[0] > 0
    distance_between_pvrows = \
        pvarray.pvrows[1].centroid.x - pvarray.pvrows[0].centroid.x
    assert distance_between_pvrows == 5.0
    assert pvarray.n_ts_surfaces == 40

    # Orient the array the other way
    params.update({'surface_azimuth': 270.})
    pvarray = OrderedPVArray.transform_from_dict_of_scalars(params)
    assert pvarray.pvrows[0].front.n_vector[0] < 0
    assert pvarray.n_ts_surfaces == 40


def test_plot_ordered_pvarray():
    """Test that ordered pv array plotting works correctly"""
    is_ci = os.environ.get('CI', False)
    if not is_ci:
        import matplotlib.pyplot as plt

        # Create base params
        params = {
            'n_pvrows': 3,
            'pvrow_height': 2.5,
            'pvrow_width': 2.,
            'surface_azimuth': 90.,  # east oriented modules / point right
            'axis_azimuth': 0.,  # axis of rotation towards North
            'surface_tilt': 20.,
            'gcr': 0.4,
            'solar_zenith': 20.,
            'solar_azimuth': 90.,  # sun located in the east
            'rho_ground': 0.2,
            'rho_front_pvrow': 0.01,
            'rho_back_pvrow': 0.03
        }

        # Plot simple ordered pv array
        ordered_pvarray = OrderedPVArray.transform_from_dict_of_scalars(params)
        f, ax = plt.subplots()
        ordered_pvarray.plot(ax)
        plt.show()

        # Plot discretized ordered pv array
        params.update({'cut': {0: {'front': 5}, 1: {'back': 3}},
                       'surface_azimuth': 270.})  # point left
        ordered_pvarray = OrderedPVArray.transform_from_dict_of_scalars(params)
        f, ax = plt.subplots()
        ordered_pvarray.plot(ax)
        plt.show()


def test_discretization_ordered_pvarray(discr_params):
    pvarray = OrderedPVArray.transform_from_dict_of_scalars(discr_params)
    pvrows = pvarray.pvrows

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
    pvarray = OrderedPVArray.transform_from_dict_of_scalars(params)
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


def test_ordered_pvarray_gnd_shadow_casting(params):
    """Test shadow casting on ground, no inter-row shading"""

    # Test front shading on right
    ordered_pvarray = OrderedPVArray.transform_from_dict_of_scalars(params)
    # Check shadow casting on ground
    assert len(ordered_pvarray.ground.list_segments[0]
               .shaded_collection.list_surfaces) == 3
    assert len(ordered_pvarray.ground.list_segments[0]
               .illum_collection.list_surfaces) == 7
    assert ordered_pvarray.ground.shaded_length == 6.385066634855473


def test_ordered_pvarray_gnd_pvrow_shadow_casting_right(params_direct_shading):

    # Test front shading on right
    ordered_pvarray = OrderedPVArray.transform_from_dict_of_scalars(
        params_direct_shading)
    # Check shadow casting on ground
    assert len(ordered_pvarray.ground.list_segments[0]
               .shaded_collection.list_surfaces) == 2
    assert len(ordered_pvarray.ground.list_segments[0]
               .illum_collection.list_surfaces) == 4
    np.testing.assert_allclose(ordered_pvarray.ground.length,
                               MAX_X_GROUND - MIN_X_GROUND)

    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[0].front.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[1].front.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[2].front.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[0].back.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[1].back.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[2].back.shaded_length, 0.)


def test_ordered_pvarray_gnd_pvrow_shadow_casting_left(params_direct_shading):

    params_direct_shading.update({'solar_azimuth': 270,
                                  'surface_azimuth': 270})
    # Test front shading on right
    ordered_pvarray = OrderedPVArray.transform_from_dict_of_scalars(
        params_direct_shading)
    # Check shadow casting on ground
    assert len(ordered_pvarray.ground.list_segments[0]
               .shaded_collection.list_surfaces) == 2
    assert len(ordered_pvarray.ground.list_segments[0]
               .illum_collection.list_surfaces) == 4
    np.testing.assert_allclose(ordered_pvarray.ground.length,
                               MAX_X_GROUND - MIN_X_GROUND)

    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[2].front.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[1].front.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[0].front.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[2].back.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[1].back.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[0].back.shaded_length, 0.)


def test_ordered_pvarray_gnd_pvrow_shadow_casting_back(params_direct_shading):

    params_direct_shading.update({'solar_azimuth': 270,
                                  'surface_tilt': 120})

    # Test front shading on right
    ordered_pvarray = OrderedPVArray.transform_from_dict_of_scalars(
        params_direct_shading)
    # Check shadow casting on ground
    assert len(ordered_pvarray.ground.list_segments[0]
               .shaded_collection.list_surfaces) == 2
    assert len(ordered_pvarray.ground.list_segments[0]
               .illum_collection.list_surfaces) == 4
    np.testing.assert_allclose(ordered_pvarray.ground.length,
                               MAX_X_GROUND - MIN_X_GROUND)

    # Shading length should be identical as in previous test for front surface,
    # but now with back surface
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[2].back.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[1].back.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[0].back.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[2].front.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[1].front.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[0].front.shaded_length, 0.)


def test_ordered_pvarray_gnd_pvrow_shadow_casting_right_n_seg(
        params_direct_shading):

    params_direct_shading.update({'cut': {1: {'front': 7}}})
    # Test front shading on right
    ordered_pvarray = OrderedPVArray.transform_from_dict_of_scalars(
        params_direct_shading)
    # Check shadow casting on ground
    assert len(ordered_pvarray.ground.list_segments[0]
               .shaded_collection.list_surfaces) == 2
    assert len(ordered_pvarray.ground.list_segments[0]
               .illum_collection.list_surfaces) == 4
    np.testing.assert_allclose(ordered_pvarray.ground.length,
                               MAX_X_GROUND - MIN_X_GROUND)

    # Test pvrow sides: should be the same as without segments
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[0].front.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[1].front.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[2].front.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[0].back.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[1].back.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[2].back.shaded_length, 0.)

    # Test individual segments
    center_row = ordered_pvarray.pvrows[1]
    list_pvsegments = center_row.front.list_segments
    fully_shaded_segment = list_pvsegments[-1]
    partial_shaded_segment = list_pvsegments[-2]
    assert fully_shaded_segment.illum_collection.is_empty
    np.testing.assert_almost_equal(
        fully_shaded_segment.shaded_collection.length,
        list_pvsegments[0].length)
    assert partial_shaded_segment.shaded_collection.length > 0
    assert partial_shaded_segment.illum_collection.length > 0
    sum_lengths = (partial_shaded_segment.illum_collection.length +
                   partial_shaded_segment.shaded_collection.length)
    np.testing.assert_almost_equal(sum_lengths, list_pvsegments[0].length)


def test_ordered_pvarray_gnd_pvrow_shadow_casting_back_n_seg(
        params_direct_shading):

    params_direct_shading.update({'cut': {1: {'back': 7}},
                                  'solar_azimuth': 270,
                                  'surface_tilt': 120})
    # Test front shading on right
    ordered_pvarray = OrderedPVArray.transform_from_dict_of_scalars(
        params_direct_shading)
    # Check shadow casting on ground
    assert len(ordered_pvarray.ground.list_segments[0]
               .shaded_collection.list_surfaces) == 2
    assert len(ordered_pvarray.ground.list_segments[0]
               .illum_collection.list_surfaces) == 4
    np.testing.assert_allclose(ordered_pvarray.ground.length,
                               MAX_X_GROUND - MIN_X_GROUND)

    # Shading length should be identical as in previous test for front surface,
    # but now with back surface
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[2].back.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[1].back.shaded_length, 0.33333333333333254)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[0].back.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[2].front.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[1].front.shaded_length, 0.)
    np.testing.assert_almost_equal(
        ordered_pvarray.pvrows[0].front.shaded_length, 0.)

    # Test individual segments
    center_row = ordered_pvarray.pvrows[1]
    list_pvsegments = center_row.back.list_segments
    fully_shaded_segment = list_pvsegments[-1]
    partial_shaded_segment = list_pvsegments[-2]
    assert fully_shaded_segment.illum_collection.is_empty
    np.testing.assert_almost_equal(
        fully_shaded_segment.shaded_collection.length,
        list_pvsegments[0].length)
    assert partial_shaded_segment.shaded_collection.length > 0
    assert partial_shaded_segment.illum_collection.length > 0
    sum_lengths = (partial_shaded_segment.illum_collection.length +
                   partial_shaded_segment.shaded_collection.length)
    np.testing.assert_almost_equal(sum_lengths, list_pvsegments[0].length)


def test_ordered_pvarray_cuts_for_pvrow_view(ordered_pvarray):
    """Test that pvarray ground is cut correctly"""

    n_surfaces_ground_before_cut = 7
    ground_length = 200.0

    n_surfaces_1 = ordered_pvarray.ground.n_surfaces
    len_1 = ordered_pvarray.ground.length

    assert n_surfaces_1 == n_surfaces_ground_before_cut + 3
    np.testing.assert_allclose(len_1, ground_length)


def test_ordered_pvarray_list_surfaces(ordered_pvarray):
    """Check that getting a correct list of surfaces"""
    n_surfaces = ordered_pvarray.n_surfaces
    list_surfaces = ordered_pvarray.all_surfaces

    assert isinstance(list_surfaces, list)
    assert len(list_surfaces) == n_surfaces
    assert isinstance(list_surfaces[0], PVSurface)


def test_build_surface_registry(ordered_pvarray):
    """Test that building surface registry correctly"""
    reg = ordered_pvarray.surface_registry

    assert reg.shape[0] == ordered_pvarray.n_surfaces
    assert reg.shape[1] == len(ordered_pvarray.registry_cols)


def test_get_all_surface_indices(ordered_pvarray):

    # Check surface indices before indexing
    surf_indices = ordered_pvarray.surface_indices
    assert surf_indices == [None] * ordered_pvarray.n_surfaces

    # Check surface indices after indexing
    ordered_pvarray.index_all_surfaces()
    surf_indices = ordered_pvarray.surface_indices
    np.testing.assert_array_equal(surf_indices,
                                  range(ordered_pvarray.n_surfaces))


def test_get_all_ts_surface_indices(ordered_pvarray):
    """Check that the ts surface indices are created correctly, and that
    are properly used to make dictionary of ts surfaces"""
    # Check ts surface indices: indexing should happen at indexing
    dict_ts_surfaces = ordered_pvarray.dict_ts_surfaces
    indices = list(dict_ts_surfaces.keys())
    expected_list_indices = set([ts_surf.index for ts_surf
                                 in ordered_pvarray.all_ts_surfaces])
    assert set(indices) == expected_list_indices
    assert isinstance(indices[0], int)
    assert np.max(indices) == (ordered_pvarray.n_ts_surfaces - 1)


def test_param_names(params):

    param_names = ['qinc']
    pvarray = OrderedPVArray.transform_from_dict_of_scalars(
        params, param_names=param_names)

    # Set all surfaces parameters to 1
    pvarray.update_params({'qinc': 1})

    # Check that all surfaces of the correct surface params
    all_surfaces = pvarray.all_surfaces
    for surf in all_surfaces:
        assert surf.param_names == param_names
        assert surf.get_param('qinc') == 1

    # Check weighted values
    np.testing.assert_almost_equal(
        pvarray.ground.get_param_weighted('qinc'), 1)
    np.testing.assert_almost_equal(
        pvarray.ground.get_param_ww('qinc'),
        pvarray.ground.length)
    for pvrow in pvarray.pvrows:
        # Front
        np.testing.assert_almost_equal(
            pvrow.front.get_param_weighted('qinc'), 1)
        np.testing.assert_almost_equal(
            pvrow.front.get_param_ww('qinc'), pvrow.front.length)
        # Back
        np.testing.assert_almost_equal(
            pvrow.back.get_param_weighted('qinc'), 1)
        np.testing.assert_almost_equal(
            pvrow.back.get_param_ww('qinc'), pvrow.back.length)


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

    pvarray = OrderedPVArray.transform_from_dict_of_scalars(params)

    ground_seg = pvarray.ground.list_segments[0]
    # there should be no visible shadow on the ground
    assert len(ground_seg.shaded_collection.list_surfaces) == 0
    # all of the edge points should be outside of range of ground geometry
    for edge_pt in pvarray.edge_points:
        assert not contains(pvarray.ground.original_linestring, edge_pt)


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
    pvarray_w_direct_shading = OrderedPVArray.transform_from_dict_of_scalars(
        params)

    # Check that 3 shadows on ground
    assert (pvarray_w_direct_shading.ground.list_segments[0]
            .shaded_collection.n_surfaces) == 5
    # Check that there is no shading on the center pv row
    pvrow = pvarray_w_direct_shading.pvrows[1]
    assert (pvrow.front.list_segments[0]
            .shaded_collection.n_surfaces) == 0


def test_coords_ground_shadows():

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
    pvarray = OrderedPVArray.transform_from_dict_of_scalars(params)

    # Test that ground is created successfully
    assert isinstance(pvarray.ground, PVGround)
    np.testing.assert_equal(pvarray.ground.length,
                            MAX_X_GROUND - MIN_X_GROUND)
    np.testing.assert_equal(pvarray.pvrows[0].length, 2)
    np.testing.assert_equal(pvarray.pvrows[1].length, 2)
    np.testing.assert_equal(pvarray.pvrows[2].length, 2)

    # Test the front and back sides
    assert len(pvarray.pvrows) == 3
    np.testing.assert_array_equal(
        pvarray.pvrows[0].front.n_vector, -pvarray.pvrows[0].back.n_vector)
    np.testing.assert_allclose(pvarray.pvrows[1].front.shaded_length,
                               0.05979874)
    assert pvarray.gcr == params['gcr']
    assert np.abs(pvarray.rotation_vec) == params['surface_tilt']
    assert pvarray.pvrows[0].front.n_vector[0] < 0
    distance_between_pvrows = \
        pvarray.pvrows[1].centroid.x - pvarray.pvrows[0].centroid.x
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
