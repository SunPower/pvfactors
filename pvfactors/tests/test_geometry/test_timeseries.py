import os
from pvfactors.geometry.timeseries import TsPointCoords, TsLineCoords
from pvfactors.geometry.pvrow import TsPVRow
from pvfactors.geometry.pvground import TsGround, TsGroundElement
import pandas as pd
import numpy as np
from pvfactors.geometry.pvrow import PVRow
from pvfactors.geometry.base import \
    BaseSide, PVSegment, PVSurface, ShadeCollection
from pvfactors.config import MIN_X_GROUND, MAX_X_GROUND


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

    # check segment index
    assert len(ts_pvrow.front.list_segments) == 3
    assert [s.index for s in ts_pvrow.front.list_segments] == [0, 1, 2]

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


# def test_plot_ts_pvrow():

#     is_ci = os.environ.get('CI', False)
#     if not is_ci:
#         import matplotlib.pyplot as plt

#         # Create a PV row
#         xy_center = (0, 2)
#         width = 2.
#         df_inputs = pd.DataFrame({
#             'rotation_vec': [20., -30., 0.],
#             'shaded_length_front': [1.3, 0., 1.9],
#             'shaded_length_back': [0, 0.3, 0.6]})
#         cut = {'front': 3, 'back': 4}

#         ts_pvrow = TsPVRow.from_raw_inputs(
#             xy_center, width, df_inputs.rotation_vec,
#             cut, df_inputs.shaded_length_front,
#             df_inputs.shaded_length_back)

#         # Plot it at ts 0
#         f, ax = plt.subplots()
#         ts_pvrow.plot_at_idx(0, ax)
#         plt.show()

#         # Plot it at ts 1
#         f, ax = plt.subplots()
#         ts_pvrow.plot_at_idx(1, ax)
#         plt.show()

#         # Plot it at ts 2: flat case
#         f, ax = plt.subplots()
#         ts_pvrow.plot_at_idx(2, ax)
#         plt.show()


# def test_ts_pvrow_to_geometry():
#     """Check that the geometries are created correctly"""

#     xy_center = (0, 2)
#     width = 2.
#     df_inputs = pd.DataFrame({
#         'rotation_vec': [20., -30., 0.],
#         'shaded_length_front': [1.3, 0., 1.9],
#         'shaded_length_back': [0, 0.3, 0.6]})
#     cut = {'front': 3, 'back': 4}
#     param_names = ['test1', 'test2']

#     ts_pvrow = TsPVRow.from_raw_inputs(
#         xy_center, width, df_inputs.rotation_vec,
#         cut, df_inputs.shaded_length_front,
#         df_inputs.shaded_length_back, param_names=param_names)

#     pvrow = ts_pvrow.at(0)
#     # Check classes of geometries
#     assert isinstance(pvrow, PVRow)
#     assert isinstance(pvrow.front, BaseSide)
#     assert isinstance(pvrow.back, BaseSide)
#     assert isinstance(pvrow.front.list_segments[0], PVSegment)
#     assert isinstance(pvrow.back.list_segments[0].illum_collection,
#                       ShadeCollection)
#     assert isinstance(pvrow.front.list_segments[1].illum_collection
#                       .list_surfaces[0], PVSurface)
#     # Check some values
#     np.testing.assert_allclose(pvrow.front.shaded_length, 1.3)
#     front_surface = (pvrow.front.list_segments[1].illum_collection
#                      .list_surfaces[0])
#     back_surface = (pvrow.back.list_segments[1].illum_collection
#                     .list_surfaces[0])
#     n_vector_front = front_surface.n_vector
#     n_vector_back = back_surface.n_vector
#     expected_n_vec_front = np.array([-0.68404029, 1.87938524])
#     np.testing.assert_allclose(n_vector_front, expected_n_vec_front)
#     np.testing.assert_allclose(n_vector_back, - expected_n_vec_front)
#     assert front_surface.param_names == param_names
#     assert back_surface.param_names == param_names


def test_ts_ground_from_ts_pvrow():
    """Check that ground geometries are created correctly from ts pvrow"""

    # Create a ts pv row
    xy_center = (0, 2)
    width = 2.
    df_inputs = pd.DataFrame({
        'rotation_vec': [20., -90., 0.],
        'shaded_length_front': [1.3, 0., 1.9],
        'shaded_length_back': [0, 0.3, 0.6]})
    cut = {'front': 3, 'back': 4}
    param_names = ['test1', 'test2']

    ts_pvrow = TsPVRow.from_raw_inputs(
        xy_center, width, df_inputs.rotation_vec,
        cut, df_inputs.shaded_length_front,
        df_inputs.shaded_length_back, param_names=param_names)

    # Create ground from it
    alpha_vec = np.deg2rad([80., 90., 70.])
    ts_ground = TsGround.from_ts_pvrows_and_angles(
        [ts_pvrow], alpha_vec, df_inputs.rotation_vec, param_names=param_names)

    assert len(ts_ground.shadow_elements) == 1
    # # Check at specific times
    # ground_0 = ts_ground.at(0)
    # assert ground_0.n_surfaces == 4
    # assert ground_0.list_segments[0].shaded_collection.n_surfaces == 1
    # ground_1 = ts_ground.at(1)  # vertical, sun above
    # assert ground_1.n_surfaces == 2  # only 2 illuminated surfaces
    # assert ground_1.list_segments[0].shaded_collection.n_surfaces == 0
    # assert ground_1.shaded_length == 0  # no shadow (since shadow length 0ish)
    # np.testing.assert_allclose(ground_0.shaded_length, 1.7587704831436)
    # np.testing.assert_allclose(ts_ground.at(2).shaded_length, width)  # flat
    # # Check that all have surface params
    # for surf in ground_0.all_surfaces:
    #     assert surf.param_names == param_names


def test_ts_ground_overlap():

    shadow_coords = np.array([
        [[[0, 0], [0, 0]], [[2, 1], [0, 0]]],
        [[[1, 2], [0, 0]], [[5, 5], [0, 0]]]
    ])
    overlap = [True, False]

    # Test without overlap
    ts_ground = TsGround.from_ordered_shadows_coords(shadow_coords)
    np.testing.assert_allclose(ts_ground.shadow_elements[0].b2.x, [2, 1])

    # Test with overlap
    ts_ground = TsGround.from_ordered_shadows_coords(shadow_coords,
                                                     flag_overlap=overlap)
    np.testing.assert_allclose(ts_ground.shadow_elements[0].b2.x, [1, 1])


# def test_ts_ground_to_geometry():

#     # There should be an overlap
#     shadow_coords = np.array([
#         [[[0, 0], [0, 0]], [[2, 1], [0, 0]]],
#         [[[1, 2], [0, 0]], [[5, 5], [0, 0]]]
#     ])
#     overlap = [True, False]
#     cut_point_coords = [TsPointCoords.from_array(np.array([[2, 2], [0, 0]]))]

#     # Test with overlap
#     ts_ground = TsGround.from_ordered_shadows_coords(
#         shadow_coords, flag_overlap=overlap, cut_point_coords=cut_point_coords)

#     # Run some checks for index 0
#     pvground = ts_ground.at(0, merge_if_flag_overlap=False,
#                             with_cut_points=False)
#     assert pvground.n_surfaces == 4
#     assert pvground.list_segments[0].illum_collection.n_surfaces == 2
#     assert pvground.list_segments[0].shaded_collection.n_surfaces == 2
#     assert pvground.list_segments[0].shaded_collection.length == 5
#     np.testing.assert_allclose(pvground.shaded_length, 5)

#     # Run some checks for index 1
#     pvground = ts_ground.at(1, with_cut_points=False)
#     assert pvground.n_surfaces == 5
#     assert pvground.list_segments[0].illum_collection.n_surfaces == 3
#     assert pvground.list_segments[0].shaded_collection.n_surfaces == 2
#     assert pvground.list_segments[0].shaded_collection.length == 4
#     np.testing.assert_allclose(pvground.shaded_length, 4)

#     # Run some checks for index 0, when merging
#     pvground = ts_ground.at(0, merge_if_flag_overlap=True,
#                             with_cut_points=False)
#     assert pvground.n_surfaces == 3
#     assert pvground.list_segments[0].illum_collection.n_surfaces == 2
#     assert pvground.list_segments[0].shaded_collection.n_surfaces == 1
#     assert pvground.list_segments[0].shaded_collection.length == 5
#     np.testing.assert_allclose(pvground.shaded_length, 5)

#     # Run some checks for index 0, when merging and with cut points
#     pvground = ts_ground.at(0, merge_if_flag_overlap=True,
#                             with_cut_points=True)
#     assert pvground.n_surfaces == 4
#     assert pvground.list_segments[0].illum_collection.n_surfaces == 2
#     assert pvground.list_segments[0].shaded_collection.n_surfaces == 2
#     assert pvground.list_segments[0].shaded_collection.length == 5
#     np.testing.assert_allclose(pvground.shaded_length, 5)


def test_shadows_coords_left_right_of_cut_point():
    """Test that coords left and right of cut point are created correctly"""
    # Ground inputs
    shadow_coords = np.array([
        [[[0], [0]], [[2], [0]]],
        [[[3], [0]], [[5], [0]]]
    ], dtype=float)
    overlap = [False]

    # --- Create timeseries ground
    cut_point = TsPointCoords([2.5], [0])
    ts_ground = TsGround.from_ordered_shadows_coords(
        shadow_coords, flag_overlap=overlap,
        cut_point_coords=[cut_point])

    # Get left and right shadows
    shadows_left = ts_ground.shadow_coords_left_of_cut_point(0)
    shadows_right = ts_ground.shadow_coords_right_of_cut_point(0)

    # Reformat for testing
    shadows_left = [shadow.as_array for shadow in shadows_left]
    shadows_right = [shadow.as_array for shadow in shadows_right]

    expected_shadows_left = [shadow_coords[0],
                             [cut_point.as_array, cut_point.as_array]]
    expected_shadows_right = [[cut_point.as_array, cut_point.as_array],
                              shadow_coords[1]]

    # Test that correct
    np.testing.assert_allclose(shadows_left, expected_shadows_left)
    np.testing.assert_allclose(shadows_right, expected_shadows_right)

    # --- Case where pv rows are flat, cut point are inf
    cut_point = TsPointCoords([np.inf], [0])
    ts_ground = TsGround.from_ordered_shadows_coords(
        shadow_coords, flag_overlap=overlap,
        cut_point_coords=[cut_point])

    # Get right shadows
    shadows_right = ts_ground.shadow_coords_right_of_cut_point(0)

    # Test that correct
    maxi = MAX_X_GROUND
    expected_shadows_right = np.array([[[[maxi], [0.]], [[maxi], [0.]]],
                                       [[[maxi], [0.]], [[maxi], [0.]]]])
    shadows_right = [shadow.as_array for shadow in shadows_right]
    np.testing.assert_allclose(shadows_right, expected_shadows_right)

    # --- Case where pv rows are flat, cut point are - inf
    cut_point = TsPointCoords([- np.inf], [0])
    ts_ground = TsGround.from_ordered_shadows_coords(
        shadow_coords, flag_overlap=overlap,
        cut_point_coords=[cut_point])

    # Get left shadows
    shadows_left = ts_ground.shadow_coords_left_of_cut_point(0)

    # Test that correct
    mini = MIN_X_GROUND
    expected_shadows_left = np.array([[[[mini], [0.]], [[mini], [0.]]],
                                      [[[mini], [0.]], [[mini], [0.]]]])
    shadows_left = [shadow.as_array for shadow in shadows_left]
    np.testing.assert_allclose(shadows_left, expected_shadows_left)


def test_ts_ground_elements_surfaces():
    """Check timeseries ground elements are created correctly"""

    # Create timeseries coords
    gnd_element_coords = TsLineCoords.from_array(
        np.array([[[-1, -1], [0, 0]], [[1, 1], [0, 0]]]))
    pt_coords_1 = TsPointCoords.from_array(np.array([[-0.5, -1], [0, 0]]))
    pt_coords_2 = TsPointCoords.from_array(np.array([[0.5, 0], [0, 0]]))

    # Create gnd element
    gnd_element = TsGroundElement(
        gnd_element_coords,
        list_ordered_cut_pts_coords=[pt_coords_1, pt_coords_2])

    # Check that structures contain the correct number of ts surfaces
    assert len(gnd_element.surface_list) == 3
    assert len(gnd_element.surface_dict[0]['left']) == 1
    assert len(gnd_element.surface_dict[1]['left']) == 2
    assert len(gnd_element.surface_dict[0]['right']) == 2
    assert len(gnd_element.surface_dict[1]['right']) == 1
    # Check that the objects are the same
    assert (gnd_element.surface_list[0]
            == gnd_element.surface_dict[0]['left'][0])
    assert (gnd_element.surface_list[0]
            == gnd_element.surface_dict[1]['left'][0])
    assert (gnd_element.surface_list[1]
            == gnd_element.surface_dict[0]['right'][0])
    assert (gnd_element.surface_list[1]
            == gnd_element.surface_dict[1]['left'][1])
    assert (gnd_element.surface_list[2]
            == gnd_element.surface_dict[0]['right'][1])
    assert (gnd_element.surface_list[2]
            == gnd_element.surface_dict[1]['right'][0])
    # Now check surfaces lengths
    np.testing.assert_allclose(gnd_element.surface_list[0].length, [0.5, 0])
    np.testing.assert_allclose(gnd_element.surface_list[1].length, [1, 1])
    np.testing.assert_allclose(gnd_element.surface_list[2].length, [0.5, 1])
    # Check coords of surfaces
    np.testing.assert_allclose(gnd_element.surface_list[0].b1.x, [-1, -1])
    np.testing.assert_allclose(gnd_element.surface_list[0].b2.x, [-0.5, -1])
