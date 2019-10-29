from pvfactors.viewfactors.calculator import VFCalculator
from pvfactors.viewfactors.timeseries import VFTsMethods
import numpy as np


def test_ts_vf_matrix(ordered_pvarray):
    """Test that timeseries vf matrix is calculated correctly"""
    vfcalculator = VFCalculator()
    vf_matrix = vfcalculator.build_ts_vf_matrix(ordered_pvarray)

    # Check that correct size
    assert vf_matrix.shape == (41, 41, 1)

    # get all indices where surfaces have positive length
    list_idx = [surf.index for surf in ordered_pvarray.all_ts_surfaces
                if surf.length[0] > 0]
    print(vf_matrix[:, :, 0][np.ix_(list_idx, list_idx)])


def test_vf_pvrow_to_gnd_surf_obstruction_hottel(ordered_pvarray):
    """Check that calculation of ts view factors between pv rows and ground
    is working correctly"""
    # pv array is tilted to right
    pvarray = ordered_pvarray
    tilted_to_left = pvarray.rotation_vec > 0
    vf_ts_methods = VFTsMethods()

    # left pvrow & back side & left surface
    pvrow_idx = 0
    left_gnd_surfaces = (
        ordered_pvarray.ts_ground.ts_surfaces_side_of_cut_point('left',
                                                                pvrow_idx))
    gnd_surf = left_gnd_surfaces[0]
    pvrow_surf = pvarray.ts_pvrows[pvrow_idx].back.all_ts_surfaces[0]
    ts_length = pvrow_surf.length
    vf1, vf2 = vf_ts_methods.vf_pvrow_surf_to_gnd_surf_obstruction_hottel(
        pvrow_surf, pvrow_idx, 3, tilted_to_left, pvarray.ts_pvrows,
        gnd_surf, ts_length, is_back=True, is_left=True)
    # check
    np.testing.assert_allclose(vf1, [0.33757309])
    np.testing.assert_allclose(vf2, [0.31721494009])

    # middle pv row & front side & right surface
    pvrow_idx = 1
    right_gnd_surfaces = (
        ordered_pvarray.ts_ground.ts_surfaces_side_of_cut_point('right',
                                                                pvrow_idx))
    gnd_surf = right_gnd_surfaces[12]
    pvrow_surf = pvarray.ts_pvrows[pvrow_idx].front.all_ts_surfaces[0]
    ts_length = pvrow_surf.length
    vf1, vf2 = vf_ts_methods.vf_pvrow_surf_to_gnd_surf_obstruction_hottel(
        pvrow_surf, pvrow_idx, 3, tilted_to_left, pvarray.ts_pvrows,
        gnd_surf, ts_length, is_back=False, is_left=False)
    # check
    np.testing.assert_allclose(vf1, [0.00502823674])
    np.testing.assert_allclose(vf2, [0.0020112947])

    # middle pv row & back side & right surface
    pvrow_idx = 1
    right_gnd_surfaces = (
        ordered_pvarray.ts_ground.ts_surfaces_side_of_cut_point('right',
                                                                pvrow_idx))
    gnd_surf = right_gnd_surfaces[12]
    pvrow_surf = pvarray.ts_pvrows[pvrow_idx].back.all_ts_surfaces[0]
    ts_length = pvrow_surf.length
    vf1, vf2 = vf_ts_methods.vf_pvrow_surf_to_gnd_surf_obstruction_hottel(
        pvrow_surf, pvrow_idx, 3, tilted_to_left, pvarray.ts_pvrows,
        gnd_surf, ts_length, is_back=True, is_left=False)
    # check
    np.testing.assert_allclose(vf1, [0.])
    np.testing.assert_allclose(vf2, [0.])

    # right pv row & back side & left surface
    pvrow_idx = 2
    right_gnd_surfaces = (
        ordered_pvarray.ts_ground.ts_surfaces_side_of_cut_point('left',
                                                                pvrow_idx))
    gnd_surf = right_gnd_surfaces[0]
    pvrow_surf = pvarray.ts_pvrows[pvrow_idx].back.all_ts_surfaces[0]
    ts_length = pvrow_surf.length
    vf1, vf2 = vf_ts_methods.vf_pvrow_surf_to_gnd_surf_obstruction_hottel(
        pvrow_surf, pvrow_idx, 3, tilted_to_left, pvarray.ts_pvrows,
        gnd_surf, ts_length, is_back=True, is_left=True)
    # check
    np.testing.assert_allclose(vf1, [0.01191152457])
    np.testing.assert_allclose(vf2, [0.01119317174])
