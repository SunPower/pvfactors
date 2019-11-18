from pvfactors.viewfactors.aoimethods import \
    AOIMethods, faoi_fn_from_pvlib_sandia
from pvfactors.geometry.timeseries import TsPointCoords
from pvfactors.geometry.pvarray import OrderedPVArray
from pvfactors.viewfactors.vfmethods import VFTsMethods
import numpy as np


def test_faoi_fn_from_pvlib():
    """Check that faoi function creator produces correct results"""
    module_example = 'Canadian_Solar_CS5P_220M___2009_'
    faoi_fn = faoi_fn_from_pvlib_sandia(module_example)

    # Make sure the convention to measure angles from 0 to 180 deg works
    np.testing.assert_allclose(faoi_fn([10, 20, 50]), faoi_fn([170, 160, 130]))
    # Check some value consistency
    np.testing.assert_allclose(faoi_fn([10, 20, 50, 90]),
                               [0.597472, 0.856388, 1., 1.])
    # Check values outside of acceptable range: should be zero
    np.testing.assert_allclose(faoi_fn([-10, 190]), [0., 0.])


def test_ts_aoi_methods(pvmodule_canadian):
    """Checks
    - can create aoi methods correctly
    - vf_aoi_integrand matrix makes sense and stays consistent
    - vf_aoi values stay consistent"""
    n_timestamps = 6  # using 5 timestamps
    n_points = 6  # using only 6 sections for the integral from 0 to 180 deg
    faoi_fn = faoi_fn_from_pvlib_sandia(pvmodule_canadian)
    # use same faoi fn for front and back
    aoi_methods = AOIMethods(faoi_fn, faoi_fn, n_integral_sections=n_points)
    aoi_methods.fit(n_timestamps)
    # Check that function was passed correctly
    assert callable(aoi_methods.faoi_fn_front)
    assert callable(aoi_methods.faoi_fn_back)
    assert aoi_methods.aoi_angles_low.shape == (n_timestamps, n_points)
    assert aoi_methods.aoi_angles_high.shape == (n_timestamps, n_points)
    assert aoi_methods.integrand_front.shape == (n_timestamps, n_points)
    assert aoi_methods.integrand_back.shape == (n_timestamps, n_points)

    # Create some dummy angle values
    low_angles = np.array([0., 2., 108., 72., 179., 0.])
    high_angles = np.array([31., 30., 144., 144., 180., 180.])

    # Check that integrand is calculated correctly
    faoi_integrand = aoi_methods._calculate_vfaoi_integrand(
        low_angles, high_angles)
    expected_integrand = \
        [[0.05056557, 0.18255583, 0., 0., 0., 0.],
         [0.05056557, 0., 0., 0., 0., 0.],
         [0., 0., 0., 0.25, 0.18255583, 0.],
         [0., 0., 0.25, 0.25, 0.18255583, 0.],
         [0., 0., 0., 0., 0., 0.05056557],
         [0.05056557, 0.18255583, 0.25, 0.25, 0.18255583, 0.05056557]]
    np.testing.assert_allclose(faoi_integrand, expected_integrand)

    # Check that faoi values calculated correctly
    vf_aoi = aoi_methods._calculate_vf_aoi_wedge_level(low_angles, high_angles)
    expected_vf_aoi = [0.2331214, 0.05056557, 0.43255583, 0.68255583,
                       0.05056557, 0.96624281]
    np.testing.assert_allclose(vf_aoi, expected_vf_aoi)


def test_sanity_check(pvmodule_canadian):
    """Sanity check: make sure than when faoi = 1 everywhere, the calculated
    view factor values make sense"""
    n_timestamps = 3  # using 5 timestamps
    n_points = 300  # using only 6 sections for the integral from 0 to 180 deg
    def faoi_fn(aoi_angles): return np.ones_like(aoi_angles)
    aoi_methods = AOIMethods(faoi_fn, faoi_fn, n_integral_sections=n_points)
    aoi_methods.fit(n_timestamps)
    # Create some dummy angle values
    low_angles = np.array([0., 90., 0.])
    high_angles = np.array([180., 180., 90.])

    # Check that faoi values calculated correctly
    vf_aoi = aoi_methods._calculate_vf_aoi_wedge_level(low_angles, high_angles)
    expected_vf_aoi = [1., 0.5, 0.5]
    np.testing.assert_allclose(vf_aoi, expected_vf_aoi)


def test_vf():
    """Make sure that view factor from infinitesimal strip
    to parallel infinite strip is calculated correctly"""
    # Input AOI angles
    aoi_1 = [0, 90, 45, 0]
    aoi_2 = [90, 0, 135, 10]
    # Calculate view factors and check values
    vf = AOIMethods._vf(aoi_1, aoi_2)
    expected_vf = [0.5, 0.5, 0.70710678, 0.00759612]
    np.testing.assert_allclose(vf, expected_vf, atol=0, rtol=1e-6)


def test_calculate_aoi_angles():
    """Make sure calculation of AOI angles is correct"""
    u_vector = np.array([[1, 2, 3], [0, 0, 0]])
    centroid = TsPointCoords(np.array([0.5, 1, 1]), np.array([0, 0, 0]))
    point = TsPointCoords(np.array([1, 0, 1]), np.array([0.5, 1, 5]))

    aoi_angles = AOIMethods._calculate_aoi_angles(u_vector, centroid, point)
    expected_aoi_angles = [45, 135, 90]
    np.testing.assert_allclose(aoi_angles, expected_aoi_angles)


def test_vf_aoi_pvrow_gnd_benchmark_no_obstruction():
    """Check that the NREL view factors are close to the truth for small
    segments, but further away when segments are large
    Assumption: no obstruction"""

    # Create vf ts methods
    vf_ts_methods = VFTsMethods()
    # Create aoi methods
    n_timestamps = 1
    n_points = 300  # using only 6 sections for the integral from 0 to 180 deg
    def faoi_fn(aoi_angles): return np.ones_like(aoi_angles)
    aoi_methods = AOIMethods(faoi_fn, faoi_fn, n_integral_sections=n_points)
    aoi_methods.fit(n_timestamps)

    # Get parameters
    n_pvrows = 3
    pvrow_idx = 0  # leftmost pv row
    side = 'back'
    segment_idx = 0  # top segment
    idx_gnd_surf = 0  # first shadow left

    # -- Check situation when the segments are small, and no obstruction at all
    discretization = {pvrow_idx: {'back': 10}}
    pvrow_surf, pvrow_idx, tilted_to_left, ts_pvrows, \
        gnd_surf, ts_length, _ = _get_vf_method_inputs(
            discretization, pvrow_idx, side, segment_idx, idx_gnd_surf)
    # --- There is no obstruction
    # Calculate using VFTsMethods
    vf1_vf, _ = vf_ts_methods.vf_pvrow_surf_to_gnd_surf_obstruction_hottel(
        pvrow_surf, pvrow_idx, n_pvrows, tilted_to_left, ts_pvrows,
        gnd_surf, ts_length, is_back=True, is_left=True)
    # Calculate using AOIMethods
    vf1_nrel = aoi_methods._vf_aoi_pvrow_surf_to_gnd_surf_obstruction(
        pvrow_surf, pvrow_idx, n_pvrows, tilted_to_left, ts_pvrows,
        gnd_surf, ts_length, is_back=True, is_left=True)

    # The segments are small, so there should be good agreement between methods
    np.testing.assert_allclose(vf1_vf, [0.29276516])
    np.testing.assert_allclose(vf1_nrel, [0.29674707])
    np.testing.assert_allclose(vf1_vf, vf1_nrel, atol=0, rtol=1.4e-2)

    # -- Check situation when the segments are large, and no obstruction at all
    discretization = {pvrow_idx: {'back': 1}}
    pvrow_surf, pvrow_idx, tilted_to_left, ts_pvrows, \
        gnd_surf, ts_length, _ = _get_vf_method_inputs(
            discretization, pvrow_idx, side, segment_idx, idx_gnd_surf)
    # --- There is no obstruction
    # Calculate using VFTsMethods
    vf1_vf, _ = vf_ts_methods.vf_pvrow_surf_to_gnd_surf_obstruction_hottel(
        pvrow_surf, pvrow_idx, n_pvrows, tilted_to_left, ts_pvrows,
        gnd_surf, ts_length, is_back=True, is_left=True)
    # Calculate using AOIMethods
    vf1_nrel = aoi_methods._vf_aoi_pvrow_surf_to_gnd_surf_obstruction(
        pvrow_surf, pvrow_idx, n_pvrows, tilted_to_left, ts_pvrows,
        gnd_surf, ts_length, is_back=True, is_left=True)

    # The segments are small, the view factors are different
    np.testing.assert_allclose(vf1_vf, [0.2898575])
    np.testing.assert_allclose(vf1_nrel, [0.30886451])
    np.testing.assert_allclose(vf1_vf, vf1_nrel, atol=0, rtol=6.2e-2)


def test_vf_aoi_pvrow_gnd_benchmark_with_obstruction():
    """Check that the NREL view factors are close to the truth for small
    segments, but further away when segments are large
    Assumption: the left pvrow is an obstruction

    Note: when there is obstruction, the interpretation of the differences
    is a little more difficult
    """

    # Create vf ts methods
    vf_ts_methods = VFTsMethods()
    # Create aoi methods
    n_timestamps = 1
    n_points = 300  # using only 6 sections for the integral from 0 to 180 deg
    def faoi_fn(aoi_angles): return np.ones_like(aoi_angles)
    aoi_methods = AOIMethods(faoi_fn, faoi_fn, n_integral_sections=n_points)
    aoi_methods.fit(n_timestamps)

    # Get parameters
    n_pvrows = 3
    pvrow_idx = 1  # center pv row
    side = 'back'
    segment_idx = 0  # top segment
    idx_gnd_surf = 0  # first shadow left

    # -- Check situation when the segments are small, and no obstruction at all
    discretization = {pvrow_idx: {'back': 10}}
    pvrow_surf, pvrow_idx, tilted_to_left, ts_pvrows, \
        gnd_surf, ts_length, _ = _get_vf_method_inputs(
            discretization, pvrow_idx, side, segment_idx, idx_gnd_surf,
            gcr=0.8)
    # --- There is no obstruction
    # Calculate using VFTsMethods
    vf1_vf, _ = vf_ts_methods.vf_pvrow_surf_to_gnd_surf_obstruction_hottel(
        pvrow_surf, pvrow_idx, n_pvrows, tilted_to_left, ts_pvrows,
        gnd_surf, ts_length, is_back=True, is_left=True)
    # Calculate using AOIMethods
    vf1_nrel = aoi_methods._vf_aoi_pvrow_surf_to_gnd_surf_obstruction(
        pvrow_surf, pvrow_idx, n_pvrows, tilted_to_left, ts_pvrows,
        gnd_surf, ts_length, is_back=True, is_left=True)

    # The segments are small, so there should be good agreement between methods
    # but there is an obstruction
    np.testing.assert_allclose(vf1_vf, [0.16756797])
    np.testing.assert_allclose(vf1_nrel, [0.17880681])
    np.testing.assert_allclose(vf1_vf, vf1_nrel, atol=0, rtol=6.3e-2)

    # -- Check situation when the segments are large, and no obstruction at all
    discretization = {pvrow_idx: {'back': 1}}
    pvrow_surf, pvrow_idx, tilted_to_left, ts_pvrows, \
        gnd_surf, ts_length, _ = _get_vf_method_inputs(
            discretization, pvrow_idx, side, segment_idx, idx_gnd_surf)
    # --- There is no obstruction
    # Calculate using VFTsMethods
    vf1_vf, _ = vf_ts_methods.vf_pvrow_surf_to_gnd_surf_obstruction_hottel(
        pvrow_surf, pvrow_idx, n_pvrows, tilted_to_left, ts_pvrows,
        gnd_surf, ts_length, is_back=True, is_left=True)
    # Calculate using AOIMethods
    # Since it uses the centroid, here the surface is not even considered
    # to be obstructed...
    vf1_nrel = aoi_methods._vf_aoi_pvrow_surf_to_gnd_surf_obstruction(
        pvrow_surf, pvrow_idx, n_pvrows, tilted_to_left, ts_pvrows,
        gnd_surf, ts_length, is_back=True, is_left=True)

    # The segments are small, the view factors are different
    # but somehow, due to the obstruction, they tend to agree more
    np.testing.assert_allclose(vf1_vf, [0.05269675])
    np.testing.assert_allclose(vf1_nrel, [0.05338835])
    np.testing.assert_allclose(vf1_vf, vf1_nrel, atol=0, rtol=1.3e-2)


def _get_vf_method_inputs(discretization, pvrow_idx, side, segment_idx,
                          idx_gnd_surf, gcr=0.4):
    """Helper function to be able to play with discretization and vf method
    inputs"""
    params = {
        'n_pvrows': 3,
        'pvrow_height': 1.5,
        'pvrow_width': 1.,
        'surface_tilt': 20.,
        'surface_azimuth': 180.,
        'gcr': gcr,
        'solar_zenith': 20.,
        'solar_azimuth': 90.,  # sun located in the east
        'axis_azimuth': 0.,  # axis of rotation towards North
        'rho_ground': 0.2,
        'rho_front_pvrow': 0.01,
        'rho_back_pvrow': 0.03,
    }
    # Update discretization scheme
    params.update({'cut': discretization})
    # pv array is tilted to right
    pvarray = OrderedPVArray.fit_from_dict_of_scalars(params)
    tilted_to_left = pvarray.rotation_vec > 0

    # left pvrow & back side & left surface
    left_gnd_surfaces = (
        pvarray.ts_ground.ts_surfaces_side_of_cut_point('left',
                                                        pvrow_idx))
    gnd_surf = left_gnd_surfaces[idx_gnd_surf]
    pvrow_side = getattr(pvarray.ts_pvrows[pvrow_idx], side)
    pvrow_surf = pvrow_side.all_ts_surfaces[segment_idx]
    ts_length = pvrow_surf.length

    return (pvrow_surf, pvrow_idx, tilted_to_left, pvarray.ts_pvrows,
            gnd_surf, ts_length, pvarray)


def test_vf_aoi_pvrow_to_sky(params):
    """Check aoi methods for vf_aoi from pvrow to sky"""
    # Create aoi methods
    n_timestamps = 1
    n_points = 300  # using only 6 sections for the integral from 0 to 180 deg
    def faoi_fn(aoi_angles): return np.ones_like(aoi_angles)
    aoi_methods = AOIMethods(faoi_fn, faoi_fn, n_integral_sections=n_points)
    aoi_methods.fit(n_timestamps)

    # Create pv array
    params.update({'cut': {0: {'front': 3}, 1: {'back': 2}}})
    pvarray = OrderedPVArray.fit_from_dict_of_scalars(params)

    # Initialize vf_aoi_matrix
    n_ts_surfaces = pvarray.n_ts_surfaces
    n_steps = n_timestamps
    vf_aoi_matrix = np.zeros((n_ts_surfaces + 1, n_ts_surfaces + 1, n_steps),
                             dtype=float)

    # Calculate pvrow to sky vf_aoi values
    tilted_to_left = pvarray.rotation_vec > 0
    aoi_methods.vf_aoi_pvrow_to_sky(
        pvarray.ts_pvrows, pvarray.ts_ground, tilted_to_left, vf_aoi_matrix)

    sky_column = np.squeeze(vf_aoi_matrix[:, -1, :])
    expected_sky_column = [0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0.,
                           0., 0., 0., 0.96679021, 0.,
                           0.95461805, 0., 0.93560691, 0., 0.03511176,
                           0., 0.95461805, 0., 0.02611579, 0.,
                           0.01841872, 0., 0.97552826, 0., 0.02134025,
                           0., 0.]
    np.testing.assert_array_almost_equal(sky_column, expected_sky_column,
                                         decimal=7)


def test_rho_from_faoi_fn(pvmodule_canadian):
    """Check that can correctly calculate rho from faoi function"""
    faoi_fn = faoi_fn_from_pvlib_sandia(pvmodule_canadian)
    n_points = 300
    # use same faoi fn for front and back
    aoi_methods = AOIMethods(faoi_fn, faoi_fn, n_integral_sections=n_points)

    # Should be identical for front and back
    rho_out = aoi_methods.rho_from_faoi_fn(True)
    np.testing.assert_allclose(rho_out, 0.02900688, atol=0, rtol=1e-6)
    rho_out = aoi_methods.rho_from_faoi_fn(False)
    np.testing.assert_allclose(rho_out, 0.02900688, atol=0, rtol=1e-6)
